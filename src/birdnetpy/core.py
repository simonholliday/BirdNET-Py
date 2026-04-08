import os
import warnings

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
warnings.filterwarnings("ignore", message='The value of the smallest subnormal*')

import asyncio
import collections
import datetime
import importlib
import librosa
import logging
import numba
import numpy
import sounddevice
import soundfile
import ai_edge_litert.interpreter
import threading
import time
import typing
import wave

logger = logging.getLogger(__name__)

Detection = collections.namedtuple('Detection', ['index', 'english_name', 'latin_name', 'is_bird', 'is_human', 'confidence'])

class Listener:

	@staticmethod
	@numba.njit
	def _custom_sigmoid (x, sensitivity=1.0):
		"""Apply a sigmoid function with adjustable sensitivity to convert logits to probabilities."""

		return 1 / (1 + numpy.exp(-sensitivity * x))

	@staticmethod
	@numba.njit
	def get_dbfs_peak (chunk:numpy.ndarray) -> float:
		"""Return the peak amplitude of the chunk in dBFS."""

		if len(chunk) == 0:
			return 0.0

		peak = numpy.max(numpy.abs(chunk))
		dbfs = 20 * numpy.log10(peak + 1e-10)

		return dbfs

	SUPPORTED_ANNOTATIONS = {'audacity', 'csv', 'raven', 'reaper', 'srt'}

	def __init__ (self, match_threshold:float = 0.75, silence_threshold_dbfs:typing.Optional[float] = None, callback_function:typing.Optional[typing.Callable] = None, audio_output_dir:typing.Optional[str] = None, exclude_label_file_path:typing.Optional[str] = None, model_file_path:typing.Optional[str] = None, latitude:typing.Optional[float] = None, longitude:typing.Optional[float] = None, species_threshold:float = 0.03, annotate:typing.Optional[str] = None) -> None:

		"""
		match_threshold: The lowest confidence level we want to see matches for (between 0 and 1).
		silence_threshold_dbfs: If defined, we will check whether there is any signal in the sampled audio which exceeds this level, and if not, it will not be passed to the BirdNET model (a value in dBFS e.g. -60).
		callback_function: This function will be called any time one or more bird is detected in an audio chunk. It should accept three arguments: a list of Detection objects, a wav file path (or None), and a timecode in seconds (or None for live streaming).
		audio_output_dir: An optional directory to store the analyzed audio when there are detections. Omit or specify `None` if you don't want to keep the audio.
		exclude_label_file_path: An optional path to a list of labels which will be excluded from detection.
		model_file_path: An optional path to a TFLite model file. Three variants are bundled: FP32 (highest accuracy, ~50MB), FP16 (default, good balance, ~25MB), and INT8 (smallest, ~39MB). If omitted, the bundled FP16 model is used.
		latitude: Optional latitude for geographic species filtering. Must be provided together with longitude.
		longitude: Optional longitude for geographic species filtering. Must be provided together with latitude.
		species_threshold: Minimum probability from the geographic model to include a species (default 0.03). Only used when latitude and longitude are provided.
		annotate: Optional annotation format for file analysis. Supported: 'audacity' (creates a label track .txt file alongside the audio file). More formats may be added in future.
		"""

		if annotate and annotate not in self.SUPPORTED_ANNOTATIONS:
			raise ValueError('Unsupported annotation format: %s (supported: %s)' % (annotate, ', '.join(sorted(self.SUPPORTED_ANNOTATIONS))))

		self.annotate = annotate

		self.lock = threading.Lock()

		buffer_size_s = 1 # Optimal for Raspberry Pi Zero 2 without "input overflow" errors.
		window_overlap_size_s = 0.5

		self.window_size_s = 3.0 # Required for BirdNET
		self.sample_rate_hz = 48000 # The BirdNET model is trained with 48kHz files

		self.match_threshold = match_threshold
		self.silence_threshold_dbfs = silence_threshold_dbfs
		self.callback_function = callback_function

		# Geographic filtering

		if (latitude is None) != (longitude is None):
			raise ValueError('Both latitude and longitude must be provided, or neither')

		self.latitude = latitude
		self.longitude = longitude
		self.species_threshold = species_threshold
		self.geo_species:typing.Optional[typing.Set[int]] = None
		self._geo_last_refreshed:float = 0.0
		self._geo_week:int = 0

		self.audio_output_dir = None

		if audio_output_dir:

			audio_output_dir = audio_output_dir.rstrip('/\\')

			if not os.path.isdir(audio_output_dir):
				raise FileNotFoundError('Audio output directory does not exist: %s' % (audio_output_dir))

			if not os.access(audio_output_dir, os.W_OK):
				raise PermissionError('Audio output directory is not writeable: %s' % (audio_output_dir))

			self.audio_output_dir = audio_output_dir

		self.step_size_s = self.window_size_s - window_overlap_size_s

		self.window_samples = int(self.window_size_s * self.sample_rate_hz)
		self.step_samples = int(self.step_size_s * self.sample_rate_hz)

		self.buffer_samples = int(buffer_size_s * self.sample_rate_hz)

		# Keep a note of the last detection timestamp for each species
		self.last_detection_timestamps:typing.Dict[int, float] = {}

		# Resolve audio model path

		if model_file_path:

			if not os.path.isfile(model_file_path):
				raise FileNotFoundError('Model file does not exist: %s' % (model_file_path))

			self._model_file_path = model_file_path

		else:

			self._model_file_path = str(importlib.resources.files("birdnetpy.birdnet") / "BirdNET_GLOBAL_6K_V2.4_Model_FP16.tflite")

		# Import labels first (no model needed)

		label_file_path = str(importlib.resources.files("birdnetpy.birdnet") / "labels_en.txt")
		non_bird_label_file_path = str(importlib.resources.files("birdnetpy") / "labels_non_birds.txt")

		self._import_labels(label_file_path, non_bird_label_file_path, exclude_label_file_path)

		# Geographic filtering is deferred to first use in birdcatcher (live) or analyze_file (file).
		# This avoids loading the geo model at init only to reload it with a different week later.

		if self.latitude is None:
			logger.warning('No latitude/longitude provided — geographic filtering is disabled, all species are candidates')

		# Load the audio model

		logger.info('Using model: %s' % (os.path.basename(self._model_file_path)))
		self._load_model(self._model_file_path)

		# Validate that the model output dimension matches the label count

		output_classes = self.output_details[0]['shape'][1]

		if output_classes != self.num_source_labels:
			logger.warning('Model output size (%d) does not match label count (%d)' % (output_classes, self.num_source_labels))

	def _load_model (self, file_path:str) -> None:

		"""Load the TFLite model from the given file path."""

		logger.debug('Loading model')

		self.interpreter = ai_edge_litert.interpreter.Interpreter(model_path=file_path, experimental_delegates=None)
		self.interpreter.allocate_tensors()
		self.input_details = self.interpreter.get_input_details()
		self.output_details = self.interpreter.get_output_details()

	@staticmethod
	def _date_to_week (d:datetime.date) -> int:
		"""Convert a date to BirdNET week number (1–48)."""

		return min(48, max(1, int(d.timetuple().tm_yday / 7.625) + 1))

	def _unload_model (self) -> None:
		"""Unload the current audio model to free memory."""

		if hasattr(self, 'interpreter'):
			del self.interpreter
			del self.input_details
			del self.output_details

	def _refresh_geo_filter (self, week:int) -> None:

		"""
		Generate a geographic species filter using the MData model.
		Only one TFLite model is kept in memory at a time: the audio model is unloaded
		before the MData model is loaded, and reloaded afterwards.
		"""

		# Unload the audio model if it is currently loaded

		self._unload_model()

		# Load the MData model

		geo_model_path = str(importlib.resources.files("birdnetpy.birdnet") / "BirdNET_GLOBAL_6K_V2.4_MData_Model_V2_FP16.tflite")

		lat = typing.cast(float, self.latitude)
		lon = typing.cast(float, self.longitude)

		logger.info('Loading geographic model for lat=%.2f, lon=%.2f, week=%d' % (lat, lon, week))

		geo_interpreter = ai_edge_litert.interpreter.Interpreter(model_path=geo_model_path, experimental_delegates=None)
		geo_interpreter.allocate_tensors()

		geo_input = geo_interpreter.get_input_details()[0]
		geo_output = geo_interpreter.get_output_details()[0]

		# Run inference

		sample = numpy.array([[lat, lon, week]], dtype='float32')
		geo_interpreter.set_tensor(geo_input['index'], sample)
		geo_interpreter.invoke()

		probabilities = geo_interpreter.get_tensor(geo_output['index'])[0]

		# Build the set of allowed label indices

		self.geo_species = set()

		for index in range(len(probabilities)):

			if probabilities[index] >= self.species_threshold and index in self.model_labels:
				self.geo_species.add(index)

		if week == -1:
			logger.info('Geographic filter: %d species for this location (year-round)' % (len(self.geo_species)))
		else:
			logger.info('Geographic filter: %d species for this location and time of year' % (len(self.geo_species)))

		# Unload the MData model

		del geo_interpreter

		self._geo_week = week
		self._geo_last_refreshed = time.time()

	def _load_label_file (self, label_file_path:typing.Optional[str] = None) -> typing.Tuple[typing.Set[str], typing.Dict[int, str]]:

		"""
		Load items from label_file_path into a dict with each item's row number as its index, and a de-duplicated set.
		Comment lines (starting with #) and blank lines are skipped without incrementing the index.
		"""

		labels_set:typing.Set[str] = set()
		labels_dict:typing.Dict[int, str] = {}

		if label_file_path is None:
			return labels_set, labels_dict

		with open(label_file_path, 'r', encoding='utf-8') as f:

			index = 0

			for line in f:

				if line.startswith('#') or not line.strip():
					continue

				label = line.strip()

				if label in labels_set:
					logger.warning('Duplicate label "%s" found in %s' % (label, label_file_path))

				labels_dict[index] = label
				labels_set.add(label)
				index += 1

		return labels_set, labels_dict

	def _import_labels (self, label_file_path:str, non_bird_label_file_path:typing.Optional[str] = None, exclude_label_file_path:typing.Optional[str] = None) -> None:

		"""
		Import the label file. If exclude_label_file_path is specified, any items contained in that file will be excluded.
		Human entries and those which are not birds are flagged.
		"""

		logger.debug('Importing labels')

		_, model_labels = self._load_label_file(label_file_path)

		self.num_source_labels = len(model_labels)
		logger.debug('Label file contains %d item%s' % (self.num_source_labels, '' if self.num_source_labels == 1 else 's'))

		non_bird_labels, _ = self._load_label_file(non_bird_label_file_path)
		exclude_labels, _ = self._load_label_file(exclude_label_file_path)

		num_exclude_labels = len(exclude_labels)

		if num_exclude_labels:
			logger.debug('Exclusion filter contains %d item%s' % (num_exclude_labels, '' if num_exclude_labels == 1 else 's'))

		self.model_labels:typing.Dict[int, typing.Tuple[str, str, bool, bool]] = {}

		for index, model_label in model_labels.items():

			if len(exclude_labels):

				if model_label in exclude_labels:
					exclude_labels.remove(model_label)
					continue

			latin_name, english_name = model_label.split('_', 1)

			is_bird = model_label not in non_bird_labels
			is_human = english_name.startswith('Human')

			self.model_labels[index] = (latin_name, english_name, is_bird, is_human)

		# If all of the exclude_labels were valid, we should have none left

		num_exclude_labels = len(exclude_labels)

		if num_exclude_labels:
			logger.debug('Exclusion filter contains %d invalid item%s, which were not found in the source labels file' % (num_exclude_labels, '' if num_exclude_labels == 1 else 's'))

		num_imported_labels = len(self.model_labels)

		logger.debug('Imported %s label%s' % (num_imported_labels, '' if num_imported_labels == 1 else 's'))

	def _save_wav (self, file_path:str, analysis_buffer:numpy.ndarray, samplerate:int = 48000) -> None:

		"""Save the analysis buffer as a 16-bit PCM WAV file."""

		# Convert float32 [-1.0, 1.0] to int16
		audio_int16 = numpy.clip(analysis_buffer * 32767, -32768, 32767).astype('<i2')

		logger.debug('Writing audio file %s' % (file_path))

		with wave.open(file_path, 'wb') as wf:

			wf.setnchannels(1)
			wf.setsampwidth(2) # 16-bit PCM
			wf.setframerate(samplerate)

			wf.writeframes(audio_int16.tobytes())

	def _open_annotation_file (self, audio_file_path:str) -> typing.Optional[typing.IO[str]]:

		"""Open an annotation file for streaming writes. Returns the file handle, or None if annotation is not enabled."""

		if not self.annotate:
			return None

		base, _ = os.path.splitext(audio_file_path)

		extensions = {
			'audacity': '.audacity-labels.txt',
			'csv': '.birdnetpy.csv',
			'raven': '.raven-selections.txt',
			'reaper': '.reaper-markers.csv',
			'srt': '.srt',
		}

		self._annotation_file_path = base + extensions[self.annotate]
		self._annotation_counter = 0

		f = open(self._annotation_file_path, 'w', encoding='utf-8')

		# Write headers for formats that need them

		if self.annotate == 'csv':
			f.write('start_s,end_s,english_name,latin_name,confidence\n')

		elif self.annotate == 'raven':
			f.write('Selection\tView\tChannel\tBegin Time (s)\tEnd Time (s)\tLow Freq (Hz)\tHigh Freq (Hz)\tAnnotation\n')

		elif self.annotate == 'reaper':
			f.write('#,Name,Start,End,Length,Color\n')

		return f

	@staticmethod
	def _format_reaper_time (seconds:float) -> str:
		"""Format seconds as H:MM:SS.mmm for Reaper CSV."""

		h = int(seconds // 3600)
		m = int((seconds % 3600) // 60)
		s = seconds % 60

		return '%d:%02d:%06.3f' % (h, m, s)

	@staticmethod
	def _format_srt_time (seconds:float) -> str:
		"""Format seconds as HH:MM:SS,mmm for SRT subtitles."""

		h = int(seconds // 3600)
		m = int((seconds % 3600) // 60)
		s = int(seconds % 60)
		ms = int((seconds % 1) * 1000)

		return '%02d:%02d:%02d,%03d' % (h, m, s, ms)

	def _write_annotation (self, annotation_file:typing.IO[str], detections:typing.List[Detection], timecode_s:float) -> None:

		"""Write detections to the annotation file in the configured format. Flushes after each write."""

		half_window = self.window_size_s / 2
		window_start = max(0.0, timecode_s - half_window)
		window_end = timecode_s + half_window

		if self.annotate == 'audacity':

			for detection in detections:

				annotation_file.write('%f\t%f\t%s (%.0f%%)\n' % (window_start, window_end, detection.english_name, 100 * detection.confidence))

		elif self.annotate == 'csv':

			for detection in detections:

				annotation_file.write('%f,%f,%s,%s,%.2f\n' % (window_start, window_end, detection.english_name, detection.latin_name, detection.confidence))

		elif self.annotate == 'raven':

			for detection in detections:

				self._annotation_counter += 1
				annotation_file.write('%d\tSpectrogram 1\t1\t%f\t%f\t0\t%d\t%s (%.0f%%)\n' % (self._annotation_counter, window_start, window_end, self.sample_rate_hz // 2, detection.english_name, 100 * detection.confidence))

		elif self.annotate == 'reaper':

			start_str = self._format_reaper_time(window_start)
			end_str = self._format_reaper_time(window_end)
			length_str = self._format_reaper_time(self.window_size_s)

			for detection in detections:

				self._annotation_counter += 1
				annotation_file.write('R%d,%s (%.0f%%),%s,%s,%s,\n' % (self._annotation_counter, detection.english_name, 100 * detection.confidence, start_str, end_str, length_str))

		elif self.annotate == 'srt':

			start_str = self._format_srt_time(window_start)
			end_str = self._format_srt_time(window_end)

			for detection in detections:

				self._annotation_counter += 1
				annotation_file.write('%d\n%s --> %s\n%s (%.0f%%)\n\n' % (self._annotation_counter, start_str, end_str, detection.english_name, 100 * detection.confidence))

		annotation_file.flush()

	def birdcatcher (self, analysis_buffer:numpy.ndarray, timecode_s:typing.Optional[float] = None) -> None:

		"""
		Run inference on the analysis buffer and invoke the callback with any detections.
		If timecode_s is provided (file mode), it is used for dedup and passed to the callback.
		If None (live mode), wall-clock time is used and timecode_s is passed as None.
		"""

		with self.lock:

			start_time = time.perf_counter()

			current_timestamp = timecode_s if timecode_s is not None else time.time()

			# Refresh the geographic filter every 12 hours for long-running sessions
			if self.latitude is not None and (current_timestamp - self._geo_last_refreshed) > 43200:

				self._refresh_geo_filter(self._date_to_week(datetime.date.today()))
				self._load_model(self._model_file_path)

			# Avoid triggering the same identification on successive windows, since windows overlap.
			max_last_detection_timestamp = current_timestamp - self.window_size_s

			input_data = numpy.expand_dims(analysis_buffer, axis=0)

			self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
			self.interpreter.invoke()

			output_data = self.interpreter.get_tensor(self.output_details[0]['index'])

			logits = numpy.squeeze(output_data)
			confidences = self._custom_sigmoid(logits)

			indices = numpy.where(confidences >= self.match_threshold)[0]

			if indices.size < 1:
				return

			detections = []

			for index in indices:

				if index not in self.model_labels:
					continue

				if self.geo_species is not None and index not in self.geo_species:
					continue

				if index in self.last_detection_timestamps and self.last_detection_timestamps[index] > max_last_detection_timestamp:
					logger.debug('Skipping duplicate detection: %s' % (self.model_labels[index][1]))
					continue

				self.last_detection_timestamps[index] = current_timestamp

				latin, english, is_bird, is_human = self.model_labels[index]
				detections.append(Detection(index, english, latin, is_bird, is_human, confidences[index]))

			if self.callback_function and detections:

				if self.audio_output_dir:

					wav_file_path = os.path.join(self.audio_output_dir, time.strftime('%Y%m%d-%H%M%S') + '.wav')
					self._save_wav(file_path=wav_file_path, analysis_buffer=analysis_buffer, samplerate=self.sample_rate_hz)

				else:

					wav_file_path = None

				self.callback_function(detections, wav_file_path, timecode_s)

			end_time = time.perf_counter()

			logger.debug('Analysis took %0.2f seconds' % (end_time-start_time))

	def _process_file_chunk (self, chunk:numpy.ndarray, analysis_buffer:numpy.ndarray, samples_filled:int, source_samples_read:int, source_rate:int) -> typing.Tuple[numpy.ndarray, int]:

		"""
		Append a chunk of resampled audio to the analysis buffer and run detection when full.
		Returns the updated (analysis_buffer, samples_filled).
		"""

		chunk_len = len(chunk)

		if chunk_len >= self.window_samples:

			# Chunk is larger than the window (unlikely but handle it)
			analysis_buffer = chunk[-self.window_samples:].copy()
			samples_filled = self.window_samples

		elif samples_filled + chunk_len >= self.window_samples:

			# Shift and append
			analysis_buffer = numpy.roll(analysis_buffer, -chunk_len)
			analysis_buffer[-chunk_len:] = chunk
			samples_filled = self.window_samples

		else:

			# Still filling the initial window
			analysis_buffer[samples_filled:samples_filled + chunk_len] = chunk
			samples_filled += chunk_len
			return analysis_buffer, samples_filled

		# Timecode is the midpoint of the analysis window, since we can't know
		# where in the 3-second window the call occurred.
		# source_samples_read tracks samples at the original file rate.
		source_position_s = source_samples_read / source_rate
		timecode_s = source_position_s - (self.window_size_s / 2)

		if self.silence_threshold_dbfs:

			peak_dbfs = self.get_dbfs_peak(analysis_buffer)

			if peak_dbfs < self.silence_threshold_dbfs:
				logger.debug('Ignoring silent chunk at %.1fs' % (timecode_s))
				return analysis_buffer, samples_filled

		self.birdcatcher(analysis_buffer.copy(), timecode_s)

		return analysis_buffer, samples_filled

	def analyze_file (self, file_path:str, analysis_date:typing.Optional[datetime.date] = None) -> None:

		"""
		Analyze a pre-recorded audio file by streaming it in chunks. The audio is resampled
		to 48kHz if necessary and processed using the same sliding window as live streaming.
		Detections are passed to the callback with a timecode_s value indicating the midpoint
		of the analysis window (in seconds).

		Supports any audio format handled by soundfile (WAV, FLAC, OGG, etc.) and via
		librosa for formats requiring decoding (MP3, etc.).

		analysis_date: Optional date for geographic filtering. If lat/lon is set and no date
		is provided, year-round filtering is used (all species that could appear at that
		location in any season).
		"""

		if not os.path.isfile(file_path):
			raise FileNotFoundError('Audio file does not exist: %s' % (file_path))

		logger.info('Analyzing audio file: %s' % (file_path))

		# Reset dedup timestamps for file analysis
		self.last_detection_timestamps.clear()

		# Refresh geographic filter for the analysis date if lat/lon is set

		if self.latitude is not None:

			if analysis_date:
				week = self._date_to_week(analysis_date)
			else:
				logger.warning('No analysis_date provided — using year-round geographic filtering')
				week = -1

			if week != self._geo_week:

				self._refresh_geo_filter(week)
				self._load_model(self._model_file_path)

		# If annotation is enabled, open the file and wrap the callback to write detections as they arrive

		annotation_file = self._open_annotation_file(file_path)
		original_callback = self.callback_function

		if annotation_file:

			def annotating_callback (detections:typing.List[Detection], wav_file_path:typing.Optional[str] = None, timecode_s:typing.Optional[float] = None) -> None:

				self._write_annotation(annotation_file, detections, typing.cast(float, timecode_s))

				if original_callback:
					original_callback(detections, wav_file_path, timecode_s)

			self.callback_function = annotating_callback

		try:

			try:

				sf = soundfile.SoundFile(file_path)

			except soundfile.SoundFileError:

				# Fall back to librosa for formats soundfile can't open directly (e.g. MP3)
				self._analyze_file_via_librosa(file_path)
				return

			with sf:

				source_rate = sf.samplerate
				needs_resample = source_rate != self.sample_rate_hz

				logger.info('File: %dHz, %d channel%s, %.1f seconds%s' % (
					source_rate,
					sf.channels,
					'' if sf.channels == 1 else 's',
					sf.frames / source_rate,
					' (will resample to %dHz)' % (self.sample_rate_hz) if needs_resample else ''
				))

				# Read in chunks of step_samples (2.5s at target rate).
				# For resampling files, read proportionally more source samples.

				if needs_resample:
					source_chunk_size = int(self.step_samples * source_rate / self.sample_rate_hz)
				else:
					source_chunk_size = self.step_samples

				analysis_buffer = numpy.zeros(self.window_samples, dtype=numpy.float32)
				samples_filled = 0
				source_samples_read = 0

				while True:

					raw = sf.read(source_chunk_size, dtype='float32', always_2d=True)

					if len(raw) == 0:
						break

					# Mix to mono
					chunk = numpy.mean(raw, axis=1)

					# Track samples at the source rate for accurate timecodes
					source_samples_read += len(chunk)

					# Resample if necessary
					if needs_resample:
						chunk = librosa.resample(chunk, orig_sr=source_rate, target_sr=self.sample_rate_hz)

					analysis_buffer, samples_filled = self._process_file_chunk(
						chunk, analysis_buffer, samples_filled, source_samples_read, source_rate
					)

		finally:

			self.callback_function = original_callback

			if annotation_file:

				annotation_file.close()
				logger.info('Annotation file created: %s' % (self._annotation_file_path))

		logger.info('File analysis complete')

	def _analyze_file_via_librosa (self, file_path:str) -> None:

		"""
		Fallback for audio formats that soundfile cannot open directly (e.g. MP3).
		Uses librosa.stream to process the file in chunks without loading it entirely.
		"""

		logger.info('Using librosa decoder for: %s' % (file_path))

		# Get the native sample rate using librosa (not soundfile, which may not support this format)
		source_rate = int(librosa.get_samplerate(file_path))
		needs_resample = source_rate != self.sample_rate_hz

		analysis_buffer = numpy.zeros(self.window_samples, dtype=numpy.float32)
		samples_filled = 0
		source_samples_read = 0

		# librosa.stream returns blocks at the file's native sample rate
		for chunk in librosa.stream(file_path, block_length=1, frame_length=self.step_samples, hop_length=self.step_samples, mono=True):

			# Track samples at the source rate for accurate timecodes
			source_samples_read += len(chunk)

			if needs_resample:
				chunk = librosa.resample(chunk, orig_sr=source_rate, target_sr=self.sample_rate_hz)

			chunk = chunk.astype(numpy.float32)

			analysis_buffer, samples_filled = self._process_file_chunk(
				chunk, analysis_buffer, samples_filled, source_samples_read, source_rate
			)

		logger.info('File analysis complete')

	async def listen (self) -> None:

		"""Continuously capture audio and run detection on each analysis window."""

		loop = asyncio.get_running_loop()
		queue:asyncio.Queue[numpy.ndarray] = asyncio.Queue()

		def callback (indata, frames, stream_time, status):

			if status:
				logger.warning("Sounddevice status: %s" % (status))

			loop.call_soon_threadsafe(queue.put_nowait, indata[:, 0].copy())

		stream = sounddevice.InputStream(
			samplerate = self.sample_rate_hz,
			channels = 1,
			dtype = 'int16',
			blocksize = self.buffer_samples,
			latency = 'high',
			callback = callback
		)

		stream.start()

		logger.info('Streaming audio...')

		samples_since_last_window = 0

		try:

			analysis_buffer = numpy.zeros(self.window_samples, dtype=numpy.float32)

			while True:

				chunk_int16 = await queue.get()
				chunk_float32 = chunk_int16.astype(numpy.float32) / 32768.0

				analysis_buffer = numpy.roll(analysis_buffer, -self.buffer_samples)
				analysis_buffer[-self.buffer_samples:] = chunk_float32

				samples_since_last_window += self.buffer_samples

				if samples_since_last_window < self.step_samples:
					continue

				# Update counter for the next analysis window
				samples_since_last_window -= self.step_samples

				peak_dbfs = self.get_dbfs_peak(analysis_buffer)

				if self.silence_threshold_dbfs and (peak_dbfs < self.silence_threshold_dbfs):
					logger.debug('Ignoring silent chunk')
					continue

				await loop.run_in_executor(None, self.birdcatcher, analysis_buffer.copy())

		except KeyboardInterrupt:

			logger.info('Stopping')

		finally:

			stream.stop()
			stream.close()
