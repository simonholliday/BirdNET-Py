import warnings

warnings.filterwarnings("ignore", message='The value of the smallest subnormal*')

import asyncio
import collections
import importlib
import librosa
import logging
import numba
import numpy
import os
import sounddevice
import soundfile
import tflite_runtime.interpreter
import threading
import time
import typing
import wave

logging.basicConfig (
	level = logging.INFO, # DEBUG, INFO, WARNING, ERROR, CRITICAL
	handlers = [
		logging.StreamHandler()
	],
	encoding = 'utf-8'
)

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

	@staticmethod
	@numba.njit
	def get_dbfs_rms (chunk:numpy.ndarray) -> float:
		"""Return the RMS level of the chunk in dBFS."""

		if len(chunk) == 0:
			return 0.0

		rms = numpy.sqrt(numpy.mean(chunk**2))
		dbfs = 20 * numpy.log10(rms + 1e-10)

		return dbfs

	def __init__ (self, match_threshold:float = 0.75, silence_threshold_dbfs:typing.Optional[float] = None, callback_function:typing.Optional[typing.Callable] = None, audio_output_dir:typing.Optional[str] = None, exclude_label_file_path:typing.Optional[str] = None, model_file_path:typing.Optional[str] = None) -> None:

		"""
		match_threshold: The lowest confidence level we want to see matches for (between 0 and 1).
		silence_threshold_dbfs: If defined, we will check whether there is any signal in the sampled audio which exceeds this level, and if not, it will not be passed to the BirdNET model (a value in dBFS e.g. -60).
		callback_function: This function will be called any time one or more bird is detected in an audio chunk. It should accept three arguments: a list of Detection objects, a wav file path (or None), and a timecode in seconds (or None for live streaming).
		audio_output_dir: An optional directory to store the analyzed audio when there are detections. Omit or specify `None` if you don't want to keep the audio.
		exclude_label_file_path: An optional path to a list of labels which will be excluded from detection.
		model_file_path: An optional path to a TFLite model file. Three variants are bundled: FP32 (highest accuracy, ~50MB), FP16 (default, good balance, ~25MB), and INT8 (smallest, ~39MB). If omitted, the bundled FP16 model is used.
		"""

		self.lock = threading.Lock()

		buffer_size_s = 1 # Optimal for Raspberry Pi Zero 2 without "input overflow" errors.
		window_overlap_size_s = 0.5

		self.window_size_s = 3.0 # Required for BirdNET
		self.sample_rate_hz = 48000 # The BirdNET model is trained with 48kHz files

		self.match_threshold = match_threshold
		self.silence_threshold_dbfs = silence_threshold_dbfs
		self.callback_function = callback_function

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

		# Load model and labels

		if model_file_path:

			if not os.path.isfile(model_file_path):
				raise FileNotFoundError('Model file does not exist: %s' % (model_file_path))

			tflite_file_path = model_file_path

		else:

			tflite_file_path = str(importlib.resources.files("birdnetpy.birdnet") / "BirdNET_GLOBAL_6K_V2.4_Model_FP16.tflite")

		label_file_path = str(importlib.resources.files("birdnetpy.birdnet") / "labels_en.txt")
		non_bird_label_file_path = str(importlib.resources.files("birdnetpy") / "labels_non_birds.txt")

		self._load_model(tflite_file_path)
		self._import_labels(label_file_path, non_bird_label_file_path, exclude_label_file_path)

		# Validate that the model output dimension matches the label count

		output_classes = self.output_details[0]['shape'][1]

		if output_classes != self.num_source_labels:
			logger.warning('Model output size (%d) does not match label count (%d)' % (output_classes, self.num_source_labels))

	def _load_model (self, file_path:str) -> None:

		"""Load the TFLite model from the given file path."""

		logger.info('Loading model')

		self.interpreter = tflite_runtime.interpreter.Interpreter(model_path=file_path, experimental_delegates=None)
		self.interpreter.allocate_tensors()
		self.input_details = self.interpreter.get_input_details()
		self.output_details = self.interpreter.get_output_details()

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

		logger.info('Importing labels')

		_, model_labels = self._load_label_file(label_file_path)

		self.num_source_labels = len(model_labels)
		logger.info('Label file contains %d item%s' % (self.num_source_labels, '' if self.num_source_labels == 1 else 's'))

		non_bird_labels, _ = self._load_label_file(non_bird_label_file_path)
		exclude_labels, _ = self._load_label_file(exclude_label_file_path)

		num_exclude_labels = len(exclude_labels)

		if num_exclude_labels:
			logger.info('Exclusion filter contains %d item%s' % (num_exclude_labels, '' if num_exclude_labels == 1 else 's'))

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
			logger.info('Exclusion filter contains %d invalid item%s, which were not found in the source labels file' % (num_exclude_labels, '' if num_exclude_labels == 1 else 's'))

		num_imported_labels = len(self.model_labels)

		logger.info('Imported %s label%s' % (num_imported_labels, '' if num_imported_labels == 1 else 's'))

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

	def birdcatcher (self, analysis_buffer:numpy.ndarray, timecode_s:typing.Optional[float] = None) -> None:

		"""
		Run inference on the analysis buffer and invoke the callback with any detections.
		If timecode_s is provided (file mode), it is used for dedup and passed to the callback.
		If None (live mode), wall-clock time is used and timecode_s is passed as None.
		"""

		with self.lock:

			start_time = time.perf_counter()

			current_timestamp = timecode_s if timecode_s is not None else time.time()

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

	def analyze_file (self, file_path:str) -> None:

		"""
		Analyze a pre-recorded audio file by streaming it in chunks. The audio is resampled
		to 48kHz if necessary and processed using the same sliding window as live streaming.
		Detections are passed to the callback with a timecode_s value indicating the midpoint
		of the analysis window (in seconds).

		Supports any audio format handled by soundfile (WAV, FLAC, OGG, etc.) and via
		librosa for formats requiring decoding (MP3, etc.).
		"""

		if not os.path.isfile(file_path):
			raise FileNotFoundError('Audio file does not exist: %s' % (file_path))

		logger.info('Analyzing audio file: %s' % (file_path))

		# Reset dedup timestamps for file analysis
		self.last_detection_timestamps.clear()

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

		logger.info('File analysis complete')

	def _analyze_file_via_librosa (self, file_path:str) -> None:

		"""
		Fallback for audio formats that soundfile cannot open directly (e.g. MP3).
		Uses librosa.stream to process the file in chunks without loading it entirely.
		"""

		logger.info('Using librosa decoder for: %s' % (file_path))

		# Read the source sample rate once before streaming
		info = soundfile.info(file_path)
		source_rate = info.samplerate
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

		def callback (indata, frames, time, status):

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
