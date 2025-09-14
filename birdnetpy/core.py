import asyncio
import collections
import logging
import numpy
import os
import pathlib
import sounddevice
import sys
import tflite_runtime.interpreter
import threading
import time
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

	def __init__ (self, match_threshold:float=0.75, silence_threshold_dbfs:float=None, callback_function:object=None, audio_output_dir:str=None, exclude_label_file_path:str=None):

		"""
		match_threshold: The lowest confidence level we want to see matches for (between 0 and 1).
		silence_threshold_dbfs: If defined, we will check whether there is any signal in the sampled audio which exceeds this level, and if not, it will not be passed to the BirdNET model (a value in dBFS e.g. -60).
		callback_function: This function will be called any time one or more bird is detected in an audio chunk. It should accept a list of Detection objects and a wav file path as its arguments.
		audio_output_dir: An optional directory to store the analyzed audio when there are detections. Omit or specify `None` if you don't want to keep the audio.
		exclude_label_file_path: An optional path to a list of labels which will be excluded from detection.
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
				logger.critical('Audio output directory does not exist: %s' % (audio_output_dir))
				sys.exit(1)

			if not os.access(audio_output_dir, os.W_OK):
				logger.critical('Audio output directory is not writeable: %s' % (audio_output_dir))
				sys.exit(1)

			self.audio_output_dir = audio_output_dir

		self.step_size_s = self.window_size_s - window_overlap_size_s

		self.window_samples = int(self.window_size_s * self.sample_rate_hz)
		self.step_samples = int(self.step_size_s * self.sample_rate_hz)

		self.buffer_samples = int(buffer_size_s * self.sample_rate_hz)

		# Keep a note of the last detection timestamp for each species
		self.last_detection_timestamps = {}

		# Load model and labels

		module_dir = str(pathlib.Path(__file__).parent)

		tflite_file_path = module_dir + '/birdnet/BirdNET_GLOBAL_6K_V2.4_Model_FP16.tflite'
		label_file_path = module_dir + '/birdnet/labels_en.txt'
		non_bird_label_file_path = module_dir + '/labels_non_birds.txt'

		if not self._load_model(tflite_file_path) or not self._import_labels(label_file_path, non_bird_label_file_path, exclude_label_file_path):

			logger.critical('Setup failed')
			sys.exit(1)

	def _load_model (self, file_path:str):

		logger.info('Loading model')

		try:

			self.interpreter = tflite_runtime.interpreter.Interpreter(model_path=file_path, experimental_delegates=None)
			self.interpreter.allocate_tensors()
			self.input_details = self.interpreter.get_input_details()
			self.output_details = self.interpreter.get_output_details()
			self.input_type = self.input_details[0]['dtype']

			return True

		except Exception as e:

			logger.critical('Something went wrong loading the model: %s' % (e))

		return False

	def _load_label_file (self, label_file_path:str=None) -> dict:

		"""
		Load items from label_file_path into a dict with each item's row number as its index, and a de-duplicated set
		Return (set, dict)
		"""

		try:

			labels_set = set()
			labels_dict = {}

			if label_file_path is None:
				return labels_set, labels_dict

			with open(label_file_path, 'r', encoding='utf-8') as f:

				for index, line in enumerate(f):

					if line.startswith('#'):
						continue

					label = line.strip()

					if label in labels_set:
						logger.warning('Duplicate label "%s" found in %s' % (label, label_file_path))

					labels_dict[index] = label
					labels_set.add(label)

			return labels_set, labels_dict

		except Exception as e:

			logger.error('Something went wrong loading label file: %s' % (label_file_path))

		return False

	def _import_labels (self, label_file_path:str, non_bird_label_file_path:str=None, exclude_label_file_path:str=None):

		"""
		Import the the label file. If exclude_label_file_path is specified, any items contained in that file will be excluded.
		Human entries and those which are not birds are flagged.
		"""

		logger.info('Importing labels')

		try:

			_, model_labels = self._load_label_file(label_file_path)

			num_model_labels = len(model_labels)
			logger.info('Label file contains %d item%s' % (num_model_labels, '' if num_model_labels == 1 else 's'))

			non_bird_labels, _ = self._load_label_file(non_bird_label_file_path)
			exclude_labels, _ = self._load_label_file(exclude_label_file_path)

			num_exclude_labels = len(exclude_labels)

			if num_exclude_labels:
				logger.info('Exclusion filter contains %d item%s' % (num_exclude_labels, '' if num_exclude_labels == 1 else 's'))

			self.model_labels = {}

			for index, model_label in model_labels.items():

				if len(exclude_labels):

					if model_label in exclude_labels:
						exclude_labels.remove(model_label)
						continue

				latin_name, english_name = model_label.split('_', 1)

				is_bird = model_label not in non_bird_labels
				is_human = model_label.startswith('Human ')

				self.model_labels[index] = (latin_name, english_name, is_bird, is_human)

			# If all of the exclude_labels were valid, we should have none left

			num_exclude_labels = len(exclude_labels)

			if num_exclude_labels:
				logger.info('Exclusion filter contains %d invalid item%s, which were not found in the source labels file' % (num_exclude_labels, '' if num_exclude_labels == 1 else 's'))

			num_imported_labels = len(self.model_labels)

			logger.info('Imported %s label%s' % (num_imported_labels, '' if num_imported_labels == 1 else 's'))

			return True

		except Exception as e:

			logger.critical('Something went wrong loading the labels: %s' % (e))

		return False

	def _custom_sigmoid (self, x, sensitivity=1.0):

		return 1 / (1 + numpy.exp(-sensitivity * x))

	def get_dbfs_peak (self, chunk:numpy.ndarray) -> float:

		if len(chunk) == 0:
			return 0.0

		peak = numpy.max(numpy.abs(chunk))
		dbfs = 20 * numpy.log10(peak + 1e-10)

		return dbfs

	def get_dbfs_rms (self, chunk:numpy.ndarray) -> float:

		if len(chunk) == 0:
			return 0.0

		rms = numpy.sqrt(numpy.mean(chunk**2))
		dbfs = 20 * numpy.log10(rms + 1e-10)

		return dbfs

	def save_wav (self, file_path:str, analysis_buffer:numpy.ndarray, samplerate:int=48000):

		# Convert float32 [-1.0, 1.0] to int16
		audio_int16 = numpy.clip(analysis_buffer * 32767, -32768, 32767).astype('<i2')

		logger.debug('Writing audio file %s' % (file_path))

		with wave.open(file_path, 'wb') as wf:

			wf.setnchannels(1)
			wf.setsampwidth(2) # 16-bit PCM
			wf.setframerate(samplerate)

			wf.writeframes(audio_int16.tobytes())

	def birdcatcher (self, analysis_buffer:numpy.ndarray):

		with self.lock:

			start_time = time.perf_counter()

			current_timestamp = time.time()

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
					logger.info('Skip')
					continue

				self.last_detection_timestamps[index] = current_timestamp

				latin, english, is_bird, is_human = self.model_labels[index]
				detections.append(Detection(index, english, latin, is_bird, is_human, confidences[index]))

			if self.callback_function:

				if self.audio_output_dir:

					wav_file_path = self.audio_output_dir + '/' + time.strftime('%Y%m%d-%H%M%S') + '.wav'
					self.save_wav(file_path=wav_file_path, analysis_buffer=analysis_buffer, samplerate=self.sample_rate_hz)

				else:

					wav_file_path = None

				self.callback_function(detections, wav_file_path)

			end_time = time.perf_counter()

			logger.debug('Analysis took %0.2f seconds' % (end_time-start_time))

	async def listen (self):

		loop = asyncio.get_running_loop()
		queue = asyncio.Queue()

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

				# logger.debug('Peak dBFS: %0.2f', peak_dbfs)

				if self.silence_threshold_dbfs and (peak_dbfs < self.silence_threshold_dbfs):
					logger.debug('Ignoring silent chunk')
					continue

				loop.run_in_executor(None, self.birdcatcher, analysis_buffer.copy())

		except KeyboardInterrupt:

			logger.info('Stopping')

		finally:

			stream.stop()
			stream.close()
