import asyncio
import collections
import logging
import numpy
import pathlib
import sounddevice
import sys
import tflite_runtime.interpreter
import time

logging.basicConfig (
	level = logging.INFO, # DEBUG, INFO, WARNING, ERROR, CRITICAL
	handlers = [
		logging.StreamHandler()
	],
	encoding = 'utf-8'
)

logger = logging.getLogger(__name__)

Detection = collections.namedtuple('Detection', ['index', 'english_name', 'latin_name', 'confidence'])

class Listener:

	def __init__ (self, match_threshold:float=0.75, silence_threshold_dbfs:float=None, callback_function:object=None):

		"""
		match_threshold: The lowest confidence level we want to see matches for (between 0 and 1).
		silence_threshold_dbfs: If defined, we will check whether there is any signal in the sampled audio which exceeds this level, and if not, it will not be passed to the BirdNET model (a value in dBFS e.g. -60).
		callback_function: This function will be called any time one or more bird is detected in an audio chunk. It should accept a list of Detection objects as its argument.
		"""

		buffer_size_s = 1
		window_size_s = 3.0
		overlap_size_s = 0.5

		module_dir = str(pathlib.Path(__file__).parent)

		tflite_file_path = module_dir + '/birdnet/BirdNET_GLOBAL_6K_V2.4_Model_FP16.tflite'
		label_file_path = module_dir + '/birdnet/labels_en.txt'
		# non_bird_label_file_path = module_dir + '/labels_not_birds.txt'

		self.sample_rate_hz = 48000 # The BirdNET model is trained with 48kHz files

		self.match_threshold = match_threshold
		self.silence_threshold_dbfs = silence_threshold_dbfs
		self.callback_function = callback_function

		# No more configurable items

		self.step_size_s = window_size_s - overlap_size_s

		self.window_samples = int(window_size_s * self.sample_rate_hz)
		self.step_samples = int(self.step_size_s * self.sample_rate_hz)

		self.buffer_samples = int(buffer_size_s * self.sample_rate_hz)

		if not self._load_model(tflite_file_path) or not self._load_labels(label_file_path):

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

	def _load_labels (self, file_path:str):

		logger.info('Loading labels')

		try:

			self.labels = {}

			with open(file_path, 'r', encoding='utf-8') as f:

				for idx, line in enumerate(f):

					latin_name, english_name = line.strip().split('_', 1)
					self.labels[idx] = (latin_name, english_name)

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

	async def birdcatcher (self, chunk:numpy.ndarray):

		start_time = time.perf_counter()

		input_data = numpy.expand_dims(chunk, axis=0)

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

			latin, english = self.labels.get(index, ('Unknown', 'Unknown'))
			detections.append(Detection(index, english, latin, confidences[index]))

		if self.callback_function:
			self.callback_function(detections)

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
			callback = callback
		)

		stream.start()

		logger.info('Capturing audio...')

		samples_since_last_window = 0

		try:

			analysis_buffer = numpy.empty(self.window_samples, dtype=numpy.float32)
			audio_buffer = collections.deque(maxlen = self.window_samples)

			while True:

				chunk = await queue.get()
				audio_buffer.extend(chunk)
				samples_since_last_window += len(chunk)

				if samples_since_last_window < self.step_samples or len(audio_buffer) != self.window_samples:
					continue

				analysis_buffer[:] = numpy.array(audio_buffer, dtype=numpy.float32) / 32768.0

				samples_since_last_window = 0

				peak_dbfs = self.get_dbfs_peak(analysis_buffer)

				# logger.debug('Peak dBFS: %0.2f', peak_dbfs)

				if self.silence_threshold_dbfs and (peak_dbfs < self.silence_threshold_dbfs):
					# Peak audio is below threshold
					logger.debug('Ignoring silent chunk')
					continue

				asyncio.create_task(self.birdcatcher(analysis_buffer.copy()))

		except KeyboardInterrupt:

			asyncio.gather(*asyncio.all_tasks(), return_exceptions=True)

		finally:

			stream.stop()
			stream.close()
