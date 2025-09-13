import asyncio
import os

from birdnetpy.core import Listener, Detection

def example_callback (detections:list[Detection], wav_file_path:str=None):

	for detection in detections:

		print(detection)
		print('We detected a %s with a confidence level of %0.2f%%' % (detection.english_name, 100 * detection.confidence))

	if wav_file_path and os.path.isfile(wav_file_path):

		# The user is responsible for managing the saved WAV files.
		# In this example, we'll just remove the file to prevent the disk from filling up.

		os.remove(wav_file_path)

async def main ():

	listener = Listener(match_threshold=0.8, silence_threshold_dbfs=-60.0, callback_function=example_callback, audio_output_dir='/tmp')
	await listener.listen()

if __name__ == '__main__':

	asyncio.run(main())
