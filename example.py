import asyncio

from birdnetpy.core import Listener, Detection

def example_callback (detections:list[Detection]):

	for detection in detections:
		print(detection)
		print('We detected a %s with a confidence level of %0.2f%%' % (detection.english_name, 100 * detection.confidence))

async def main ():

	listener = Listener(match_threshold=0.8, silence_threshold_dbfs=-60.0, callback_function=example_callback)
	await listener.listen()

if __name__ == '__main__':

	asyncio.run(main())
