import asyncio
import logging
import os
import typing

logging.basicConfig(level=logging.INFO)

import birdnetpy.core

def on_detection (detections:typing.List[birdnetpy.core.Detection], wav_file_path:typing.Optional[str] = None, timecode_s:typing.Optional[float] = None) -> None:

	"""Called each time one or more species is detected."""

	for detection in detections:
		print('%s (%.0f%%)' % (detection.english_name, 100 * detection.confidence))

	if wav_file_path and os.path.isfile(wav_file_path):
		os.remove(wav_file_path)

async def main () -> None:

	"""Listen to live audio and print detections."""

	listener = birdnetpy.core.Listener(
		match_threshold = 0.8,
		silence_threshold_dbfs = -60.0,
		callback_function = on_detection,
		latitude = 51.454,   # Bristol, UK
		longitude = -2.598
	)

	await listener.listen()

if __name__ == '__main__':

	asyncio.run(main())
