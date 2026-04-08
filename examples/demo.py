import asyncio
import importlib
import os
import sys
import typing

import birdnetpy.core

def example_callback (detections:typing.List[birdnetpy.core.Detection], wav_file_path:typing.Optional[str] = None, timecode_s:typing.Optional[float] = None) -> None:

	"""
	This function will be called when items are detected.
	It is passed as an argument to the Listener() init below.
	"""

	for detection in detections:

		if timecode_s is not None:

			minutes, seconds = divmod(timecode_s, 60)
			print('[%02d:%05.2f] %s (%.0f%%)' % (int(minutes), seconds, detection.english_name, 100 * detection.confidence))

		else:

			print('%s (%.0f%%)' % (detection.english_name, 100 * detection.confidence))

	if wav_file_path and os.path.isfile(wav_file_path):

		# The user is responsible for managing the saved WAV files.
		# In this example, we'll just remove the file to prevent the disk from filling up.

		os.remove(wav_file_path)

async def main () -> None:

	"""Initialize a listener and start detecting."""

	non_uk_label_file_path = str(importlib.resources.files("birdnetpy") / "labels_filter_non_uk.txt")

	listener = birdnetpy.core.Listener(
		match_threshold = 0.8,
		silence_threshold_dbfs = -60.0,
		callback_function = example_callback,
		exclude_label_file_path = non_uk_label_file_path
	)

	# If an audio file is provided as an argument, analyze it. Otherwise, listen live.

	if len(sys.argv) > 1:

		listener.analyze_file(sys.argv[1])

	else:

		await listener.listen()

if __name__ == '__main__':

	asyncio.run(main())
