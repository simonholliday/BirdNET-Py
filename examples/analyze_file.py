import datetime
import logging
import sys
import typing

logging.basicConfig(level=logging.INFO)

import birdnetpy.core

def on_detection (detections:typing.List[birdnetpy.core.Detection], wav_file_path:typing.Optional[str] = None, timecode_s:typing.Optional[float] = None) -> None:

	"""Called each time one or more species is detected."""

	for detection in detections:

		minutes, seconds = divmod(typing.cast(float, timecode_s), 60)
		print('[%02d:%05.2f] %s (%.0f%%)' % (int(minutes), seconds, detection.english_name, 100 * detection.confidence))

def main () -> None:

	"""Analyze an audio file and print detections with timecodes."""

	if len(sys.argv) < 2:
		print('Usage: python analyze_file.py <audio_file> [YYYY-MM-DD]')
		sys.exit(1)

	file_path = sys.argv[1]

	# Parse optional date argument, or default to None (year-round filtering)
	analysis_date = None

	if len(sys.argv) > 2:
		analysis_date = datetime.date.fromisoformat(sys.argv[2])

	listener = birdnetpy.core.Listener(
		match_threshold = 0.8,
		callback_function = on_detection,
		latitude = 51.454,   # Bristol, UK
		longitude = -2.598
	)

	listener.analyze_file(file_path, analysis_date=analysis_date)

if __name__ == '__main__':

	main()
