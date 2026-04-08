import argparse
import datetime
import logging
import typing

import birdnetpy.core

def _print_detection (detections:typing.List[birdnetpy.core.Detection], wav_file_path:typing.Optional[str] = None, timecode_s:typing.Optional[float] = None) -> None:

	"""Print each detection with its timecode."""

	for detection in detections:

		minutes, seconds = divmod(typing.cast(float, timecode_s), 60)
		print('[%02d:%05.2f] %s (%.0f%%)' % (int(minutes), seconds, detection.english_name, 100 * detection.confidence))

def main () -> None:

	"""Command-line interface for analyzing audio files with BirdNET-Py."""

	parser = argparse.ArgumentParser(
		prog = 'birdnetpy',
		description = 'Identify bird species in audio files using the BirdNET model.'
	)

	parser.add_argument('file_path', help='Path to the audio file to analyze')
	parser.add_argument('--location', help='Latitude,longitude for geographic filtering (e.g. 51.454,-2.598)')
	parser.add_argument('--date', help='Recording date for seasonal filtering (YYYY-MM-DD)')
	parser.add_argument('--threshold', type=float, default=0.8, help='Minimum confidence level (0-1, default 0.8)')
	parser.add_argument('--species-threshold', type=float, default=0.03, help='Minimum geographic probability to include a species (0-1, default 0.03)')
	parser.add_argument('--model', help='Path to a custom TFLite model file')
	parser.add_argument('--annotate', choices=sorted(birdnetpy.core.Listener.SUPPORTED_ANNOTATIONS), help='Write annotation file alongside the audio (e.g. audacity)')

	args = parser.parse_args()

	logging.basicConfig(level=logging.INFO)

	# Parse location

	latitude = None
	longitude = None

	if args.location:

		parts = args.location.split(',')

		if len(parts) != 2:
			parser.error('--location must be in the format LAT,LON (e.g. 51.454,-2.598)')

		try:
			latitude = float(parts[0])
			longitude = float(parts[1])
		except ValueError:
			parser.error('--location must contain valid numbers (e.g. 51.454,-2.598)')

	# Parse date

	analysis_date = None

	if args.date:

		try:
			analysis_date = datetime.date.fromisoformat(args.date)
		except ValueError:
			parser.error('--date must be in YYYY-MM-DD format')

	# Create listener and analyze

	listener = birdnetpy.core.Listener(
		match_threshold = args.threshold,
		callback_function = _print_detection,
		model_file_path = args.model,
		latitude = latitude,
		longitude = longitude,
		species_threshold = args.species_threshold,
		annotate = args.annotate
	)

	listener.analyze_file(args.file_path, analysis_date=analysis_date)

if __name__ == '__main__':

	main()
