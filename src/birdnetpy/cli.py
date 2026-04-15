import argparse
import datetime
import importlib
import logging
import typing

import birdnetpy.core

MODEL_VARIANTS = {
	'fp16': 'BirdNET_GLOBAL_6K_V2.4_Model_FP16.tflite',
	'fp32': 'BirdNET_GLOBAL_6K_V2.4_Model_FP32.tflite',
	'int8': 'BirdNET_GLOBAL_6K_V2.4_Model_INT8.tflite',
}

def _print_detection (detections:typing.List[birdnetpy.core.Detection], wav_file_path:typing.Optional[str] = None, timecode_s:typing.Optional[float] = None) -> None:

	"""Print each detection with its timecode."""

	for detection in detections:

		minutes, seconds = divmod(typing.cast(float, timecode_s), 60)
		print('[%02d:%05.2f] %s (%.0f%%)' % (int(minutes), seconds, detection.english_name, 100 * detection.confidence))

def main () -> None:

	"""Command-line interface for analyzing audio files with BirdNET-Py."""

	parser = argparse.ArgumentParser(
		prog = 'birdcatcher',
		description = 'Identify bird species in audio files using the BirdNET model.'
	)

	parser.add_argument('file_path', help='Path to the audio file to analyze')
	parser.add_argument('--location', help='Latitude,longitude for geographic filtering (e.g. 51.454,-2.598)')
	parser.add_argument('--date', help='Recording date for seasonal filtering (YYYY-MM-DD)')
	parser.add_argument('--threshold', type=float, default=0.8, help='Minimum confidence level (0-1, default 0.8)')
	parser.add_argument('--species-threshold', type=float, default=0.03, help='Minimum geographic probability to include a species (0-1, default 0.03)')
	parser.add_argument('--model', help='Path to a custom TFLite model file')
	parser.add_argument('--model-variant', choices=sorted(MODEL_VARIANTS.keys()), help='Bundled model variant to use (default fp16)')
	parser.add_argument('--annotate', choices=sorted(birdnetpy.core.Listener.SUPPORTED_ANNOTATIONS), help='Write annotation file alongside the audio (e.g. audacity)')
	parser.add_argument('--sensitivity', type=float, default=1.0, help='Detection sensitivity (0.5-1.5, default 1.0). Lower values find more birds at lower confidence; higher values are more selective')
	parser.add_argument('--highpass', type=int, default=100, help='High-pass filter frequency in Hz (default 100). Removes low-frequency noise like wind and traffic')
	parser.add_argument('--lowpass', type=int, default=None, help='Low-pass filter frequency in Hz (disabled by default). Set to e.g. 15000 to remove high-frequency noise')
	parser.add_argument('--no-highpass', action='store_true', help='Disable the default 100Hz high-pass filter')

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

	# Resolve model path

	model_file_path = args.model

	if args.model and args.model_variant:
		parser.error('--model and --model-variant are mutually exclusive')

	if args.model_variant:
		model_file_path = str(importlib.resources.files("birdnetpy.birdnet") / MODEL_VARIANTS[args.model_variant])

	# Resolve filter settings

	filter_highpass_hz = None if args.no_highpass else args.highpass
	filter_lowpass_hz = args.lowpass

	# Create listener and analyze

	listener = birdnetpy.core.Listener(
		match_threshold = args.threshold,
		callback_function = _print_detection,
		model_file_path = model_file_path,
		latitude = latitude,
		longitude = longitude,
		species_threshold = args.species_threshold,
		annotate = args.annotate,
		sensitivity = args.sensitivity,
		filter_highpass_hz = filter_highpass_hz,
		filter_lowpass_hz = filter_lowpass_hz
	)

	listener.analyze_file(args.file_path, analysis_date=analysis_date)

if __name__ == '__main__':

	main()
