[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)
# BirdNET-Py

**A lightweight Python library for identifying bird species from live audio or pre-recorded files using the BirdNET model.**

BirdNET-Py detects bird species by analyzing 3-second audio windows (with a 0.5-second overlap) and returning results to a user-defined callback function. It supports two modes:

- **Live streaming** — continuously listens to a microphone or audio input and reports detections in real time.
- **File analysis** — processes pre-recorded audio files (WAV, MP3, FLAC, OGG, etc.) at any sample rate or bit depth, reporting detections with timecodes. Files are streamed from disk in small chunks, so memory usage stays low even for very large recordings.

Three quantization variants of the BirdNET V2.4 model are bundled (FP32, FP16, INT8), or you can supply your own compatible TFLite model. Geographic filtering by latitude, longitude, and time of year automatically restricts detections to species expected at your location, or you can provide a custom exclusion file.

The code is optimized for reliable operation on resource-constrained devices like the Raspberry Pi Zero 2, where analysis of a 3-second audio window typically completes in under 0.8 seconds. If an audio output directory is specified, the analysis buffer is saved as a WAV file whenever a detection is made.

Privacy Note: Users should be respectful of privacy. If the `is_human` property is `True` in any detections, the corresponding audio may contain human speech.

## Installation

### Prerequisites

This project requires Python ≥3.7 and <3.12 (due to TensorFlow Lite runtime incompatibility with Python 3.12).
You’ll also need the PortAudio development headers for audio input:

```bash
sudo apt-get install portaudio19-dev
```

### Install with pip

Clone the repository and install normally:

```bash
git clone https://github.com/simonholliday/BirdNET-Py.git
cd BirdNET-Py
pip install .
```

This will install the package and its dependencies as defined in `pyproject.toml`.

If you plan to contribute or make changes to the code, you may prefer an **editable install**:

```bash
pip install -e .
```

### Python version note

At the time of writing, `tflite-runtime` is not available for Python 3.12 or newer.
If your system uses Python 3.12+ by default, you can install Python 3.11 alongside it and create a virtual environment for BirdNET-Py:

```bash
# Install Python 3.11 (Ubuntu example)
sudo apt install -y software-properties-common
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install -y python3.11 python3.11-venv

# Create and activate a Python 3.11 virtual environment
python3.11 -m venv birdnetpy
source birdnetpy/bin/activate

# Install BirdNET-Py
pip install .
```

## Examples

Two examples are included in the `examples/` directory.

### Live streaming (`examples/analyze_stream.py`)

Listens to a microphone and prints detections in real time:

```bash
python examples/analyze_stream.py
```

```python
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
```

### File analysis (`examples/analyze_file.py`)

Analyzes a pre-recorded audio file and prints detections with timecodes:

```bash
python examples/analyze_file.py /path/to/recording.wav
python examples/analyze_file.py /path/to/recording.wav 2025-06-15
```

```python
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
```

### Parameters for the Listener object

- **match_threshold**: The lowest confidence level we want to see matches for (between 0 and 1).
- **silence_threshold_dbfs**: If defined, we will check whether there is any signal in the sampled audio which exceeds this level, and if not, it will not be passed to the BirdNET model (a value in dBFS e.g. -60).
- **callback_function**: This function will be called any time one or more bird is detected in an audio chunk. It should accept three arguments: a list of Detection objects, a wav file path (or `None`), and a timecode in seconds (or `None` for live streaming).
- **audio_output_dir**: An optional directory to store the analyzed audio when there are detections. Omit or specify `None` if you don't want to keep the audio.
- **exclude_label_file_path**: An optional path to a list of labels which will be excluded from detection. Omit or specify `None` if you don't need filtering.
- **model_file_path**: An optional path to a TFLite model file. If omitted, the bundled FP16 model is used. See *Model Variants* below.
- **latitude**: Optional latitude for geographic species filtering. Must be provided together with `longitude`. See *Geographic Filtering* below.
- **longitude**: Optional longitude for geographic species filtering. Must be provided together with `latitude`.
- **species_threshold**: Minimum probability from the geographic model to include a species (default 0.03). Only used when `latitude` and `longitude` are provided.

See *Filtering* below for more information about filtering options.

### Detections

The *Detection* object is a named tuple defined as follows:

```
Detection = collections.namedtuple('Detection', ['index', 'english_name', 'latin_name', 'is_bird', 'is_human', 'confidence'])
```

The BirdNET model contains some non-bird items, and so the additional boolean `is_bird` and `is_human` properties are intended to help with classification.

### Model Variants

Three variants of the BirdNET V2.4 model are bundled:

| Variant | Size | Best for |
|---------|------|----------|
| **FP32** | ~50MB | Maximum accuracy, desktop/server hardware |
| **FP16** (default) | ~25MB | Good balance of accuracy and size |
| **INT8** | ~39MB | Optimised for edge devices and constrained hardware |

All three accept the same input format (48kHz float32 audio) and produce the same output shape.

To use a specific bundled variant:

```python
import importlib

model_path = str(importlib.resources.files("birdnetpy.birdnet") / "BirdNET_GLOBAL_6K_V2.4_Model_INT8.tflite")

listener = birdnetpy.core.Listener(
	model_file_path = model_path
)
```

You can also supply your own compatible TFLite model file via `model_file_path`.

### File Analysis

Any common audio format (WAV, MP3, FLAC, OGG, etc.) is supported, at any sample rate or bit depth — audio is automatically resampled to 48kHz as required by the BirdNET model. Files are streamed from disk in small chunks, so memory usage remains low regardless of file size.

The callback receives a `timecode_s` parameter indicating the midpoint of the 3-second analysis window (in seconds). This minimises the worst-case timing error to 1.5 seconds, since the model cannot localise the call within the window. For live streaming, this value is `None`.

### Geographic Filtering

By providing `latitude` and `longitude` when creating a Listener, BirdNET-Py uses a bundled geographic model to automatically filter detections to species that are likely present at that location and time of year. This uses the BirdNET V2.4 MData model, which predicts species occurrence based on eBird data.

```python
listener = birdnetpy.core.Listener(
	latitude = 51.454,   # Bristol, UK
	longitude = -2.598,
	callback_function = my_callback
)
```

For live streaming, the filter is refreshed automatically every 12 hours to account for seasonal changes. For file analysis, you can provide a specific date:

```python
import datetime

listener.analyze_file('recording.wav', analysis_date=datetime.date(2025, 6, 15))
```

The `species_threshold` parameter controls how selective the filter is (default 0.03, i.e. 3% probability). Lower values include more species; higher values are more restrictive.

To conserve memory on devices like the Raspberry Pi, only one TFLite model is loaded at a time — the geographic model is loaded to generate the filter, then unloaded before the audio model is loaded.

### Exclusion File Filtering

As an alternative (or in addition) to geographic filtering, you can exclude species using a plain-text list of labels via `exclude_label_file_path`. Any species on the list will be ignored during detection.

For example, the included file `labels_filter_non_uk.txt` excludes species not found in the UK. You can adapt the same method for other regions or use cases by supplying your own exclusion file.

Both filtering methods can be used together — the exclusion file is applied first, then the geographic filter further narrows the list.

## Licence

This project is released under [CC BY-NC-SA 4.0](LICENSE).

See https://creativecommons.org/licenses/by-nc-sa/4.0/ for a human-readable summary.

## Attribution

This project includes *unmodified* files from the [BirdNET-Analyzer](https://github.com/birdnet-team/BirdNET-Analyzer) project by the [BirdNET-Team](https://github.com/birdnet-team):

- `birdnet/BirdNET_GLOBAL_6K_V2.4_Model_FP32.tflite`
- `birdnet/BirdNET_GLOBAL_6K_V2.4_Model_FP16.tflite`
- `birdnet/BirdNET_GLOBAL_6K_V2.4_Model_INT8.tflite`
- `birdnet/BirdNET_GLOBAL_6K_V2.4_MData_Model_V2_FP16.tflite`
- `birdnet/labels_en.txt`

These files are provided under the terms of the [CC BY-NC-SA 4.0 licence](https://creativecommons.org/licenses/by-nc-sa/4.0/).  

## Disclaimer

BirdNET-Py uses the BirdNET model for automatic sound detection. Like all machine learning tools, it may not always provide perfectly accurate results, so please use detections as a helpful guide rather than definitive proof. This project is intended for experimentation and development, and no warranty or guarantee of accuracy is provided.
