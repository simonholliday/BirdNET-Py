[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)
# BirdNET-Py

**A lightweight Python library for identifying bird species from live audio or pre-recorded files using the BirdNET model.**

BirdNET-Py detects bird species by analyzing 3-second audio windows (with a 0.5-second overlap) and returning results to a user-defined callback function. It supports two modes:

- **Live streaming** — continuously listens to a microphone or audio input and reports detections in real time.
- **File analysis** — processes pre-recorded audio files (WAV, MP3, FLAC, OGG, etc.) at any sample rate or bit depth, reporting detections with timecodes. Files are streamed from disk in small chunks, so memory usage stays low even for very large recordings.

Three quantization variants of the BirdNET V2.4 model are bundled (FP32, FP16, INT8), or you can supply your own compatible TFLite model. Optional label filtering lets you restrict detections to species expected in your region.

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

## Example

This example is included as `examples/demo.py` in the repo.

```python
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
```

### Parameters for the Listener object

- **match_threshold**: The lowest confidence level we want to see matches for (between 0 and 1).
- **silence_threshold_dbfs**: If defined, we will check whether there is any signal in the sampled audio which exceeds this level, and if not, it will not be passed to the BirdNET model (a value in dBFS e.g. -60).
- **callback_function**: This function will be called any time one or more bird is detected in an audio chunk. It should accept three arguments: a list of Detection objects, a wav file path (or `None`), and a timecode in seconds (or `None` for live streaming).
- **audio_output_dir**: An optional directory to store the analyzed audio when there are detections. Omit or specify `None` if you don't want to keep the audio.
- **exclude_label_file_path**: An optional path to a list of labels which will be excluded from detection. Omit or specify `None` if you don't need filtering.
- **model_file_path**: An optional path to a TFLite model file. If omitted, the bundled FP16 model is used. See *Model Variants* below.

See *Filtering* below for more information about using `exclude_label_file_path`.

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

In addition to live audio streaming, you can analyze pre-recorded audio files. Any common audio format (WAV, MP3, FLAC, OGG, etc.) is supported, at any sample rate or bit depth — the audio is automatically resampled to 48kHz as required by the BirdNET model.

```python
listener.analyze_file('/path/to/recording.wav')
```

The callback receives a `timecode_s` parameter indicating the midpoint of the 3-second analysis window (in seconds). This minimises the worst-case timing error to 1.5 seconds, since the model cannot localise the call within the window. For live streaming, this value is `None`. See the example above for how to format the timecode.

Audio is streamed from disk in small chunks, so memory usage remains low regardless of file size.

The demo can be run against a file directly:

```bash
python examples/demo.py /path/to/recording.wav
```

### Filtering

Full implementations of BirdNET sometimes apply geographic and seasonal filters, using occurrence databases to restrict detections to species that are realistically present at a given place and time.

BirdNET-Py takes a simpler and more lightweight approach. Instead of relying on external data sources, it allows you to exclude species using a plain-text list of labels. This file is provided via the optional `exclude_label_file_path` argument when creating a Listener. Any species on the list will be ignored during detection.

This design keeps the code portable and easy to run on small devices, while still giving users flexibility to apply their own filters. For example, the included file `labels_filter_non_uk.txt` excludes species not found in the UK, helping to reduce false positives. You can adapt the same method for other regions or use cases by editing or supplying your own exclusion file.

## Licence

This project is released under CC BY-NC-SA 4.0

See https://creativecommons.org/licenses/by-nc-sa/4.0/ for full terms.

## Attribution

This project includes *unmodified* files from the [BirdNET-Analyzer](https://github.com/birdnet-team/BirdNET-Analyzer) project by the [BirdNET-Team](https://github.com/birdnet-team):

- `birdnet/BirdNET_GLOBAL_6K_V2.4_Model_FP32.tflite`
- `birdnet/BirdNET_GLOBAL_6K_V2.4_Model_FP16.tflite`
- `birdnet/BirdNET_GLOBAL_6K_V2.4_Model_INT8.tflite`
- `birdnet/labels_en.txt`

These files are provided under the terms of the [CC BY-NC-SA 4.0 licence](https://creativecommons.org/licenses/by-nc-sa/4.0/).  

## Disclaimer

BirdNET-Py uses the BirdNET model for automatic sound detection. Like all machine learning tools, it may not always provide perfectly accurate results, so please use detections as a helpful guide rather than definitive proof. This project is intended for experimentation and development, and no warranty or guarantee of accuracy is provided.
