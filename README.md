[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)
# BirdNET-Py

**A lightweight asynchronous audio listener and BirdNET analysis pipeline for Python development and experimentation.**

This project provides a simple class that continuously listens to an audio input, passing 3-second chunks of audio (with a 0.5-second overlap) to be analyzed by the BirdNET-Lite model. Detections are returned to a user-defined callback function.

It uses asyncio for efficient, non-blocking audio capture and offloads the CPU-intensive model inference to a separate thread. This design ensures the audio stream is never interrupted, preventing buffer overflows and lost samples.

If an audio output directory is specified, the 3-second analysis buffer is saved to a 16-bit WAV file whenever a positive detection is made. The full file path is then passed to the callback function.

It is intended as a tool to simplify access to the BirdNET model for Python developers. The code is highly optimized to ensure reliable, uninterrupted audio streaming even on devices like the Raspberry Pi Zero 2. On this hardware, analysis of a 3-second audio file is typically completed in under 0.8 seconds.

Privacy Note: Users should be respectful of privacy. If the `is_human` property is `True` in any detections, the corresponding audio file may contain human speech.

## Installation

### Prerequisites

This project requires Python ≥3.7 and <3.12 (due to TensorFlow Lite runtime incompatibility with Python 3.12). The following system package is also required:

```bash
sudo apt-get install portaudio19-dev
```

This project uses a TFLite model that is not compatible with NumPy 2.0 or newer. The requirements.txt file enforces this.

### Dependencies

The Python dependencies are listed in requirements.txt:

```
librosa
numpy<2
sounddevice
tflite-runtime
```

Install them using `pip`

```bash
pip install -r requirements.txt
```

#### Python Versions

At the time of writing, `tflite-runtime` is not available for Python 3.12 or newer. If your system uses Python 3.12+ by default, you can install Python 3.11 alongside it and create a virtual environment for BirdNET-Py:

```bash
# Install Python 3.11 (Ubuntu example)
sudo apt install -y software-properties-common
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install -y python3.11 python3.11-venv

# Create and activate a Python 3.11 virtual environment
python3.11 -m venv birdnetpy
source birdnetpy/bin/activate

# Now install requirements
pip install -r requirements.txt
```

## Example

This example is included as `example.py` in the repo.

```python
import asyncio
import os

from birdnetpy.core import Listener, Detection

def example_callback (detections:list[Detection], wav_file_path:str=None):

	"""
	This function will be called when items are detected.
	It is passed as an argument to the Listener() init below.
	"""

	for detection in detections:

		print(detection)
		print('We detected a %s with a confidence level of %0.2f%%' % (detection.english_name, 100 * detection.confidence))

	if wav_file_path and os.path.isfile(wav_file_path):

		# The user is responsible for managing the saved WAV files.
		# In this example, we'll just remove the file to prevent the disk from filling up.

		os.remove(wav_file_path)

async def main ():

	# Initialize a listener

	listener = Listener(
		match_threshold = 0.8,
		silence_threshold_dbfs = -60.0,
		callback_function = example_callback,
		audio_output_dir = '/tmp',
		exclude_label_file_path = 'birdnetpy/labels_filter_non_uk.txt'
	)

	# Listen :)

	await listener.listen()

if __name__ == '__main__':

	asyncio.run(main())
```

### Parameters for the Listener object

- **match_threshold**: The lowest confidence level we want to see matches for (between 0 and 1).
- **silence_threshold_dbfs**: If defined, we will check whether there is any signal in the sampled audio which exceeds this level, and if not, it will not be passed to the BirdNET model (a value in dBFS e.g. -60).
- **callback_function**: This function will be called any time one or more bird is detected in an audio chunk. It should accept a list of Detection objects and a wav file path as its arguments.
- **audio_output_dir**: An optional directory to store the analyzed audio when there are detections. Omit or specify `None` if you don't want to keep the audio.
- **exclude_label_file_path**: An optional path to a list of labels which will be excluded from detection. Omit or specify `None` if you don't need filtering.

See *Filtering* below for more information about using `exclude_label_file_path`.

### Detections

The *Detection* object is a named tuple defined as follows:

```
Detection = collections.namedtuple('Detection', ['index', 'english_name', 'latin_name', 'is_bird', 'is_human', 'confidence'])
```

The BirdNET model contains some non-bird items, and so the additional boolean `is_bird` and `is_human` properties are intended to help with classification.

### Filtering

Full implementations of BirdNET sometimes apply geographic and seasonal filters, using occurrence databases to restrict detections to species that are realistically present at a given place and time.

BirdNET-Py takes a simpler and more lightweight approach. Instead of relying on external data sources, it allows you to exclude species using a plain-text list of labels. This file is provided via the optional `exclude_label_file_path` argument when creating a Listener. Any species on the list will be ignored during detection.

This design keeps the code portable and easy to run on small devices, while still giving users flexibility to apply their own filters. For example, the included file `labels_filter_non_uk.txt` excludes species not found in the UK, helping to reduce false positives. You can adapt the same method for other regions or use cases by editing or supplying your own exclusion file.

## Licence

This project is released under CC BY-NC-SA 4.0

See https://creativecommons.org/licenses/by-nc-sa/4.0/ for full terms.

## Attribution

This project includes two *unmodified* files from the [BirdNET-Lite](https://github.com/birdnet-team/BirdNET-Lite) project by the [BirdNET-Team](https://github.com/birdnet-team):

- `birdnet/BirdNET_GLOBAL_6K_V2.4_Model_FP16.tflite`  
- `birdnet/labels_en.txt`  

These files are provided under the terms of the [CC BY-NC-SA 4.0 licence](https://creativecommons.org/licenses/by-nc-sa/4.0/).  

## Disclaimer

BirdNET-Py uses the BirdNET-Lite model for automatic sound detection. Like all machine learning tools, it may not always provide perfectly accurate results, so please use detections as a helpful guide rather than definitive proof. This project is intended for experimentation and development, and no warranty or guarantee of accuracy is provided.
