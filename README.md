[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)
# BirdNET-Py

**A lightweight async audio listener and BirdNET analysis pipeline for development and experimentation.**

This simple Python project implements an audio listener, which passes 3-second chunks of audio (with a 0.5 second overlap) to be analyzed by the BirdNET-Lite model, which in turn calls a user-defined function with any detections.

It is intended as a tool which can simplify access to the BirdNET model for Python developers. It has been tested on a Raspberry Pi Zero 2, with analysis of a 3-second file typically taking under 0.8s.

## Privacy

This code captures audio solely for the purpose of analysis, does not store any captured audio beyond the time it takes to analyze, and does not send any captured audio anywhere. The codebase is small enough that you can verify this yourself very quickly in case of any doubt.

## Installation

### Dependencies

```bash
sudo apt-get install python3-dev portaudio19-dev
pip install -r requirements.txt
```

#### requirements.txt

Note that this project (specifically the .tflite model) is not compatible with NumPy 2.

```
librosa
numpy<2
pyaudio
tflite-runtime
```

## Example

```python
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
```

### Parameters for the Listener object

- **match_threshold**: The lowest confidence level we want to see matches for (between 0 and 1).
- **silence_threshold_dbfs**: If defined, we will check whether there is any signal in the sampled audio which exceeds this level, and if not, it will not be passed to the BirdNET model (a value in dBFS e.g. -60).
- **callback_function**: This function will be called any time one or more bird is detected in an audio chunk. It should accept a list of Detection objects as its argument.

### Detections

The *Detection* object is a namedtuple defined as follows:

```
Detection = collections.namedtuple('Detection', ['index', 'english_name', 'latin_name', 'confidence'])
```

## Licence

This project is released under CC BY-NC-SA 4.0

See https://creativecommons.org/licenses/by-nc-sa/4.0/ for full terms.

## Attribution

This project includes two *unmodified* files from the [BirdNET-Lite](https://github.com/birdnet-team/BirdNET-Lite) project by the [BirdNET-Team](https://github.com/birdnet-team):

- `birdnet/BirdNET_GLOBAL_6K_V2.4_Model_FP16.tflite`  
- `birdnet/labels_en.txt`  

These files are provided under the terms of the [CC BY-NC-SA 4.0 licence](https://creativecommons.org/licenses/by-nc-sa/4.0/).  






