[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)
# BirdNET-Py

**Free, open-source bird species identification for Python — powered by the BirdNET machine learning model.**

BirdNET-Py is a lightweight Python library and command-line tool for automated bird sound recognition. It identifies over 6,000 bird species from audio using the [BirdNET V2.4](https://github.com/birdnet-team/BirdNET-Analyzer) deep learning model, and outputs results in formats compatible with popular free tools used in bioacoustics research.

### Key Features

- **Live monitoring** — listen to a microphone and detect birds in real time
- **Batch file analysis** — process recordings in any common audio format (WAV, MP3, FLAC, OGG) at any sample rate, with timecoded results
- **Geographic and seasonal filtering** — provide latitude, longitude, and date to automatically restrict detections to species expected at your location, using eBird occurrence data
- **Annotation export** — output detections as Audacity labels, Raven selection tables, Reaper markers, CSV data, or SRT subtitles
- **Low resource usage** — optimized for devices like the Raspberry Pi Zero 2; audio files are streamed from disk, not loaded into memory
- **Simple API** — one class, callback-driven, easy to integrate into larger projects

### Who is this for?

BirdNET-Py is designed for wildlife researchers, citizen scientists, conservation projects, and hobbyists who want to identify birds from audio recordings or live microphone input. It is free for non-commercial use and intended to support biodiversity monitoring, ecological surveys, and ornithological research.

## Installation

### Prerequisites

This project requires Python ≥3.9.
You'll also need PortAudio and libsndfile development libraries for audio input and file handling:

**Debian / Ubuntu:**
```bash
sudo apt-get install portaudio19-dev libsndfile1-dev
```

**Fedora / RHEL:**
```bash
sudo dnf install portaudio-devel libsndfile-devel
```

**Arch Linux:**
```bash
sudo pacman -S portaudio libsndfile
```

**macOS:**
```bash
brew install portaudio libsndfile
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

## Quick Start

The fastest way to analyze a recording:

```bash
birdcatcher recording.wav --location 51.454,-2.598 --date 2026-04-07
```

Add `--annotate` to export detections for use in other tools:

```bash
birdcatcher recording.wav --location 51.454,-2.598 --date 2026-04-07 --annotate audacity
birdcatcher recording.wav --location 51.454,-2.598 --date 2026-04-07 --annotate raven
birdcatcher recording.wav --location 51.454,-2.598 --date 2026-04-07 --annotate srt
```

## Command Line

After installation, the `birdcatcher` command is available for file analysis:

| Option | Description |
|--------|-------------|
| `--location LAT,LON` | Latitude and longitude for geographic species filtering |
| `--date YYYY-MM-DD` | Recording date for seasonal filtering |
| `--threshold N` | Minimum confidence level, 0–1 (default 0.8) |
| `--species-threshold N` | Minimum geographic probability to include a species, 0–1 (default 0.03) |
| `--model PATH` | Path to a custom TFLite model file |
| `--model-variant VARIANT` | Bundled model variant: `fp16` (default), `fp32`, or `int8` |
| `--annotate FORMAT` | Write an annotation file alongside the audio (see *Annotations* below) |

For full help: `birdcatcher --help`

## Annotations

When analyzing files, BirdNET-Py can generate annotation files for use in audio editors and research tools. Annotations are streamed to disk as detections are found, so partial results are preserved even if analysis is interrupted.

Use the `annotate` parameter in Python or `--annotate` on the command line:

| Format | Value | Output file | Description |
|--------|-------|-------------|-------------|
| Audacity | `audacity` | `.audacity-labels.txt` | Label track — import via File > Import > Labels |
| CSV | `csv` | `.birdnetpy.csv` | Comma-separated data with English names, Latin names, and confidence scores |
| Raven | `raven` | `.raven-selections.txt` | Selection table for Cornell Lab's [Raven Lite](https://www.ravensoundsoftware.com/software/raven-lite/) (free) or Raven Pro |
| Reaper | `reaper` | `.reaper-markers.csv` | Region markers — import via "Import markers/regions from file" |
| SRT | `srt` | `.srt` | Subtitle file — play audio in VLC and see detections as subtitles |

Annotation files are created in the same directory as the audio file.

## Python API

For programmatic usage, two example scripts are included in the `examples/` directory.

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

### Parameters

- **match_threshold**: Minimum confidence level for detections (0–1, default 0.75).
- **silence_threshold_dbfs**: Audio below this level is skipped (dBFS, e.g. -60). Omit for no silence filtering.
- **callback_function**: Called with each detection. Receives a list of `Detection` objects, an optional WAV file path, and an optional timecode in seconds.
- **audio_output_dir**: Directory to save the 3-second analysis buffer as a WAV file on each detection. Omit to disable.
- **exclude_label_file_path**: Path to a plain-text exclusion list. See *Exclusion File Filtering*.
- **model_file_path**: Path to a custom TFLite model. See *Model Variants*.
- **latitude** / **longitude**: Coordinates for geographic species filtering. See *Geographic Filtering*.
- **species_threshold**: Minimum geographic probability to include a species (0–1, default 0.03).
- **annotate**: Annotation format for file analysis. See *Annotations*.

### Detections

The `Detection` object is a named tuple:

```python
Detection = collections.namedtuple('Detection', ['index', 'english_name', 'latin_name', 'is_bird', 'is_human', 'confidence'])
```

The BirdNET model includes some non-bird sounds. The `is_bird` and `is_human` flags help classify these — if `is_human` is `True`, the audio may contain human speech.

## Model Variants

Three variants of the BirdNET V2.4 model are bundled:

| Variant | Size | Best for |
|---------|------|----------|
| **FP32** | ~50MB | Maximum accuracy, desktop/server hardware |
| **FP16** (default) | ~25MB | Good balance of accuracy and size |
| **INT8** | ~39MB | Optimised for edge devices and constrained hardware |

All three accept the same input format (48kHz float32 audio) and produce the same output shape. Use `--model-variant` on the command line or `model_file_path` in Python to select a variant.

## File Analysis

Any common audio format (WAV, MP3, FLAC, OGG, etc.) is supported, at any sample rate or bit depth — audio is automatically resampled to 48kHz as required by the BirdNET model. Files are streamed from disk in small chunks, so memory usage remains low regardless of file size.

The `timecode_s` passed to the callback indicates the midpoint of the 3-second analysis window. This minimises the worst-case timing error to 1.5 seconds, since the model cannot localise the call within the window. For live streaming, `timecode_s` is `None`.

## Geographic Filtering

By providing `latitude` and `longitude`, BirdNET-Py uses a bundled geographic model (trained on [eBird](https://ebird.org/) occurrence data) to automatically filter detections to species likely present at your location and time of year. This significantly reduces false positives.

For live streaming, the filter refreshes automatically every 12 hours. For file analysis, provide an `analysis_date` for seasonal accuracy — otherwise year-round filtering is used.

The `species_threshold` parameter controls selectivity (default 0.03). Lower values include more species; higher values are more restrictive.

To conserve memory on devices like the Raspberry Pi, only one TFLite model is loaded at a time — the geographic model is loaded to generate the filter, then unloaded before the audio model is loaded.

## Exclusion File Filtering

As an alternative (or in addition) to geographic filtering, you can exclude species using a plain-text list of labels via `exclude_label_file_path`. Any species on the list will be ignored during detection.

The included file `labels_filter_non_uk.txt` excludes species not found in the UK. You can create your own exclusion files for other regions.

Both filtering methods can be combined — the exclusion file is applied first, then the geographic filter further narrows the list.

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

BirdNET-Py uses the BirdNET deep learning model for automated bird sound identification. Like all machine learning tools, it may not always provide perfectly accurate results — please use detections as a helpful guide rather than definitive proof. This project is intended for research, education, and conservation, and no warranty or guarantee of accuracy is provided.
