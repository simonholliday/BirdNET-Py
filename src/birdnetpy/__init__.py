"""BirdNET-Py: A lightweight Python library for identifying bird species from live audio or pre-recorded files."""

import importlib.metadata

import birdnetpy.core

__version__ = importlib.metadata.version("birdnetpy")

Detection = birdnetpy.core.Detection
Listener = birdnetpy.core.Listener