import numpy
import pytest
import soundfile


@pytest.fixture
def silent_wav (tmp_path):

	"""Five seconds of silence as a 48kHz mono WAV. Use when a test needs a real audio file but no actual signal."""

	path = tmp_path / 'silent.wav'
	samples = numpy.zeros(48000 * 5, dtype=numpy.float32)

	soundfile.write(str(path), samples, 48000)

	return str(path)


@pytest.fixture
def sine_wav (tmp_path):

	"""Five seconds of a 440Hz sine tone as a 48kHz mono WAV. Useful for filter and detection tests."""

	path = tmp_path / 'sine.wav'

	t = numpy.linspace(0.0, 5.0, 48000 * 5, endpoint=False)
	samples = (0.3 * numpy.sin(2 * numpy.pi * 440 * t)).astype(numpy.float32)

	soundfile.write(str(path), samples, 48000)

	return str(path)


@pytest.fixture
def offrate_wav (tmp_path):

	"""Five seconds of silence at 44.1kHz to exercise the resample path in analyze_file."""

	path = tmp_path / 'offrate.wav'
	samples = numpy.zeros(44100 * 5, dtype=numpy.float32)

	soundfile.write(str(path), samples, 44100)

	return str(path)
