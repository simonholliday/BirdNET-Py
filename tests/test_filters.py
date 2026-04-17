import numpy

import birdnetpy.core


def _sine (freq_hz, duration_s=1.0, sample_rate_hz=48000):

	"""Generate a unit-amplitude sine wave as float32 for filter testing."""

	t = numpy.arange(int(sample_rate_hz * duration_s)) / sample_rate_hz

	return numpy.sin(2 * numpy.pi * freq_hz * t).astype(numpy.float32)


def test_no_filters_returns_input_unchanged ():
	listener = birdnetpy.core.Listener(filter_highpass_hz=None, filter_lowpass_hz=None)

	x = numpy.random.RandomState(0).randn(1000).astype(numpy.float32)
	y = listener._apply_filters(x)

	assert numpy.array_equal(y, x)


def test_highpass_attenuates_low_frequency ():
	listener = birdnetpy.core.Listener(filter_highpass_hz=1000, filter_lowpass_hz=None)

	low = _sine(50)
	high = _sine(4000)

	# Skip the filter's startup transient before comparing energies.

	low_filtered = listener._apply_filters(low)[10000:]
	high_filtered = listener._apply_filters(high)[10000:]

	assert numpy.max(numpy.abs(low_filtered)) < 0.1
	assert numpy.max(numpy.abs(high_filtered)) > 0.5


def test_lowpass_attenuates_high_frequency ():
	listener = birdnetpy.core.Listener(filter_highpass_hz=None, filter_lowpass_hz=1000)

	low = _sine(200)
	high = _sine(10000)

	low_filtered = listener._apply_filters(low)[10000:]
	high_filtered = listener._apply_filters(high)[10000:]

	assert numpy.max(numpy.abs(low_filtered)) > 0.5
	assert numpy.max(numpy.abs(high_filtered)) < 0.1


def test_both_filters_yield_bandpass ():
	listener = birdnetpy.core.Listener(filter_highpass_hz=500, filter_lowpass_hz=5000)

	too_low = _sine(100)
	in_band = _sine(2000)
	too_high = _sine(12000)

	low_filtered = listener._apply_filters(too_low)[10000:]
	mid_filtered = listener._apply_filters(in_band)[10000:]
	high_filtered = listener._apply_filters(too_high)[10000:]

	assert numpy.max(numpy.abs(low_filtered)) < 0.1
	assert numpy.max(numpy.abs(mid_filtered)) > 0.5
	assert numpy.max(numpy.abs(high_filtered)) < 0.1


def test_filter_output_is_float32 ():
	listener = birdnetpy.core.Listener(filter_highpass_hz=100)

	x = numpy.zeros(1000, dtype=numpy.float32)
	y = listener._apply_filters(x)

	assert y.dtype == numpy.float32
