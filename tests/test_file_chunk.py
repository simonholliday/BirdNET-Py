import numpy
import pytest

import birdnetpy.core


class RecordingListener(birdnetpy.core.Listener):

	"""Listener subclass that records birdcatcher() calls instead of running inference."""

	def __init__ (self, **kwargs):

		super().__init__(**kwargs)

		self.calls = []

	def birdcatcher (self, analysis_buffer, timecode_s=None):
		self.calls.append((analysis_buffer.copy(), timecode_s))


def test_initial_fill_below_window_does_not_trigger ():
	listener = RecordingListener()

	buf = numpy.zeros(listener.window_samples, dtype=numpy.float32)
	chunk = numpy.ones(1000, dtype=numpy.float32)

	buf_out, filled_out = listener._process_file_chunk(chunk, buf, 0, 1000, 48000)

	assert filled_out == 1000
	assert listener.calls == []


def test_shift_and_append_triggers_detection ():
	listener = RecordingListener()

	buf = numpy.zeros(listener.window_samples, dtype=numpy.float32)

	# Start near-full, then append enough to push the buffer over.

	samples_filled = listener.window_samples - 1000
	chunk = numpy.ones(5000, dtype=numpy.float32)
	source_samples_read = samples_filled + len(chunk)

	buf_out, filled_out = listener._process_file_chunk(chunk, buf, samples_filled, source_samples_read, 48000)

	expected_timecode = source_samples_read / 48000 - listener.window_size_s / 2

	assert filled_out == listener.window_samples
	assert len(listener.calls) == 1
	assert listener.calls[0][1] == pytest.approx(expected_timecode)


def test_chunk_larger_than_window_uses_last_window ():
	listener = RecordingListener()

	buf = numpy.zeros(listener.window_samples, dtype=numpy.float32)
	chunk = numpy.arange(listener.window_samples + 5000, dtype=numpy.float32)
	source_samples_read = len(chunk)

	buf_out, filled_out = listener._process_file_chunk(chunk, buf, 0, source_samples_read, 48000)

	assert filled_out == listener.window_samples
	assert len(listener.calls) == 1

	# The analysed buffer should be the trailing window of the oversized chunk.

	assert numpy.array_equal(listener.calls[0][0], chunk[-listener.window_samples:])


def test_silence_at_buffer_fill_does_not_call_birdcatcher ():

	"""If the silence threshold is set and the buffer is below it, birdcatcher is skipped but the buffer still advances."""

	listener = RecordingListener(silence_threshold_dbfs=-10.0)

	buf = numpy.zeros(listener.window_samples, dtype=numpy.float32)

	# Start near-full with silence (zeros), append more silence.

	samples_filled = listener.window_samples - 1000
	chunk = numpy.zeros(5000, dtype=numpy.float32)
	source_samples_read = samples_filled + len(chunk)

	_, filled_out = listener._process_file_chunk(chunk, buf, samples_filled, source_samples_read, 48000)

	assert filled_out == listener.window_samples
	assert listener.calls == []


def test_timecode_at_source_rate_for_resampled_audio ():

	"""When the file is at a non-target sample rate, the timecode uses the source rate so it maps back to wall-clock file position."""

	listener = RecordingListener()

	buf = numpy.zeros(listener.window_samples, dtype=numpy.float32)

	# Suppose the source rate was 44100Hz and 88200 source samples have been read (2 seconds of source audio).

	samples_filled = listener.window_samples - 1000
	chunk = numpy.ones(5000, dtype=numpy.float32)
	source_samples_read = 88200

	listener._process_file_chunk(chunk, buf, samples_filled, source_samples_read, 44100)

	expected_timecode = 88200 / 44100 - listener.window_size_s / 2

	assert listener.calls[0][1] == pytest.approx(expected_timecode)
