import datetime
import numpy
import pytest

import birdnetpy.core


class TestDateToWeek:

	def test_day_1_maps_to_week_1 (self):
		assert birdnetpy.core.Listener._date_to_week(datetime.date(2026, 1, 1)) == 1

	def test_mid_year_is_around_week_24 (self):

		# Day 180 / 7.625 ≈ 23.6, so week 24 after the +1 offset.

		result = birdnetpy.core.Listener._date_to_week(datetime.date(2026, 6, 29))

		assert 23 <= result <= 25

	def test_year_end_clamps_to_48 (self):
		assert birdnetpy.core.Listener._date_to_week(datetime.date(2026, 12, 31)) == 48

	def test_leap_day_366_clamps_to_48 (self):

		# 2024 is a leap year; Dec 31 is day 366.

		assert birdnetpy.core.Listener._date_to_week(datetime.date(2024, 12, 31)) == 48


class TestCustomSigmoid:

	def test_zero_maps_to_half (self):
		result = birdnetpy.core.Listener._custom_sigmoid(numpy.array([0.0]))

		assert result[0] == pytest.approx(0.5)

	def test_large_positive_approaches_one (self):
		result = birdnetpy.core.Listener._custom_sigmoid(numpy.array([100.0]))

		assert result[0] > 0.99

	def test_large_negative_approaches_zero (self):
		result = birdnetpy.core.Listener._custom_sigmoid(numpy.array([-100.0]))

		assert result[0] < 0.01

	def test_higher_sensitivity_is_more_extreme (self):

		# At the same input, higher sensitivity should push the sigmoid output further from 0.5.

		x = numpy.array([1.0])

		low = birdnetpy.core.Listener._custom_sigmoid(x, 0.5)[0]
		high = birdnetpy.core.Listener._custom_sigmoid(x, 1.5)[0]

		assert abs(high - 0.5) > abs(low - 0.5)


class TestGetDbfsPeak:

	def test_empty_chunk_returns_zero (self):
		assert birdnetpy.core.Listener.get_dbfs_peak(numpy.array([], dtype=numpy.float32)) == 0.0

	def test_full_scale_is_zero_dbfs (self):
		x = numpy.array([1.0, -1.0, 0.5], dtype=numpy.float32)

		assert birdnetpy.core.Listener.get_dbfs_peak(x) == pytest.approx(0.0, abs=1e-3)

	def test_half_scale_is_minus_six_dbfs (self):
		x = numpy.array([0.5, -0.5], dtype=numpy.float32)

		assert birdnetpy.core.Listener.get_dbfs_peak(x) == pytest.approx(-6.02, abs=0.1)


class TestFormatReaperTime:

	def test_zero (self):
		assert birdnetpy.core.Listener._format_reaper_time(0.0) == '0:00:00.000'

	def test_sub_second (self):
		assert birdnetpy.core.Listener._format_reaper_time(1.234) == '0:00:01.234'

	def test_minute_and_fraction (self):
		assert birdnetpy.core.Listener._format_reaper_time(65.5) == '0:01:05.500'

	def test_hour (self):
		assert birdnetpy.core.Listener._format_reaper_time(3723.456) == '1:02:03.456'


class TestFormatSrtTime:

	def test_zero (self):
		assert birdnetpy.core.Listener._format_srt_time(0.0) == '00:00:00,000'

	def test_sub_second (self):
		assert birdnetpy.core.Listener._format_srt_time(1.234) == '00:00:01,234'

	def test_hour (self):
		assert birdnetpy.core.Listener._format_srt_time(3723.456) == '01:02:03,456'
