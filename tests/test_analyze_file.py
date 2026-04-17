import os
import pytest

import birdnetpy.core


def test_missing_file_raises ():
	listener = birdnetpy.core.Listener()

	with pytest.raises(FileNotFoundError):
		listener.analyze_file('/nonexistent/path.wav')


def test_silent_file_produces_no_high_confidence_detections (silent_wav):
	collected = []

	def cb (detections, wav, tc):
		collected.extend(detections)

	listener = birdnetpy.core.Listener(callback_function=cb, match_threshold=0.99)
	listener.analyze_file(silent_wav)

	assert collected == []


def test_audio_model_is_loaded_after_analysis (silent_wav):
	listener = birdnetpy.core.Listener(match_threshold=0.99)

	assert not hasattr(listener, 'interpreter')

	listener.analyze_file(silent_wav)

	assert hasattr(listener, 'interpreter')


def test_non_target_sample_rate_is_resampled_without_error (offrate_wav):
	listener = birdnetpy.core.Listener(match_threshold=0.99)

	listener.analyze_file(offrate_wav)


def test_annotation_file_is_created (silent_wav):
	listener = birdnetpy.core.Listener(annotate='csv', match_threshold=0.99)
	listener.analyze_file(silent_wav)

	expected = silent_wav.replace('.wav', '.birdnetpy.csv')

	assert os.path.isfile(expected)


def test_geo_filter_refreshes_with_location_and_date (silent_wav, caplog):
	listener = birdnetpy.core.Listener(latitude=51.454, longitude=-2.598, match_threshold=0.99)

	import datetime

	with caplog.at_level('INFO', logger='birdnetpy.core'):
		listener.analyze_file(silent_wav, analysis_date=datetime.date(2026, 4, 7))

	assert any('Loading geographic model' in record.message for record in caplog.records)
	assert listener.geo_species is not None


def test_geo_filter_year_round_when_no_date (silent_wav, caplog):
	listener = birdnetpy.core.Listener(latitude=51.454, longitude=-2.598, match_threshold=0.99)

	with caplog.at_level('WARNING', logger='birdnetpy.core'):
		listener.analyze_file(silent_wav)

	assert any('year-round' in record.message for record in caplog.records)
