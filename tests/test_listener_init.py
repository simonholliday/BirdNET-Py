import os
import pytest

import birdnetpy.core


def test_defaults_construct ():
	listener = birdnetpy.core.Listener()

	assert listener.match_threshold == 0.8
	assert listener.sensitivity == 1.0
	assert listener.latitude is None
	assert listener.longitude is None


def test_sensitivity_below_range_rejected ():
	with pytest.raises(ValueError, match='[Ss]ensitivity'):
		birdnetpy.core.Listener(sensitivity=0.49)


def test_sensitivity_above_range_rejected ():
	with pytest.raises(ValueError, match='[Ss]ensitivity'):
		birdnetpy.core.Listener(sensitivity=1.51)


def test_sensitivity_range_boundaries_accepted ():
	birdnetpy.core.Listener(sensitivity=0.5)
	birdnetpy.core.Listener(sensitivity=1.5)


def test_unsupported_annotation_format_rejected ():
	with pytest.raises(ValueError, match='[Uu]nsupported annotation'):
		birdnetpy.core.Listener(annotate='xml')


def test_latitude_without_longitude_rejected ():
	with pytest.raises(ValueError):
		birdnetpy.core.Listener(latitude=51.454)


def test_longitude_without_latitude_rejected ():
	with pytest.raises(ValueError):
		birdnetpy.core.Listener(longitude=-2.598)


def test_both_lat_and_lon_accepted ():
	listener = birdnetpy.core.Listener(latitude=51.454, longitude=-2.598)

	assert listener.latitude == 51.454
	assert listener.longitude == -2.598


def test_missing_model_file_rejected ():
	with pytest.raises(FileNotFoundError):
		birdnetpy.core.Listener(model_file_path='/nonexistent/model.tflite')


def test_missing_audio_output_dir_rejected ():
	with pytest.raises(FileNotFoundError):
		birdnetpy.core.Listener(audio_output_dir='/nonexistent/dir/that/does/not/exist')


def test_non_writeable_audio_output_dir_rejected (tmp_path):
	target = tmp_path / 'readonly'
	target.mkdir()

	os.chmod(str(target), 0o500)

	try:

		with pytest.raises(PermissionError):
			birdnetpy.core.Listener(audio_output_dir=str(target))

	finally:

		# Restore permissions so tmp_path cleanup can succeed.

		os.chmod(str(target), 0o700)


def test_interpreter_not_loaded_at_init ():

	"""The audio model is loaded lazily on first use, not at construction time."""

	listener = birdnetpy.core.Listener()

	assert not hasattr(listener, 'interpreter')


def test_no_geo_warning_logs_when_location_missing (caplog):
	with caplog.at_level('WARNING', logger='birdnetpy.core'):
		birdnetpy.core.Listener()

	assert any('geographic filtering is disabled' in record.message for record in caplog.records)
