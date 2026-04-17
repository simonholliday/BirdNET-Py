import pytest
import sys

import birdnetpy.cli


def test_help_exits_cleanly (monkeypatch):
	monkeypatch.setattr(sys, 'argv', ['birdcatcher', '--help'])

	with pytest.raises(SystemExit) as exc_info:
		birdnetpy.cli.main()

	assert exc_info.value.code == 0


def test_missing_file_argument_errors (monkeypatch):
	monkeypatch.setattr(sys, 'argv', ['birdcatcher'])

	with pytest.raises(SystemExit) as exc_info:
		birdnetpy.cli.main()

	assert exc_info.value.code != 0


def test_invalid_location_format_errors (monkeypatch, silent_wav):
	monkeypatch.setattr(sys, 'argv', ['birdcatcher', silent_wav, '--location', 'abc'])

	with pytest.raises(SystemExit):
		birdnetpy.cli.main()


def test_invalid_location_numbers_errors (monkeypatch, silent_wav):
	monkeypatch.setattr(sys, 'argv', ['birdcatcher', silent_wav, '--location', 'foo,bar'])

	with pytest.raises(SystemExit):
		birdnetpy.cli.main()


def test_invalid_date_errors (monkeypatch, silent_wav):
	monkeypatch.setattr(sys, 'argv', ['birdcatcher', silent_wav, '--date', '2026-13-01'])

	with pytest.raises(SystemExit):
		birdnetpy.cli.main()


def test_model_and_model_variant_are_mutually_exclusive (monkeypatch, silent_wav, tmp_path):
	fake_model = tmp_path / 'fake.tflite'
	fake_model.touch()

	monkeypatch.setattr(sys, 'argv', ['birdcatcher', silent_wav, '--model', str(fake_model), '--model-variant', 'fp16'])

	with pytest.raises(SystemExit):
		birdnetpy.cli.main()


def test_valid_args_run_analysis (monkeypatch, silent_wav):

	"""A valid CLI invocation on a silent file should run analyze_file to completion."""

	monkeypatch.setattr(sys, 'argv', ['birdcatcher', silent_wav, '--threshold', '0.99'])

	birdnetpy.cli.main()
