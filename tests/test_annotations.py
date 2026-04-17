import csv
import pytest

import birdnetpy.core


def _detection (index, english_name, latin_name, confidence, is_bird=True, is_human=False):
	return birdnetpy.core.Detection(index, english_name, latin_name, is_bird, is_human, confidence)


def test_no_annotation_returns_none (tmp_path):
	audio = tmp_path / 'test.wav'
	audio.touch()

	listener = birdnetpy.core.Listener()

	assert listener._open_annotation_file(str(audio)) is None


def test_audacity_format (tmp_path):
	audio = tmp_path / 'test.wav'
	audio.touch()

	listener = birdnetpy.core.Listener(annotate='audacity')
	f = listener._open_annotation_file(str(audio))

	try:
		listener._write_annotation(f, [_detection(0, 'European Greenfinch', 'Chloris chloris', 0.95)], 5.0)
	finally:
		f.close()

	content = (tmp_path / 'test.audacity-labels.txt').read_text()

	assert '3.500000\t6.500000' in content
	assert 'European Greenfinch (95%)' in content


def test_csv_format_escapes_commas_in_names (tmp_path):
	audio = tmp_path / 'test.wav'
	audio.touch()

	listener = birdnetpy.core.Listener(annotate='csv')
	f = listener._open_annotation_file(str(audio))

	try:
		listener._write_annotation(f, [_detection(0, 'Species, with comma', 'Genus, sp', 0.95)], 5.0)
	finally:
		f.close()

	with open(tmp_path / 'test.birdnetpy.csv') as fh:
		rows = list(csv.reader(fh))

	assert rows[0] == ['start_s', 'end_s', 'english_name', 'latin_name', 'confidence']
	assert rows[1][2] == 'Species, with comma'
	assert rows[1][3] == 'Genus, sp'
	assert rows[1][4] == '0.95'


def test_raven_format (tmp_path):
	audio = tmp_path / 'test.wav'
	audio.touch()

	listener = birdnetpy.core.Listener(annotate='raven')
	f = listener._open_annotation_file(str(audio))

	try:
		listener._write_annotation(f, [_detection(0, 'A', 'B', 0.9)], 10.0)
		listener._write_annotation(f, [_detection(1, 'C', 'D', 0.8)], 15.0)
	finally:
		f.close()

	content = (tmp_path / 'test.raven-selections.txt').read_text()
	lines = content.strip().split('\n')

	assert lines[0].startswith('Selection\tView\tChannel')
	assert lines[1].split('\t')[0] == '1'
	assert lines[2].split('\t')[0] == '2'


def test_reaper_format_escapes_commas_in_names (tmp_path):
	audio = tmp_path / 'test.wav'
	audio.touch()

	listener = birdnetpy.core.Listener(annotate='reaper')
	f = listener._open_annotation_file(str(audio))

	try:
		listener._write_annotation(f, [_detection(0, 'Species, with comma', 'Genus sp.', 0.9)], 3600.5)
	finally:
		f.close()

	with open(tmp_path / 'test.reaper-markers.csv') as fh:
		rows = list(csv.reader(fh))

	assert rows[0] == ['#', 'Name', 'Start', 'End', 'Length', 'Color']
	assert rows[1][0] == 'R1'
	assert 'Species, with comma' in rows[1][1]


def test_srt_format (tmp_path):
	audio = tmp_path / 'test.wav'
	audio.touch()

	listener = birdnetpy.core.Listener(annotate='srt')
	f = listener._open_annotation_file(str(audio))

	try:
		listener._write_annotation(f, [_detection(0, 'A', 'B', 0.9)], 5.0)
		listener._write_annotation(f, [_detection(1, 'C', 'D', 0.8)], 10.0)
	finally:
		f.close()

	content = (tmp_path / 'test.srt').read_text()

	assert '1\n00:00:03,500 --> 00:00:06,500\nA (90%)' in content
	assert '2\n00:00:08,500 --> 00:00:11,500\nC (80%)' in content


def test_annotation_file_is_flushed (tmp_path):

	"""The annotation writer flushes after each call so partial results survive a mid-run crash."""

	audio = tmp_path / 'test.wav'
	audio.touch()

	listener = birdnetpy.core.Listener(annotate='audacity')
	f = listener._open_annotation_file(str(audio))

	try:

		listener._write_annotation(f, [_detection(0, 'A', 'B', 0.9)], 5.0)

		# Read while the file handle is still open — flush should have happened.

		content = (tmp_path / 'test.audacity-labels.txt').read_text()

		assert 'A (90%)' in content

	finally:
		f.close()


def test_window_start_clamped_to_zero (tmp_path):

	"""If a detection lands earlier than half a window, window_start should clamp to 0 rather than going negative."""

	audio = tmp_path / 'test.wav'
	audio.touch()

	listener = birdnetpy.core.Listener(annotate='audacity')
	f = listener._open_annotation_file(str(audio))

	try:
		listener._write_annotation(f, [_detection(0, 'A', 'B', 0.9)], 0.5)
	finally:
		f.close()

	content = (tmp_path / 'test.audacity-labels.txt').read_text()

	assert content.startswith('0.000000\t')
