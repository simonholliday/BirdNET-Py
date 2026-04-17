import pytest

import birdnetpy.core


def test_bundled_labels_load ():
	listener = birdnetpy.core.Listener()

	assert listener.num_source_labels > 6000
	assert len(listener.model_labels) > 6000


def test_non_bird_labels_are_flagged ():
	listener = birdnetpy.core.Listener()

	non_birds = [entry for entry in listener.model_labels.values() if not entry[2]]

	assert len(non_birds) > 0


def test_human_labels_are_flagged ():
	listener = birdnetpy.core.Listener()

	humans = [entry for entry in listener.model_labels.values() if entry[3]]

	assert len(humans) > 0


def test_exclude_file_removes_labels (tmp_path):
	listener = birdnetpy.core.Listener()

	# Pick any known label and construct the source line for it.

	index, (latin, english, _, _) = next(iter(listener.model_labels.items()))
	source_label = '%s_%s' % (latin, english)

	exclude_file = tmp_path / 'exclude.txt'
	exclude_file.write_text(source_label + '\n', encoding='utf-8')

	filtered = birdnetpy.core.Listener(exclude_label_file_path=str(exclude_file))

	assert index not in filtered.model_labels


def test_load_label_file_handles_none_path ():
	listener = birdnetpy.core.Listener()

	labels_set, labels_dict = listener._load_label_file(None)

	assert labels_set == set()
	assert labels_dict == {}


def test_load_label_file_skips_comments_and_blanks (tmp_path):
	listener = birdnetpy.core.Listener()

	f = tmp_path / 'labels.txt'
	f.write_text('# header comment\n\nfoo\nbar\n# trailing comment\n\n', encoding='utf-8')

	labels_set, labels_dict = listener._load_label_file(str(f))

	assert labels_set == {'foo', 'bar'}
	assert labels_dict == {0: 'foo', 1: 'bar'}


def test_load_label_file_warns_on_duplicates (tmp_path, caplog):
	listener = birdnetpy.core.Listener()

	f = tmp_path / 'labels.txt'
	f.write_text('foo\nbar\nfoo\n', encoding='utf-8')

	with caplog.at_level('WARNING', logger='birdnetpy.core'):
		listener._load_label_file(str(f))

	assert any('Duplicate label' in record.message for record in caplog.records)
