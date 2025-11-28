import pytest
from configargparse import ArgumentError, ArgumentParser

from chemprop.cli.train import add_train_args, process_train_args


def _make_parser():
    parser = ArgumentParser()
    return add_train_args(parser)


def test_task_weights_loaded_from_file(tmp_path):
    weights_file = tmp_path / "weights.json"
    weights_file.write_text("[1, 0.5, 3]")

    args = _make_parser().parse_args(["--task-weights-path", str(weights_file)])
    args = process_train_args(args)

    assert args.task_weights == [1.0, 0.5, 3.0]


def test_task_weights_file_conflicts_with_inline(tmp_path):
    weights_file = tmp_path / "weights.txt"
    weights_file.write_text("1 2 3")

    args = _make_parser().parse_args(
        ["--task-weights", "1", "2", "3", "--task-weights-path", str(weights_file)]
    )

    with pytest.raises(ArgumentError):
        process_train_args(args)
