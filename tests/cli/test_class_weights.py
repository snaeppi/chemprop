"""Tests for CLI functionality of class weights feature."""

import json
from pathlib import Path

from configargparse import ArgumentError
import pytest
import torch

from chemprop.cli.main import main
from chemprop.models.model import MPNN

pytestmark = pytest.mark.CLI
EPOCHS = "1"
WARMUP = "0"


def build_train_args(data_path: str, task_type: str = "classification") -> list[str]:
    return [
        "chemprop",
        "train",
        "-i",
        data_path,
        "--epochs",
        EPOCHS,
        "--warmup-epochs",
        WARMUP,
        "--num-workers",
        "0",
        "--accelerator",
        "cpu",
        "--task-type",
        task_type,
    ]


@pytest.fixture
def data_path(data_dir):
    return str(data_dir / "classification" / "mol.csv")


@pytest.fixture
def class_weights_json(tmp_path):
    """Create a temporary JSON file with class weights."""
    weights = {
        "NR-AhR": [1.0, 2.0],
        "NR-ER": [1.0, 3.0],
        "SR-ARE": [1.0, 1.5],
        "SR-MMP": [2.0, 1.0],
    }
    json_path = tmp_path / "class_weights.json"
    with open(json_path, "w") as f:
        json.dump(weights, f)
    return str(json_path)


def test_train_with_class_weights_json(monkeypatch, data_path, class_weights_json):
    """Test training with class weights from JSON file."""
    args = build_train_args(data_path)
    args += ["--class-weights-path", class_weights_json]

    with monkeypatch.context() as m:
        m.setattr("sys.argv", args)
        main()


def test_train_with_pos_weight(monkeypatch, data_path):
    """Test training with pos-weight argument."""
    args = build_train_args(data_path)
    args += ["--pos-weight", "2.0", "3.0", "1.5", "4.0"]

    with monkeypatch.context() as m:
        m.setattr("sys.argv", args)
        main()


def test_train_with_auto_class_weights_balanced(monkeypatch, data_path):
    """Test training with automatic balanced class weights."""
    args = build_train_args(data_path)
    args += ["--auto-class-weights", "balanced"]

    with monkeypatch.context() as m:
        m.setattr("sys.argv", args)
        main()


def test_train_with_auto_class_weights_inverse(monkeypatch, data_path):
    """Test training with automatic inverse frequency class weights."""
    args = build_train_args(data_path)
    args += ["--auto-class-weights", "inverse"]

    with monkeypatch.context() as m:
        m.setattr("sys.argv", args)
        main()


def test_train_class_weights_mutually_exclusive(
    monkeypatch, data_path, class_weights_json
):
    """Test that class-weights-path and pos-weight are mutually exclusive."""
    args = [
        "chemprop",
        "train",
        "-i",
        data_path,
        "--epochs",
        EPOCHS,
        "--num-workers",
        "0",
        "--task-type",
        "classification",
        "--class-weights-path",
        class_weights_json,
        "--pos-weight",
        "2.0",
        "3.0",
        "1.5",
        "4.0",
    ]

    with monkeypatch.context() as m:
        m.setattr("sys.argv", args)
        with pytest.raises(ArgumentError):
            main()


def test_train_class_weights_and_auto_mutually_exclusive(
    monkeypatch, data_path, class_weights_json
):
    """Test that class-weights-path and auto-class-weights are mutually exclusive."""
    args = build_train_args(data_path)
    args += ["--class-weights-path", class_weights_json, "--auto-class-weights", "balanced"]

    with monkeypatch.context() as m:
        m.setattr("sys.argv", args)
        with pytest.raises(ArgumentError):
            main()


def test_train_pos_weight_wrong_task_type(monkeypatch, data_path):
    """Test that pos-weight fails for non-classification tasks."""
    args = build_train_args(data_path, task_type="multiclass")
    args += [
        "--multiclass-num-classes",
        "3",
        "--pos-weight",
        "2.0",
        "3.0",
        "1.5",
        "4.0",
    ]

    with monkeypatch.context() as m:
        m.setattr("sys.argv", args)
        with pytest.raises(ArgumentError):
            main()


def test_train_class_weights_file_not_found(monkeypatch, data_path):
    """Test that training fails with non-existent class weights file."""
    args = build_train_args(data_path)
    args += ["--class-weights-path", "/nonexistent/path/to/weights.json"]

    with monkeypatch.context() as m:
        m.setattr("sys.argv", args)
        with pytest.raises(ArgumentError):
            main()


def test_train_pos_weight_wrong_number(monkeypatch, data_path):
    """Test that pos-weight fails when number doesn't match tasks."""
    args = build_train_args(data_path)
    args += ["--pos-weight", "2.0"]  # Only one weight for 4 tasks

    with monkeypatch.context() as m:
        m.setattr("sys.argv", args)
        with pytest.raises(ArgumentError):
            main()


def test_train_multiclass_with_class_weights(monkeypatch, tmp_path):
    """Test training multiclass with class weights."""
    # Create a simple multiclass dataset
    data_path = tmp_path / "multiclass.csv"
    with open(data_path, "w") as f:
        f.write("smiles,target\n")
        f.write("C,0\n")
        f.write("CC,1\n")
        f.write("CCC,2\n")
        f.write("CCCC,0\n")
        f.write("CCCCC,1\n")

    # Create class weights
    class_weights_json = tmp_path / "multiclass_weights.json"
    with open(class_weights_json, "w") as f:
        json.dump({"target": [1.0, 2.0, 3.0]}, f)

    args = build_train_args(str(data_path), task_type="multiclass")
    args += [
        "--multiclass-num-classes",
        "3",
        "--class-weights-path",
        str(class_weights_json),
        "--split-sizes",
        "0.6",
        "0.2",
        "0.2",
        "--batch-size",
        "1",
    ]

    with monkeypatch.context() as m:
        m.setattr("sys.argv", args)
        main()


def test_train_multiclass_with_auto_class_weights(monkeypatch, tmp_path):
    """Test training multiclass with automatic class weights."""
    # Create a simple multiclass dataset
    data_path = tmp_path / "multiclass.csv"
    with open(data_path, "w") as f:
        f.write("smiles,target\n")
        f.write("C,0\n")
        f.write("CC,1\n")
        f.write("CCC,2\n")
        f.write("CCCC,0\n")
        f.write("CCCCC,1\n")

    args = build_train_args(str(data_path), task_type="multiclass")
    args += [
        "--multiclass-num-classes",
        "3",
        "--auto-class-weights",
        "balanced",
        "--split-sizes",
        "0.6",
        "0.2",
        "0.2",
        "--batch-size",
        "1",
    ]

    with monkeypatch.context() as m:
        m.setattr("sys.argv", args)
        main()


def test_train_pos_weight_sets_buffer(monkeypatch, data_path, tmp_path):
    """Ensure pos_weight maps to the loss buffer for all tasks."""
    pos_weights = ["1.0", "2.0", "3.0", "4.0"]
    args = build_train_args(data_path)
    args += ["--pos-weight", *pos_weights, "--save-dir", str(tmp_path)]

    with monkeypatch.context() as m:
        m.setattr("sys.argv", args)
        main()

    checkpoint_path = tmp_path / "model_0" / "checkpoints" / "last.ckpt"
    model = MPNN.load_from_checkpoint(checkpoint_path)
    expected = torch.tensor([float(w) for w in pos_weights])
    assert torch.allclose(model.criterion.pos_weight.squeeze(), expected)


def test_train_class_weights_set_pos_weight(monkeypatch, data_path, tmp_path):
    """Ensure class weights map correctly to BCE pos_weight."""
    weights = {
        "NR-AhR": [1.0, 4.0],
        "NR-ER": [0.5, 2.0],
        "SR-ARE": [2.0, 1.0],
        "SR-MMP": [1.0, 1.25],
    }
    class_weights_path = tmp_path / "class_weights.json"
    with open(class_weights_path, "w") as f:
        json.dump(weights, f)

    args = build_train_args(data_path)
    args += [
        "--class-weights-path",
        str(class_weights_path),
        "--save-dir",
        str(tmp_path),
    ]

    with monkeypatch.context() as m:
        m.setattr("sys.argv", args)
        main()

    checkpoint_path = tmp_path / "model_0" / "checkpoints" / "last.ckpt"
    model = MPNN.load_from_checkpoint(checkpoint_path)
    expected = torch.tensor([4.0, 4.0, 0.5, 1.25])
    assert torch.allclose(model.criterion.pos_weight.squeeze(), expected)
