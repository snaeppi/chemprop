from types import SimpleNamespace

import numpy as np
import pytest
import torch

from chemprop.cli.train import compute_class_weights_from_data, prepare_class_weights


class DummyDataset:
    def __init__(self, Y):
        self.Y = Y


class DummyMulticomponentDataset:
    def __init__(self, Y):
        self.Y = Y
        self.datasets = [DummyDataset(Y)]


def test_compute_class_weights_binary_with_nans():
    Y = np.array([[1, 0], [1, 0], [0, 1], [np.nan, 1]], dtype=float)
    dataset = DummyDataset(Y)

    weights = compute_class_weights_from_data(
        dataset, method="balanced", task_type="classification"
    )

    expected = torch.tensor([[1.5, 0.75], [1.0, 1.0]])
    assert torch.allclose(weights, expected)


def test_compute_class_weights_multiclass_inverse_counts():
    Y = np.array([[0], [1], [2], [2]], dtype=float)
    dataset = DummyMulticomponentDataset(Y)

    weights = compute_class_weights_from_data(
        dataset, method="inverse", task_type="multiclass", n_classes=3
    )

    expected = torch.tensor([[4.0, 4.0, 2.0]])
    assert torch.allclose(weights, expected)


def test_prepare_class_weights_respects_target_order():
    args = SimpleNamespace(
        class_weights_dict={"b": [1.0, 3.0], "a": [2.0, 4.0]},
        pos_weight=None,
        auto_class_weights="none",
        task_type="classification",
        multiclass_num_classes=2,
    )

    weights = prepare_class_weights(args, target_columns=["a", "b"])

    expected = torch.tensor([[2.0, 4.0], [1.0, 3.0]])
    assert torch.equal(weights, expected)


def test_prepare_class_weights_requires_train_data_for_auto():
    args = SimpleNamespace(
        class_weights_dict=None,
        pos_weight=None,
        auto_class_weights="balanced",
        task_type="classification",
        multiclass_num_classes=2,
    )

    with pytest.raises(ValueError):
        prepare_class_weights(args, target_columns=["a"], train_dset=None)
