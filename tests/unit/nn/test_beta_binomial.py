import math

import pytest
import torch
from scipy.stats import betabinom

from chemprop.nn import BetaBinomialFFN, BetaBinomialLoss


def test_predictor_constraints():
    batch_size = 5
    n_tasks = 3
    input_dim = 10

    model = BetaBinomialFFN(n_tasks=n_tasks, input_dim=input_dim, hidden_dim=32)

    inputs = torch.randn(batch_size, input_dim)

    # Inference: returns probability and concentration
    preds = model(inputs)
    p, conc = torch.unbind(preds, dim=2)

    assert p.shape == (batch_size, n_tasks)
    assert conc.shape == (batch_size, n_tasks)
    assert torch.all(p > 0) and torch.all(p < 1)
    assert torch.all(conc > 0)

    # Training: returns alpha, beta
    alpha, beta = torch.unbind(model.train_step(inputs), dim=2)
    assert torch.all(alpha > 0)
    assert torch.all(beta > 0)


def test_loss_matches_scipy():
    pytest.importorskip("scipy")

    loss_fn = BetaBinomialLoss()

    alpha_val = 5.0
    beta_val = 5.0
    preds = torch.tensor([[[alpha_val, beta_val]]], dtype=torch.float32)  # (1, 1, 2)

    n_val = 10.0
    rate_val = 0.5
    k_val = 5.0

    weights = torch.tensor([[n_val]], dtype=torch.float32)
    targets = torch.tensor([[rate_val]], dtype=torch.float32)
    mask = torch.ones_like(targets, dtype=torch.bool)
    lt_mask = torch.zeros_like(targets, dtype=torch.bool)
    gt_mask = torch.zeros_like(targets, dtype=torch.bool)

    torch_loss = loss_fn._calc_unreduced_loss(preds, targets, mask, weights, lt_mask, gt_mask)

    scipy_logprob = betabinom.logpmf(k_val, n_val, alpha_val, beta_val)
    expected_loss = -scipy_logprob / n_val

    assert math.isclose(torch_loss.item(), expected_loss, rel_tol=0, abs_tol=1e-5)


def test_gradient_flow():
    batch_size = 4
    n_tasks = 2
    input_dim = 6

    model = BetaBinomialFFN(n_tasks=n_tasks, input_dim=input_dim, hidden_dim=16)
    loss_fn = BetaBinomialLoss()

    inputs = torch.randn(batch_size, input_dim, requires_grad=True)
    preds = model.train_step(inputs)

    targets = torch.rand(batch_size, n_tasks)
    weights = torch.randint(1, 20, (batch_size,), dtype=torch.float)
    mask = torch.ones_like(targets, dtype=torch.bool)
    lt_mask = torch.zeros_like(targets, dtype=torch.bool)
    gt_mask = torch.zeros_like(targets, dtype=torch.bool)

    loss = loss_fn(preds, targets, mask, weights, lt_mask, gt_mask)
    loss.backward()

    assert inputs.grad is not None
    assert not torch.isnan(inputs.grad).any()


def test_extreme_values():
    loss_fn = BetaBinomialLoss()

    preds = torch.tensor([[[1.0, 1.0]]])  # alpha=1, beta=1 (uniform prior)
    mask = torch.ones([1, 1], dtype=torch.bool)
    lt_mask = torch.zeros([1, 1], dtype=torch.bool)
    gt_mask = torch.zeros([1, 1], dtype=torch.bool)

    # k = 0, n = 10
    targets_zero = torch.tensor([[0.0]])
    weights_ten = torch.tensor([[10.0]])
    loss_zero = loss_fn._calc_unreduced_loss(preds, targets_zero, mask, weights_ten, lt_mask, gt_mask)
    assert not torch.isnan(loss_zero).any()

    # k = n, n = 10
    targets_one = torch.tensor([[1.0]])
    loss_one = loss_fn._calc_unreduced_loss(preds, targets_one, mask, weights_ten, lt_mask, gt_mask)
    assert not torch.isnan(loss_one).any()

    assert torch.isclose(loss_zero, loss_one)
