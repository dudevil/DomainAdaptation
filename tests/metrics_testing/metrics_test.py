import torch
import numpy as np
from metrics import accuracy_score


def test_accuracy_score_simple():
    y_true_np = np.array([1, 0, 1, 0])
    y_pred_np = np.array([1, 1, 0, 0])
    y_true = torch.Tensor(y_true_np)
    y_pred = torch.Tensor(y_pred_np)
    acc = accuracy_score(y_pred, y_true)
    assert acc == 0.5


def test_accuracy_score_2d():
    y_true_np = np.array([
        [1, 0, 1, 0],
        [1, 0, 1, 0],
    ])
    y_pred_np = np.array([
        [1, 1, 0, 0],
        [1, 1, 0, 0],
    ])
    y_true = torch.Tensor(y_true_np)
    y_pred = torch.Tensor(y_pred_np)
    acc = accuracy_score(y_pred, y_true)
    assert acc == 0.5
