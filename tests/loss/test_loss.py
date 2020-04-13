import pytest
import numpy as np
import torch

from loss import _loss_DANN, _loss_DANN_splitted
import configs.dann_config as dann_config


def test__loss_DANN():
    dann_config.IS_UNSUPERVISED = False
    class_logits = torch.Tensor(
        np.array([
            [1.0, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
            [10, 11, 12]
        ]))
    logprob_target = torch.Tensor(np.array([-5.1, 5, -6, 8]))
    instances_labels = np.array([1, 2, -100, 2], dtype='int')
    is_target = np.array([0, 1, 0, 1], dtype='int')
    assert abs(0.7448 - _loss_DANN(class_logits,
                                   logprob_target,
                                   instances_labels,
                                   is_target, 1, 1)) < 1e-4
    return True


def test__loss_DANN_splitted():
    dann_config.IS_UNSUPERVISED = False
    class_logits_src = torch.Tensor(
        np.array([
            [1.0, 2, 3],
            [7, 8, 9],
        ]))
    class_logits_trg = torch.Tensor(
        np.array([
            [4, 5, 6],
            [10, 11, 12]
        ]))
    logprob_target_src = torch.Tensor(np.array([-5.1, -6]))
    logprob_target_trg = torch.Tensor(np.array([5, 8]))
    true_labels_src = np.array([1, -100], dtype='int')
    true_labels_trg = np.array([2, 2], dtype='int')
    actual_loss, _ = _loss_DANN_splitted(
        class_logits_src,
        class_logits_trg,
        logprob_target_src,
        logprob_target_trg,
        true_labels_src,
        true_labels_trg, 1, 1
    )
    expected_loss = 0.7448
    assert abs(expected_loss - actual_loss) < 1e-4
    return True