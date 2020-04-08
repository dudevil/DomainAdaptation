import torch
import numpy as np
from utils import measure_metric
from metrics import AccuracyScore


class MockModel:
    def __init__(self, predicts):
        self.predicts = predicts
        self.ind = 0

    def predict(self, X):
        self.ind += 1
        return torch.Tensor(self.predicts[self.ind - 1])


def dateset_gen(dataset):
    for X, y in dataset:
        yield torch.Tensor(X), torch.Tensor(y)


def test_measure_metric():
    dataset_data1 = [
        [1, np.array([0, 0, 0, 0])],
        [1, np.array([1, 1, 1, 1])],
    ]

    pred1 = [
        np.array([1, 1, 0, 0]),
        np.array([1, 1, 0, 0]),
    ]

    pred2 = [
        np.array([1, 1, 1, 1]),
        np.array([0, 0, 0, 0]),
    ]

    model1 = MockModel(pred1)
    dataset1 = dateset_gen(dataset_data1)

    model2 = MockModel(pred2)
    dataset2 = dateset_gen(dataset_data1)

    argument_dict = [
        {
            'topic': "A-C",
            'load_model': lambda: model1,
            'load_dataset': lambda: dataset1
        },
        {
            'topic': "A-W",
            'load_model': lambda: model2,
            'load_dataset': lambda: dataset2
        }
    ]

    res = measure_metric(AccuracyScore, argument_dict)
    assert res == {'A-C': 0.5, 'A-W': 0.0}
