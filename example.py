import matplotlib.pyplot as plt
import numpy as np
from trainer import Trainer
from loss import loss_DANN
from models import DANNModel
from data_loader import *
from metrics import AccuracyScore

train_gen_s, val_gen_s, test_gen_s =  create_data_generators("office-31",
                                                       'amazon',
                                                             batch_size=16)

train_gen_t, val_gen_t, test_gen_t =  create_data_generators("office-31",
                                                       'dslr',
                                                      batch_size=16)
model = DANNModel()
def only_loss(*args, **kwargs):
    loss, rich_loss = loss_DANN(*args, **kwargs)
    print(f"loss: {rich_loss}")
    return loss

class MMM:
    def __init__(self, metric):
        self.metric = metric

    def reset(self):
        self.metric.reset()

    def __call__(self, *args, **kwargs):
        print(f"Call metric with args:\n{args}\n{kwargs}")
        print("metric: ", self.metric(*args, **kwargs))

acc = AccuracyScore()
mmm = MMM(acc)

tr = Trainer(model, only_loss)
tr.fit(train_gen_s, train_gen_t,
       validation_data=[val_gen_s, val_gen_t],
       metrics=[mmm],
       steps_per_epoch=3)
