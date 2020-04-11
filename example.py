from trainer import Trainer
from loss import loss_DANN
from models import DANNModel
from data_loader import create_data_generators
from metrics import AccuracyScoreFromLogits
from utils.callbacks import simple_callback


def only_loss(*args, **kwargs):
    loss, rich_loss = loss_DANN(*args, **kwargs)
    print(f"loss: {rich_loss}")
    return loss


class DebugMetric:
    score = 1
    name = 'test'

    def __init__(self, metric):
        self.metric = metric

    def reset(self):
        self.metric.reset()

    def __call__(self, *args, **kwargs):
        pass
        print(f"Call metric with args:\n{args}\n{kwargs}")
        print("metric: ", self.metric(*args, **kwargs))


if __name__ == '__main__':
    train_gen_s, val_gen_s, test_gen_s = create_data_generators("office-31",
                                                                'amazon',
                                                                batch_size=16,
                                                                infinite_train=True)

    train_gen_t, val_gen_t, test_gen_t = create_data_generators("office-31",
                                                                'dslr',
                                                                batch_size=16,
                                                                infinite_train=True)
    model = DANNModel()
    acc = AccuracyScoreFromLogits()
    mmm = DebugMetric(acc)

    tr = Trainer(model, only_loss)
    tr.fit(train_gen_s, train_gen_t,
           n_epochs=5,
           validation_data=[val_gen_s, val_gen_t],
           metrics=[acc],
           steps_per_epoch=1,
           callback=simple_callback)
