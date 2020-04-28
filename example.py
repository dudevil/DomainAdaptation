import torch
import os

from trainer import Trainer
from loss import loss_DANN
from models import DANNModel
from data_loader import create_data_generators
from metrics import AccuracyScoreFromLogits
from utils.callbacks import simple_callback, print_callback, ModelSaver, HistorySaver
from utils.schedulers import LRSchedulerSGD
import configs.dann_config as dann_config

# os.environ['CUDA_VISIBLE_DEVICES'] = '4, 5'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def debug_loss(*args, **kwargs):
    loss, rich_loss = loss_DANN(*args, **kwargs)
    loss_string = '   '.join(['{}: {:.5f}\t'.format(k, float(v)) for k, v in rich_loss.items()])
    print(f"step_loss: {loss_string}")
    return loss, rich_loss


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
    train_gen_s, val_gen_s, test_gen_s = create_data_generators(dann_config.DATASET,
                                                                dann_config.SOURCE_DOMAIN,
                                                                batch_size=dann_config.BATCH_SIZE,
                                                                infinite_train=True,
                                                                image_size=dann_config.IMAGE_SIZE,
                                                                num_workers=dann_config.NUM_WORKERS,
                                                                device=device)

    train_gen_t, val_gen_t, test_gen_t = create_data_generators(dann_config.DATASET,
                                                                dann_config.TARGET_DOMAIN,
                                                                batch_size=dann_config.BATCH_SIZE,
                                                                infinite_train=True,
                                                                image_size=dann_config.IMAGE_SIZE,
                                                                num_workers=dann_config.NUM_WORKERS,
                                                                device=device)

    model = DANNModel().to(device)
    acc = AccuracyScoreFromLogits()
    mmm = DebugMetric(acc)

    scheduler = LRSchedulerSGD()
    tr = Trainer(model, loss_DANN)
    tr.fit(train_gen_s, train_gen_t,
           n_epochs=dann_config.N_EPOCHS,
           validation_data=[val_gen_s, val_gen_t],
           metrics=[acc],
           steps_per_epoch=dann_config.STEPS_PER_EPOCH,
           val_freq=dann_config.VAL_FREQ,
           opt='sgd',
           opt_kwargs={'lr': 0.01, 'momentum': 0.9},
           lr_scheduler=scheduler,
           callbacks=[print_callback(watch=["loss", "domain_loss", "val_loss",
                                            "val_domain_loss", 'trg_metrics', 'src_metrics']),
                      ModelSaver('DANN', dann_config.SAVE_MODEL_FREQ),
                      HistorySaver('log_with_sgd', dann_config.VAL_FREQ,
                                   extra_losses={'domain_loss': ['domain_loss', 'val_domain_loss'],
                                                 'train_domain_loss': ['domain_loss_on_src', 'domain_loss_on_trg']})])
