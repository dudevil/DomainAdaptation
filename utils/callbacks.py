import os


def simple_callback(model, epoch_log, current_epoch, total_epoch):
    train_loss = epoch_log['loss']
    trg_metrics = epoch_log['trg_metrics']
    src_metrics = epoch_log['src_metrics']
    message_head = f'Epoch {current_epoch+1}/{total_epoch}\n'
    message_loss = 'loss: {:<10}\t'.format(train_loss)
    message_src_metrics = ' '.join(['val_src_{}: {:<10}\t'.format(k, v) for k, v in src_metrics.items()])
    message_trg_metrics = ' '.join(['val_trg_{}: {:<10}\t'.format(k, v) for k, v in trg_metrics.items()])
    print(message_head + message_loss + message_src_metrics + message_trg_metrics)

class ModelSaver:
    def __init__(self, model_type, path="checkpoints"):
        self.model_type = model_type
        self.path = path
        if not os.path.exists(path):
            os.makedirs(path)
        if not os.path.exists(os.path.join(path, model_type)):
            os.makedirs(os.path.join(path, model_type))

    def __call__(self, model, epoch_log, current_epoch, total_epoch):
        import torch
        filename = os.path.join(self.path, self.model_type, "epoch_{}.pt".format(current_epoch))
        torch.save(model.state_dict(), filename)
