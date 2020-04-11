import torch


class Trainer:
    def __init__(self, model, loss):
        self.model = model
        self.loss = loss
        self.epoch = 0
        self.last_epoch_history = None

    def _reset_last_epoch_history(self):
        self.last_epoch_history = {
            'loss': 0.0,
            'src_metrics': {},
            'trg_metrics': {}
        }

    def train_on_batch(self, src_batch, trg_batch, opt):
        self.model.train()
        batch = self._merge_batches(src_batch, trg_batch)
        metadata = {'epoch': self.epoch, 'n_epochs': self.n_epochs}
        loss = self.loss(self.model, batch, **metadata)
        self.last_epoch_history['loss'] += loss

        opt.zero_grad()
        loss.backward()
        opt.step()

    def _merge_batches(self, src_batch, trg_batch):
        src_images, src_classes = src_batch
        trg_images, trg_classes = trg_batch
        batch = dict()
        batch['src_images'] = src_images
        batch['trg_images'] = trg_images
        batch['src_classes'] = src_classes
        batch['trg_classes'] = trg_classes
        return batch

    def fit(self, src_data, trg_data, n_epochs=1000, steps_per_epoch=100, val_freq=1,
            opt='adam', opt_kwargs=None, validation_data=None, metrics=None, callback=None):
        self.n_epochs = n_epochs

        if opt_kwargs is None:
            opt_kwargs = dict()

        if opt == 'adam':
            opt = torch.optim.Adam(self.model.parameters(), **opt_kwargs)
        else:
            raise NotImplementedError

        for self.epoch in range(self.epoch, n_epochs):
            self._reset_last_epoch_history()
            for step, (src_batch, trg_batch) in enumerate(zip(src_data, trg_data)):
                if step == steps_per_epoch:
                    break
                self.train_on_batch(src_batch, trg_batch, opt)

            if metrics is not None and self.epoch % val_freq == 0:
                src_val_data, trg_val_data = validation_data
                self.model.eval()
                if src_val_data is not None:
                    src_metrics = self.score(src_val_data, metrics)
                    self.last_epoch_history['src_metrics'] = src_metrics
                if trg_val_data is not None:
                    trg_metrics = self.score(trg_val_data, metrics)
                    self.last_epoch_history['trg_metrics'] = trg_metrics

            if callback is not None:
                self.last_epoch_history['loss'] /= steps_per_epoch
                callback(self.last_epoch_history, self.epoch, n_epochs)

    def score(self, data, metrics):
        for metric in metrics:
            metric.reset()

        data.reload_iterator()
        for images, true_classes in data:
            pred_classes = self.model.predict(images)
            for metric in metrics:
                metric(true_classes, pred_classes)
        return {metric.name: metric.score for metric in metrics}

    def predict(self, data):
        predictions = []
        for batch in data:
            predictions.append(self.model.predict(batch))
        return torch.cat(predictions)
