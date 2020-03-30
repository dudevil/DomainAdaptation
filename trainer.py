import torch


class Trainer:
    def __init__(self, model, loss):
        self.model = model
        self.loss = loss

    def train_on_batch(self, batch, opt):
        loss = self.loss(self.model, batch)

        opt.zero_grad()
        loss.backward()
        opt.step()

    def fit(self, dataloader, n_epochs, opt_kwargs=dict(), opt='adam'):
        if opt == 'adam':
            opt = torch.optim.Adam(self.model.parameters(), **opt_kwargs)
        else:
            raise NotImplementedError

        for epoch in range(n_epochs):
            for batch in dataloader:
                self.train_on_batch(batch, opt)

    def predict_on_batch(self, batch):
        pass

    def predict(self, dataloader):
        predictions = []
        for batch in dataloader:
            predictions.append(self.predict_on_batch(batch))
        return torch.stack(predictions)
