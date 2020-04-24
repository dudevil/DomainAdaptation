class LRSchedulerSGD:
    def __init__(self, mu_0=0.01, alpha=10, beta=0.75):
        self.mu_0 = mu_0
        self.alpha = alpha
        self.beta = beta

    def step(self, opt, current_epoch, total_epoch):
        p = (current_epoch + 1.) / total_epoch
        mu_p = self.mu_0 / ((1 + self.alpha * p) ** self.beta)
        for group in opt.param_groups:
            group['lr'] = mu_p
