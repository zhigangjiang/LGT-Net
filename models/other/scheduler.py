"""
@Date: 2021/09/14
@description:
"""


class WarmupScheduler:
    def __init__(self, optimizer, lr_pow, init_lr, warmup_lr, warmup_step, max_step, **kwargs):
        self.lr_pow = lr_pow
        self.init_lr = init_lr
        self.running_lr = init_lr
        self.warmup_lr = warmup_lr
        self.warmup_step = warmup_step
        self.max_step = max_step
        self.optimizer = optimizer

    def step_update(self, cur_step):
        if cur_step < self.warmup_step:
            frac = cur_step / self.warmup_step
            step = self.warmup_lr - self.init_lr
            self.running_lr = self.init_lr + step * frac
        else:
            frac = (float(cur_step) - self.warmup_step) / (self.max_step - self.warmup_step)
            scale_running_lr = max((1. - frac), 0.) ** self.lr_pow
            self.running_lr = self.warmup_lr * scale_running_lr

        if self.optimizer is not None:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.running_lr


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    scheduler = WarmupScheduler(optimizer=None,
                                lr_pow=4,
                                init_lr=0.0000003,
                                warmup_lr=0.00003,
                                warmup_step=10000,
                                max_step=100000)

    x = []
    y = []
    for i in range(100000):
        if i == 10000-1:
            print()
        scheduler.step_update(i)
        x.append(i)
        y.append(scheduler.running_lr)
    plt.plot(x, y, linewidth=1)
    plt.show()
