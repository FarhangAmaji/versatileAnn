import pytorch_lightning as pl


class StoreEpochData(pl.Callback):
    def __init__(self):
        super().__init__()
        self.epochData = {}

    def _metricUpdate(self, phase, trainer):
        metrics = {key: value for key, value in trainer.callback_metrics.items() if
                   key.startswith(phase)}
        if phase not in self.epochData:
            self.epochData.update({phase: {}})

        self.epochData[phase].update({trainer.current_epoch: metrics})

    def on_train_epoch_end(self, trainer, pl_module):
        phase = 'train'
        self._metricUpdate(phase, trainer)

    def on_validation_epoch_end(self, trainer, pl_module):
        phase = 'val'
        self._metricUpdate(phase, trainer)

    def on_test_epoch_end(self, trainer, pl_module):
        phase = 'test'
        self._metricUpdate(phase, trainer)

    def on_predict_epoch_end(self, trainer, pl_module):
        phase = 'predict'
        self._metricUpdate(phase, trainer)


class WarmUpScheduler(LambdaLR):
    """
    Gradual warm-up learning rate scheduler.

    Args:
        optimizer (torch.optim.Optimizer): Wrapped optimizer.
        warmUpEpochs (int): Number of epochs for warming up.
        last_epoch (int, optional): The index of last epoch. Default: -1.
    """

    def __init__(self, optimizer, warmUpEpochs, last_epoch=-1):
        def lr_lambda(current_epoch):
            if current_epoch < warmUpEpochs:
                return float(current_epoch) / float(warmUpEpochs)
            return 1.0

        super().__init__(optimizer, lr_lambda, last_epoch=last_epoch)


