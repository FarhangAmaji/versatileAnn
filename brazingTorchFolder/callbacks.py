import copy

import pytorch_lightning as pl
from torch.optim.lr_scheduler import LambdaLR


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


class SchedulerChanger:
    """Context manager to temporarily change PyTorch Lightning module's schedulers.

    Args:
        module (LightningModule): The PyTorch Lightning module instance.
        new_schedulers (list): The new list of schedulers to use within the context.
    """

    def __init__(self, module, new_schedulers):
        self.module = module
        self.original_schedulers = copy.deepcopy(module.schedulers)  # Deep copy
        self.new_schedulers = module.from_schedulers(new_schedulers) if hasattr(module,
                                                                                'from_schedulers') else new_schedulers

    def __enter__(self):
        self.module.schedulers = self.new_schedulers
        return self.module

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.module.schedulers = self.original_schedulers


class ReduceLROnPlateauScheduler(pl.callbacks.lr_scheduler.ReduceLROnPlateau):
    def __init__(self, monitor, mode='min', factor=0.1, patience=10, verbose=False,
                 threshold=1e-4, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-8):
        super().__init__(monitor, mode, factor, patience, verbose, threshold, threshold_mode,
                         cooldown, min_lr, eps)

    def on_validation_end(self, trainer, pl_module):
        self._reduce_lr(trainer, pl_module)
