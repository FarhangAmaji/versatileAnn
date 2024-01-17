import pytorch_lightning as pl


class StoreEpochData(pl.Callback):
    def __init__(self):
        super().__init__()
        self.epochData = {}

    def on_train_epoch_end(self, trainer, pl_module):
        stepPhase = 'train'
        self._metricUpdate(stepPhase, trainer)

    def on_validation_epoch_end(self, trainer, pl_module):
        stepPhase = 'val'
        self._metricUpdate(stepPhase, trainer)

    def on_test_epoch_end(self, trainer, pl_module):
        stepPhase = 'test'
        self._metricUpdate(stepPhase, trainer)

    def on_predict_epoch_end(self, trainer, pl_module):
        stepPhase = 'predict'
        self._metricUpdate(stepPhase, trainer)

    def _metricUpdate(self, stepPhase, trainer):
        metrics = {key: value for key, value in trainer.callback_metrics.items() if
                   key.startswith(stepPhase)}
        if stepPhase not in self.epochData:
            self.epochData.update({stepPhase: {}})
        self.epochData[stepPhase].update({trainer.current_epoch: metrics})
