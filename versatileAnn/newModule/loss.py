from abc import ABC

class _NewWrapper_loss(ABC):  # kkk1 do it later
    def __init__(self, **kwargs):
        pass
    @staticmethod
    def _printLossChanges(callbacks_):
        epochValData = callbacks_[0].epochData['val']
        epochNum = list(epochValData.keys())
        epochNum = epochNum[-1] + 1
        metrics = list(epochValData[0].keys())
        infoMsg = f'over {epochNum} epochs:'
        infoMsg += ''.join([
            f'\n    "{metric}" changed from {epochValData[0][metric]:.5f} to {epochValData[epochNum - 1][metric]:.5f}'
            for metric in metrics])
        Warn.info(infoMsg)
