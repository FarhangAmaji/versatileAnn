
class progressBarTempOff:
    def __init__(self, instance):
        self.instance = instance

    def __enter__(self):
        self.originalVal = self.instance.showProgressBar
        self.instance.showProgressBar = False

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.instance.showProgressBar = self.originalVal


class _NewWrapper_loss(ABC):  # kkk1 do it later
    def __init__(self, **kwargs):

    def _flattenTargetsNOutputs_data(self, forwardOutputs, targets):
        # cccDevStruct
        #  because _NestedDictStruct also used here doesn't make it obligatory
        #  to have VAnnTsDataloader. and works with any dataloader

        # cccDevAlgo
        #  in rare cases there is a possibility that the structure of forwardOutputs get changed
        #  even though it slows down the code, I think its better to to keep it safe.
        #  so if it fails to fillData to _outputsStruct we try to remake _outputsStruct
        try:
            outputsFlatData = self._getFlattedData(forwardOutputs)
        except:
            # kkk
            #  where the message "once" should be applied(in the case of next features
            #  which implements this) this part also should run do the checks
            #  of targetsNForwardOutputs having same pattern again
            self._outputsStruct = _NestedDictStruct(forwardOutputs)
            outputsFlatData = self._getFlattedData(forwardOutputs)



        if isinstance(forwardOutputs, dict):
            # cccDevAlgo
            #  this is for the feature of letting user not to use all keys of targets of dataloader
            forwardOutputs_keys = list(forwardOutputs.keys())
            # only keep keys which are in outputs
            targets = {key: targets[key] for key in forwardOutputs.keys()}

        targetsFlatData = self._getFlattedData(targets)
        assert len(outputsFlatData) == len(targetsFlatData), \
            'mismatch in lens of outputsFlatData and targetsFlatData'
        return outputsFlatData, targetsFlatData

    def _getFlattedData(self, data):
        outputsFlatData = self._outputsStruct
        outputsFlatData.fillWithData(data)
        outputsFlatData = outputsFlatData.toList()
        return outputsFlatData

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

# kkk maybe make a file for contextManagers in NewWrapper utils folder
class progressBarTempOff:
    def __init__(self, instance):
        self.instance = instance

    def __enter__(self):
        self.originalVal = self.instance.showProgressBar
        self.instance.showProgressBar = False

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.instance.showProgressBar = self.originalVal
