
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

    def _warnIf_forwardOutputsNTargets_haveNoSamePattern(self, forwardOutputs, targets):
        # kkk
        #  but the user should be warned "once" that he/she has not
        #  followed targets pattern; requires lots of things to take care for; like runTemp variable
        # kkk
        #  may move it to _flattenTargetsNOutputs_data where that targets _outputsStruct fillWithData;
        #  and 1. make a try block there this is because not to slow down here by checking it everyRun
        #  or 2. even better, so when giving error "once" is done we put it there
        #  note even "once" feature is not done 3. we can put it in setter of _outputsStruct(not preferred)
        if isinstance(targets, dict):
            if not isinstance(forwardOutputs, dict):
                targetsKeys = joinListWithComma(list(targets.keys()))
                raise ValueError(
                    f'targets of dataloader is a dict with {targetsKeys}' + \
                    ' so forwardOutputs of forward method must follow the same pattern')
            else:
                noNestedDictMsg = 'must not have more than one level of nesting dictionaries.' + \
                                  ' should be a normal dict with no dict inside'
                if isNestedDict(targets):
                    raise ValueError('targets of dataloader' + noNestedDictMsg)
                if isNestedDict(targets):
                    raise ValueError('outputs of forward' + noNestedDictMsg)

                # addTest2
                # check if forwardOutputs has keys not defined in targets
                _, keysNotDefined = areItemsOfList1_InList2(
                    list(forwardOutputs.keys()), list(targets.keys()), giveNotInvolvedItems=True)
                if keysNotDefined:
                    spelling_forWordKey = spellPluralS(keysNotDefined, 'key')  # just for details
                    keysNotDefined = joinListWithComma(keysNotDefined)
                    raise ValueError(
                        f"you have defined {keysNotDefined} {spelling_forWordKey} in output of forward" + \
                        " which are not in targets of dataloader")

                # addTest2
                # warn if targets has keys not defined in forwardOutputs
                # the case users don't not to use all of the keys in the targets of dataloader
                _, keysNotDefined2 = areItemsOfList1_InList2(
                    list(targets.keys()), list(forwardOutputs.keys()), giveNotInvolvedItems=True)
                if keysNotDefined2:
                    # goodToHave1#kkk
                    #  warn just once for each train run
                    spelling_forWordKey = spellPluralS(keysNotDefined2, 'key')  # just for details
                    keysNotDefined2 = joinListWithComma(keysNotDefined2)
                    Warn.error(
                        f'targets of dataloader has {keysNotDefined2} {spelling_forWordKey} which' + \
                        ' are not taken care for in outputs of forward')

        else:
            pass
            # do it later
            # kkk what about when they are tensor

    def _getLossName(self, stepPhase, loss_):
        return snakeToCamel(stepPhase + type(loss_).__name__)


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
