import copy
from typing import List, Union

import torch
import torch.nn as nn

from dataPrep.dataloader import _NestedDictStruct
from utils.typeCheck import argValidator
from utils.vAnnGeneralUtils import snakeToCamel, areItemsOfList1_InList2, joinListWithComma, \
    spellPluralS, isNestedDict
from utils.warnings import Warn

class _NewWrapper_loss:
    def __init__(self, **kwargs):
        self.showProgressBar = kwargs.get('showProgressBar', True)
        self.lossFuncs = []

    @property
    def lossFuncs(self):
        return self._lossFuncs

    @lossFuncs.setter
    @argValidator
    def lossFuncs(self, value: List[nn.modules.loss._Loss]):
        self._lossFuncs = value

    @property
    def _outputsStruct(self):
        # always return a copy, so it would be untouched
        return copy.deepcopy(self.__outputsStruct)

    @_outputsStruct.setter
    @argValidator
    def _outputsStruct(self, value: Union[_NestedDictStruct, None]):
        self.__outputsStruct = value

    @argValidator
    def _calculateLosses(self, loss, forwardOutputs: Union[torch.Tensor, dict],
                         targets: Union[torch.Tensor, dict]):

        # warn or error that targets and forwardOutputs based on their dict keys
        self._warnIf_forwardOutputsNTargets_haveNoSamePattern(forwardOutputs, targets)

        # bugPotentialCheck1 addTest1
        #  gives error without setting lossFuncs;
        #  should think is having lossFuncs is a must or I can make it optional
        calculatedLosses = []
        if not self.lossFuncs:  # guard clause #kkk this is related to allowing self.lossFuncs to be empty
            return calculatedLosses, loss

        if not hasattr(self, '_outputsStruct') or not self._outputsStruct:
            self._outputsStruct = _NestedDictStruct(forwardOutputs)
            # cccDevStruct
            #  this is making a structure for nestedDicts(also works with other types of data);
            #  so later in _getFlattedData we can fill it with data
            # cccDevStruct
            #  to get _outputsStruct only once in order not to
            #  slow down the code by doing same thing every time
            # cccDevAlgo
            #  note the self._outputsStruct is built upon forwardOutputs and not targets
            #  as we also let users not to use all of the keys in the targets of dataloader
            #  and leave some unused
            # kkk
            #  where the message "once" should be applied(in the case of next features
            #  which implements this) this part also should run do the checks
            #  of targetsNForwardOutputs having same pattern again

        outputsFlatData, targetsFlatData = self._flattenTargetsNOutputs_data(forwardOutputs,
                                                                             targets)

        for i, lossFunc_ in enumerate(self.lossFuncs):
            lossRes = sum(lossFunc_(output, target) for output, target in
                          zip(outputsFlatData, targetsFlatData))

            calculatedLosses.append(lossRes)

            # cccDevAlgo
            #  only results of first loss function is returned; therefore, backprop is done on those
            if i == 0:
                returnedLoss = calculatedLosses[0]

        return returnedLoss, calculatedLosses

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

    def _logLosses(self, calculatedLosses, stepPhase):
        # kkk
        #  take care of unsatisfaction with logs of preRunTests here
        # kkk2
        #  should I add more options like to be able to log.
        #  the problem with this is that after features are added they are
        #  outside of trainer so I may add my implementation of trainer
        # kkk2
        #  on_step=True for low important runs is not necessary but its important
        #  for main runs: but I avoid complexity and put on_step one for train
        on_step = True if stepPhase == 'train' else False
        for i, loss_ in enumerate(self.lossFuncs):
            # goodToHave3
            #  somewhere it logs `epochs`(and brings it in tensorboard) which I dont want
            self.log(self._getLossName(stepPhase, loss_),
                     calculatedLosses[i].to(torch.device('cpu')).item(),
                     # kkk3 why still metrics are in cuda
                     on_epoch=True, on_step=on_step, prog_bar=self.showProgressBar)

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
