import copy
from typing import Union

import torch

from dataPrep.dataloader import _NestedDictStruct
from projectUtils.dataTypeUtils.dict import isNestedDict
from projectUtils.dataTypeUtils.list import areItemsOfList1_InList2
from projectUtils.dataTypeUtils.str import snakeToCamel, joinListWithComma, spellPluralS
from projectUtils.misc import _allowOnlyCreationOf_ChildrenInstances
from projectUtils.typeCheck import argValidator
from projectUtils.warnings import Warn


class _BrazingTorch_loss_inner:
    def __init__(self):

        # not allowing this class to have direct instance
        _allowOnlyCreationOf_ChildrenInstances(self, _BrazingTorch_loss_inner)

    @property
    def _outputsStruct(self):
        # always return a copy, so it would be untouched
        return copy.deepcopy(self.__outputsStruct)

    @_outputsStruct.setter
    @argValidator
    def _outputsStruct(self, value: Union[_NestedDictStruct, None]):
        self.__outputsStruct = value

    # ---- _calculateLosses
    @argValidator
    def _calculateLosses(self, forwardOutputs: Union[torch.Tensor, dict],
                         targets: Union[torch.Tensor, dict]):
        # bugPotn1 addTest1
        #  gives error without setting lossFuncs;
        #  should think is having lossFuncs is a must or I can make it optional
        calculatedLosses = []
        if not self.lossFuncs:  # guard clause #kkk this is related to allowing self.lossFuncs to be empty
            return None, calculatedLosses

        if not hasattr(self, '_outputsStruct') or not self._outputsStruct:
            self._outputsStruct = _NestedDictStruct(forwardOutputs)

            # warn or error that targets and forwardOutputs based on their dict keys
            self._warnIf_forwardOutputsNTargets_haveNoSamePattern(forwardOutputs, targets)
            # ccc1
            #  this is making a structure for nestedDicts(also works with other types of data);
            #  so later in _getFlattedData we can fill it with data
            # ccc1
            #  to get _outputsStruct only once in order not to
            #  slow down the code by doing same thing every time
            # ccc1
            #  note the self._outputsStruct is built upon forwardOutputs and not targets
            #  as we also let users not to use all of the keys in the targets of dataloader
            #  and leave some unused

        outputsFlatData, targetsFlatData = self._flattenTargetsNOutputs_data(forwardOutputs,
                                                                             targets)

        for i, lossFunc_ in enumerate(self.lossFuncs):
            # adding all different losses to a single number which is total loss
            lossRes = sum(lossFunc_(output, target) for output, target in
                          zip(outputsFlatData, targetsFlatData))

            calculatedLosses.append(lossRes)

            # ccc1
            #  only results of first loss function is returned; therefore, backprop is done on those
            if i == 0:
                returnedLoss = calculatedLosses[0]
                # add regularizations to loss
                # ccc1
                #  the regularization is not just added in 'train' phase. even though in
                #  other phases it might slow down but in order to have same scale of losses
                #  in all phases, it's better to add regularizations to loss in all phases
                if 'operationalRegularizations_isSet' not in self._tempVarRun_allPhases_hidden:
                    # cccWhat
                    #  note _setOperationalRegularizations is set once at beginning of each run
                    #  which means it's calculated only once at each run
                    #  also renews _operationalRegularizations because the different regularizations
                    #  may be added between different runs
                    self._setOperationalRegularizations()
                    self._tempVarRun_allPhases_hidden['operationalRegularizations_isSet'] = True

                returnedLoss = self.addRegularizationsToLoss(returnedLoss)

        return returnedLoss, calculatedLosses

    def _flattenTargetsNOutputs_data(self, forwardOutputs, targets):
        # ccc1
        #  because _NestedDictStruct also used here doesn't make it obligatory
        #  to have VAnnTsDataloader. and works with any dataloader

        try:
            outputsFlatData = self._getFlattedData(forwardOutputs)
        except:
            # ccc1
            #  in rare cases there is a possibility that the structure of forwardOutputs get
            #  changed. even though it slows down the code, I think its better to to keep it safe.
            #  so if it fails to fillData to _outputsStruct we try to remake _outputsStruct
            self._outputsStruct = _NestedDictStruct(forwardOutputs)
            self._warnIf_forwardOutputsNTargets_haveNoSamePattern(forwardOutputs, targets)
            outputsFlatData = self._getFlattedData(forwardOutputs)

        if isinstance(forwardOutputs, dict):
            # ccc1
            #  this is for the feature of letting user not to use all keys of targets of dataloader
            # only keep keys which are in outputs
            targets = {key: targets[key] for key in forwardOutputs.keys()}

        targetsFlatData = self._getFlattedData(targets)
        if len(outputsFlatData) != len(targetsFlatData):
            raise ValueError('mismatch in lens of outputsFlatData and targetsFlatData')
        return outputsFlatData, targetsFlatData

    def _getFlattedData(self, data):
        outputsFlatData = self._outputsStruct
        outputsFlatData.fillWithData(data)
        outputsFlatData = outputsFlatData.toList()
        return outputsFlatData

    def _warnIf_forwardOutputsNTargets_haveNoSamePattern(self, forwardOutputs, targets):
        # ccc1
        #  note this is called only when _outputsStruct is getting set
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
                    spelling_forWordKey = spellPluralS(keysNotDefined2, 'key')  # just for details
                    keysNotDefined2 = joinListWithComma(keysNotDefined2)
                    Warn.error(
                        f'targets of dataloader has {keysNotDefined2} {spelling_forWordKey} which' + \
                        ' are not taken care for in outputs of forward')

        else:
            pass
            # do it later
            # kkk what about when they are tensor

    # ----
    def _logLosses(self, calculatedLosses, phase):
        # addTest2 for phaseBased_logOptions
        # kkk
        #  take care of unsatisfaction with logs of preRunTests here
        # ccc1
        #  on_step=True for low important runs is not necessary but its important
        #  for main runs: but I avoid complexity and put on_step one for train
        logOptions = {"on_step": True if phase == 'train' else False,
                      'on_epoch': True, 'prog_bar': True}
        if hasattr(self, '_logOptions'):
            # pass _logOptions with phaseBased _logOptions format
            # ccc1
            #  for more info about phaseBased _logOptions take a look at modelFitter
            for akl, aklV in self._logOptions.items():
                if isinstance(aklV, dict):
                    if phase in aklV.keys():
                        logOptions.update({akl: aklV[phase]})
                    elif 'else' in aklV.keys():
                        logOptions.update({akl: aklV['else']})
                else:
                    logOptions.update({akl: self._logOptions[akl]})

        for i, loss_ in enumerate(self.lossFuncs):
            self.log(self._getLossName(phase, loss_),
                     calculatedLosses[i].to(torch.device('cpu')).item(),
                     **logOptions)
            # goodToHave3
            #  somewhere it logs `epochs`(and brings it in tensorboard) which I dont want
            # goodToHave3
            #  why still metrics are in cuda

    @staticmethod
    def _printFirstNLast_valLossChanges(callbacks_):
        epochValData = callbacks_[0].epochData['val']
        epochNum = list(epochValData.keys())
        epochNum = epochNum[-1] + 1
        metrics = list(epochValData[0].keys())
        infoMsg = f'over {epochNum} epochs:'
        infoMsg += ''.join([
            f'\n    "{metric}" changed from {epochValData[0][metric]:.5f} to {epochValData[epochNum - 1][metric]:.5f}'
            for metric in metrics])
        Warn.info(infoMsg)

    def _getLossName(self, phase, loss_):
        return snakeToCamel(phase + type(loss_).__name__)

    # ----
    def _setLossFuncs_ifNot(self, lossFuncs):
        if lossFuncs:
            self.lossFuncs = lossFuncs
            # cccUsage
            #  only first loss is used for backpropagation and others are just for logging
            # ccc1
            #  in the case outside of trainModel lossFuncs is been set, so if not passed would use them
            # anyway self.lossFuncs must be set

        if not self.lossFuncs:
            raise ValueError('lossFuncs must have set self.lossFuncs before running ' + \
                             'preRunTests or pass them to it')
