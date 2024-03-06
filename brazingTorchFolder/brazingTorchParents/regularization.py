from typing import Optional, Union

import torch
from torch import nn

from brazingTorchFolder.brazingTorchParents.innerClassesWithoutPublicMethods.regularization_inner import \
    _BrazingTorch_regularization_inner
from brazingTorchFolder.utilsFolder.lossRegulator import LossRegulator
from projectUtils.globalVars import regularizationTypes
from projectUtils.misc import _allowOnlyCreationOf_ChildrenInstances
from projectUtils.typeCheck import argValidator
from projectUtils.warnings import Warn


class _BrazingTorch_regularization(_BrazingTorch_regularization_inner):
    _regularizationTypes = regularizationTypes
    nullRegulator = LossRegulator(LossRegulator.nullDictValue)

    @argValidator
    def __init__(self, generalRegularization: Optional[Union[LossRegulator, dict]] = None,
                 **kwargs):

        # not allowing this class to have direct instance
        _allowOnlyCreationOf_ChildrenInstances(self, _BrazingTorch_regularization)

        # ccc1
        #  - this module(BrazingTorch) by default sets some features like generalRegularization
        #       or gradient clipping
        #  - we have generalRegularization and specific layer generalRegularization
        #  - we have generalRegularization anyway but in _setOperationalRegularizations if the layer
        #       has layer specific so that one is gonna be applied
        # mustHave1
        #  do similiar things for gradient clipping
        if generalRegularization or self.noAdditionalOptions:
            self.generalRegularization = generalRegularization
        else:
            self.generalRegularization = {'type': 'l2', 'value': 1e-3}
            Warn.info('generalRegularization is not provided; so it is set to default ' + \
                      '"l2 regularization" with value of 1e-3' + \
                      '\nyou may either pass noAdditionalOptions=True to model or ' + \
                      'call .noGeneralRegularization method on model.' + \
                      '\nor set .generalRegularization to another value for i.e. ' + \
                      '{"type":"l1","value":.02}')

        self._specificLayerRegularization = {}
        self._operationalRegularizations = {}

    # ---- general regularization
    @property
    def generalRegularization(self):
        return self._generalRegularization

    @generalRegularization.setter
    @argValidator
    def generalRegularization(self, value: Optional[Union[LossRegulator, dict]]):
        # kkk2 if this is being set check that optimizer doesn't have any weight_decay
        # kkk2 in both check on the other one
        if isinstance(value, dict):
            self._generalRegularization = LossRegulator(value)
        elif isinstance(value, LossRegulator):
            self._generalRegularization = value
        else:  # None
            self._generalRegularization = LossRegulator(LossRegulator.nullDictValue)

        # it's not allowed to have weight_decay in optimizer and generalRegularization(ofc not None
        # version) together
        # addTest2
        if self._generalRegularization.type != 'None':
            if hasattr(self, '_optimizer'):
                if self._optimizer is not None:
                    if 'weight_decay' in self._optimizer.param_groups[0].keys():
                        if self._optimizer.param_groups[0]['weight_decay'] != 0:
                            self._optimizer.param_groups[0]['weight_decay'] = 0
                            Warn.warn("because the generalRegularization is set the " + \
                                      "weight_decay of optimizer has been set to 0")

    def noGeneralRegularization(self):
        self.generalRegularization = LossRegulator(LossRegulator.nullDictValue)

    # ---- specific layer regularizations
    @argValidator
    def addLayerRegularization(self, regDict: dict):
        # goodToHave1
        #  similar to goodToHave of _register_VAnnCustomLayers_regularizations:
        #  again won't work if the layer is not the main layer; even though
        #  adding it works but won't get affected
        # cccUsage
        #  this is for adding regularization to non-VAnnCustomLayer
        #  the format of regDict is {layer:{'type':type,'value':value}}
        #  or {layer:RegulatorObject}
        #  note this func can be used in model __init__ or even after that
        for layer, regVal in regDict.items():
            if not isinstance(layer, nn.Module):
                raise ValueError(f'{layer} layer must be an instance of nn.Module')

            # check regVal has correct reg format
            if not isinstance(regVal, LossRegulator):
                # assumes it's a dict with correct format
                regDict[layer] = LossRegulator(regVal)
                # if it has error the LossRegulator constructor will raise error

        for layer, regVal in regDict.items():
            # check does this layer exist in modules
            foundLayer = False
            for existingLayerName, existingLayer in self._modules.items():
                if layer is existingLayer:
                    foundLayer = True
                    layerName = existingLayerName
                    break
            if not foundLayer:
                raise ValueError(f'{layer} is not in layers of this model')

            self._specificLayerRegularization[layerName] = regVal

    def addRegularizationsToLoss(self, loss):
        addedReg = torch.tensor(0., requires_grad=True)
        for name, param in self.named_parameters():
            layerName = name.split('.')[0]
            if layerName in self._operationalRegularizations.keys():
                regularization = self._operationalRegularizations[layerName]

                addedReg = addedReg + regularization.addRegularizationToParam(param)
        return loss + addedReg
