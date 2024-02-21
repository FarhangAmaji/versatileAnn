from typing import Optional, Union

import torch
from torch import nn

from projectUtils.globalVars import regularizationTypes
from projectUtils.typeCheck import argValidator
from projectUtils.generalUtils import _allowOnlyCreationOf_ChildrenInstances
from projectUtils.warnings import Warn
from brazingTorchFolder.layers.customLayers import VAnnCustomLayer
from brazingTorchFolder.lossRegulator import LossRegulator


class _BrazingTorch_regularization:
    _regularizationTypes = regularizationTypes
    nullRegulator = LossRegulator(LossRegulator.nullDictValue)

    @argValidator
    def __init__(self, generalRegularization: Optional[Union[LossRegulator, dict]] = None,
                 **kwargs):

        # not allowing this class to have direct instance
        _allowOnlyCreationOf_ChildrenInstances(self, _BrazingTorch_regularization)

        # cccDevAlgo
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
    def _register_VAnnCustomLayers_regularizations(self):
        # goodToHave1
        #  now only detects if VAnnCustomLayer is the main layer in model but if
        #  VAnnCustomLayer is in a class and that class is the main layer in model
        #  regularization won't get detected; think about self.modules() and not self._modules
        # cccDevAlgo
        #  VAnnCustomLayers can have regularization on their layer
        # cccDevStruct
        #  vars(self)['_modules'](ofc not used here but may be used elsewhere) is similar to
        #  self._modules. these contain only the direct children modules of the model
        for layerName, layer in self._modules.items():
            if isinstance(layer, VAnnCustomLayer):
                if layer.regularization:  # Llr1
                    if layerName not in self._specificLayerRegularization.keys():
                        self._specificLayerRegularization[layerName] = layer.regularization

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

    # ----
    def _setOperationalRegularizations(self):
        # note _operationalRegularizations is set on each run not to slow down
        # also renewed if there are changes on different runs
        self._operationalRegularizations = {}  # reset

        self._register_VAnnCustomLayers_regularizations()
        hasGeneralRegularization = False if self.generalRegularization.type == 'None' else True

        for name, param in self.named_parameters():
            layerName = name.split('.')[0]
            # note name looks like 'layer1Name.layer.0.weight' which is self.layer1Name.layer[0].weight
            # note layer1Name is the name of attribute of the model class so layerName here gets
            # name of attribute of the model:layer1Name

            # if the layer has a regularization of it's own then that one is applied
            # otherwise the general regularization if available is applied
            if layerName in self._specificLayerRegularization.keys():
                self._operationalRegularizations[layerName] = \
                    self._specificLayerRegularization[layerName]
            else:
                if hasGeneralRegularization:
                    self._operationalRegularizations[layerName] = self.generalRegularization

    def addRegularizationsToLoss(self, loss):
        addedReg = torch.tensor(0., requires_grad=True)
        for name, param in self.named_parameters():
            layerName = name.split('.')[0]
            if layerName in self._operationalRegularizations.keys():
                regularization = self._operationalRegularizations[layerName]

                addedReg = addedReg + regularization.addRegularizationToParam(param)
        return loss + addedReg
