from typing import Optional, Union

import torch
from torch import nn

from utils.customErrors import InternalLogicError
from utils.globalVars import regularizationTypes
from utils.typeCheck import argValidator
from utils.vAnnGeneralUtils import _allowOnlyCreationOf_ChildrenInstances
from utils.warnings import Warn
from versatileAnn.layers.customLayers import VAnnCustomLayer


class LossRegularizator:
    nullDictValue = {'type': 'None', 'value': None}

    @argValidator
    def __init__(self, value: Union[dict, None]):
        correctFormatMsg = 'correct regularization format examples: ' + \
                           '\n{"type":"l1","value":.02}, {"type":"l2","value":.02}, ' + \
                           '{"type":"None","value":None}'
        if not value:
            self.type = "None"
            self.value = None
            return

        if 'type' not in value.keys():
            Warn.error(correctFormatMsg)
            raise ValueError('for creating LossRegularizator object "type" key is required')
        if value['type'] not in regularizationTypes:
            Warn.error(correctFormatMsg)
            raise ValueError('regularization type must be one of ' + \
                             "'l1', 'l2', 'None'(str)")

        if value['type'] == 'None':
            self.type = "None"
            self.value = None
            return

        if 'value' not in value.keys():
            Warn.error(correctFormatMsg)
            raise ValueError('for l1 and l2 regularizations defining dict must have "value" key')
        if not isinstance(value['value'], (float)):
            raise ValueError('regularization value must be float')

        self.type = value['type']
        self.value = value['value']

    # cccDevAlgo disable changing type and value
    @property
    def type(self):
        return self._type

    @type.setter
    def type(self, value):
        raise AttributeError('type of LossRegularizator object is not allowed to be changed')

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value):
        raise AttributeError('value of LossRegularizator object is not allowed to be changed')

    # ----

    def addRegularizationToParam(self, param):
        # goodToHave3 add argValidator
        # kkk2 these need to device?
        if self.type == 'None':
            return torch.tensor(0)
        elif self.type == 'l1':
            return torch.linalg.norm(param, 1) * self.value  # kkk2 is this correct for l1
        elif self.type == 'l2':
            return torch.norm(param) * self.value  # kkk2 is this correct for l2
        raise InternalLogicError('sth has gone wrong and type is not one of ' + \
                                 "'l1', 'l2', 'None'(str)")


class _NewWrapper_regularization:
    _regularizationTypes = regularizationTypes
    nullRegulator = LossRegularizator(LossRegularizator.nullDictValue)

    @argValidator
    def __init__(self, generalRegularization: Optional[Union[LossRegularizator, dict]] = None,
                 **kwargs):

        # not allowing this class to have direct instance
        _allowOnlyCreationOf_ChildrenInstances(self, _NewWrapper_regularization)

        # kkk1 do _setOperationalRegularizations in each run
        # kkk3 think about renaming some internal regularization to reg
        # cccDevAlgo
        #  - this module(NewWrapper) by default sets some features like generalRegularization
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
    def generalRegularization(self, value: Optional[LossRegularizator, dict]):
        # kkk2 if this is being set check that optimizer doesn't have any weight_decay
        # kkk2 in both check on the other one
        if isinstance(value, dict):
            self._generalRegularization = LossRegularizator(value)
        elif isinstance(value, LossRegularizator):
            self._generalRegularization = value
        else:  # None
            self._generalRegularization = LossRegularizator(LossRegularizator.nullDictValue)

    def noGeneralRegularization(self):
        self.generalRegularization = LossRegularizator(LossRegularizator.nullDictValue)

