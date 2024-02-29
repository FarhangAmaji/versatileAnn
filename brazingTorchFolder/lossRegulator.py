from typing import Union

import torch

from projectUtils.customErrors import InternalLogicError
from projectUtils.globalVars import regularizationTypes
from projectUtils.typeCheck import argValidator
from projectUtils.warnings import Warn


class LossRegulator:
    nullDictValue = {'type': 'None', 'value': None}

    @argValidator
    def __init__(self, value: Union[dict, None]):
        correctFormatMsg = 'correct regularization format examples: ' + \
                           '\n{"type":"l1","value":.02}, {"type":"l2","value":.02}, ' + \
                           '{"type":"None","value":None}'
        if not value:
            self._type = "None"
            self._value = None
            return

        if 'type' not in value.keys():
            Warn.error(correctFormatMsg)
            raise ValueError('for creating LossRegulator object "type" key is required')
        if value['type'] not in regularizationTypes:
            Warn.error(correctFormatMsg)
            raise ValueError('regularization type must be one of ' + \
                             "'l1', 'l2', 'None'(str)")

        if value['type'] == 'None':
            self._type = "None"
            self._value = None
            return

        if 'value' not in value.keys():
            Warn.error(correctFormatMsg)
            raise ValueError('for l1 and l2 regularizations defining dict must have "value" key')
        if not isinstance(value['value'], (float)):
            raise ValueError('regularization value must be float')

        self._type = value['type']
        self._value = value['value']

    # ccc1 disable changing type and value
    @property
    def type(self):
        return self._type

    @type.setter
    def type(self, value):
        raise AttributeError('type of LossRegulator object is not allowed to be changed')

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value):
        raise AttributeError('value of LossRegulator object is not allowed to be changed')

    # ----
    @argValidator
    def addRegularizationToParam(self, param: torch.nn.parameter.Parameter):
        # bugPotn3
        #  these need to device?
        if self.type == 'None':
            return torch.tensor(0)
        elif self.type == 'l1':
            return torch.linalg.norm(param, 1) * self.value
        elif self.type == 'l2':
            return torch.norm(param) * self.value
        raise InternalLogicError('sth has gone wrong and type is not one of ' + \
                                 "'l1', 'l2', 'None'(str)")

    def __str__(self):
        dict_ = {'type': self.type, 'value': self.value}
        return f"LossRegulator{dict_}"
