
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


