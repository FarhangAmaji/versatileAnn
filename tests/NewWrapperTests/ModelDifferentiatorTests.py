import unittest

from torch import nn

from tests.baseTest import BaseTestClass
from tests.newWrapperTests.ModelDifferentiatorTests_dummyClassDefs.m1 import NNDummyModule1
from tests.newWrapperTests.ModelDifferentiatorTests_dummyClassDefs.m2 import NNDummyModule2, \
    NNDummyModule3, NNDummyModule4
from versatileAnn.newModule.newWrapper import NewWrapper


class Parent2p2p1p1:
    def __init__(self):
        self.layp2p2p1p1 = 34
        self.layp2p2p1p12 = NNDummyModule3()


class Parent2p2p1(Parent2p2p1p1):
    def __init__(self):
        self.layp2p2p1 = 15


class Parent2p2p2:
    def __init__(self):
        self.layp2p2p2 = 16


class Parent2p2(Parent2p2p1, Parent2p2p2):
    def __init__(self, midLayer2p2):
        self.layp2p2 = 5


class Parent2p1:
    def __init__(self, midLayer2p1):
        self.layp2p1 = 47


class Parent2(Parent2p1, Parent2p2):
    # kkk nn.Module should not be a parent; kkk why
    def __init__(self, midLayer2):
        self.layp21 = NNDummyModule1()
        self.layp22 = NNDummyModule2()
        self.layp23 = nn.Linear(1, midLayer2)
        self.layp24 = nn.Linear(midLayer2, 1)

    def forward(self, inputs, targets):
        x = self.layp22(self.layp21(inputs))
        return self.layp24(self.layp23(x))


class Parent1p1:
    def __init__(self):
        self.layp1p1 = 4


class Parent1(NewWrapper, Parent1p1):
    def __init__(self, midLayerp1):
        self.layp11 = NNDummyModule1()
        self.layp12 = NNDummyModule2()
        self.layp13 = nn.Linear(1, midLayerp1)
        self.layp14 = nn.Linear(midLayerp1, 1)

    def forward(self, inputs, targets):
        x = self.layp12(self.layp11(inputs))
        return self.layp14(self.layp13(x))


class NNDummy(Parent1, Parent2):
    def __init__(self, midLayer1):
        self.lay1 = NNDummyModule1()
        self.lay2 = NNDummyModule2()
        self.classDefExample = NNDummyModule4
        self.lay3 = nn.Linear(1, midLayer1)
        self.lay4 = nn.Linear(midLayer1, 1)

    def forward(self, inputs, targets):
        x = self.lay2(self.lay1(inputs))
        x = self.lay4(self.lay3(x))
        return self.layp2p2p1p12(x)


class ModelDifferentiatorTests(BaseTestClass):
    def test1(self):
        model = NNDummy(midLayer1=5, midLayer2=6, midLayerp1=18, midLayer2p1=7, midLayer2p2=8,
                        lossFuncs=[nn.MSELoss(), nn.L1Loss()])
        classDefinitions = model._getAllNeededDefinitions(model)
        expectedClassDefinitions = r"""class Parent2p2p2:
    def __init__(self):
        self.layp2p2p2 = 16

class Parent1p1:
    def __init__(self):
        self.layp1p1 = 4

class Parent2p2p1p1:
    def __init__(self):
        self.layp2p2p1p1 = 34
        self.layp2p2p1p12 = NNDummyModule3()

class Parent1(NewWrapper, Parent1p1):
    def __init__(self, midLayerp1):
        self.layp11 = NNDummyModule1()
        self.layp12 = NNDummyModule2()
        self.layp13 = nn.Linear(1, midLayerp1)
        self.layp14 = nn.Linear(midLayerp1, 1)

    def forward(self, inputs, targets):
        x = self.layp12(self.layp11(inputs))
        return self.layp14(self.layp13(x))

class Parent2p2p1(Parent2p2p1p1):
    def __init__(self):
        self.layp2p2p1 = 15

class Parent2p2(Parent2p2p1, Parent2p2p2):
    def __init__(self, midLayer2p2):
        self.layp2p2 = 5

class Parent2p1:
    def __init__(self, midLayer2p1):
        self.layp2p1 = 47

class Parent2(Parent2p1, Parent2p2):
    # kkk nn.Module should not be a parent; kkk why
    def __init__(self, midLayer2):
        self.layp21 = NNDummyModule1()
        self.layp22 = NNDummyModule2()
        self.layp23 = nn.Linear(1, midLayer2)
        self.layp24 = nn.Linear(midLayer2, 1)

    def forward(self, inputs, targets):
        x = self.layp22(self.layp21(inputs))
        return self.layp24(self.layp23(x))

class NNDummy(Parent1, Parent2):
    def __init__(self, midLayer1):
        self.lay1 = NNDummyModule1()
        self.lay2 = NNDummyModule2()
        self.classDefExample = NNDummyModule4
        self.lay3 = nn.Linear(1, midLayer1)
        self.lay4 = nn.Linear(midLayer1, 1)

    def forward(self, inputs, targets):
        x = self.lay2(self.lay1(inputs))
        x = self.lay4(self.lay3(x))
        return self.layp2p2p1p12(x)

class NNDummyModule3(nn.Module):
    def __init__(self):
        super(NNDummyModule3, self).__init__()
        self.lay21 = nn.Linear(1, 4)
        self.lay22 = nn.Linear(4, 1)

    def forward(self, inputs, targets):
        return self.lay22(self.lay21(inputs))

class NNDummyModule1(nn.Module):
    def __init__(self):
        super(NNDummyModule1, self).__init__()
        self.lay11 = nn.Linear(1, 3)
        self.lay12 = nn.Linear(3, 1)

    def forward(self, inputs, targets):
        return self.lay12(self.lay11(inputs))

class NNDummyModule2(nn.Module):
    def __init__(self):
        super(NNDummyModule2, self).__init__()
        self.lay21 = nn.Linear(1, 4)
        self.lay22 = nn.Linear(4, 1)

    def forward(self, inputs, targets):
        return self.lay22(self.lay21(inputs))

class DotDict:
    def __init__(self, data):
        if not hasattr(data, 'keys') or not callable(getattr(data, 'keys')):
            raise ValueError("Input data must be a type that supports keys (e.g., a dictionary)")
        self._data = data

    def keys(self):
        return self._data.keys()

    def values(self):
        return self._data.values()

    def items(self):
        return self._data.items()

    @property
    def dict(self):
        return {key: self._data[key] for key in self.keys()}

    def copy(self):
        return DotDict(self._data.copy())

    def __len__(self):
        return len(self.keys())

    def __getattr__(self, key):
        if key in self._data.keys():
            return self._data[key]
        else:
            raise AttributeError(f"'DotDict' object has no attribute '{key}'")

    def __getitem__(self, key):
        if key in self._data.keys():
            return self._data[key]
        else:
            raise KeyError(key)

    def get(self, key, default=None):
        return self._data.get(key, default)

    def setDefault(self, key, default=None):
        if key not in self._data:
            self._data[key] = default
        return self._data[key]

    def __setitem__(self, key, value):
        self._data[key] = value

    def __iter__(self):
        return iter(self._data.items())

    def __repr__(self):
        return 'DotDict: ' + str(self.dict)

class LossRegularizator:
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
            raise ValueError('for creating LossRegularizator object "type" key is required')
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
    @argValidator
    def addRegularizationToParam(self, param: torch.nn.parameter.Parameter):
        # bugPotentialCheck3
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
        return f"LossRegularizator{dict_}"

class NNDummyModule4:
    def __init__(self):
        self.a2 = 24

    def md(self):
        return ''
"""
        classDefinitions == expectedClassDefinitions
        self.assertEqual(expectedClassDefinitions, classDefinitions)


# ---- run test
if __name__ == '__main__':
    unittest.main()
