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


class ClassForStaticMethod_forParent2p1:
    def __init__(self):
        self.ke = 27

    @staticmethod
    def static_Methodp2p1():
        print('staticmethod for Parent2p1')


class Parent2p1:
    def __init__(self, midLayer2p1):
        self.layp2p1 = 47
        self.statMethp2p1 = ClassForStaticMethod_forParent2p1.static_Methodp2p1


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


def aFuncDefForParent1():
    print('aFuncDefForParent1')


class Parent1(NewWrapper, Parent1p1):
    def __init__(self, midLayerp1):
        self.layp11 = NNDummyModule1()
        self.layp12 = NNDummyModule2()
        self.layp13 = nn.Linear(1, midLayerp1)
        self.layp14 = nn.Linear(midLayerp1, 1)
        self.p1FuncDef = aFuncDefForParent1

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
        definitions = model._getAllNeededDefinitions(model)
        expectedDefinitions = [
            'class Parent2p2p2:\n    def __init__(self):\n        self.layp2p2p2 = 16\n',
            'class Parent1p1:\n    def __init__(self):\n        self.layp1p1 = 4\n',
            'class Parent2p2p1p1:\n    def __init__(self):\n        self.layp2p2p1p1 = 34\n        self.layp2p2p1p12 = NNDummyModule3()\n',
            'class Parent1(NewWrapper, Parent1p1):\n    def __init__(self, midLayerp1):\n        self.layp11 = NNDummyModule1()\n        self.layp12 = NNDummyModule2()\n        self.layp13 = nn.Linear(1, midLayerp1)\n        self.layp14 = nn.Linear(midLayerp1, 1)\n        self.p1FuncDef = aFuncDefForParent1\n\n    def forward(self, inputs, targets):\n        x = self.layp12(self.layp11(inputs))\n        return self.layp14(self.layp13(x))\n',
            'class Parent2p2p1(Parent2p2p1p1):\n    def __init__(self):\n        self.layp2p2p1 = 15\n',
            'class Parent2p2(Parent2p2p1, Parent2p2p2):\n    def __init__(self, midLayer2p2):\n        self.layp2p2 = 5\n',
            'class Parent2p1:\n    def __init__(self, midLayer2p1):\n        self.layp2p1 = 47\n        self.statMethp2p1 = ClassForStaticMethod_forParent2p1.static_Methodp2p1\n',
            'class Parent2(Parent2p1, Parent2p2):\n    # kkk nn.Module should not be a parent; kkk why\n    def __init__(self, midLayer2):\n        self.layp21 = NNDummyModule1()\n        self.layp22 = NNDummyModule2()\n        self.layp23 = nn.Linear(1, midLayer2)\n        self.layp24 = nn.Linear(midLayer2, 1)\n\n    def forward(self, inputs, targets):\n        x = self.layp22(self.layp21(inputs))\n        return self.layp24(self.layp23(x))\n',
            'class NNDummy(Parent1, Parent2):\n    def __init__(self, midLayer1):\n        self.lay1 = NNDummyModule1()\n        self.lay2 = NNDummyModule2()\n        self.classDefExample = NNDummyModule4\n        self.lay3 = nn.Linear(1, midLayer1)\n        self.lay4 = nn.Linear(midLayer1, 1)\n\n    def forward(self, inputs, targets):\n        x = self.lay2(self.lay1(inputs))\n        x = self.lay4(self.lay3(x))\n        return self.layp2p2p1p12(x)\n',
            'class NNDummyModule3(nn.Module):\n    def __init__(self):\n        super(NNDummyModule3, self).__init__()\n        self.lay21 = nn.Linear(1, 4)\n        self.lay22 = nn.Linear(4, 1)\n        self.insMeth = NNDummyModule3ClassForInstanceMethod.instanceMethod3\n\n    def forward(self, inputs, targets):\n        return self.lay22(self.lay21(inputs))\n',
            'class NNDummyModule1(nn.Module):\n    def __init__(self):\n        super(NNDummyModule1, self).__init__()\n        self.lay11 = nn.Linear(1, 3)\n        self.lay12 = nn.Linear(3, 1)\n        self.statMeth = NNDummyModule1ClassForStaticAndInstanceMethod.static_Method1\n        self.instanceMeth = NNDummyModule1ClassForStaticAndInstanceMethod.instanceMeth1\n\n    def forward(self, inputs, targets):\n        return self.lay12(self.lay11(inputs))\n',
            'class NNDummyModule2(nn.Module):\n    def __init__(self):\n        super(NNDummyModule2, self).__init__()\n        self.lay21 = nn.Linear(1, 4)\n        self.lay22 = nn.Linear(4, 1)\n        self.statMeth2 = NNDummyModule2ClassForStaticMethod.static_Method2\n\n    def forward(self, inputs, targets):\n        return self.lay22(self.lay21(inputs))\n',
            'class DotDict:\n    def __init__(self, data):\n        if not hasattr(data, \'keys\') or not callable(getattr(data, \'keys\')):\n            raise ValueError("Input data must be a type that supports keys (e.g., a dictionary)")\n        self._data = data\n\n    def keys(self):\n        return self._data.keys()\n\n    def values(self):\n        return self._data.values()\n\n    def items(self):\n        return self._data.items()\n\n    @property\n    def dict(self):\n        return {key: self._data[key] for key in self.keys()}\n\n    def copy(self):\n        return DotDict(self._data.copy())\n\n    def __len__(self):\n        return len(self.keys())\n\n    def __getattr__(self, key):\n        if key in self._data.keys():\n            return self._data[key]\n        else:\n            raise AttributeError(f"\'DotDict\' object has no attribute \'{key}\'")\n\n    def __getitem__(self, key):\n        if key in self._data.keys():\n            return self._data[key]\n        else:\n            raise KeyError(key)\n\n    def get(self, key, default=None):\n        return self._data.get(key, default)\n\n    def setDefault(self, key, default=None):\n        if key not in self._data:\n            self._data[key] = default\n        return self._data[key]\n\n    def __setitem__(self, key, value):\n        self._data[key] = value\n\n    def __iter__(self):\n        return iter(self._data.items())\n\n    def __repr__(self):\n        return \'DotDict: \' + str(self.dict)\n',
            'class LossRegularizator:\n    nullDictValue = {\'type\': \'None\', \'value\': None}\n\n    @argValidator\n    def __init__(self, value: Union[dict, None]):\n        correctFormatMsg = \'correct regularization format examples: \' + \\\n                           \'\\n{"type":"l1","value":.02}, {"type":"l2","value":.02}, \' + \\\n                           \'{"type":"None","value":None}\'\n        if not value:\n            self._type = "None"\n            self._value = None\n            return\n\n        if \'type\' not in value.keys():\n            Warn.error(correctFormatMsg)\n            raise ValueError(\'for creating LossRegularizator object "type" key is required\')\n        if value[\'type\'] not in regularizationTypes:\n            Warn.error(correctFormatMsg)\n            raise ValueError(\'regularization type must be one of \' + \\\n                             "\'l1\', \'l2\', \'None\'(str)")\n\n        if value[\'type\'] == \'None\':\n            self._type = "None"\n            self._value = None\n            return\n\n        if \'value\' not in value.keys():\n            Warn.error(correctFormatMsg)\n            raise ValueError(\'for l1 and l2 regularizations defining dict must have "value" key\')\n        if not isinstance(value[\'value\'], (float)):\n            raise ValueError(\'regularization value must be float\')\n\n        self._type = value[\'type\']\n        self._value = value[\'value\']\n\n    # cccDevAlgo disable changing type and value\n    @property\n    def type(self):\n        return self._type\n\n    @type.setter\n    def type(self, value):\n        raise AttributeError(\'type of LossRegularizator object is not allowed to be changed\')\n\n    @property\n    def value(self):\n        return self._value\n\n    @value.setter\n    def value(self, value):\n        raise AttributeError(\'value of LossRegularizator object is not allowed to be changed\')\n\n    # ----\n    @argValidator\n    def addRegularizationToParam(self, param: torch.nn.parameter.Parameter):\n        # bugPotentialCheck3\n        #  these need to device?\n        if self.type == \'None\':\n            return torch.tensor(0)\n        elif self.type == \'l1\':\n            return torch.linalg.norm(param, 1) * self.value\n        elif self.type == \'l2\':\n            return torch.norm(param) * self.value\n        raise InternalLogicError(\'sth has gone wrong and type is not one of \' + \\\n                                 "\'l1\', \'l2\', \'None\'(str)")\n\n    def __str__(self):\n        dict_ = {\'type\': self.type, \'value\': self.value}\n        return f"LossRegularizator{dict_}"\n',
            "class NNDummyModule4:\n    def __init__(self):\n        self.a2 = 24\n\n    def md(self):\n        return ''\n",
            "def aFuncDefForParent1():\n    print('aFuncDefForParent1')\n"]
        self.assertEqual(expectedDefinitions, definitions)


# ---- run test
if __name__ == '__main__':
    unittest.main()
