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
    # bugPotentialCheck1
    #  nn.Module should not be a parent
    # goodToHave2
    #  why nn.Module should not be a parent
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
                        lossFuncs=[nn.MSELoss(), nn.L1Loss()], testPrints=True)

        # kkk if added to postInit and the results differ should still use this example
        def innerFunc():
            definitions = model._getAllNeededDefinitions(model)
            return definitions

        expectedDefinitions = [
            {'Parent1p1': 'class Parent1p1:\n    def __init__(self):\n        self.layp1p1 = 4\n'},
            {
                'Parent2p1': 'class Parent2p1:\n    def __init__(self, midLayer2p1):\n        self.layp2p1 = 47\n        self.statMethp2p1 = ClassForStaticMethod_forParent2p1.static_Methodp2p1\n'},
            {
                'Parent2p2p2': 'class Parent2p2p2:\n    def __init__(self):\n        self.layp2p2p2 = 16\n'},
            {
                'Parent2p2p1p1': 'class Parent2p2p1p1:\n    def __init__(self):\n        self.layp2p2p1p1 = 34\n        self.layp2p2p1p12 = NNDummyModule3()\n'},
            {
                'NNDummyModule3': 'class NNDummyModule3(nn.Module):\n    def __init__(self):\n        super(NNDummyModule3, self).__init__()\n        self.lay21 = nn.Linear(1, 4)\n        self.lay22 = nn.Linear(4, 1)\n        self.insMeth = NNDummyModule3ClassForInstanceMethod.instanceMethod3\n\n    def forward(self, inputs, targets):\n        return self.lay22(self.lay21(inputs))\n'},
            {
                'NNDummyModule1': 'class NNDummyModule1(nn.Module):\n    def __init__(self):\n        super(NNDummyModule1, self).__init__()\n        self.lay11 = nn.Linear(1, 3)\n        self.lay12 = nn.Linear(3, 1)\n        self.statMeth = NNDummyModule1ClassForStaticAndInstanceMethod.static_Method1\n        self.instanceMeth = NNDummyModule1ClassForStaticAndInstanceMethod.instanceMeth1\n\n    def forward(self, inputs, targets):\n        return self.lay12(self.lay11(inputs))\n'},
            {
                'NNDummyModule2': 'class NNDummyModule2(nn.Module):\n    def __init__(self):\n        super(NNDummyModule2, self).__init__()\n        self.lay21 = nn.Linear(1, 4)\n        self.lay22 = nn.Linear(4, 1)\n        self.statMeth2 = NNDummyModule2ClassForStaticMethod.static_Method2\n\n    def forward(self, inputs, targets):\n        return self.lay22(self.lay21(inputs))\n'},
            {
                'NNDummyModule4': "class NNDummyModule4:\n    def __init__(self):\n        self.a2 = 24\n\n    def md(self):\n        return ''\n"},
            {
                'NNDummyModule3ClassForInstanceMethod': "class NNDummyModule3ClassForInstanceMethod:\n    def __init__(self):\n        self.ke = 47\n\n    def instanceMethod3(self):\n        print('instance method for NNDummyModule3')\n"},
            {
                'NNDummyModule1ClassForStaticAndInstanceMethod': "class NNDummyModule1ClassForStaticAndInstanceMethod:\n    def __init__(self):\n        self.ke = 78\n\n    @staticmethod\n    def static_Method1():\n        print('staticmethod for NNDummyModule1')\n\n    def instanceMeth1(self):\n        print('instancemethod for NNDummyModule1')\n"},
            {
                'NNDummyModule2ClassForStaticMethod': "class NNDummyModule2ClassForStaticMethod:\n    def __init__(self):\n        self.ke = 43\n\n    @staticmethod\n    def static_Method2():\n        print('staticmethod for NNDummyModule2')\n"},
            {
                'ClassForStaticMethod_forParent2p1': "class ClassForStaticMethod_forParent2p1:\n    def __init__(self):\n        self.ke = 27\n\n    @staticmethod\n    def static_Methodp2p1():\n        print('staticmethod for Parent2p1')\n"},
            {
                'Parent1': 'class Parent1(NewWrapper, Parent1p1):\n    def __init__(self, midLayerp1):\n        self.layp11 = NNDummyModule1()\n        self.layp12 = NNDummyModule2()\n        self.layp13 = nn.Linear(1, midLayerp1)\n        self.layp14 = nn.Linear(midLayerp1, 1)\n        self.p1FuncDef = aFuncDefForParent1\n\n    def forward(self, inputs, targets):\n        x = self.layp12(self.layp11(inputs))\n        return self.layp14(self.layp13(x))\n'},
            {
                'Parent2p2p1': 'class Parent2p2p1(Parent2p2p1p1):\n    def __init__(self):\n        self.layp2p2p1 = 15\n'},
            {
                'Parent2p2': 'class Parent2p2(Parent2p2p1, Parent2p2p2):\n    def __init__(self, midLayer2p2):\n        self.layp2p2 = 5\n'},
            {
                'Parent2': 'class Parent2(Parent2p1, Parent2p2):\n    # bugPotentialCheck1\n    #  nn.Module should not be a parent\n    # goodToHave2\n    #  why nn.Module should not be a parent\n    def __init__(self, midLayer2):\n        self.layp21 = NNDummyModule1()\n        self.layp22 = NNDummyModule2()\n        self.layp23 = nn.Linear(1, midLayer2)\n        self.layp24 = nn.Linear(midLayer2, 1)\n\n    def forward(self, inputs, targets):\n        x = self.layp22(self.layp21(inputs))\n        return self.layp24(self.layp23(x))\n'},
            {
                'NNDummy': 'class NNDummy(Parent1, Parent2):\n    def __init__(self, midLayer1):\n        self.lay1 = NNDummyModule1()\n        self.lay2 = NNDummyModule2()\n        self.classDefExample = NNDummyModule4\n        self.lay3 = nn.Linear(1, midLayer1)\n        self.lay4 = nn.Linear(midLayer1, 1)\n\n    def forward(self, inputs, targets):\n        x = self.lay2(self.lay1(inputs))\n        x = self.lay4(self.lay3(x))\n        return self.layp2p2p1p12(x)\n'},
            {'aFuncDefForParent1': "def aFuncDefForParent1():\n    print('aFuncDefForParent1')\n"}]

        expectedPrint = ""
        definitions = self.assertPrint(innerFunc, expectedPrint)
        self.assertEqual(expectedDefinitions, definitions)


# ---- run test
if __name__ == '__main__':
    unittest.main()
