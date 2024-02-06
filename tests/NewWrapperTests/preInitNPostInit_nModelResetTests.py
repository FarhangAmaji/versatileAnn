import unittest

from torch import nn

from tests.baseTest import BaseTestClass
from utils.initParentClasses import checkIfAClassIs_initingItsParentClasses_inItsInit
from versatileAnn.newModule.newWrapper import NewWrapper


class newWrapperTests_preInitNPostInit_nModelReset(BaseTestClass):

    def classDefinitionsSetup(self):
        class OtherParentOfChild:
            def __init__(self, opoc):
                print('OtherParentOfChild init')
                self.opoc = opoc

        class Child(NewWrapper, OtherParentOfChild):
            def __init__(self, modelName, opoc, **kwargs):
                print('Child;Before calling Parent __init__', self.__class__.__name__)
                NewWrapper.__init__(self, modelName=modelName, **kwargs)
                OtherParentOfChild.__init__(self, opoc=opoc)
                print('Child;After calling Parent __init__', self.__class__.__name__)

        class OtherParentOfGrandChild:
            def __init__(self, opogc):
                self.opogc = opogc
                print('OtherParentOfGrandChild init')

        class GrandChild(Child, OtherParentOfGrandChild):
            def __init__(self, modelName, opoc, opogc, **kwargs):
                print('GrandChild;Before calling Child __init__', self.__class__.__name__)
                Child.__init__(self, modelName=modelName, opoc=opoc, **kwargs)
                OtherParentOfGrandChild.__init__(self, opogc=opogc)
                self.nnLayers = nn.Sequential(nn.Linear(1, 1))
                print('GrandChild __init__;After calling Child __init__', self.__class__.__name__)

            def forward(self, inputs, targets):
                pass

        return Child, GrandChild, OtherParentOfChild, OtherParentOfGrandChild

    def testObjectCreation_withPrintingSteps(self):
        Child, GrandChild, OtherParentOfChild, OtherParentOfGrandChild = self.classDefinitionsSetup()

        def innerFunc():
            model = GrandChild(modelName='ModelKog2s', opoc='opoc', opogc='opogc',
                               additionalArg='additionalArg', testPrints=True)

            self.assertTrue(isinstance(model, GrandChild))

        expectedPrint = """NewWrapper __new__ method initiated for "GrandChild" class

 Warning: defining __init__ in subclasses of NewWrapper
    you have initiated parent classes in your __init__.
    "GrandChild" class is one of them.
    this may cause error because parent classes are initiated automatically.
    so you may want to remove the __init__ of parent classes (even using "super()") from your __init__.

 Warning: defining __init__ in subclasses of NewWrapper
    you have initiated parent classes in your __init__.
    "Child" class is one of them.
    this may cause error because parent classes are initiated automatically.
    so you may want to remove the __init__ of parent classes (even using "super()") from your __init__.

Warning: using args in subclasses of NewWrapper with similar argnames to NewWrapper args
    "modelName" arg is used in the classes you have defined. and also exist in required args of NewWrapper.
    this may cause conflict if are used for other purposes than passing to NewWrapper.you may want to change the name of this arg.
OtherParentOfChild init
OtherParentOfGrandChild init
Child;Before calling Parent __init__ GrandChild
NewWrapper init
emptyMethod_usedForDisabling__init__s
Child;After calling Parent __init__ GrandChild
GrandChild;Before calling Child __init__ GrandChild
emptyMethod_usedForDisabling__init__s
emptyMethod_usedForDisabling__init__s
GrandChild __init__;After calling Child __init__ GrandChild
_NewWrapper_postInit func
GrandChild
"""
        self.assertPrint(innerFunc, expectedPrint)

    def testInitArgs(self):
        Child, GrandChild, OtherParentOfChild, OtherParentOfGrandChild = self.classDefinitionsSetup()

        model = GrandChild(modelName='ModelKog2s', opoc='opoc', opogc='opogc',
                           additionalArg='additionalArg', testPrints=True)

        del model._initArgs['__plSeed__']
        self.assertTrue(model._initArgs['initPassedKwargs'], {'modelName': 'ModelKog2s',
                                          'opoc': 'opoc',
                                          'opogc': 'opogc',
                                          'additionalArg': 'additionalArg',
                                          'testPrints': True})

    def testClassInitiation_multipleTimes(self):
        """
        check that class can be initiated multiple times
        """
        Child, GrandChild, OtherParentOfChild, OtherParentOfGrandChild = self.classDefinitionsSetup()

        model = GrandChild(modelName='ModelKog2s', opoc='opoc', opogc='opogc',
                           additionalArg='additionalArg', testPrints=True)
        self.assertTrue(isinstance(model, GrandChild))

        model2 = GrandChild(modelName='ModelKog2s', opoc='opoc', opogc='opogc',
                            additionalArg='additionalArg', testPrints=True)
        self.assertTrue(isinstance(model2, GrandChild))

    def testResetModel(self):
        # mustHave1
        #  check does it correctly work after training some epochs, then after reseting would give
        #  the same result
        Child, GrandChild, OtherParentOfChild, OtherParentOfGrandChild = self.classDefinitionsSetup()

        model = GrandChild(modelName='ModelKog2s', opoc='opoc', opogc='opogc',
                           additionalArg='additionalArg', testPrints=True)

        def innerFunc(model):
            newModel = model.resetModel()
            self.assertTrue(isinstance(model, GrandChild))

        expectedPrint = """NewWrapper __new__ method initiated for "GrandChild" class

 Warning: defining __init__ in subclasses of NewWrapper
    you have initiated parent classes in your __init__.
    "GrandChild" class is one of them.
    this may cause error because parent classes are initiated automatically.
    so you may want to remove the __init__ of parent classes (even using "super()") from your __init__.

 Warning: defining __init__ in subclasses of NewWrapper
    you have initiated parent classes in your __init__.
    "Child" class is one of them.
    this may cause error because parent classes are initiated automatically.
    so you may want to remove the __init__ of parent classes (even using "super()") from your __init__.

Warning: using args in subclasses of NewWrapper with similar argnames to NewWrapper args
    "modelName" arg is used in the classes you have defined. and also exist in required args of NewWrapper.
    this may cause conflict if are used for other purposes than passing to NewWrapper.you may want to change the name of this arg.
OtherParentOfChild init
OtherParentOfGrandChild init
Child;Before calling Parent __init__ GrandChild
NewWrapper init
emptyMethod_usedForDisabling__init__s
Child;After calling Parent __init__ GrandChild
GrandChild;Before calling Child __init__ GrandChild
emptyMethod_usedForDisabling__init__s
emptyMethod_usedForDisabling__init__s
GrandChild __init__;After calling Child __init__ GrandChild
"""
        self.assertPrint(innerFunc, expectedPrint, model=model)

    def testInitsToOriginal(self):
        """
        do init get back to their original after _NewWrapper_postInit
        """
        Child, GrandChild, OtherParentOfChild, OtherParentOfGrandChild = self.classDefinitionsSetup()

        model = GrandChild(modelName='ModelKog2s', opoc='opoc', opogc='opogc',
                           additionalArg='additionalArg', testPrints=True)

        def innerFunc():
            OtherParentOfChild(opoc='opoc')

        expectedPrint = "OtherParentOfChild init"
        self.assertPrint(innerFunc, expectedPrint)

    def classDefinitionsSetup2(self):
        class Parent2(NewWrapper):
            def __init__(self):
                pass

        class ChildWithSuper(Parent2):
            def __init__(self, **kwargs):
                super().__init__()
                self.nnLayers = nn.Sequential(nn.Linear(1, 1))

            def forward(self, inputs, targets):
                pass

        return Parent2, ChildWithSuper

    def testCreate2differentClasses_inheritedFromNewWrapper(self):
        """
        the purpose is to see if creating 2 different classes would mess up or not
        """
        _, ChildWithSuper = self.classDefinitionsSetup2()
        _, GrandChild, _, _ = self.classDefinitionsSetup()

        model = ChildWithSuper(modelName='ModelKog2s',
                               additionalArg='additionalArg', testPrints=True)
        self.assertTrue(isinstance(model, ChildWithSuper))

        model2 = GrandChild(modelName='ModelKog2s', opoc='opoc', opogc='opogc',
                            additionalArg='additionalArg', testPrints=True)
        self.assertTrue(isinstance(model2, GrandChild))

    def classDefinitionsSetup3(self):
        class Parent:
            def __init__(self):
                pass

        class ChildWithSuper(Parent):
            def __init__(self):
                super().__init__()

        class ChildWithParent(Parent):
            def __init__(self):
                Parent.__init__(self)

        class ChildWithoutParent:
            def __init__(self):
                pass

        return Parent, ChildWithSuper, ChildWithParent, ChildWithoutParent

    def test_checkIfAClassIs_initingItsParentClasses_inItsInit(self):
        Parent, ChildWithSuper, ChildWithParent, ChildWithoutParent = self.classDefinitionsSetup3()
        func = checkIfAClassIs_initingItsParentClasses_inItsInit

        self.assertFalse(func(Parent))
        self.assertTrue(func(ChildWithSuper))
        self.assertTrue(func(ChildWithParent))
        self.assertFalse(func(ChildWithoutParent))


# ---- run test
if __name__ == '__main__':
    unittest.main()
