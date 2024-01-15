import unittest

from torch import nn

from tests.baseTest import BaseTestClass
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

        self.expectedPrint1 = """NewWrapper __new__ method initiated for "GrandChild" class
you have initiated parent classes in your __init__.
GrandChild class is one of them.
this may cause error because parent classes are initiated automatically.
so you may want to remove the __init__ of parent classes from your __init__.
you have initiated parent classes in your __init__.
Child class is one of them.
this may cause error because parent classes are initiated automatically.
so you may want to remove the __init__ of parent classes from your __init__.
you have initiated parent classes in your __init__.
Child class is one of them.
this may cause error because parent classes are initiated automatically.
so you may want to remove the __init__ of parent classes from your __init__.
you have initiated parent classes in your __init__.
Child class is one of them.
this may cause error because parent classes are initiated automatically.
so you may want to remove the __init__ of parent classes from your __init__.
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
_NewWrapper_postInit func GrandChild
"""

        return Child, GrandChild, OtherParentOfChild, OtherParentOfGrandChild

    def testObjectCreation_withPrintingSteps(self):
        Child, GrandChild, OtherParentOfChild, OtherParentOfGrandChild = self.classDefinitionsSetup()

        def testFunc():
            model = GrandChild(modelName='ModelKog2s', opoc='opoc', opogc='opogc',
                               additionalArg='additionalArg', testPrints=True)

            self.assertTrue(isinstance(model, GrandChild))

        expectedPrint = self.expectedPrint1
        self.assertPrint(testFunc, expectedPrint)

    def testInitArgs(self):
        Child, GrandChild, OtherParentOfChild, OtherParentOfGrandChild = self.classDefinitionsSetup()

        model = GrandChild(modelName='ModelKog2s', opoc='opoc', opogc='opogc',
                           additionalArg='additionalArg', testPrints=True)

        del model._initArgs['__plSeed__']
        self.assertTrue(model._initArgs, {'modelName': 'ModelKog2s',
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


# ---- run test
if __name__ == '__main__':
    unittest.main()
