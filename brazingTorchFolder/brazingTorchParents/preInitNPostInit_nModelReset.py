import copy
from random import random

import pytorch_lightning as pl

from brazingTorchFolder.brazingTorchParents.innerClassesWithoutPublicMethods.preInitNPostInit_nModelReset_inner1 import \
    _BrazingTorch_preInitNPostInit_nModelReset_inner1
from projectUtils.misc import _allowOnlyCreationOf_ChildrenInstances


class _BrazingTorch_preInitNPostInit_nModelReset(_BrazingTorch_preInitNPostInit_nModelReset_inner1):
    """
    local definition of 'last child of all':
        the class which its instance is getting initialized by user.
        in brazingTorchTests_preInitNPostInit_nModelReset, in classDefinitionsSetup,
        the GrandChild is the 'last child of all'
    """

    def __init__(self):
        # not allowing this class to have direct instance
        _allowOnlyCreationOf_ChildrenInstances(self, _BrazingTorch_preInitNPostInit_nModelReset)

    def _BrazingTorch_postInit(self, **kwargs):
        # kkk should it be a public method?
        self._printTestPrints('_BrazingTorch_postInit func', self.__class__.__name__)

        # putting back original inits
        # addTest2
        for pc, pcInfo in self._parentClasses_tillBrazingTorch_inits.items():
            pcInfo['classObj'].__init__ = pcInfo['originalInit']

        if self.getAllNeededDefinitions:
            self._getAllNeededDefinitions(self)

    def resetModel(self, withPastSeed=True, attrsToKeep=None):
        # addTest1
        #  test if the model params are reset after some epochs of training
        # kkk add modes here(if not added by default)

        # cccUsage
        #  this is not inplace so u have to do `self = self.resetModel()`
        # bugPotn1
        #  this is a major feature but very prone to bugs, specially the attributes which are set
        #  after __init__ may be lost
        # mustHave1
        #  after writing whole BrazingTorch code, this model reset must be revised to keep
        #  __init__ kwargs or attributes added or replaced init kwargs. specailly the attributes
        #  which have properties to set. in general any attribue that I think user may change
        # mustHave2
        #  attrsToKeep should be applied so that the attributes are kept in the same state as they are
        # mustHave2
        #  also add warning that [attr1, attr2, ...] are not kept in the same state as they are
        #  - timeOut message can be useful here
        # ccc1
        #  note the __init_subclass__ and _BrazingTorch_postInit are not called; only __new__ is called
        # kkk
        #  wherer and why prints sth like ##teamcity[testStdErr timestamp='2024-01-30T00:29:05.138' flowId='preRunTests_Tests.preRunTests_Tests.testTraining_step' locationHint='python<F:\projects\public github projects\private repos\versatileAnnModule\tests\brazingTorchTests>://preRunTests_Tests.preRunTests_Tests.testTraining_step' name='testTraining_step' nodeId='3' out='F:\projects\public github projects\private repos\versatileAnnModule\projectUtils\warnings.py:21: CusWarn: |n|[22m|[30m|[44m generalRegularization is not provided; so it is set to default "l2 regularization" with value of 1e-3|nyou may either pass noAdditionalOptions=True to model or call .noGeneralRegularization method on model.|nor set .generalRegularization to another value for i.e. {"type":"l1","value":.02} |[0m|n  warnings.warn(warningMessage, CusWarn)|n' parentNodeId='2']
        attrsToKeep = attrsToKeep or {}

        # bugPotn1
        #  in past we didn't have optimizer and _optimizerInitArgs in attrsKeptByDefault_names but
        #  worked fine but now it doesn't. so had to add them to it
        attrsKeptByDefault_names = ['lossFuncs', 'optimizer', '_optimizerInitArgs']
        for atk in attrsKeptByDefault_names:
            attrsToKeep[atk] = copy.deepcopy(getattr(self, atk))

        classOfSelf = type(self)

        kwargsToReset = self._initArgs.copy()
        if not withPastSeed:
            newRandomSeed = random.randint(1, 2 ** 31 - 1)
            pl.seed_everything(newRandomSeed)
        else:
            pl.seed_everything(kwargsToReset['__plSeed__'])

        kwargsToReset.pop('__plSeed__')

        newObj = classOfSelf.__new__(classOfSelf, **kwargsToReset['initPassedKwargs'])
        newObj.__init__(
            **kwargsToReset['initPassedKwargs'])  # kkk why both new and __init__ are used here

        for atk, atkVal in attrsToKeep.items():
            setattr(newObj, atk, atkVal)
        return newObj
