import copy
from random import random

import pytorch_lightning as pl

from utils.customErrors import ImplementationError
from utils.initParentClasses import initClasses_withAllArgs
from utils.generalUtils import _allowOnlyCreationOf_ChildrenInstances
from brazingTorchFolder.brazingTorchParents.preInitNPostInit_nModelReset_inner import \
    _BrazingTorch_preInitNPostInit_nModelReset_inner


class _BrazingTorch_preInitNPostInit_nModelReset(_BrazingTorch_preInitNPostInit_nModelReset_inner):
    """
    local definition of 'last child of all':
        the class which its instance is getting initialized by user.
        in brazingTorchTests_preInitNPostInit_nModelReset, in classDefinitionsSetup,
        the GrandChild is the 'last child of all'
    """
    # cccDevStruct
    #  this is called even before __init_subclass__
    classesCalledBy_init_subclass_ = []

    def __init__(self):
        # not allowing this class to have direct instance
        _allowOnlyCreationOf_ChildrenInstances(self, _BrazingTorch_preInitNPostInit_nModelReset)

    def __init_subclass__(cls, **kwargs):
        # cccDevAlgo
        #  in _findAllParentClasses_tillBrazingTorch we need BrazingTorch class object.
        #  so along some other classes, we store BrazingTorch class object here.

        # cccDevStruct
        #  this method is keep only to pass BrazingTorchObj to __new__
        #  note this is called even before 'last child of all'
        #  note this is even called for BrazingTorch itself, and its the first one to be called;
        #  but it's super important that BrazingTorch is apparently is called only for the first
        #  time!!! and not in next calls. therefore in _managingClassVariableSpace when releasing
        #  classesCalledBy_init_subclass_ we still keeping its first element(BrazingTorch)

        _BrazingTorch_preInitNPostInit_nModelReset.classesCalledBy_init_subclass_.append(cls)

        # give error if the user defined classes have their __new__ method
        if len(_BrazingTorch_preInitNPostInit_nModelReset.classesCalledBy_init_subclass_) > 1:
            _BrazingTorch_Obj = \
                _BrazingTorch_preInitNPostInit_nModelReset.classesCalledBy_init_subclass_[0]
            if cls.__new__ is not _BrazingTorch_Obj.__new__:
                raise ImplementationError(f'"{cls}" class is not allowed to have __new__ method.')

    def __new__(cls, **kwargs):
        # cccDevStruct
        #  - __new__ acts as `preInit` step, also enables to have `postInit`.
        #    this is for a more clean setup. so users don't need to init parent classes themselves.
        #    even adviced to not do so. and user can just define their model related things
        #    in __init__ and forward
        #  - this method initiates the object. also calls __init__ of parent classes
        #    with their related keyword args(kwargs) automatically.
        #  - we get parent classes of `last child of all`('cls' when running the code)
        #      in 2 different parts:
        #         1. parent classes of `last child of all` up till `BrazingTorch` class. these are
        #         user defined classes.
        #         2. parent classes of  `BrazingTorch` class

        # goodToHave1
        #  detect if the super() is called or not in children; also maybe the __init__s of parent classes are directly callled
        # mustHave1 make initArgs
        print(f'BrazingTorch __new__ method initiated for "{cls.__name__}" class')

        # we know the first item in .classesCalledBy_init_subclass_ is the BrazingTorch class object
        _BrazingTorch_Obj = cls.classesCalledBy_init_subclass_[0]

        # check if the user has defined forward method or not
        if cls.forward is _BrazingTorch_Obj.forward:
            raise ImplementationError(f'"{cls}" class must have "forward" method reImplemented.')

        # we get seed to just be sure this is the same seed applied in the model
        _plSeed__ = pl.seed_everything()

        initiatedObj = super().__new__(cls)
        # set 'testPrints' before other kwargs just to be able to use printTestPrints
        if 'testPrints' in kwargs:
            initiatedObj.testPrints = kwargs['testPrints']
        else:
            initiatedObj.testPrints = False

        argsOf_parentClasses_tillBrazingTorch, parentClasses_tillBrazingTorch = \
            cls._getArgsOfParentClasses_tillBrazingTorch(_BrazingTorch_Obj, cls, kwargs, initiatedObj)

        # warn/advice the users to not __init__ their parent classes in their code because it's
        # done automatically here, and may cause errors
        cls._warnUsersAgainstExplicitParentInitialization(parentClasses_tillBrazingTorch,
                                                          initiatedObj)

        # parent classes of BrazingTorch
        argsOf_parentClassesOfBrazingTorch, parentClassesOfBrazingTorch = cls._get_parentClassesOfBrazingTorch(
            _BrazingTorch_Obj, kwargs)

        # get parent classes of `last child of all` upto BrazingTorch, also args of those classes
        allArgs = cls._combineArgsOfParentClasses_ofTillBrazingTorch_withParentsOfBrazingTorch(
            argsOf_parentClassesOfBrazingTorch, argsOf_parentClasses_tillBrazingTorch, initiatedObj)

        # cccDevStruct
        #  init parent classes of `last child of all` upto BrazingTorch except _BrazingTorch_optimizer
        #  _BrazingTorch_optimizer is initiated few lines later, after initing parent classes till BrazingTorch
        #  because after those, the neural network layer are defined and as the optimizer needs
        #  the parameters to be defined, so it is initiated after that.
        initClasses_withAllArgs(initiatedObj, parentClassesOfBrazingTorch,
                                allArgs, exceptions=['_BrazingTorch_optimizer'])

        cls._initParentClasses_tillBrazingTorch_withDisablingTheirInits(
            allArgs, cls, initiatedObj, parentClasses_tillBrazingTorch)

        # initializing _BrazingTorch_optimizer
        initClasses_withAllArgs(initiatedObj, parentClassesOfBrazingTorch,
                                allArgs, just=['_BrazingTorch_optimizer'])

        # set initArgs, which is used for model reset
        # kkk
        #  1. should I have change initArgs for some of it's attrs?!
        #  2. or maybe have sth similar to it so that one keeps essential values; for i.e.
        #  if the optimizer is changed the new optimizer should be in this similar variable
        cls._setInitArgs(_plSeed__, initiatedObj, kwargs, cls.__name__)

        cls._managingClassVariableSpace(cls, initiatedObj)

        return initiatedObj

    @staticmethod
    def _managingClassVariableSpace(cls, initiatedObj):
        # cccDevStruct
        #  moving classesCalledBy_init_subclass_ from _BrazingTorch_preInitNPostInit_nModelReset to
        #  cls. it's ok that classesCalledBy_init_subclass_ exist in cls, as it's definition is
        #  fixed, but as the _BrazingTorch_preInitNPostInit_nModelReset can be used in other classes,
        #  so it must be cleaned
        #  note also read comments of __init_subclass__
        if _BrazingTorch_preInitNPostInit_nModelReset.classesCalledBy_init_subclass_:
            cls.classesCalledBy_init_subclass_ = _BrazingTorch_preInitNPostInit_nModelReset.classesCalledBy_init_subclass_
            _BrazingTorch_preInitNPostInit_nModelReset.classesCalledBy_init_subclass_ = [
                _BrazingTorch_preInitNPostInit_nModelReset.classesCalledBy_init_subclass_[0]]

        # cccDevStruct
        #  now we have the object, so we move cls._parentClasses_tillBrazingTorch_inits to
        #  initiatedObj, to clean class variable space.
        #  note in _getArgsOfParentClasses_tillBrazingTorch we temporarily put
        #  _parentClasses_tillBrazingTorch_inits in cls, because at that moment we don't have initiatedObj
        initiatedObj._parentClasses_tillBrazingTorch_inits = cls._parentClasses_tillBrazingTorch_inits
        del cls._parentClasses_tillBrazingTorch_inits

    def _BrazingTorch_postInit(self, **kwargs):
        self.printTestPrints('_BrazingTorch_postInit func', self.__class__.__name__)

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
        # bugPotentialCheck1
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
        # cccDevStruct
        #  note the __init_subclass__ and _BrazingTorch_postInit are not called; only __new__ is called
        # kkk
        #  wherer and why prints sth like ##teamcity[testStdErr timestamp='2024-01-30T00:29:05.138' flowId='preRunTests_Tests.preRunTests_Tests.testTraining_step' locationHint='python<F:\projects\public github projects\private repos\versatileAnnModule\tests\brazingTorchTests>://preRunTests_Tests.preRunTests_Tests.testTraining_step' name='testTraining_step' nodeId='3' out='F:\projects\public github projects\private repos\versatileAnnModule\utils\warnings.py:21: CusWarn: |n|[22m|[30m|[44m generalRegularization is not provided; so it is set to default "l2 regularization" with value of 1e-3|nyou may either pass noAdditionalOptions=True to model or call .noGeneralRegularization method on model.|nor set .generalRegularization to another value for i.e. {"type":"l1","value":.02} |[0m|n  warnings.warn(warningMessage, CusWarn)|n' parentNodeId='2']
        attrsToKeep = attrsToKeep or {}

        # bugPotentialCheck1
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
        newObj.__init__(**kwargsToReset['initPassedKwargs'])# kkk why both new and __init__ are used here

        for atk, atkVal in attrsToKeep.items():
            setattr(newObj, atk, atkVal)
        return newObj
