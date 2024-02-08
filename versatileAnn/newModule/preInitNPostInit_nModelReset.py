import copy
from random import random

import pytorch_lightning as pl

from utils.customErrors import ImplementationError
from utils.initParentClasses import initClasses_withAllArgs
from utils.vAnnGeneralUtils import _allowOnlyCreationOf_ChildrenInstances
from versatileAnn.newModule.preInitNPostInit_nModelReset_inner import \
    _NewWrapper_preInitNPostInit_nModelReset_inner


class _NewWrapper_preInitNPostInit_nModelReset(_NewWrapper_preInitNPostInit_nModelReset_inner):
    """
    local definition of 'last child of all':
        the class which its instance is getting initialized by user.
        in newWrapperTests_preInitNPostInit_nModelReset, in classDefinitionsSetup,
        the GrandChild is the 'last child of all'
    """
    # cccDevStruct
    #  this is called even before __init_subclass__
    classesCalledBy_init_subclass_ = []

    def __init__(self):
        # not allowing this class to have direct instance
        _allowOnlyCreationOf_ChildrenInstances(self, _NewWrapper_preInitNPostInit_nModelReset)

    def __init_subclass__(cls, **kwargs):
        # cccDevAlgo
        #  in _findAllParentClasses_tillNewWrapper we need NewWrapper class object.
        #  so along some other classes, we store NewWrapper class object here.

        # cccDevStruct
        #  this method is keep only to pass NewWrapperObj to __new__
        #  note this is called even before 'last child of all'
        #  note this is even called for NewWrapper itself, and its the first one to be called;
        #  but it's super important that NewWrapper is apparently is called only for the first
        #  time!!! and not in next calls. therefore in _managingClassVariableSpace when releasing
        #  classesCalledBy_init_subclass_ we still keeping its first element(NewWrapper)

        _NewWrapper_preInitNPostInit_nModelReset.classesCalledBy_init_subclass_.append(cls)

        # give error if the user defined classes have their __new__ method
        if len(_NewWrapper_preInitNPostInit_nModelReset.classesCalledBy_init_subclass_) > 1:
            _NewWrapper_Obj = \
                _NewWrapper_preInitNPostInit_nModelReset.classesCalledBy_init_subclass_[0]
            if cls.__new__ is not _NewWrapper_Obj.__new__:
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
        #         1. parent classes of `last child of all` up till `NewWrapper` class. these are
        #         user defined classes.
        #         2. parent classes of  `NewWrapper` class

        # goodToHave1
        #  detect if the super() is called or not in children; also maybe the __init__s of parent classes are directly callled
        # mustHave1 make initArgs
        print(f'NewWrapper __new__ method initiated for "{cls.__name__}" class')

        # we know the first item in .classesCalledBy_init_subclass_ is the NewWrapper class object
        _NewWrapper_Obj = cls.classesCalledBy_init_subclass_[0]

        # check if the user has defined forward method or not
        if cls.forward is _NewWrapper_Obj.forward:
            raise ImplementationError(f'"{cls}" class must have "forward" method reImplemented.')

        # we get seed to just be sure this is the same seed applied in the model
        _plSeed__ = pl.seed_everything()

        initiatedObj = super().__new__(cls)
        # set 'testPrints' before other kwargs just to be able to use printTestPrints
        if 'testPrints' in kwargs:
            initiatedObj.testPrints = kwargs['testPrints']
        else:
            initiatedObj.testPrints = False

        argsOf_parentClasses_tillNewWrapper, parentClasses_tillNewWrapper = \
            cls._getArgsOfParentClasses_tillNewWrapper(_NewWrapper_Obj, cls, kwargs, initiatedObj)

        # warn/advice the users to not __init__ their parent classes in their code because it's
        # done automatically here, and may cause errors
        cls._warnUsersAgainstExplicitParentInitialization(parentClasses_tillNewWrapper,
                                                          initiatedObj)

        # parent classes of NewWrapper
        argsOf_parentClassesOfNewWrapper, parentClassesOfNewWrapper = cls._get_parentClassesOfNewWrapper(
            _NewWrapper_Obj, kwargs)

        # get parent classes of `last child of all` upto NewWrapper, also args of those classes
        allArgs = cls._combineArgsOfParentClasses_ofTillNewWrapper_withParentsOfNewWrapper(
            argsOf_parentClassesOfNewWrapper, argsOf_parentClasses_tillNewWrapper, initiatedObj)

        # cccDevStruct
        #  init parent classes of `last child of all` upto NewWrapper except _NewWrapper_optimizer
        #  _NewWrapper_optimizer is initiated few lines later, after initing parent classes till NewWrapper
        #  because after those, the neural network layer are defined and as the optimizer needs
        #  the parameters to be defined, so it is initiated after that.
        initClasses_withAllArgs(initiatedObj, parentClassesOfNewWrapper,
                                allArgs, exceptions=['_NewWrapper_optimizer'])

        cls._initParentClasses_tillNewWrapper_withDisablingTheirInits(
            allArgs, cls, initiatedObj, parentClasses_tillNewWrapper)

        # initializing _NewWrapper_optimizer
        initClasses_withAllArgs(initiatedObj, parentClassesOfNewWrapper,
                                allArgs, just=['_NewWrapper_optimizer'])

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
        #  moving classesCalledBy_init_subclass_ from _NewWrapper_preInitNPostInit_nModelReset to
        #  cls. it's ok that classesCalledBy_init_subclass_ exist in cls, as it's definition is
        #  fixed, but as the _NewWrapper_preInitNPostInit_nModelReset can be used in other classes,
        #  so it must be cleaned
        #  note also read comments of __init_subclass__
        if _NewWrapper_preInitNPostInit_nModelReset.classesCalledBy_init_subclass_:
            cls.classesCalledBy_init_subclass_ = _NewWrapper_preInitNPostInit_nModelReset.classesCalledBy_init_subclass_
            _NewWrapper_preInitNPostInit_nModelReset.classesCalledBy_init_subclass_ = [
                _NewWrapper_preInitNPostInit_nModelReset.classesCalledBy_init_subclass_[0]]

        # cccDevStruct
        #  now we have the object, so we move cls._parentClasses_tillNewWrapper_inits to
        #  initiatedObj, to clean class variable space.
        #  note in _getArgsOfParentClasses_tillNewWrapper we temporarily put
        #  _parentClasses_tillNewWrapper_inits in cls, because at that moment we don't have initiatedObj
        initiatedObj._parentClasses_tillNewWrapper_inits = cls._parentClasses_tillNewWrapper_inits
        del cls._parentClasses_tillNewWrapper_inits

    def _NewWrapper_postInit(self, **kwargs):
        self.printTestPrints('_NewWrapper_postInit func', self.__class__.__name__)

        # putting back originial inits
        for pc, pcInfo in self._parentClasses_tillNewWrapper_inits.items():
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
        #  after writing whole NewWrapper code, this model reset must be revised to keep
        #  __init__ kwargs or attributes added or replaced init kwargs. specailly the attributes
        #  which have properties to set. in general any attribue that I think user may change
        # mustHave2
        #  attrsToKeep should be applied so that the attributes are kept in the same state as they are
        # mustHave2
        #  also add warning that [attr1, attr2, ...] are not kept in the same state as they are
        #  - timeOut message can be useful here
        # cccDevStruct
        #  note the __init_subclass__ and _NewWrapper_postInit are not called; only __new__ is called
        # kkk
        #  wherer and why prints sth like ##teamcity[testStdErr timestamp='2024-01-30T00:29:05.138' flowId='preRunTests_Tests.preRunTests_Tests.testTraining_step' locationHint='python<F:\projects\public github projects\private repos\versatileAnnModule\tests\newWrapperTests>://preRunTests_Tests.preRunTests_Tests.testTraining_step' name='testTraining_step' nodeId='3' out='F:\projects\public github projects\private repos\versatileAnnModule\utils\warnings.py:21: CusWarn: |n|[22m|[30m|[44m generalRegularization is not provided; so it is set to default "l2 regularization" with value of 1e-3|nyou may either pass noAdditionalOptions=True to model or call .noGeneralRegularization method on model.|nor set .generalRegularization to another value for i.e. {"type":"l1","value":.02} |[0m|n  warnings.warn(warningMessage, CusWarn)|n' parentNodeId='2']
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
