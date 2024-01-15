import pytorch_lightning as pl

from utils.customErrors import ImplementationError
from utils.initParentClasses import initClasses_withAllArgs
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
                raise ImplementationError(f'"{cls} class is not allowed to have __new__ method."')

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
        # we get seed to just be sure this is the same seed applied in the model
        _plSeed__ = pl.seed_everything()

        # we know the first item in .classesCalledBy_init_subclass_ is the NewWrapper class object
        _NewWrapper_Obj = cls.classesCalledBy_init_subclass_[0]

        argsOf_parentClasses_tillNewWrapper, parentClasses_tillNewWrapper = \
            cls._getArgsOfParentClasses_tillNewWrapper(_NewWrapper_Obj, cls, kwargs)

        # warn/advice the users to not __init__ their parent classes in their code because it's
        # done automatically here, and may cause errors
        cls._warnUsersAgainstExplicitParentInitialization(parentClasses_tillNewWrapper)

        # parent classes of NewWrapper
        argsOf_parentClassesOfNewWrapper, parentClassesOfNewWrapper = cls._get_parentClassesOfNewWrapper(
            _NewWrapper_Obj, kwargs)

        # get parent classes of `last child of all` upto NewWrapper, also args of those classes
        allArgs = cls._combineArgsOfParentClasses_ofTillNewWrapper_withParentsOfNewWrapper(
            argsOf_parentClassesOfNewWrapper, argsOf_parentClasses_tillNewWrapper)

        initiatedObj = super().__new__(cls)

        # cccDevStruct
        #  init parent classes of `last child of all` upto NewWrapper except _NewWrapper_optimizer
        #  _NewWrapper_optimizer is initiated few lines later, after initing parent classes till NewWrapper
        #  because after those, the neural network layer are defined and as the optimizer needs
        #  the parameters to be defined, so it is initiated after that.
        initClasses_withAllArgs(initiatedObj, parentClassesOfNewWrapper,
                                allArgs, exceptions=['_NewWrapper_optimizer'])

        _NewWrapper_preInitNPostInit_nModelReset._initParentClasses_tillNewWrapper_withDisablingTheirInits(
            allArgs, cls, initiatedObj, parentClasses_tillNewWrapper)

        # initializing _NewWrapper_optimizer
        initClasses_withAllArgs(initiatedObj, parentClassesOfNewWrapper,
                                allArgs, just=['_NewWrapper_optimizer'])

        # set initArgs, which is used for model reset
        cls._setInitArgs(_plSeed__, initiatedObj, kwargs)

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
            # addTest2
