from utils.initParentClasses import initClasses_withAllArgs
from versatileAnn.newModule.preInitNPostInit_nModelReset_inner import \
    _NewWrapper_preInitNPostInit_nModelReset_inner


class _NewWrapper_preInitNPostInit_nModelReset(_NewWrapper_preInitNPostInit_nModelReset_inner):
    classesCalledBy_init_subclass_ = []
    _parentClasses_tillNewWrapper_inits = None

    def __init_subclass__(cls, **kwargs):
        # cccDevAlgo
        #  in _findAllParentClasses_tillNewWrapper we need NewWrapper class object.
        #  so along some other classes, we store NewWrapper class object here.

        # cccDevStruct
        #  note this is called even before 'last child of all'
        #  note this is even called for NewWrapper itself, and its the first one to be called
        #  this method is keep only to pass NewWrapperObj to __new__

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
