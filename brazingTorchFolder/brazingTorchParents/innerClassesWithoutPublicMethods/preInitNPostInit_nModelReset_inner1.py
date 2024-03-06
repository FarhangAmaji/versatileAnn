import pytorch_lightning as pl

from brazingTorchFolder.brazingTorchParents.innerClassesWithoutPublicMethods.preInitNPostInit_nModelReset_inner2 import \
    _BrazingTorch_preInitNPostInit_nModelReset_inner2
from projectUtils.customErrors import ImplementationError
from projectUtils.initParentClasses import initClasses_withAllArgs
from projectUtils.misc import _allowOnlyCreationOf_ChildrenInstances


class _BrazingTorch_preInitNPostInit_nModelReset_inner1(
    _BrazingTorch_preInitNPostInit_nModelReset_inner2):
    """
    local definition of 'last child of all':
        the class which its instance is getting initialized by user.
        in brazingTorchTests_preInitNPostInit_nModelReset, in classDefinitionsSetup,
        the GrandChild is the 'last child of all'
    """

    def __init__(self):
        # not allowing this class to have direct instance
        _allowOnlyCreationOf_ChildrenInstances(self,
                                               _BrazingTorch_preInitNPostInit_nModelReset_inner1)

    def __new__(cls, **kwargs):
        # ccc1
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
        # _BrazingTorch_Obj = cls.classesCalledBy_init_subclass_[0]#kkk

        # we get seed to just be sure this is the same seed applied in the model
        if 'seed' in kwargs and kwargs['seed']:
            _plSeed__ = kwargs['seed']
            pl.seed_everything(_plSeed__)
        else:
            _plSeed__ = pl.seed_everything()

        initiatedObj = super().__new__(cls)
        initiatedObj.seed = _plSeed__

        _BrazingTorch_Obj = initiatedObj._getBrazingTorch_classObject()
        # check if the user has defined forward method or not
        if cls.forward is _BrazingTorch_Obj.forward:
            raise ImplementationError(f'"{cls}" class must have "forward" method reImplemented.')

        # set 'testPrints' before other kwargs just to be able to use _printTestPrints
        if 'testPrints' in kwargs:
            initiatedObj.testPrints = kwargs['testPrints']
        else:
            initiatedObj.testPrints = False

        argsOf_parentClasses_tillBrazingTorch, parentClasses_tillBrazingTorch = \
            cls._getArgsOfParentClasses_tillBrazingTorch(_BrazingTorch_Obj, cls, kwargs,
                                                         initiatedObj)

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

        # ccc1
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

        return initiatedObj
