import copy

import pytorch_lightning as pl

from projectUtils.customErrors import InternalLogicError
from projectUtils.initParentClasses import getArgsOfClasses, exclude_selfNArgsNKwargs_fromAllArgs, \
    getArgsRelatedToAClass_fromAllArgs, orderClassNames_soChildIsAlways_afterItsParents, \
    checkIfAClassIs_initingItsParentClasses_inItsInit
from projectUtils.warnings import Warn


class _BrazingTorch_preInitNPostInit_nModelReset_inner:

    @staticmethod
    def _getArgsOfParentClasses_tillBrazingTorch(_BrazingTorch_Obj, cls, originalKwargs, self):
        parentClasses_tillBrazingTorch = cls._findAllParentClasses_tillBrazingTorch(cls,
                                                                                    _BrazingTorch_Obj)

        # cccDevStruct
        #  beside the class object we also store its '__init__' because at the end of __new__,
        #  init are disabled but later at _BrazingTorch_postInit,
        #  these stored inits are going to be replaced back
        cls._parentClasses_tillBrazingTorch_inits = {
            clsName: {'originalInit': copy.deepcopy(classObj.__init__), 'classObj': classObj} for
            clsName, classObj in parentClasses_tillBrazingTorch.items()}

        argsOf_parentClasses_tillBrazingTorch = getArgsOfClasses(parentClasses_tillBrazingTorch,
                                                                 originalKwargs)

        # *args for subclasses of BrazingTorch are not applied
        if 'args' in argsOf_parentClasses_tillBrazingTorch:
            warnMsg = '\nWarning: using *args for subclasses of BrazingTorch' + \
                      '\n    "*args" for subclasses of BrazingTorch are not applied.' + \
                      "\n    this warning is not always True, but it's better to double check" + \
                      " that you have not used *args in your __init__"
            Warn.error(warnMsg)
            self.printTestPrints(warnMsg)
            # cccDevStruct
            #  we don't make error and just give warning, because sometimes the user have not
            #  included *args in their __init__ but because of inheritance from `object`,
            #  __init__ has *args
        argsOf_parentClasses_tillBrazingTorch = exclude_selfNArgsNKwargs_fromAllArgs(
            argsOf_parentClasses_tillBrazingTorch)
        return argsOf_parentClasses_tillBrazingTorch, parentClasses_tillBrazingTorch

    @staticmethod
    def _findAllParentClasses_tillBrazingTorch(cls_, BrazingTorch_Obj, parentClasses: dict = None):
        # this method is similar to findParentClasses_OfAClass_tillAnotherClass in projectUtils/initParentClasses.py

        parentClasses = parentClasses or {}
        # goodToHave3 bugPotentialCheck2
        #  some classes may have same .__name__ but are actually different classes
        #  but I am not counting for that(maybe later).
        #  so for now each class is going to be captured in a dict with {class.__name__:classObj}

        if str(cls_) == str(BrazingTorch_Obj):
            return parentClasses
        elif cls_ is pl.LightningModule:
            return parentClasses
        elif cls_ is object:
            return parentClasses

        parentClasses.update({cls_.__name__: cls_})
        parentsOfThisClass = cls_.__bases__
        for potc in parentsOfThisClass:
            parentClasses = _BrazingTorch_preInitNPostInit_nModelReset_inner._findAllParentClasses_tillBrazingTorch(
                potc,
                BrazingTorch_Obj,
                parentClasses)
        return parentClasses

    @staticmethod
    def _get_parentClassesOfBrazingTorch(BrazingTorch_Obj, originalKwargs):
        parentClassesOfBrazingTorch = {pc.__name__: pc for pc in BrazingTorch_Obj.__bases__}
        argsOf_parentClassesOfBrazingTorch = getArgsOfClasses(parentClassesOfBrazingTorch,
                                                              originalKwargs)
        argsOf_parentClassesOfBrazingTorch = exclude_selfNArgsNKwargs_fromAllArgs(
            argsOf_parentClassesOfBrazingTorch)

        # cccDevStruct
        #  note this is only for development error detection
        #  args of parent classes of BrazingTorch must not have similar names
        for arg, argVal in argsOf_parentClassesOfBrazingTorch.items():
            if len(argVal['classes']) > 1:  # LBTEam1
                raise InternalLogicError(
                    "internalError: this is for development:" +
                    "\nparentClasses of BrazingTorch must not have args with similar names in their __init__."
                    f'\narg "{arg}" is used in more than one base classes of BrazingTorch: {argVal["classes"]}.')
        return argsOf_parentClassesOfBrazingTorch, parentClassesOfBrazingTorch

    @staticmethod
    def _combineArgsOfParentClasses_ofTillBrazingTorch_withParentsOfBrazingTorch(
            argsOf_parentClassesOfBrazingTorch, argsOf_parentClasses_tillBrazingTorch, self):
        allArgs = {**argsOf_parentClasses_tillBrazingTorch}
        for arg, argVal in argsOf_parentClassesOfBrazingTorch.items():
            if arg not in allArgs:
                allArgs[arg] = argVal
            else:
                allArgs[arg]['classes'].extend(argVal['classes'])
                warnMsg = '\nWarning: using args in subclasses of BrazingTorch with similar argnames to BrazingTorch args' + \
                          f'\n    "{arg}" arg is used in the classes you have defined. ' + \
                          'and also exist in required args of BrazingTorch.' + \
                          '\n    this may cause conflict if are used for other purposes than passing to BrazingTorch.' + \
                          'you may want to change the name of this arg.'
                Warn.warn(warnMsg)
                self.printTestPrints(warnMsg)
        return allArgs

    @staticmethod
    def _initParentClasses_tillBrazingTorch_withDisablingTheirInits(allArgs, cls,
                                                                    initiatedObj,
                                                                    parentClasses_tillBrazingTorch):
        # parent classes which are more base(upper parents) are __init__ed first
        parentClasses_tillBrazingTorch_names_ordered = orderClassNames_soChildIsAlways_afterItsParents(
            parentClasses_tillBrazingTorch)

        for i, clsName in enumerate(parentClasses_tillBrazingTorch_names_ordered):
            classRelatedArgs = getArgsRelatedToAClass_fromAllArgs(clsName, allArgs)
            clsObj = parentClasses_tillBrazingTorch[clsName]
            clsObj.__init__(initiatedObj, **classRelatedArgs)

            # cccDevStruct
            #  - inits are disabled in order to:
            #       1. not to get inited twice
            #       2. also not to mess model's parameters (therefore optimizer's params)
            #  - __init__s are set back to their originalInit later at _BrazingTorch_postInit
            if clsObj is not cls:
                clsObj.__init__ = cls._emptyMethod_usedForDisabling__init__s
            else:
                # replace lastChildOfAll's __init__ with _BrazingTorch_postInit
                clsObj.__init__ = cls._BrazingTorch_postInit
                # cccDevStruct
                #  in past I called line "clsObj.__init__(initiatedObj, **classRelatedArgs)"
                #  in order to call postInit manually. but postInit is called automatically and
                #  must not be called manually because it would mess model's parameters
                #  (therefore optimizer's params)
                #  also the disabling __init__s would not be applied if the postInit
                #  is called manually

    @staticmethod
    def _setInitArgs(_plSeed__, initiatedObj, kwargs, clsTypeName):
        # bugPotentialCheck1 #addTest1
        #  if ._initArgs have some nn.module does it work
        kwargs_ = kwargs or {}
        _initArgs = {}

        _initArgs['initPassedKwargs'] = kwargs_.copy()
        _initArgs['clsTypeName'] = clsTypeName
        _initArgs['__plSeed__'] = _plSeed__
        initiatedObj._initArgs = _initArgs

    @staticmethod
    def _emptyMethod_usedForDisabling__init__s(self, **kwargs):
        self.printTestPrints('emptyMethod_usedForDisabling__init__s')

    @staticmethod
    def _warnUsersAgainstExplicitParentInitialization(parentClasses_tillBrazingTorch, self):
        for clsName, clsObj in parentClasses_tillBrazingTorch.items():
            if checkIfAClassIs_initingItsParentClasses_inItsInit(clsObj):
                warnMsg = '\n Warning: defining __init__ in subclasses of BrazingTorch' + \
                          '\n    you have initiated parent classes in your __init__.' + \
                          f'\n    "{clsName}" class is one of them.' + \
                          '\n    this may cause error because parent classes are initiated automatically.' + \
                          '\n    so you may want to remove the __init__ of parent classes (even using "super()") from your __init__.'
                Warn.warn(warnMsg)
                self.printTestPrints(warnMsg)
