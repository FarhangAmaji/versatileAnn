import copy

import pytorch_lightning as pl

from utils.customErrors import InternalLogicError
from utils.initParentClasses import getArgsOfClasses, exclude_selfNArgsNKwargs_fromAllArgs, \
    getArgsRelatedToAClass_fromAllArgs, orderClassNames_soChildIsAlways_afterItsParents, \
    checkIfAClassIs_initingItsParentClasses_inItsInit
from utils.warnings import Warn


class _NewWrapper_preInitNPostInit_nModelReset_inner:

    @staticmethod
    def _getArgsOfParentClasses_tillNewWrapper(_NewWrapper_Obj, cls, originalKwargs):
        parentClasses_tillNewWrapper = cls._findAllParentClasses_tillNewWrapper(cls,
                                                                                _NewWrapper_Obj)

        # cccDevStruct
        #  beside the class object we also store its '__init__' because at the end of __new__,
        #  init are disabled but later at _NewWrapper_postInit,
        #  these stored inits are going to be replaced back
        cls._parentClasses_tillNewWrapper_inits = {
            clsName: {'originalInit': copy.deepcopy(classObj.__init__), 'classObj': classObj} for
            clsName, classObj in parentClasses_tillNewWrapper.items()}

        argsOf_parentClasses_tillNewWrapper = getArgsOfClasses(parentClasses_tillNewWrapper,
                                                               originalKwargs)

        # *args for subclasses of NewWrapper are not applied
        if 'args' in argsOf_parentClasses_tillNewWrapper:
            Warn.error('"*args" for subclasses of NewWrapper are not applied.' +
                       "\nthis warning is not always True, but it's better to double check" +
                       " that you have not used *args in your __init__")
            # cccDevStruct
            #  we don't make error and just give warning, because sometimes the user have not
            #  included *args in their __init__ but because of inheritance from `object`,
            #  __init__ has *args
        exclude_selfNArgsNKwargs_fromAllArgs(argsOf_parentClasses_tillNewWrapper)
        return argsOf_parentClasses_tillNewWrapper, parentClasses_tillNewWrapper

    @staticmethod
    def _findAllParentClasses_tillNewWrapper(cls_, NewWrapper_Obj, parentClasses: dict = None):
        # this method is similar to findParentClasses_OfAClass_tillAnotherClass in utils/initParentClasses.py

        parentClasses = parentClasses or {}
        # goodToHave3 bugPotentialCheck2
        #  some classes may have same .__name__ but are actually different classes
        #  but I am not counting for that(maybe later).
        #  so for now each class is going to be captured in a dict with {class.__name__:classObj}

        if str(cls_) == str(NewWrapper_Obj):
            return parentClasses
        elif cls_ is pl.LightningModule:
            return parentClasses
        elif cls_ is object:
            return parentClasses

        parentClasses.update({cls_.__name__: cls_})
        parentsOfThisClass = cls_.__bases__
        for potc in parentsOfThisClass:
            parentClasses = _NewWrapper_preInitNPostInit_nModelReset_inner._findAllParentClasses_tillNewWrapper(
                potc,
                NewWrapper_Obj,
                parentClasses)
        return parentClasses

    @staticmethod
    def _get_parentClassesOfNewWrapper(NewWrapper_Obj, originalKwargs):
        parentClassesOfNewWrapper = {pc.__name__: pc for pc in NewWrapper_Obj.__bases__}
        argsOf_parentClassesOfNewWrapper = getArgsOfClasses(parentClassesOfNewWrapper,
                                                            originalKwargs)
        exclude_selfNArgsNKwargs_fromAllArgs(argsOf_parentClassesOfNewWrapper)

        # cccDevStruct
        #  note this is only for development error detection
        #  args of parent classes of NewWrapper must not have similar names
        for arg, argVal in argsOf_parentClassesOfNewWrapper.items():
            if len(argVal['classes']) > 1:
                raise InternalLogicError(
                    "internalError: this is for development:" +
                    "\nparentClasses of NewWrapper must not have args with similar names in their __init__."
                    f'\narg "{arg}" is used in more than one base classes of NewWrapper: {argVal["classes"]}.')
        return argsOf_parentClassesOfNewWrapper, parentClassesOfNewWrapper

    @staticmethod
    def _combineArgsOfParentClasses_ofTillNewWrapper_withParentsOfNewWrapper(
            argsOf_parentClassesOfNewWrapper, argsOf_parentClasses_tillNewWrapper):
        allArgs = {**argsOf_parentClasses_tillNewWrapper}
        for arg, argVal in argsOf_parentClassesOfNewWrapper.items():
            if arg not in allArgs:
                allArgs[arg] = argVal
            else:
                allArgs[arg]['classes'].extend(argVal['classes'])
                Warn.warn(
                    f'"{arg}" arg is used in the classes you have defined. ' +
                    'and also exist in required args of NewWrapper.' +
                    '\nthis may cause conflict if are used for other purposes than passing to NewWrapper.' +
                    'you may want to change the name of this arg.')
        return allArgs

    @staticmethod
    def _initParentClasses_tillNewWrapper_withDisablingTheirInits(allArgsWithValues, cls,
                                                                  initiatedObj,
                                                                  parentClasses_tillNewWrapper):
        # parent classes which are more base are __init__ed first
        parentClasses_tillNewWrapper_names_ordered = orderClassNames_soChildIsAlways_afterItsParents(
            parentClasses_tillNewWrapper)
        for i, clsName in enumerate(parentClasses_tillNewWrapper_names_ordered):
            classRelatedArgs = getArgsRelatedToAClass_fromAllArgs(clsName, allArgsWithValues)
            clsObj = parentClasses_tillNewWrapper[clsName]
            clsObj.__init__(initiatedObj, **classRelatedArgs)

            # inits are disabled so not to get inited twice; they are set back to their originalInit,
            # at _NewWrapper_postInit
            # bugPotentialCheck2
            #  returning initiatedObj in __new__ made to __init__s disabled here to be called again.
            #  this can be seen when debugging through when creating an instance.
            #  note before adding 'return initiatedObj' in __new__, this was not happening.
            #  - another odd thing is that this recalling of __init__s is not detected by
            #  self.assertPrint of testObjectCreation of newWrapperTests_preInitNPostInit_nModelReset
            if clsObj is not cls:
                clsObj.__init__ = cls._emptyMethod_usedForDisabling__init__s
            else:
                # replace lastChildOfAll's __init__ with _NewWrapper_postInit
                clsObj.__init__ = cls._NewWrapper_postInit
                clsObj.__init__(initiatedObj, **classRelatedArgs)

    @staticmethod
    def _emptyMethod_usedForDisabling__init__s(self, **kwargs):
        self.printTestPrints('emptyMethod_usedForDisabling__init__s')

    @staticmethod
    def _warnUsersAgainstExplicitParentInitialization(parentClasses_tillNewWrapper):
        for clsName, clsObj in parentClasses_tillNewWrapper.items():
            if checkIfAClassIs_initingItsParentClasses_inItsInit(clsObj):
                Warn.warn('you have initiated parent classes in your __init__.' +
                          f'\n{clsName} class is one of them.' +
                          '\nthis may cause error because parent classes are initiated automatically.' +
                          '\nso you may want to remove the __init__ of parent classes from your __init__.')