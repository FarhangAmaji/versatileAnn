
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
            # addTest2
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

