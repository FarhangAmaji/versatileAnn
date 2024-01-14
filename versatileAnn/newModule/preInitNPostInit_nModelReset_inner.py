
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

