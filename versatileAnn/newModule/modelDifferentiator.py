import inspect

from utils.initParentClasses import orderClassNames_soChildIsAlways_afterItsParents
from utils.vAnnGeneralUtils import _allowOnlyCreationOf_ChildrenInstances, isCustomClass, \
    isFunctionOrMethod, _ifFunctionOrMethod_returnIsClass_AlsoActualClassOrFunc, \
    NoneToNullValueOrValue, isCustomFunction, DotDict, getProjectDirectory, \
    findClassObject_inADirectory, joinListWithComma
from utils.warnings import Warn
from versatileAnn.utils import LossRegularizator


# kkk comment and clean this and its tests + m1 +m2
# kkk add this to postInit
# kkk does adding to postInit + initArgs will solve some not finding class definitions with 'globals().get(className)'
# kkk add to preRunTests + also seed
# kkk add to mainModelRun
class _NewWrapper_modelDifferentiator:
    def __init__(self, **kwargs):
        # not allowing this class to have direct instance
        _allowOnlyCreationOf_ChildrenInstances(self, _NewWrapper_modelDifferentiator)

    def _getAllNeededDefinitions(self, obj):
        # goodToHave3
        #  maybe object in this func is not needed
        # addTest1
        # Method to get all class definitions in the correct order

        # Get unordered class definitions and extract class names and dependencies
        classesDict, visitedClasses = self._getAllNeededClasses(obj)
        # we are just doing _getAllNeededClasses twice so maybe warns in # Lwnt may prevent because they may be added later in loops
        classesDict, visitedClasses = self._getAllNeededClasses(obj,
                                                                visitedClasses=visitedClasses,
                                                                classesDict=classesDict,
                                                                secondTime=True)

        # Order the class definitions based on dependencies
        orderedClasses = orderClassNames_soChildIsAlways_afterItsParents(classesDict)
        classDefinitions = self._getClassesDefinitions(classesDict, orderedClasses)

        funcDefinitions = self._getAttributesFuncDefinitions(obj)
        return classDefinitions + funcDefinitions

    def _getAllNeededClasses(self, obj, visitedClasses=None, classesDict=None,
                             secondTime=False, visitedWarns=None):
        # Method to get all class definitions needed for an object

        # this func is recursively called so in order not fall in to infinite loop, visitedClasses
        # is declared
        visitedClasses = NoneToNullValueOrValue(visitedClasses, set())
        classesDict = NoneToNullValueOrValue(classesDict, {})
        visitedWarns = NoneToNullValueOrValue(visitedWarns, set())
        # duty1: Get definitions of the main object's class and its parent classes
        self._getClassAndItsParentsDefinitions(obj.__class__, visitedClasses,
                                               classesDict)

        # duty2: getting class definitions for attributes(object variables)
        # cccDevAlgo
        #  the attributes and recursively their attributes are looped through
        #  as some attributes are dict and don't have their anything in their __dict__
        if isinstance(obj, dict):
            self._attributesLoop(classesDict, obj, visitedClasses, visitedWarns, secondTime)
        elif isFunctionOrMethod(obj)[0]:
            self._addClassOf_staticOrInstanceMethods_orWarnIfNotPossible(classesDict, obj,
                                                                         secondTime,
                                                                         visitedClasses,
                                                                         visitedWarns)
        elif hasattr(obj, '__dict__'):
            objVars = vars(obj)
            self._attributesLoop(classesDict, objVars, visitedClasses, visitedWarns,
                                 secondTime)
        elif not hasattr(obj, '__dict__'):
            return classesDict

        return classesDict, visitedClasses

    def _addClassOf_staticOrInstanceMethods_orWarnIfNotPossible(self, classesDict, obj, secondTime,
                                                                visitedClasses, visitedWarns):
        # cccDevAlgo
        #  this is a rare case but it's involved
        #  assume that some attribute is just definition of a static/instance method of some class
        #  so in order to have the all components that let this class to be created; the
        #  definition of that method should be included so therefore we include the definition of
        #  its class
        #  note the definitions may be needed to make a standalone object on model load(it's either
        #  a feature or on the other case of not implemented in this project, you can use
        #  definitions to utilize definitions as a standalone model)
        #  standalone means u can create the object of class without having the code in ur files
        # kkk modify above later
        isClass, classOrFuncObject = _ifFunctionOrMethod_returnIsClass_AlsoActualClassOrFunc(
            obj)
        if isClass:  # getting class object of static or instance method
            if classOrFuncObject and inspect.isclass(classOrFuncObject):
                # successful in getting class object
                self._getClass_ifUserDefined_ifNotGotBefore(classOrFuncObject, visitedClasses,
                                                            classesDict)
            else:  # failed to get class object
                classNameOf_staticOrInstanceMethod = obj.__qualname__.split('.')[0]
                visitedClasses_names = [c.__name__ for c in visitedClasses]
                if classNameOf_staticOrInstanceMethod not in visitedClasses_names and secondTime:
                    foundClasses = findClassObject_inADirectory(getProjectDirectory(),
                                                                classNameOf_staticOrInstanceMethod,
                                                                printOff=self.testPrints)
                    if len(foundClasses['classObjects']) == 1:
                        self._getClassAndItsParentsDefinitions(foundClasses['classObjects'][0],
                                                               visitedClasses,
                                                               classesDict)
                    elif len(foundClasses['classObjects']) > 1:
                        if classNameOf_staticOrInstanceMethod not in visitedWarns:
                            visitedWarns.add(classNameOf_staticOrInstanceMethod)
                            warnMsg = f"{classNameOf_staticOrInstanceMethod} exists in " + \
                                      f"{joinListWithComma(foundClasses['filePaths'])}; " + \
                                      "so a single definition for it cannot be included"
                            self.printTestPrints(warnMsg)
                            Warn.error(warnMsg)  # Lwnt
                    else:
                        if obj.__qualname__ not in visitedWarns:
                            visitedWarns.add(obj.__qualname__)
                            warnMsg = f'{classNameOf_staticOrInstanceMethod} definition is not included.'

                            self.printTestPrints(warnMsg)
                            Warn.error(warnMsg)  # Lwnt

    def _attributesLoop(self, classesDict, objVars, visitedClasses, visitedWarns, secondTime):
        for varName, varValue in objVars.items():

            if inspect.isclass(varValue):
                # this is a rare case to just have class definition Object and not an instance of it
                # but it's been included anyway
                self._getClassAndItsParentsDefinitions(varValue, visitedClasses,
                                                       classesDict)
            else:
                self._getAllNeededClasses(varValue, visitedClasses,
                                          classesDict, visitedWarns=visitedWarns,
                                          secondTime=secondTime)

    def _getClassAndItsParentsDefinitions(self, cls_, visitedClasses, classesDict):
        # Helper function to get definitions of a class and its parent classes recursively

        if not inspect.isclass(cls_):
            raise ValueError(f'{cls_} is not a class')

        self._getClass_ifUserDefined_ifNotGotBefore(cls_, visitedClasses, classesDict)
        self._getParentClasses(cls_, visitedClasses, classesDict)

    def _getParentClasses(self, cls_, visited=None, classesDict=None):
        # Helper function to get definitions of parent classes recursively
        # bugPotentialCheck2
        #  is possible to get in to infinite loop

        visited = NoneToNullValueOrValue(visited, set())
        classesDict = NoneToNullValueOrValue(classesDict, {})

        # not allowing to collect NewWrapper and it's parent definitions
        if self._isCls_NewWrapperClass(cls_):
            return
        for parentClass in cls_.__bases__:
            self._getClass_ifUserDefined_ifNotGotBefore(parentClass, visited, classesDict)

        for parentClass in cls_.__bases__:
            self._getParentClasses(parentClass, visited, classesDict)
        return classesDict

    def _getClass_ifUserDefined_ifNotGotBefore(self, cls_, visitedClasses, classesDict):
        # Helper function to get class object if not processed before

        if cls_ not in visitedClasses:
            visitedClasses.add(cls_)
            classObj = self._getClass_ifUserDefined(cls_)
            if classObj:
                classesDict.update({classObj.__name__: classObj})

    def _getClass_ifUserDefined(self, cls_):
        # jjj
        # Helper function to get the definition of a custom class

        if self._isCls_NewWrapperClass(cls_):
            return None
        if cls_ in [LossRegularizator, DotDict]:
            # prevent classes defined in this project
            return None
        if isCustomClass(cls_):
            return cls_
        return None

    @staticmethod
    def _getClassesDefinitions(classesDict, orderedClasses):
        classDefinitions = []
        for clsName in orderedClasses:
            classDefinitions.append(inspect.getsource(classesDict[clsName]))
        return classDefinitions

    def _getAttributesFuncDefinitions(self, obj, visitedFuncs=None, funcDefinitions=None):
        # cccDevAlgo
        #  this is a rare case but it's involved
        #  assume that some attribute is just definition of a func
        #  so in order to have the all components that let this class to be created; the
        #  definition of that function should be included
        visitedFuncs = NoneToNullValueOrValue(visitedFuncs, set())
        funcDefinitions = NoneToNullValueOrValue(funcDefinitions, [])

        if isinstance(obj, dict):
            for varName, varValue in obj.items():

                if not inspect.isclass(varValue):
                    self._getAttributesFuncDefinitions(varValue, visitedFuncs,
                                                       funcDefinitions)
        elif isFunctionOrMethod(obj)[0]:
            isClass, classOrFuncObject = _ifFunctionOrMethod_returnIsClass_AlsoActualClassOrFunc(
                obj)
            if not isClass:
                if classOrFuncObject not in visitedFuncs:
                    if isCustomFunction(classOrFuncObject):
                        visitedFuncs.add(classOrFuncObject)
                        funcDefinitions.append(inspect.getsource(classOrFuncObject))
        elif hasattr(obj, '__dict__'):
            objVars = vars(obj)
            for varName, varValue in objVars.items():

                if not inspect.isclass(varValue):
                    self._getAttributesFuncDefinitions(varValue, visitedFuncs,
                                                       funcDefinitions)
        elif not hasattr(obj, '__dict__'):
            return funcDefinitions

        return funcDefinitions
