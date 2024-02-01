import inspect

from utils.vAnnGeneralUtils import _allowOnlyCreationOf_ChildrenInstances, isCustomClass, \
    isFunctionOrMethod, _ifFunctionOrMethod_returnIsClass_AlsoActualClassOrFunc, \
    NoneToNullValueOrValue, isCustomFunction, DotDict
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
        unorderedClasses, visitedClasses = self._getAllNeededClassDefinitions(obj)
        # we are just doing _getAllNeededClassDefinitions twice so maybe warns in # Lwnt may prevent because they may be added later in loops
        unorderedClasses, visitedClasses = self._getAllNeededClassDefinitions(obj,
                                                                              visitedClasses=visitedClasses,
                                                                              classDefinitions=unorderedClasses,
                                                                              secondTime=True)

        # Order the class definitions based on dependencies
        classDefinitionsWithInfo = self._findClassNamesAndDependencies(unorderedClasses)
        orderedClasses = self._getOrderedClassDefinitions(classDefinitionsWithInfo)

        funcDefinitions = self._getAttributesFuncDefinitions(obj)
        return orderedClasses + funcDefinitions

    def _getAllNeededClassDefinitions(self, obj, visitedClasses=None, classDefinitions=None,
                                      secondTime=False, visitedWarns=None):
        # Method to get all class definitions needed for an object

        # this func is recursively called so in order not fall in to infinite loop, visitedClasses
        # is declared
        visitedClasses = NoneToNullValueOrValue(visitedClasses, set())
        classDefinitions = NoneToNullValueOrValue(classDefinitions, [])
        visitedWarns = NoneToNullValueOrValue(visitedWarns, set())
        # duty1: Get definitions of the main object's class and its parent classes
        self._getClassAndItsParentsDefinitions(obj.__class__, visitedClasses,
                                               classDefinitions)

        # duty2: getting class definitions for attributes(object variables)
        # kkk add comment about we are not looping through parent vars(as maybe we don't access to their objects directly) but as their attributes exist in our ojbect(which is inherited from them) we can get definitions they need
        # kkk if later wanted to add a loop through parent we may use __qualname__ to created something like visited for classes and attributes
        if isinstance(obj, dict):
            self._attributesLoop(classDefinitions, obj, visitedClasses, visitedWarns, secondTime)
        elif isFunctionOrMethod(obj)[0]:
            self._addStaticOrInstanceMethods_orWarnIfNotPossible(classDefinitions, obj, secondTime,
                                                                 visitedClasses, visitedWarns)
        elif hasattr(obj, '__dict__'):
            objVars = vars(obj)
            self._attributesLoop(classDefinitions, objVars, visitedClasses, visitedWarns,
                                 secondTime)
        elif not hasattr(obj, '__dict__'):
            return classDefinitions

        return classDefinitions, visitedClasses

    def _addStaticOrInstanceMethods_orWarnIfNotPossible(self, classDefinitions, obj, secondTime,
                                                        visitedClasses, visitedWarns):
        isClass, classOrFuncObject = _ifFunctionOrMethod_returnIsClass_AlsoActualClassOrFunc(
            obj)
        if isClass:  # getting class object of static or instance method
            if classOrFuncObject and inspect.isclass(
                    classOrFuncObject):  # successful in getting class object
                self._getClassDefinitions_ifNotGotBefore(classOrFuncObject, visitedClasses,
                                                         classDefinitions)
            else:  # failed to get class object
                classNameOf_staticOrInstanceMethod = obj.__qualname__.split('.')[0]
                visitedClasses_names = [c.__name__ for c in visitedClasses]
                if classNameOf_staticOrInstanceMethod not in visitedClasses_names:
                    if obj.__qualname__ not in visitedWarns and secondTime:
                        visitedWarns.add(obj.__qualname__)
                        warnMsg = f'{classNameOf_staticOrInstanceMethod} definition is not included.'
                        self.printTestPrints(warnMsg)
                        Warn.error(warnMsg)  # Lwnt

    def _attributesLoop(self, classDefinitions, objVars, visitedClasses, visitedWarns, secondTime):
        for varName, varValue in objVars.items():

            if inspect.isclass(varValue):
                # this is a rare case to just have class definition Object and not an instance of it
                # but it's been included anyway
                self._getClassAndItsParentsDefinitions(varValue, visitedClasses,
                                                       classDefinitions)
            else:
                self._getAllNeededClassDefinitions(varValue, visitedClasses,
                                                   classDefinitions, visitedWarns=visitedWarns,
                                                   secondTime=secondTime)

    def _getClassAndItsParentsDefinitions(self, cls_, visitedClasses, classDefinitions):
        # Helper function to get definitions of a class and its parent classes recursively

        if not inspect.isclass(cls_):
            raise ValueError(f'{cls_} is not a class')

        self._getClassDefinitions_ifNotGotBefore(cls_, visitedClasses, classDefinitions)
        self._getParentClassDefinitions(cls_, visitedClasses, classDefinitions)

    def _getParentClassDefinitions(self, cls_, visited=None, classDefinitions=None):
        # Helper function to get definitions of parent classes recursively

        visited = NoneToNullValueOrValue(visited, set())
        classDefinitions = NoneToNullValueOrValue(classDefinitions, [])

        # not allowing to collect NewWrapper and it's parent definitions
        if self._isCls_NewWrapperClass(cls_):
            return []
        for parentClass in cls_.__bases__:
            self._getClassDefinitions_ifNotGotBefore(parentClass, visited, classDefinitions)

        for parentClass in cls_.__bases__:
            self._getParentClassDefinitions(parentClass, visited, classDefinitions)
        return classDefinitions

    def _getClassDefinitions_ifNotGotBefore(self, cls_, visitedClasses, classDefinitions):
        # Helper function to get class definitions if not processed before

        if cls_ not in visitedClasses:
            visitedClasses.add(cls_)
            classDefinition = self._getCustomClassDefinition(cls_)
            if classDefinition:
                classDefinitions.append(classDefinition)

    def _getCustomClassDefinition(self, cls_):
        # Helper function to get the definition of a custom class

        if self._isCls_NewWrapperClass(cls_):
            return None
        if cls_ in [LossRegularizator, DotDict]:
            # prevent classes defined in this project
            return None
        if isCustomClass(cls_):
            return inspect.getsource(cls_)
        return None

    @staticmethod
    def _findClassNamesAndDependencies(classDefinitions):
        # kkk check this
        # Helper function to find class names and their dependencies from class definitions
        classDefinitionsWithInfo = []

        for classDef in classDefinitions:
            # bugPotentialCHeck1
            #  this way of finding superClasses is dependent on written class code;
            #  so if the class name and superClasses are in 2 or more lines it would give error
            # Extract the class name
            classNameWithDependencies = classDef.split("class ")[1].split(":")[0].strip()

            # Find the superclass names (if any)
            if '(' in classNameWithDependencies:
                className = classNameWithDependencies.split("(")[0].strip()
                superclasses = [s.strip() for s in
                                classNameWithDependencies.split("(")[1].split(")")[0].split(",")]
                superclasses = [s.split('=')[1].strip() if '=' in s else s for s in superclasses]

                classDefinitionsWithInfo.append(
                    {'className': className, 'dep': superclasses, 'def': classDef})
            else:
                className = classNameWithDependencies
                classDefinitionsWithInfo.append(
                    {'className': className, 'dep': [], 'def': classDef})
        return classDefinitionsWithInfo

    @staticmethod
    def _getOrderedClassDefinitions(classDefinitionsWithInfo):
        # Helper function to order class definitions based on their dependencies
        # kkk check this
        changes = 1
        while changes != 0:
            changes = 0
            for i in range(len(classDefinitionsWithInfo)):
                dependencies = classDefinitionsWithInfo[i]['dep']
                for k in range(len(dependencies)):
                    for j in range(len(classDefinitionsWithInfo)):
                        if dependencies[k] == classDefinitionsWithInfo[j]['className']:
                            if j > i:
                                classDefinitionsWithInfo[j], classDefinitionsWithInfo[i] = \
                                    classDefinitionsWithInfo[i], classDefinitionsWithInfo[j]
                                changes += 1
        classDefinitions = []
        for i in range(len(classDefinitionsWithInfo)):
            classDefinitions.append(classDefinitionsWithInfo[i]['def'])
        return classDefinitions

    def _getAttributesFuncDefinitions(self, obj, visitedFuncs=None, funcDefinitions=None):
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
