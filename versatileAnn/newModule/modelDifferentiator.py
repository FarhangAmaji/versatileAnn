import inspect

from utils.vAnnGeneralUtils import _allowOnlyCreationOf_ChildrenInstances, isCustomClass, \
    isFunctionOrMethod, _ifFunctionOrMethod_returnIsClass_AlsoActualClassOrFunc, \
    NoneToNullValueOrValue, isCustomFunction
from utils.warnings import Warn


class _NewWrapper_modelDifferentiator:
    def __init__(self, **kwargs):
        # not allowing this class to have direct instance
        _allowOnlyCreationOf_ChildrenInstances(self, _NewWrapper_modelDifferentiator)

    # kkk save profiler on architecture definitions/seed
    def _getAllNeededDefinitions(self, obj):
        # goodToHave3
        #  maybe object in this func is not needed
        # addTest1
        # Method to get all class definitions in the correct order
        # kkk probably somewhere in postInit this should be called 'self._getAllNeededDefinitions(self)'

        # Get unordered class definitions and extract class names and dependencies
        unorderedClasses = self._getAllNeededClassDefinitions(obj)

        # Order the class definitions based on dependencies
        classDefinitionsWithInfo = self._findClassNamesAndDependencies(unorderedClasses)
        orderedClasses = self._getOrderedClassDefinitions(classDefinitionsWithInfo)

        funcDefinitions = self._getAttributesFuncDefinitions(obj)
        return orderedClasses + funcDefinitions

    def _getAllNeededClassDefinitions(self, obj, visitedClasses=None, classDefinitions=None):
        # Method to get all class definitions needed for an object

        # this func is recursively called so in order not fall in to infinite loop, visitedClasses
        # is declared
        visitedClasses = NoneToNullValueOrValue(visitedClasses, set())
        classDefinitions = NoneToNullValueOrValue(classDefinitions, [])
        # duty1: Get definitions of the main object's class and its parent classes
        # kkk does it get custom class defs or all classes?
        self._getClassAndItsParentsDefinitions(obj.__class__, visitedClasses,
                                               classDefinitions)

        # duty2: getting class definitions for attributes(object variables)
        # kkk should visitedClasses be extended or in inner funcs appended as list is mutable
        # kkk add comment about we are not looping through parent vars(as maybe we don't access to their objects directly) but as their attributes exist in our ojbect(which is inherited from them) we can get definitions they need
        # kkk if later wanted to add a loop through parent we may use __qualname__ to created something like visited for classes and attributes
        if isinstance(obj, dict):
            self._attributesLoop(classDefinitions, obj, visitedClasses)
        elif isFunctionOrMethod(obj)[0]:
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
                        # kkk maybe do it twice so if still not involved give the warn
                        # kkk add giveWarning with qualname
                        Warn.error(
                            f'{classNameOf_staticOrInstanceMethod} definition is not included.')
        elif hasattr(obj, '__dict__'):
            objVars = vars(obj)
            self._attributesLoop(classDefinitions, objVars, visitedClasses)
        elif not hasattr(obj, '__dict__'):
            return classDefinitions

        return classDefinitions

    def _attributesLoop(self, classDefinitions, objVars, visitedClasses):
        for varName, varValue in objVars.items():

            if inspect.isclass(varValue):
                # this is a rare case to just have class definition Object and not an instance of it
                # but it's been included anyway
                self._getClassAndItsParentsDefinitions(varValue, visitedClasses,
                                                       classDefinitions)
            else:
                self._getAllNeededClassDefinitions(varValue, visitedClasses,
                                                   classDefinitions)

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

        # kkk if it should be NewWrapper, so should placed there
        if self._isCls_NewWrapperClass(cls_):
            # kkk should it be NewWrapper or _NewWrapper_modelDifferentiator; does this new architecture make problems?
            return None
        if isCustomClass(cls_):
            return inspect.getsource(cls_)
        return None

    @staticmethod
    def _findClassNamesAndDependencies(classDefinitions):
        # kkk check this
        # Helper function to find class names and their dependencies from class definitions
        classDefinitionsWithInfo = []

        for i in range(len(classDefinitions)):
            classDef = classDefinitions[i]
            # Extract the class name
            classNameWithDependencies = classDef.split("class ")[1].split(":")[0].strip()

            # Find the superclass names (if any)
            if '(' in classNameWithDependencies:
                class_name = classNameWithDependencies.split("(")[0].strip()
                superclasses = [s.strip() for s in
                                classNameWithDependencies.split("(")[1].split(")")[0].split(
                                    ",")]
                superclasses = [s.split('=')[1].strip() if '=' in s else s for s in
                                superclasses]

                classDefinitionsWithInfo.append(
                    {'class_name': class_name, 'dep': superclasses, 'def': classDef})
            else:
                class_name = classNameWithDependencies
                classDefinitionsWithInfo.append(
                    {'class_name': class_name, 'dep': [], 'def': classDef})
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
                        if dependencies[k] == classDefinitionsWithInfo[j]['class_name']:
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
        elif isFunctionOrMethod(obj)[0]:  # kkk
            isClass, classOrFuncObject = _ifFunctionOrMethod_returnIsClass_AlsoActualClassOrFunc(
                obj)
            if not isClass:
                if classOrFuncObject not in visitedFuncs:
                    if isCustomFunction(classOrFuncObject):
                        visitedFuncs.add(classOrFuncObject)
                        funcDefinitions.append(inspect.getsource(classOrFuncObject))
                    # kkk if the func is a custom user defined func should be added
        elif hasattr(obj, '__dict__'):
            objVars = vars(obj)
            for varName, varValue in objVars.items():

                if not inspect.isclass(varValue):
                    self._getAttributesFuncDefinitions(varValue, visitedFuncs,
                                                       funcDefinitions)
        elif not hasattr(obj, '__dict__'):
            return funcDefinitions

        return funcDefinitions
