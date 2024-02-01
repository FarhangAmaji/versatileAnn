import inspect

from utils.vAnnGeneralUtils import _allowOnlyCreationOf_ChildrenInstances


class _NewWrapper_modelDifferentiator:
    def __init__(self, **kwargs):
        # not allowing this class to have direct instance
        _allowOnlyCreationOf_ChildrenInstances(self, _NewWrapper_modelDifferentiator)

    # kkk save profiler on architecture definitions/seed
    def _getAllNeededDefinitions(self, obj):
        # addTest1
        # Method to get all class definitions in the correct order
        # kkk probably somewhere in postInit this should be called 'self._getAllNeededDefinitions(self)'

        # Get unordered class definitions and extract class names and dependencies
        unorderedClasses = self._getAllNeededClassDefinitions(obj)

        # Order the class definitions based on dependencies
        classDefinitionsWithInfo = self._findClassNamesAndDependencies(unorderedClasses)
        orderedClasses = self._getOrderedClassDefinitions(classDefinitionsWithInfo)
        return '\n'.join(orderedClasses)

    def _getAllNeededClassDefinitions(self, obj, visitedClasses=None, classDefinitions=None):
        # Method to get all class definitions needed for an object

        # this func is recursively called so in order not fall in to infinite loop, visitedClasses
        # is declared
        visitedClasses = visitedClasses or set()
        classDefinitions = classDefinitions or []
        # duty1: Get definitions of the main object's class and its parent classes
        # kkk does it get custom class defs or all classes?
        self._getClassAndItsParentsDefinitions(obj.__class__, visitedClasses,
                                               classDefinitions)

        # duty2: getting class definitions for attributes(object variables)
        # kkk should visitedClasses be extended or in inner funcs appended as list is mutable
        #kkk add comment about we are not looping through parent vars(as maybe we don't access to their objects directly) but as their attributes exist in our ojbect(which is inherited from them) we can get definitions they need
        #kkk if later wanted to add a loop through parent we may use __qualname__ to created something like visited for classes and attributes
        if isinstance(obj, dict):
            self._attributesLoop(classDefinitions, obj, visitedClasses)
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

        visited = visited or set()
        classDefinitions = classDefinitions or []

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
        if self._isCls_NewWrapperClass(
                cls_):  # kkk should it be NewWrapper or _NewWrapper_modelDifferentiator; does this new architecture make problems?
            return None
        if self.isCustomClass(cls_):
            return inspect.getsource(cls_)
        return None

    @staticmethod
    def isCustomClass(cls_):
        # kkk move it to utils
        # Helper function to check if a class is a custom(user defined and not python builtin or not from packages) class

        import builtins
        import pkg_resources
        import types
        if cls_ is None or cls_ is types.NoneType:  # kkk
            return False
        moduleName = getattr(cls_, '__module__', '')
        return (
                isinstance(cls_, type) and
                not (
                        cls_ in builtins.__dict__.values()
                        or any(moduleName.startswith(package.key) for package in
                               pkg_resources.working_set)
                        or moduleName.startswith('collections')
                )
        ) and not issubclass(cls_, types.FunctionType)

    @staticmethod
    def _findClassNamesAndDependencies(classDefinitions):
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
