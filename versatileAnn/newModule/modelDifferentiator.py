import inspect

from utils.initParentClasses import orderClassNames_soChildIsAlways_afterItsParents
from utils.vAnnGeneralUtils import _allowOnlyCreationOf_ChildrenInstances, isCustomClass, \
    isFunctionOrMethod, _ifFunctionOrMethod_returnIsClass_AlsoActualClassOrFunc, \
    setDefaultIfNone, isCustomFunction, DotDict, getProjectDirectory, \
    findClassObject_inADirectory, joinListWithComma
from utils.warnings import Warn
from versatileAnn.utils import LossRegularizator


# kkk user manuall add to class Definitions
# kkk final check that if the class can be recreated standalone from definitions
# kkk add this to postInit
# kkk does adding to postInit + initArgs will solve some not finding class definitions with 'globals().get(className)'
# kkk add to preRunTests + also seed
# kkk add to mainModelRun
class _NewWrapper_modelDifferentiator:
    """
    # cccWhat
    this class can be used for:
        1. check if the model definitions differ or not
        2. recreate model from strings(text) of definition of custom(user defined) classes or
            custom functions, so if the result of _getAllNeededDefinitions is passed else where the
            model can be recreated wihtout having the files, and only from those definition
            which are string

    - _getAllNeededClasses is the main func of this class and other methods are just
        utility for that one

    gets:
    - parent classes(case1):
            so this code detects what classes are used for creating "this class".
            - "this class" means the class which the user has inherited from NewWrapper, and
            probably other classes
    - classes/classes of static/instance methods or functions, related to attributes or variables(case2):
            even classes which their instances were used in attributes or variables of
            this class. for i.e. when defining model the user creates
            'self.transformer = TransformerModule()' so the definition of TransformerModule
            gets included
    - some rare cases:
        case2.1. assume the user includes a instanceMethod/staticMethod as a func to do sth. for i.e.
        "self.compareAccuracies=LossFuncsUtilsClass.accuracy" is added to model. so the definition
        of LossFuncsUtilsClass is also added to later "this class" can be created without any error
        case2.1.1: even though probably this can detect all cases but just in case maybe this code
                cannot detect existence of some class, it would give some warn(just once(not to
                irritate users with repeated warnings)) that "this class" couldn't
                detect for i.e. 'ClassA' so u should include it yourself.
        case2.2:
            assume 'self.transformerClassDefinition = TransformerModule' is exists just to have
            class object and not its instance
        3. similar to last case; assume the user includes a independent(regular func and not
            method of a class) func so its definitions is also gets added
    """

    def __init__(self, **kwargs):
        # not allowing this class to have direct instance
        _allowOnlyCreationOf_ChildrenInstances(self, _NewWrapper_modelDifferentiator)

    def _getAllNeededDefinitions(self, obj):
        # goodToHave3
        #  maybe object in this func is not needed and by default it should run on self
        # Method to get all class or func definitions in the correct order

        # Get class dict({className:classObject}) of classes needed to create "this class"
        classesDict, visitedClasses = self._getAllNeededClasses(obj)
        # cccWhy
        #  we are just doing _getAllNeededClasses twice so maybe warns in # Lwnt
        #  may be prevented because the class they want to warn which is not included may be added
        #  later at _getAllNeededClasses so the warning is not warning correctly because the class is
        #  added later
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
        visitedClasses = setDefaultIfNone(visitedClasses, set())
        classesDict = setDefaultIfNone(classesDict, {})
        visitedWarns = setDefaultIfNone(visitedWarns, set())
        # cccWhy3
        #  visitedWarns and secondTime are declared in order to give warning for not being
        #  able to include classes of some static/instance method, only once

        # duty1: Get definitions of the main object's class and its parent classes, if not already
        self._getClassAndItsParentsDefinitions(obj.__class__, visitedClasses,
                                               classesDict)

        # duty2: getting class definitions for attributes(object variables)
        # cccWhy2
        #  the attributes and recursively their attributes are looped through
        #  - as some attributes are dict and don't have their anything in their __dict__
        if isinstance(obj, dict):
            self._attributesLoop(classesDict, obj, visitedClasses, visitedWarns, secondTime)
        elif isFunctionOrMethod(obj)[0]:  # get class of static or instance methods
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

    # ---- case2 methods
    def _attributesLoop(self, classesDict, objVars, visitedClasses, visitedWarns, secondTime):
        for varName, varValue in objVars.items():

            if inspect.isclass(varValue):
                # attribute is not an instance of some class and is its definition
                # this is a rare case(case2.2) to just have class definition Object and
                # not an instance of it. but it's been included anyway
                self._getClassAndItsParentsDefinitions(varValue, visitedClasses,
                                                       classesDict)
            else:
                # attribute is an instance of some class
                self._getAllNeededClasses(varValue, visitedClasses,
                                          classesDict, visitedWarns=visitedWarns,
                                          secondTime=secondTime)

    def _addClassOf_staticOrInstanceMethods_orWarnIfNotPossible(self, classesDict, obj, secondTime,
                                                                visitedClasses, visitedWarns):
        # cccWhat
        #  related to case2.1 and if the attribute is a static or an instance method of some class

        # cccWhat
        # check if the class is in global variables
        isClass, classOrFuncObject = _ifFunctionOrMethod_returnIsClass_AlsoActualClassOrFunc(
            obj)
        if isClass:
            # if its False maybe it's the case3(regular func and not a method from a class)
            if classOrFuncObject and inspect.isclass(classOrFuncObject):
                # successful in getting class object of static or instance method from global variables
                self._getClass_ifUserDefined_ifNotGotBefore(classOrFuncObject, visitedClasses,
                                                            classesDict)
            else:
                # cccWhat
                #  failed to get class object from global variables
                #  - this is for the case that the class definition is not
                #   imported directly in the file that "this class" is defined
                # - therefore tries to search the directory of this project and this if the class
                #   is defined anywhere in files
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
                        # there is more one definition of class with the same name in the files
                        # in the directory of this project; so gives warning
                        if classNameOf_staticOrInstanceMethod not in visitedWarns:
                            visitedWarns.add(classNameOf_staticOrInstanceMethod)
                            warnMsg = f"{classNameOf_staticOrInstanceMethod} exists in " + \
                                      f"{joinListWithComma(foundClasses['filePaths'])}; " + \
                                      "so a single definition for it cannot be included"
                            self.printTestPrints(warnMsg)
                            Warn.error(warnMsg)  # Lwnt
                    else:
                        # even has failed to find the class definition from the files in the
                        # directory of this project; so gives warning
                        if obj.__qualname__ not in visitedWarns:
                            visitedWarns.add(obj.__qualname__)
                            warnMsg = f'{classNameOf_staticOrInstanceMethod} definition is not included.'

                            self.printTestPrints(warnMsg)
                            Warn.error(warnMsg)  # Lwnt

    # ---- mainly case1 methods ofc used in other cases
    def _getClassAndItsParentsDefinitions(self, cls_, visitedClasses, classesDict):
        # Helper function recursively gets class Object of cls_ and its parent class objects
        # includes them to classesDict if it has not before
        # note also tries to detect if the class is not defined by the user so then not to include it

        if not inspect.isclass(cls_):
            raise ValueError(f'{cls_} is not a class')

        self._getClass_ifUserDefined_ifNotGotBefore(cls_, visitedClasses, classesDict)
        self._getParentClasses(cls_, visitedClasses, classesDict)

    def _getParentClasses(self, cls_, visited=None, classesDict=None):
        # Helper function to get definitions of parent classes recursively
        # bugPotentialCheck2
        #  is possible to get in to infinite loop

        visited = setDefaultIfNone(visited, set())
        classesDict = setDefaultIfNone(classesDict, {})

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
        # Helper function to check if the cls_ a custom(user defined) class

        if self._isCls_NewWrapperClass(cls_):
            return None
        if cls_ in [LossRegularizator, DotDict]:
            # prevent utility classes defined for NewWrapper
            return None
        if isCustomClass(cls_):
            return cls_
        return None

    # ---- case3 methods
    def _getAttributesFuncDefinitions(self, obj, visitedFuncs=None, funcDefinitions=None):
        # cccDevAlgo
        #  case3
        #  this is a rare case but it's involved
        visitedFuncs = setDefaultIfNone(visitedFuncs, set())
        funcDefinitions = setDefaultIfNone(funcDefinitions, [])

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
                        funcDefinitions.append(
                            {classOrFuncObject.__name__: inspect.getsource(classOrFuncObject)})
        elif hasattr(obj, '__dict__'):
            objVars = vars(obj)
            for varName, varValue in objVars.items():

                if not inspect.isclass(varValue):
                    self._getAttributesFuncDefinitions(varValue, visitedFuncs,
                                                       funcDefinitions)
        elif not hasattr(obj, '__dict__'):
            return funcDefinitions

        return funcDefinitions

    # ----
    @staticmethod
    def _getClassesDefinitions(classesDict, orderedClasses):
        classDefinitions = []
        for clsName in orderedClasses:
            classDefinitions.append({clsName: inspect.getsource(classesDict[clsName])})
        return classDefinitions
