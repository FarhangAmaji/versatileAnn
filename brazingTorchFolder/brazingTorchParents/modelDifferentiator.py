import inspect
from typing import List

from brazingTorchFolder.lossRegulator import LossRegulator
from projectUtils.dataTypeUtils.dotDict_npDict import DotDict
from projectUtils.dataTypeUtils.str import joinListWithComma
from projectUtils.initParentClasses import orderClassNames_soChildIsAlways_afterItsParents
from projectUtils.misc import _allowOnlyCreationOf_ChildrenInstances, isCustomClass, \
    isFunctionOrMethod, _ifFunctionOrMethod_returnIsClass_AlsoActualClassOrFunc, \
    setDefaultIfNone, isCustomFunction, getProjectDirectory, \
    findClassObject_inADirectory, getClassObjectFromFile
from projectUtils.typeCheck import argValidator
from projectUtils.warnings import Warn


# kkk add to mainModelRun
# goodToHave3
#  this class can be a separate class of its own
class _BrazingTorch_modelDifferentiator:
    """
    # cccWhat
    this class can be used for:
        1. check if the model definitions differ or not
        2. recreate model from strings(text) of definition of custom(user defined) classes or
            custom functions, so if the result of _getAllNeededDefinitions is passed else where the
            model can be recreated without having the files, and only from those definition
            which are string

    - _getAllNeededClasses is the main func of this class and other methods are just
        utility for that one
    - user also may need to use 'addDefinitionsTo_allDefinitions'

    gets:
    - parent classes(case1):
            so this code detects what classes are used for creating "this class".
            - "this class" means the class which the user has inherited from BrazingTorch, and
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
        3. similar to last case; assume the user includes an independent(regular func and not
            method of a class) func so its definitions is also gets added

    also does final check in order to see if can definitions executed or not
    """

    def __init__(self, getAllNeededDefinitions=True, **kwargs):
        self.getAllNeededDefinitions = getAllNeededDefinitions

        # not allowing this class to have direct instance
        _allowOnlyCreationOf_ChildrenInstances(self, _BrazingTorch_modelDifferentiator)

    def _getAllNeededDefinitions(self, obj):
        # Method to get all class or func definitions in the correct order
        # goodToHave3
        #  maybe object in this func is not needed and by default it should run on self
        # goodToHave3
        #  maybe should do similar thing for _initArgs as some layers may be passed with args

        # Get class dict({className:classObject}) of classes needed to create "this class"
        classesDict, visitedClasses, _ = self._getAllNeededClasses(obj)
        # cccWhy
        #  we are just doing _getAllNeededClasses twice so maybe warns in # Lwnt
        #  may be prevented because the class they want to warn which is not included may be added
        #  later at _getAllNeededClasses so the warning is not warning correctly because the class is
        #  added later
        classesDict, visitedClasses, visitedWarns = self._getAllNeededClasses(obj,
                                                                              visitedClasses=visitedClasses,
                                                                              classesDict=classesDict,
                                                                              secondTime=True)

        # Order the class definitions based on dependencies
        orderedClasses = orderClassNames_soChildIsAlways_afterItsParents(classesDict)
        classDefinitions = self._getClassesDefinitions(classesDict, orderedClasses)

        funcDefinitions = self._getAttributesFuncDefinitions(obj)

        self.allDefinitions = classDefinitions + funcDefinitions

        self.warnsFrom_getAllNeededDefinitions = visitedWarns
        if not visitedWarns:
            self._modelDifferentiator_sanityCheck()
        else:
            self.allDefinitions_sanity = False
            Warn.error(
                f'as you are informed {joinListWithComma(self.warnsFrom_getAllNeededDefinitions)} class definitions' + \
                ' have been failed to get included;' + \
                '\nso the final sanity check is not performed' + \
                '\nuse "addDefinitionsTo_allDefinitions" to add them manually')
        return self.allDefinitions

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

        return classesDict, visitedClasses, visitedWarns

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
                    # bugPotentialCheck1
                    #  passing getProjectDirectory() to findClassObject_inADirectory has a similar
                    #  potential error which should only search for files in some folders and
                    #  not all folders; for i.e. if the user has .venv folder in it's project, this
                    #  code would loop over files and folders in '.venv' and make this code slower,
                    #  or even break the code
                    #  - so solution comes to my mind is that I should make folder names sth between hardcoded and
                    #  automatic(I mean to hardcode folders in getProjectDirectory(), which are
                    #  allowed to loop through|note I also want to allow users to make some folders
                    #  for themselves to run their projects). note this solution maybe also solution
                    #  to #LMDCC

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

        # not allowing to collect BrazingTorch and it's parent definitions
        if self._isCls_BrazingTorchClass(cls_):
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

        if self._isCls_BrazingTorchClass(cls_):
            return None
        if cls_ in [LossRegulator, DotDict]:  # LMDCC
            # goodToHave3
            #  maybe I should not do this, as standAlone may use sth like freezing dependency; also
            #  for funcs (if next goodToHave; few lines below is implemented)
            # bugPotentialCheck2
            #  think about it later
            #  add all other classes defined in the project;
            #  obviously we want to do it everytime here, so either we have to hard code them some
            #  (ofc we may add some func to runAllTests.py to update them every time that is run)
            # goodToHave2
            #  do similar thing in _getAttributesFuncDefinitions when detects custom func and don't
            #  allow funcs defined in the project to be included
            # prevent utility classes defined for BrazingTorch
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
    def _getClassesDefinitions(classesDict, orderedClasses, printOff=False):
        classDefinitions = []
        for clsName in orderedClasses:
            try:
                clsCode = inspect.getsource(classesDict[clsName])
            except OSError:  # try another way to get sourceCode
                filePathOfClass = inspect.getfile(object)
                clsCode = getClassObjectFromFile(clsName, filePathOfClass, printOff=printOff)

            if clsCode:
                classDefinitions.append({clsName: clsCode})
        return classDefinitions

    def _modelDifferentiator_sanityCheck(self):
        # goodToHave2
        #  with initArgs create an instance of "this class" in _modelDifferentiator_sanityCheck
        # mustHave2
        #  restructure needed
        #  should tries several times so if the order of dependencies are not ok the code tries
        #  it best to automatically handle it
        self.allDefinitions = self._cleanListOfDefinitions_fromBadIndent(self.allDefinitions)

        BrazingTorch = self._getBrazingTorch_classObject()
        try:
            for i, definition in enumerate(self.allDefinitions):
                for className, classCode in definition.items():
                    exec(classCode)

            # check does having these definitions, enable to create another instance of "this class"
            if self._initArgs:
                type(self)(**self._initArgs['initPassedKwargs'], getAllNeededDefinitions=False)

            self.allDefinitions_sanity = True
            return True  # returning sanity of allDefinitions
        except Exception as e:
            print(f"Error failed to execute all definitions: {e}")
            self.allDefinitions_sanity = False
            return False

    @classmethod
    def _cleanListOfDefinitions_fromBadIndent(cls, allDefinitions):
        for i, definition in enumerate(allDefinitions):
            for className, classCode in definition.items():
                classCode = cls.removeIndents_fromCodeStringDefinition(classCode)
                allDefinitions[i] = {className: classCode}

        return allDefinitions

    # ---- addDefinitionsTo_allDefinitions methods
    @argValidator
    def addDefinitionsTo_allDefinitions(self, definitions: List[str]):
        # goodToHave3
        #  - later make a new warning here so if the input definitions differ
        #       from those warned at #Lwnt to be added; this warning says
        #       'class x1,x2,... were needed but u have added class'z' also'
        #       - for this should use self.warnsFrom_getAllNeededDefinitions
        justDefsOf_allDefinitions = [next(iter(d.values())) for d in self.allDefinitions]
        definitions_fromAllDefinitions_NDefinitions = justDefsOf_allDefinitions + definitions

        defsDict = self._getDefDict_fromDefinitionsList(
            definitions_fromAllDefinitions_NDefinitions)

        self.allDefinitions = defsDict
        self._modelDifferentiator_sanityCheck()

        return self.allDefinitions

    def _getDefDict_fromDefinitionsList(self, definitions):
        # cccUsage
        #  definitions is a list of strings of class/func definitions like "def func1():\n    print('func1')\n"
        # tries several times so if the order of dependencies are not ok the code tries
        #  it best to automatically handle it
        BrazingTorch = self._getBrazingTorch_classObject()

        for i, definition in enumerate(definitions):
            definitions[i] = self.removeIndents_fromCodeStringDefinition(definition)

        defsDictList = []
        remainingDefinitions = definitions.copy()

        loopLimit = len(definitions) ** 2 + 2
        limitCounter = 0

        while remainingDefinitions and limitCounter <= loopLimit:
            for i, definition in enumerate(remainingDefinitions):
                limitCounter += 1
                oldLocals = set(locals().keys())
                try:
                    exec(definition)
                    newLocals = {name: obj for name, obj in locals().items() if
                                 name not in oldLocals}
                    for delVar in ['oldLocals', '__builtins__', 'cleanedDefinition']:
                        if delVar in newLocals.keys():
                            del newLocals[delVar]

                    if len(newLocals) == 1:
                        definitionObjectName = list(newLocals.keys())[0]
                        defsDictList.append({definitionObjectName: definition})
                        remainingDefinitions.remove(definition)
                    else:
                        errMsg = 'note u must pass definitions of different objects separately.'
                        errMsg += f'\napparently for definition number {i} {joinListWithComma(list(newLocals.keys()))} are passed'
                        Warn.error(errMsg)
                        raise RuntimeError(errMsg)
                except Exception as e:
                    continue

        for rd in remainingDefinitions:
            defsDictList.append({self.getObjectName_fromCodeString(rd): rd})

        return defsDictList

    @staticmethod
    def getObjectName_fromCodeString(code):
        lines = code.strip().split('\n')
        firstLineTokens = lines[0].split('(')

        # Check if it is a class or function definition
        if firstLineTokens[0].startswith('class'):
            # Class definition
            className = firstLineTokens[0].split()[1]
            return className
        elif firstLineTokens[0].startswith('def'):
            # Function definition
            functionName = firstLineTokens[0].split()[1]
            return functionName
        else:
            # Unknown type or not a class/function definition
            return lines[0]

    @staticmethod
    def removeIndents_fromCodeStringDefinition(definition):
        probableIndentSpaces = len(definition) - len(definition.lstrip(' '))
        definitionLines = definition.split('\n')

        cleanedLines = [line[probableIndentSpaces:] for line in definitionLines]
        cleanedDefinition = '\n'.join(cleanedLines)
        return cleanedDefinition
