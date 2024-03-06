from typing import List

from brazingTorchFolder.brazingTorchParents.innerClassesWithoutPublicMethods.modelDifferentiator_inner import \
    _BrazingTorch_modelDifferentiator_inner
from projectUtils.misc import _allowOnlyCreationOf_ChildrenInstances
from projectUtils.typeCheck import argValidator


# kkk add to mainModelRun
# goodToHave3
#  this class can be a separate class of its own
class _BrazingTorch_modelDifferentiator(_BrazingTorch_modelDifferentiator_inner):
    # kkk check docstring here and in inner
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

    @staticmethod
    def removeIndents_fromCodeStringDefinition(definition):
        # ccc3
        #  this method is a utility method which can be used only needed
        probableIndentSpaces = len(definition) - len(definition.lstrip(' '))
        definitionLines = definition.split('\n')

        cleanedLines = [line[probableIndentSpaces:] for line in definitionLines]
        cleanedDefinition = '\n'.join(cleanedLines)
        return cleanedDefinition
