from projectUtils.customErrors import ImplementationError
from projectUtils.dataTypeUtils.dotDict_npDict import DotDict
from projectUtils.dataTypeUtils.tensor import getTorchDevice
from projectUtils.misc import _allowOnlyCreationOf_ChildrenInstances
from projectUtils.typeCheck import argValidator


class _BrazingTorch_baseAttributes:
    @argValidator
    def __init__(self, modelName: str = '',
                 noAdditionalOptions: bool = False, testPrints=False, seed=None):
        # kkk why seed is not used
        # bugPotn1 # goodToHave3
        #  the 'mps' torch device used for macbooks is not working with precision=64; so if the
        #  pytorch lightning precision is 64 then should lower that to 32

        # bugPotn1
        #  setting device may not be compatible, of get use the best out of pytorch lightning
        #  multiple gpu or distributed capabilities
        #  - maybe if args related to trainer like `gpus`(maybe `gpus` is depreacated and now is
        #  `accelarator` or `devices`; anyway) is provided then don't set the device here and warn user to do it
        #  himself

        self.to(getTorchDevice().type)
        self._setModelName(modelName)
        self.noAdditionalOptions = noAdditionalOptions
        # cccUsage
        #  if noAdditionalOptions is True, adds gradientClipping=0.1
        #  and generalRegularization of type "l2" and 0.001 value
        self.testPrints = testPrints
        self.phases = DotDict({key: key for key in ['train', 'val', 'test', 'predict']})

        # ccc3 cccDev
        #  this one is related to _BrazingTorch_preInitNPostInit_nModelReset
        #  the user must not have defined __new__ method
        _BrazingTorch_Obj = self._getBrazingTorch_classObject()
        if type(self).__new__ is not _BrazingTorch_Obj.__new__:
            raise ImplementationError(
                f'"{type(self)}" class is not allowed to have __new__ method.')

        # not allowing this class to have direct instance
        _allowOnlyCreationOf_ChildrenInstances(self, _BrazingTorch_baseAttributes)

    def _setModelName(self, modelName):
        if not modelName:
            if self.__class__.__name__ == 'BrazingTorch':
                raise ValueError('modelName must be provided if not inherited form BrazingTorch')
            self.modelName = self.__class__.__name__
        else:
            self.modelName = modelName

    def _printTestPrints(self, *args):
        # ccc2
        #  only prints for test purposes
        #  specially as the Warn doesn't get catched by BaseTestClass.assertPrint
        if self.testPrints:
            for arg in args:
                print(arg)
