import pytorch_lightning as pl

from utils.dataTypeUtils.dotDict_npDict import DotDict
from utils.dataTypeUtils.tensor import getTorchDevice
from utils.generalUtils import _allowOnlyCreationOf_ChildrenInstances
from utils.typeCheck import argValidator


class _BrazingTorch_baseAttributes:
    @argValidator
    def __init__(self, modelName: str = '',
                 noAdditionalOptions: bool = False, testPrints=False, seed=None):
        # bugPotentialCheck1 # goodToHave3
        #  the 'mps' torch device used for macbooks is not working with precision=64; so if the
        #  pytorch lightning precision is 64 then should lower that to 32

        # bugPotentialCheck1
        #  setting device may not be compatible, of get use the best out of pytorch lightning
        #  multiple gpu or distributed capabilities

        self.to(getTorchDevice().type)
        self._setModelName(modelName)
        self.noAdditionalOptions = noAdditionalOptions
        # cccUsage
        #  if noAdditionalOptions is True, adds gradientClipping=0.1
        #  and generalRegularization of type "l2" and 0.001 value
        self.testPrints = testPrints
        self.phases = DotDict({key: key for key in ['train', 'val', 'test', 'predict']})

        # not allowing this class to have direct instance
        _allowOnlyCreationOf_ChildrenInstances(self, _BrazingTorch_baseAttributes)

    def _setModelName(self, modelName):
        if not modelName:
            if self.__class__.__name__ == 'BrazingTorch':
                raise ValueError('modelName must be provided if not inherited form BrazingTorch')
            self.modelName = self.__class__.__name__
        else:
            self.modelName = modelName

    def printTestPrints(self, *args):
        # only prints for test purposes
        if self.testPrints:
            for arg in args:
                print(arg)
