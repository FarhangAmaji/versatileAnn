from utils.typeCheck import argValidator
from utils.vAnnGeneralUtils import getTorchDevice, DotDict, _allowOnlyCreationOf_ChildrenInstances


class _NewWrapper_properties:
    # kkk rename this properties doesnt make sense
    @argValidator
    def __init__(self, modelName: str = '', devMode: bool = True,
                 noAdditionalOptions: bool = False, testPrints=False):
        # bugPotentialCheck1 # goodToHave3
        #  the 'mps' torch device used for macbooks is not working with precision=64; so if the
        #  pytorch lightning precision is 64 then should lower that to 32

        # bugPotentialCheck1
        #  setting device may not be compatible, of get use the best out of pytorch lightning
        #  multiple gpu or distributed capabilities
        self.to(getTorchDevice().type)
        self.losses = []
        self._setModelName(modelName)
        self.noAdditionalOptions = noAdditionalOptions
        self.devMode = devMode  # kkk2 do I need it? if I detected has not run pretests then run them and dont need devMode
        self.testPrints = testPrints
        self.phases = DotDict({key: key for key in ['train', 'val', 'test', 'predict']})

        # not allowing this class to have direct instance
        _allowOnlyCreationOf_ChildrenInstances(self, _NewWrapper_properties)

        if devMode:
            pass  # kkk?
        else:
            pass  # kkk?

    def _setModelName(self, modelName):
        if not modelName:
            if self.__class__.__name__ == 'NewWrapper':
                raise ValueError('modelName must be provided if not inherited form NewWrapper')
            self.modelName = self.__class__.__name__
        else:
            self.modelName = modelName

    @property
    def devMode(self):
        return self._devMode

    @devMode.setter
    @argValidator
    def devMode(self, value: bool):
        self._devMode = value

    def printTestPrints(self, *args):
        # only prints for test purposes
        if self.testPrints:
            for arg in args:
                print(arg)
