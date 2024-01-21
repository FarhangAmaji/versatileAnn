import torch

from utils.typeCheck import argValidator
from utils.vAnnGeneralUtils import getTorchDevice


class _NewWrapper_properties:
    @argValidator
    def __init__(self, modelName: str = '', devMode: bool = True, testPrints=False):
        self.to(getTorchDevice().type)
        self.losses = []
        self._setModelName(modelName)
        self.devMode = devMode  # kkk2 do I need it? if I detected has not run pretests then run them and dont need devMode
        self.testPrints = testPrints

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
