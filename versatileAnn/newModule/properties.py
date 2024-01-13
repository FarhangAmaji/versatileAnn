from abc import ABC

import torch

from utils.typeCheck import argValidator


class _NewWrapper_properties(ABC):
    @argValidator
    def __init__(self, modelName: str = '', devMode: bool = True, lr=3e-4, testPrints=False):
        self.to('cuda' if torch.cuda.is_available() else 'cpu')
        self.losses = []
        self._setModelName(modelName)
        self.devMode = devMode  # kkk2 do I need it? if I detected has not run pretests then run them and dont need devMode
        self.lr = lr  # kkk2 make property
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
        if self.testPrints:
            print(*args)
