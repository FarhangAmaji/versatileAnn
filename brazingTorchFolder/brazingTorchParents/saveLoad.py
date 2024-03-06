import pickle

import pytorch_lightning as pl

from brazingTorchFolder.brazingTorchParents.innerClassesWithoutPublicMethods.saveLoad_inner import \
    _BrazingTorch_saveLoad_inner
from projectUtils.misc import _allowOnlyCreationOf_ChildrenInstances


class _BrazingTorch_saveLoad(_BrazingTorch_saveLoad_inner):
    def __init__(self, **kwargs):
        # not allowing this class to have direct instance
        _allowOnlyCreationOf_ChildrenInstances(self, _BrazingTorch_saveLoad)

    @classmethod
    def modelStandALoneLoad(cls, loadPath: str):
        # addTest1
        # you don't need to have model code and the model gets initiated here
        # note but the dataloaders are needed

        # goodToHave3
        # if the dataloaders used to train model are builtin
        # in this project like EPF, electricity,...
        # so automatically detects them and load them

        # ccc1
        #  doesn't need to be added to device, as with initArgs the init will run and there device
        #  would be set

        with open(loadPath, 'rb') as f:
            checkpoint = pickle.load(f)

        allDefinitions = checkpoint['brazingTorch']['allDefinitions']
        _initArgs = checkpoint['brazingTorch']['_initArgs']
        warnsFrom_getAllNeededDefinitions = checkpoint['brazingTorch'][
            'warnsFrom_getAllNeededDefinitions']

        instance = cls._createInstanceFrom_loadedDefinitions(allDefinitions, _initArgs,
                                                             warnsFrom_getAllNeededDefinitions)
        return instance

    def onSaveCheckpoint(self, checkpoint: dict):
        # reimplement this method to save additional information to the checkpoint

        # ccc1
        #  note this is used in on_save_checkpoint which is placed in BrazingTorch

        # Add additional information to the checkpoint
        checkpoint['brazingTorch'] = {
            '_initArgs': self._initArgs,
            'allDefinitions': self.allDefinitions,
            'warnsFrom_getAllNeededDefinitions': self.warnsFrom_getAllNeededDefinitions,
        }
        return checkpoint

    def onLoadCheckpoint(self, checkpoint: dict):
        # Load additional information from the checkpoint

        # ccc1
        #  note this is used in on_load_checkpoint which is placed in BrazingTorch

        additionalInfo = checkpoint['brazingTorch']

        _initArgs = additionalInfo['_initArgs']
        pl.seed_everything(_initArgs['__plSeed__'])
        # I guess setting seed here doesn't really make difference
        # on the most models but some models which may use some random
        # variables in their implementation, may benefit from this

        return checkpoint
