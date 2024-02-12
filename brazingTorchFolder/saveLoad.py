import pickle

import pytorch_lightning as pl

from utils.vAnnGeneralUtils import _allowOnlyCreationOf_ChildrenInstances, joinListWithComma
from utils.warnings import Warn


class _BrazingTorch_saveLoad:
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

        with open(loadPath, 'rb') as f:
            checkpoint = pickle.load(f)

        allDefinitions = checkpoint['brazingTorch']['allDefinitions']
        _initArgs = checkpoint['brazingTorch']['_initArgs']
        warnsFrom_getAllNeededDefinitions = checkpoint['brazingTorch'][
            'warnsFrom_getAllNeededDefinitions']

        instance = cls.createInstanceFrom_loadedDefinitions(allDefinitions, _initArgs,
                                                            warnsFrom_getAllNeededDefinitions)
        return instance

    @classmethod
    def createInstanceFrom_loadedDefinitions(cls, allDefinitions, _initArgs,
                                             warnsFrom_getAllNeededDefinitions):

        allDefinitions_sanity = True
        initiatedObject = None
        executionOrder = []
        errors = []

        # clean allDefinitions
        allDefinitions = cls._cleanListOfDefinitions_fromBadIndent(allDefinitions)

        loopLimit = len(allDefinitions) ** 2 + 2
        limitCounter = 0

        remainingDefinitions = allDefinitions.copy()
        while remainingDefinitions and limitCounter <= loopLimit:
            for definition in remainingDefinitions:
                for className, classCode in definition.items():
                    limitCounter += 1
                    try:
                        exec(classCode)
                        executionOrder.append(definition)
                    except:
                        # mustHave3
                        #  in modelDifferentiator we didnt include the classes or the funcs
                        #  defined in project(as a part of brazingTorchFolder) to be included so if
                        #  the error is about them here the code should be able to detect and
                        #  import that class or func
                        pass

        # run to collect errors
        if remainingDefinitions:
            # we dont put allDefinitions_sanity = False here because
            # only ability to create instance matters

            for definition in remainingDefinitions:
                for className, classCode in definition.items():
                    try:
                        exec(classCode)
                    except Exception as e:
                        errors.append(e)

        plSeed = _initArgs['__plSeed__']
        pl.seed_everything(plSeed)

        clsTypeName = _initArgs['clsTypeName']
        try:
            initiatedObject = clsTypeName(
                **_initArgs['initPassedKwargs'], getAllNeededDefinitions=False)
        except Exception as e:
            allDefinitions_sanity = False
            errors.append(e)

        if not allDefinitions_sanity:
            Warn.error("couldn't construct an instance of model with saved parameters. " + \
                       "these errors occurred:")
            Warn.error("ofc this may not be always the case but in the past you had " + \
                       f"received warns about the {joinListWithComma(warnsFrom_getAllNeededDefinitions)} " + \
                       "are not included and you had you add their code " + \
                       "definition with addDefinitionsTo_allDefinitions to model then run it.")
            for error in errors:
                Warn.error(error)
            raise errors[-1]

        return initiatedObject

    def on_save_checkpoint(self, checkpoint: dict):
        # reimplement this method to save additional information to the checkpoint

        # Add additional information to the checkpoint
        checkpoint['brazingTorch'] = {
            '_initArgs': self._initArgs,
            'allDefinitions': self.allDefinitions,
            'warnsFrom_getAllNeededDefinitions': self.warnsFrom_getAllNeededDefinitions,
        }

    def on_load_checkpoint(self, checkpoint: dict):
        # Load additional information from the checkpoint
        additionalInfo = checkpoint['brazingTorch']

        _initArgs = additionalInfo['_initArgs']
        pl.seed_everything(_initArgs['__plSeed__'])
        # I guess setting seed here doesn't really make difference
        # on the most models but some models which may use some random
        # variables in their implementation, may benefit from this
