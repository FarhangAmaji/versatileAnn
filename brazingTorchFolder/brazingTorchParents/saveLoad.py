import os
import pickle

import pytorch_lightning as pl

from projectUtils.dataTypeUtils.dict import stringValuedDictsEqual
from projectUtils.dataTypeUtils.str import joinListWithComma
from projectUtils.misc import _allowOnlyCreationOf_ChildrenInstances
from projectUtils.misc import nFoldersBack
from projectUtils.warnings import Warn


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

        # ccc1
        #  doesn't need to be added to device, as with initArgs the init will run and there device
        #  would be set

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

    # ---- methods used to determine the architectureName of the model
    def _collectArchDicts(self, loggerPath):
        pickleFiles = []

        path = nFoldersBack(loggerPath, n=2)
        for root, dirs, files in os.walk(path):
            for file in files:
                if file == 'architecture.pkl':
                    pickleFiles.append(os.path.join(root, file))

        architectureDicts = []
        for pickleFile in pickleFiles:
            with open(pickleFile, 'rb') as f:
                architectureDict = pickle.load(f)
                architectureDict = {pickleFile: architectureDict}
                architectureDicts.append(architectureDict)

        return architectureDicts

    def _updateLoggerPath_withExistingArchName(self, architectureDicts, runName):
        uniqueArchNames = set(key for ad in architectureDicts for key in ad.keys())
        uniqueArchNames = set(uan.split(os.sep)[-3] for uan in uniqueArchNames)
        dummyLogger = pl.loggers.TensorBoardLogger(self.modelName,
                                                   name=list(uniqueArchNames)[0],
                                                   version=runName)
        loggerPath = os.path.abspath(dummyLogger.log_dir)
        return loggerPath

    def _getArchitectureDicts_withMatchedAllDefinitions(self, architectureDicts):
        # cccWhat
        # this func checks the match between self.allDefinitions and allDefinitions
        # in architectureDicts and brings back 'architectureDicts_withMatchedAllDefinitions' which
        # is a list of architectureDicts
        # cccWhy
        # 1. self.allDefinitions is a list of some dicts which have 'func or class names'
        # as key and there string definition, sth like
        #   [{'class1Parent': 'class class1Parent:\n    def __init__(self):\n        self.var1 = 1\n'},
        #   {'func1': "def func1():\n    print('func1')\n"}]
        # 2. architectureDicts is a list of dicts like
        #   {filePath:{'allDefinitions': allDefinitions, '__plSeed__': someNumber}}

        # Convert list of dicts to a single dict
        toDictConvertor = lambda list_: {k: v for d in list_ for k, v in d.items()}

        mainAllDefinitions_dict = toDictConvertor(self.allDefinitions)

        architectureDicts_withMatchedAllDefinitions = []

        for archDict in architectureDicts:
            for filePath, fileDict in archDict.items():
                allDefinitions = toDictConvertor(fileDict['allDefinitions'])

                if stringValuedDictsEqual(mainAllDefinitions_dict, allDefinitions):
                    architectureDicts_withMatchedAllDefinitions.append(archDict)

        return architectureDicts_withMatchedAllDefinitions

    def _findSeedMatch_inArchitectureDicts(self, architectureDicts_withMatchedAllDefinitions, seed,
                                           returnCheckPointPath=False):
        # ccc1
        #  this is used in _determineShouldRun_preRunTests and _determineFitRunState
        #  - main structure is suited for _determineShouldRun_preRunTests
        #  - and the returnCheckPointPath part adapts this func for _determineFitRunState which
        #  needs to check existence of the checkpoint named `BrazingTorch.ckpt`; besides, returns
        #  expectedCheckPointPath as for path
        foundSeedMatch = False
        architectureFilePath = ''
        expectedCheckPointPath = ''
        for acw in architectureDicts_withMatchedAllDefinitions:
            architectureFilePath = list(acw.keys())[0]
            if returnCheckPointPath:
                # expectedCheckPointPath is used here so in anyCase there would
                # be a value for it(even when the seed is not matched)
                expectedCheckPointPath = architectureFilePath.replace('architecture.pkl',
                                                                      'BrazingTorch.ckpt')

            if seed == acw[architectureFilePath]['seed']:
                if returnCheckPointPath:
                    if os.path.exists(expectedCheckPointPath):
                        foundSeedMatch = True
                        break
                else:
                    foundSeedMatch = True
                    break
        if returnCheckPointPath:
            return foundSeedMatch, expectedCheckPointPath
        return foundSeedMatch, architectureFilePath

    def _findAvailableArchName(self, folderToSearch):
        """
        Find the first available 'arch{i}' folder within the specified parent folder.
        """
        i = 0
        while True:
            i += 1
            archName = f'arch{i}'
            folderPath = os.path.join(folderToSearch, archName)

            if os.path.exists(folderPath) and os.path.isdir(folderPath):
                continue
            else:
                return archName

    def _saveArchitectureDict(self, loggerPath):
        architectureDict = {'allDefinitions': self.allDefinitions,
                            'seed': self._initArgs['__plSeed__']}

        os.makedirs(loggerPath, exist_ok=True)
        with open(os.path.join(loggerPath, 'architecture.pkl'), 'wb') as f:
            pickle.dump(architectureDict, f)
