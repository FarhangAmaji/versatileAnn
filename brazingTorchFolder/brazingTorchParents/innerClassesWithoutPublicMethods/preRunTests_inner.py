import os

import pytorch_lightning as pl
from pytorch_lightning.loggers import Logger

from projectUtils.misc import nFoldersBack, _allowOnlyCreationOf_ChildrenInstances
from projectUtils.warnings import Warn


class _BrazingTorch_preRunTests_inner:
    def __init__(self):

        # not allowing this class to have direct instance
        _allowOnlyCreationOf_ChildrenInstances(self, _BrazingTorch_preRunTests_inner)

    # ---- runs utils
    @staticmethod
    def _collectBestValScores_ofMetrics(callbacks_, lossRatioDecrease,
                                        mainValLossName, changingParam):
        firstEpochLoss = callbacks_[0].epochData['val'][0][mainValLossName]

        lastEpochNum = list(callbacks_[0].epochData['val'].keys())
        lastEpochNum.sort()
        lastEpochNum = lastEpochNum[-1]
        lastEpochLoss = callbacks_[0].epochData['val'][lastEpochNum][mainValLossName]
        lossRatioDecrease.update({changingParam: {'ratio': lastEpochLoss / firstEpochLoss,
                                                  '1st': firstEpochLoss, 'last': lastEpochLoss,
                                                  'score': lastEpochLoss / firstEpochLoss * lastEpochLoss}})

    def _mergeKwargsWith_runKwargs(self, mainKwargsOfPreRunTests, runKwargs):
        result = mainKwargsOfPreRunTests.copy()
        self._plKwargUpdater(result, runKwargs)
        return result

    # ---- _determineShouldRun_preRunTests
    def _determineShouldRun_preRunTests(self, force, seedSensitive):
        # addTest1

        # ccc1(same as _determineFitRunState)
        #  there was a idea about ""architectureName should be figured out in postInit"" but it may
        #  cause problems with loggerPath in preRunTests and .fit method

        # by default these values are assumed and will be
        # changed depending on the case
        shouldRun_preRunTests = True
        architectureName = 'arch1'

        dummyLogger = pl.loggers.TensorBoardLogger(self.modelName)
        loggerPath = os.path.abspath(dummyLogger.log_dir)
        # loggerPath is fullPath including 'modelName/someName/version_0'

        if os.path.exists(nFoldersBack(loggerPath, n=2)):
            # there is a model run before with the name of this model

            architectureDicts = self._collectArchDicts(loggerPath)
            # update loggerPath with now found archDicts
            if architectureDicts:
                loggerPath = self._updateLoggerPath_withExistingArchName(architectureDicts,
                                                                         'preRunTests')
            architectureDicts_withMatchedAllDefinitions = self._getArchitectureDicts_withMatchedAllDefinitions(
                architectureDicts)
            # matchedAllDefinitions means the exact same model structure as all
            # layers and their definitions are exactly the same
            # note architectureDicts matches _saveArchitectureDict

            if architectureDicts_withMatchedAllDefinitions:
                if force:
                    # the force is True so the user wants replace model's previous results therefore
                    # we have to find architectureName, so to know where are the past results
                    acw = architectureDicts_withMatchedAllDefinitions[0]
                    # ccc1
                    #  note acw looks like {filePath: architectureDict} and
                    #  architectureDict is like {'allDefinitions': {'class1':class1Definition,
                    #                                               'class2':class2Definition},
                    #                           '__plSeed__': someNumber}
                    filePath = list(acw.keys())[0]
                    architectureName = os.path.basename(nFoldersBack(filePath, n=1))
                else:
                    architectureName, shouldRun_preRunTests = self._shouldRun_preRunTests_seedSensitivePart(
                        architectureDicts_withMatchedAllDefinitions, architectureName, loggerPath,
                        seedSensitive, shouldRun_preRunTests)

            else:
                # there are models with the name of this model but with different structures
                architectureName = self._findAvailableArchName(nFoldersBack(loggerPath, n=2))

        else:
            # no model with this name in directory has never run
            pass  # so default shouldRun_preRunTests and architectureName are applied

        dummyLogger = pl.loggers.TensorBoardLogger(self.modelName,
                                                   name=architectureName,
                                                   version='preRunTests')
        loggerPath = os.path.abspath(dummyLogger.log_dir)
        # bugPotn2
        #  note each architecture regardless of seed has

        return architectureName, loggerPath, shouldRun_preRunTests

    def _shouldRun_preRunTests_seedSensitivePart(self, architectureDicts_withMatchedAllDefinitions,
                                                 architectureName, loggerPath, seedSensitive,
                                                 shouldRun_preRunTests):
        # note being here means an exact model with the same structure has run before
        if seedSensitive:
            # seedSensitive True means the user wants:
            # seedCase1:
            #       even if there is a model with the same structure but its seed
            #       differs, so run the model with the new seed (the seed
            #       passed to this run)
            # seedCase2:
            #       but if the seed passed to this run has run before so
            #       no need to run, unless 'preRunTests' has not run before

            thisModelSeed = self._initArgs['__plSeed__']
            foundSeedMatch, filePath = self.findSeedMatch_inArchitectureDicts(
                architectureDicts_withMatchedAllDefinitions, thisModelSeed)

            if foundSeedMatch:
                # seedCase2
                shouldRun_preRunTests = False  # just for clarity but may change
                if not os.path.join(filePath, 'preRunTests').exists():
                    # this exact model even with this seed has run before but
                    # its 'preRunTests' has not
                    shouldRun_preRunTests = True
                    architectureName = os.path.basename(nFoldersBack(filePath, n=1))

                if not shouldRun_preRunTests:
                    Warn.info(
                        'skipping preRunTests: this model with same structure and same seed has run before')

            if not foundSeedMatch:  # seedCase1
                # we have to find architectureName which doesn't exist,
                # in order not to overwrite the previous results
                architectureName = self._findAvailableArchName(nFoldersBack(loggerPath, n=1))
        else:
            # there are models with the name of this model
            # also same structure, and the seed is not important factor
            # so there is no need to run, unless 'preRunTests' has not run before
            shouldRun_preRunTests = False  # just for clarity but may change
            acw = architectureDicts_withMatchedAllDefinitions[0]
            filePath = list(acw.keys())[0]
            architecture_folderPath = nFoldersBack(filePath, n=1)
            preRunTests_folderPath = os.path.join(architecture_folderPath, 'preRunTests')
            if not os.path.exists(preRunTests_folderPath):
                # this exact model has run before but its 'preRunTests' has not
                architectureName = os.path.basename(architecture_folderPath)
                shouldRun_preRunTests = True

            if not shouldRun_preRunTests:
                Warn.info('skipping preRunTests: this model with same structure has run before')
        return architectureName, shouldRun_preRunTests

    # ----
    def _informTensorboardPath(self, fastDevRunKwargs, findBestBatchSizesKwargs,
                               findBestLearningRateKwargs, kwargs, overfitBatchesKwargs,
                               profilerKwargs):
        loggingPaths = self._getPreRunLoggingPaths(fastDevRunKwargs, overfitBatchesKwargs,
                                                   profilerKwargs, findBestLearningRateKwargs,
                                                   findBestBatchSizesKwargs, kwargs)
        for loggingPath in loggingPaths:
            Warn.info(
                "to see tensorboard in terminal execute: 'python -m tensorboard.main --logdir " +
                f'"{loggingPath}"' + "'")

    def _getPreRunLoggingPaths(self, fastDevRunKwargs, overfitBatchesKwargs, profilerKwargs,
                               findBestLearningRateKwargs, findBestBatchSizesKwargs, kwargs):
        loggingPaths = []
        for kwarg in [kwargs, fastDevRunKwargs, overfitBatchesKwargs, profilerKwargs,
                      findBestLearningRateKwargs, findBestBatchSizesKwargs]:
            if 'logger' in kwarg:
                logger = kwarg['logger']

                if isinstance(logger, Logger):
                    self._addLogDirTOLoggingPaths(loggingPaths, logger)
                else:  # logger may be a logger instance and be a list of Loggers
                    for logger in kwarg['logger']:
                        self._addLogDirTOLoggingPaths(loggingPaths, logger)

        if not loggingPaths:
            dummyLogger = pl.loggers.TensorBoardLogger(self.modelName, name='a')
            loggingPaths.append(nFoldersBack(os.path.abspath(dummyLogger.log_dir), n=2))
        return loggingPaths

    def _addLogDirTOLoggingPaths(self, loggingPaths, logger):
        logDir = os.path.abspath(logger.log_dir)
        if logDir not in loggingPaths:
            loggingPaths.append(logDir)
