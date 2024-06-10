import os
import unittest
from unittest.mock import Mock

from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import Logger
from torch import nn

from brazingTorchFolder.brazingTorch import BrazingTorch
from commonDatasets.commonDatasetsPrep.epfFrBe import getEpfFrBeDataloaders, \
    dataInfo as epfFrBeDataInfo
from projectUtils.misc import getProjectDirectory
from tests.baseTest import BaseTestClass
from tests.utils import simulateInput


# ccc1
#  note there are some saved files essential for this test(DetermineFitRunStateTests_mockSavedModels)
#  python import path is different when this file is run by runAllTests.py and when it's run
#  from this file itself; so assertEqual_pathCompatibile func tries to make the tests pass for both
#  cases but still some tests do pass when run directly from this file

# ---- dummy classes to be used in tests
class NNDummy1(BrazingTorch):
    def __init__(self, **kwargs):
        # this just in order to see does it run of not so 1 neuron is enough
        self.l1 = nn.Linear(4, 1)
        self.l2 = nn.Linear(1, 4)

    def forward(self, inputs, targets):
        output = {}
        output['output'] = self.l2(self.l1(targets['output']))
        return output


class NNDummy2(NNDummy1):
    def __init__(self, **kwargs):
        self.l3 = nn.Linear(5, 1)


# ---- actual tests
class FitTestsSetup(BaseTestClass):
    def setup(self, seed):
        self.seed = seed

        backcastLen = 7
        forecastLen = 4
        batchSize = 4
        dataInfo = epfFrBeDataInfo
        shuffle = False
        devTestMode = True
        self.trainDataloader, self.valDataloader, self.testDataloader, self.normalizer = getEpfFrBeDataloaders(
            backcastLen=backcastLen, forecastLen=forecastLen,
            batchSize=batchSize, shuffle=shuffle, dataInfo=dataInfo, devTestMode=devTestMode)
        # bugPotn1
        #  gives 'train is empty. the trainSeqLen seems to be high'; check is it giving warn
        #  with correct reason or not

        self.model = NNDummy1(modelName='DetermineFitRunStateTests_mockSavedModels',
                              testPrints=True, seed=self.seed,
                              lossFuncs=[nn.MSELoss(), nn.L1Loss()])


class BaseFitTests(FitTestsSetup):
    def test_basicCombination(self):
        self.setup(seed=71)
        kwargs = {'a': 1, 'b': 2}
        listOfKwargs = [{'c': 3}, {'d': 4}]
        result = self.model._getBaseFit_appliedKwargs(kwargs, listOfKwargs)
        expected = {'a': 1, 'b': 2, 'c': 3, 'd': 4}
        self.assertEqual(result, expected)

    def test_emptyInput(self):
        self.setup(seed=71)
        kwargs = {}
        listOfKwargs = []
        result = self.model._getBaseFit_appliedKwargs(kwargs, listOfKwargs)
        expected = {}
        self.assertEqual(result, expected)


class BaseFit_putTogetherPlLoggersTests(FitTestsSetup):
    def test_bothNone(self):
        self.setup(seed=71)
        result = self.model._putTogether_plLoggers(None, None)
        self.assertIsNone(result)

    def test_firstNone(self):
        self.setup(seed=71)
        logger2 = Mock(spec=Logger)
        result = self.model._putTogether_plLoggers(None, logger2)
        self.assertEqual(result, logger2)

    def test_secondNone(self):
        self.setup(seed=71)
        logger1 = Mock(spec=Logger)
        result = self.model._putTogether_plLoggers(logger1, None)
        self.assertEqual(result, logger1)

    def test_bothBool(self):
        self.setup(seed=71)
        result = self.model._putTogether_plLoggers(True, False)
        self.assertFalse(result)

    def test_firstBool(self):
        self.setup(seed=71)
        logger2 = Mock(spec=Logger)
        result = self.model._putTogether_plLoggers(True, logger2)
        self.assertEqual(result, logger2)

    def test_secondBool(self):
        self.setup(seed=71)
        logger1 = Mock(spec=Logger)
        result = self.model._putTogether_plLoggers(logger1, True)
        self.assertEqual(result, logger1)

    def test_bothSingleLogger(self):
        self.setup(seed=71)
        logger1 = Mock(spec=Logger)
        logger2 = Mock(spec=Logger)
        result = self.model._putTogether_plLoggers(logger1, logger2)
        self.assertEqual(result, [logger1, logger2])

    def test_firstListLogger(self):
        self.setup(seed=71)
        logger1 = [Mock(spec=Logger), Mock(spec=Logger)]
        logger2 = Mock(spec=Logger)
        result = self.model._putTogether_plLoggers(logger1, logger2)
        self.assertEqual(result, logger1 + [logger2])

    def test_secondListLogger(self):
        self.setup(seed=71)
        logger1 = Mock(spec=Logger)
        logger2 = [Mock(spec=Logger), Mock(spec=Logger)]
        result = self.model._putTogether_plLoggers(logger1, logger2)
        self.assertEqual(result, [logger1] + logger2)

    def test_bothListLogger(self):
        self.setup(seed=71)
        logger1 = [Mock(spec=Logger), Mock(spec=Logger)]
        logger2 = [Mock(spec=Logger), Mock(spec=Logger)]
        result = self.model._putTogether_plLoggers(logger1, logger2)
        self.assertEqual(result, logger1 + logger2)


class BaseFit_putTogetherPlCallbacksTests(FitTestsSetup):
    def test_bothNone(self):
        self.setup(seed=71)
        result = self.model._putTogether_plCallbacks(None, None)
        self.assertIsNone(result)

    def test_firstNone(self):
        self.setup(seed=71)
        callback2 = Mock(spec=Callback)
        result = self.model._putTogether_plCallbacks(None, callback2)
        self.assertEqual(result, callback2)

    def test_secondNone(self):
        self.setup(seed=71)
        callback1 = Mock(spec=Callback)
        result = self.model._putTogether_plCallbacks(callback1, None)
        self.assertEqual(result, callback1)

    def test_bothSingleCallback(self):
        self.setup(seed=71)
        callback1 = Mock(spec=Callback)
        callback2 = Mock(spec=Callback)
        result = self.model._putTogether_plCallbacks(callback1, callback2)
        self.assertEqual(result, [callback1, callback2])

    def test_firstListCallback(self):
        self.setup(seed=71)
        callback1 = [Mock(spec=Callback), Mock(spec=Callback)]
        callback2 = Mock(spec=Callback)
        result = self.model._putTogether_plCallbacks(callback1, callback2)
        self.assertEqual(result, callback1 + [callback2])

    def test_secondListCallback(self):
        self.setup(seed=71)
        callback1 = Mock(spec=Callback)
        callback2 = [Mock(spec=Callback), Mock(spec=Callback)]
        result = self.model._putTogether_plCallbacks(callback1, callback2)
        self.assertEqual(result, [callback1] + callback2)

    def test_bothListCallback(self):
        self.setup(seed=71)
        callback1 = [Mock(spec=Callback), Mock(spec=Callback)]
        callback2 = [Mock(spec=Callback), Mock(spec=Callback)]
        result = self.model._putTogether_plCallbacks(callback1, callback2)
        self.assertEqual(result, callback1 + callback2)

    def test_bothListCallbacks(self):
        self.setup(seed=71)
        callback1 = [Mock(spec=Callback), Mock(spec=Callback)]
        callback2 = [Mock(spec=Callback), Mock(spec=Callback)]
        result = self.model._putTogether_plCallbacks(callback1, callback2)
        self.assertEqual(result, callback1 + callback2)


class FitTests(FitTestsSetup):
    pass

    # def testFit(self):  # kkk do it later
    #     # ccc it's just to see does it run or not
    #     self.setup(71)
    #
    #     self.model.fit(trainDataloader=self.trainDataloader, valDataloader=self.valDataloader,
    #                    lossFuncs=[nn.MSELoss(), nn.L1Loss()], max_epochs=3)


class DetermineFitRunStateTests(FitTests):
    expectedLoggerPathSeed71 = os.path.join(getProjectDirectory(), 'tests',
                                            'DetermineFitRunStateTests_mockSavedModels', 'arch1',
                                            'mainRun_seed71')
    expectedLoggerPathSeed81 = os.path.join(getProjectDirectory(), 'tests',
                                            'DetermineFitRunStateTests_mockSavedModels', 'arch1',
                                            'mainRun_seed81')

    # mustHave1
    #  for some of these cases also add matchedSeedDict (should )
    # ccc1
    #  note test func which have _matchedSeedDict have seed=71 as there is a model saved with
    #  this seed

    def assertEqual_pathCompatibile(self, loggerPath, path2):
        pathArgs = path2.split(os.sep)
        if 'tests' in pathArgs:
            if loggerPath != path2:
                testsIndex = pathArgs.index('tests')
                pathArgsWithBrazingTorchTestsFolder = list(pathArgs[:])
                pathArgsWithBrazingTorchTestsFolder.insert(testsIndex + 1, 'brazingTorchTests')
                if ':' in pathArgsWithBrazingTorchTestsFolder[0] and not '\\' in \
                                                                         pathArgsWithBrazingTorchTestsFolder[
                                                                             0]:
                    pathArgsWithBrazingTorchTestsFolder[0] += '\\'
                self.assertEqual(loggerPath, os.path.join(*pathArgsWithBrazingTorchTestsFolder))
        else:
            self.assertEqual(loggerPath, os.path.join(*pathArgs))

    def test_resume_seedSensitive_matchedSeedDict(self):
        self.setup(seed=71)
        architectureName, loggerPath, fitRunState, checkpointPath, isModelChanged = self.model._determineFitRunState(
            seed=self.seed, resume=True, seedSensitive=True)
        self.assertEqual(fitRunState, 'resume')
        self.assertFalse(isModelChanged)
        self.assertEqual(checkpointPath,
                         os.path.join(self.expectedLoggerPathSeed71, 'BrazingTorch.ckpt'))

    def test_resume_seedSensitive_noMatchedSeedDict(self):
        self.setup(seed=81)
        architectureName, loggerPath, fitRunState, checkpointPath, isModelChanged = self.model._determineFitRunState(
            seed=self.seed, resume=True, seedSensitive=True)
        self.assertEqual(fitRunState, "beginning")
        self.assertEqual_pathCompatibile(loggerPath, self.expectedLoggerPathSeed81)  # note its 81

    def test_noResume_seedSensitive_matchedSeedDict_dontReplace(self):
        self.setup(seed=71)
        with simulateInput("this is some random input"):
            architectureName, loggerPath, fitRunState, checkpointPath, isModelChanged = self.model._determineFitRunState(
                seed=self.seed, resume=False, seedSensitive=True)

        self.assertEqual(fitRunState, "don't run")

    def test_noResume_seedSensitive_matchedSeedDict_replace(self):
        self.setup(seed=71)
        with simulateInput("yes"):
            architectureName, loggerPath, fitRunState, checkpointPath, isModelChanged = self.model._determineFitRunState(
                seed=self.seed, resume=False, seedSensitive=True)

        self.assertEqual(fitRunState, "beginning")
        self.assertEqual_pathCompatibile(loggerPath, self.expectedLoggerPathSeed71)

    def test_noResume_seedSensitive_noMatchedSeedDict(self):
        # note this one also exactly lands in the condition which
        # test_resume_seedSensitive_noMatchedSeedDict is landed and has exact same results
        self.setup(seed=81)
        architectureName, loggerPath, fitRunState, checkpointPath, isModelChanged = self.model._determineFitRunState(
            seed=self.seed, resume=False, seedSensitive=True)
        self.assertEqual(fitRunState, "beginning")
        self.assertEqual_pathCompatibile(loggerPath, self.expectedLoggerPathSeed81)  # note its 81

    def test_resume_noSeedSensitive_matchedSeedDict(self):
        self.setup(seed=71)
        architectureName, loggerPath, fitRunState, checkpointPath, isModelChanged = self.model._determineFitRunState(
            seed=self.seed, resume=True, seedSensitive=False)
        self.assertEqual(fitRunState, "resume")
        self.assertFalse(isModelChanged)
        self.assertEqual(loggerPath, self.expectedLoggerPathSeed71)
        self.assertEqual(checkpointPath,
                         os.path.join(self.expectedLoggerPathSeed71, 'BrazingTorch.ckpt'))

    def test_resume_noSeedSensitive_noMatchedSeedDict(self):
        self.setup(seed=81)
        architectureName, loggerPath, fitRunState, checkpointPath, isModelChanged = self.model._determineFitRunState(
            seed=self.seed, resume=True, seedSensitive=False)
        self.assertEqual(fitRunState, "resume")
        self.assertTrue(isModelChanged)
        self.assertEqual(loggerPath, self.expectedLoggerPathSeed71)
        # note it's not 81 and it's 71 as the model has been changed

        self.assertEqual(checkpointPath,
                         os.path.join(self.expectedLoggerPathSeed71, 'BrazingTorch.ckpt'))

    def test_noResume_noSeedSensitive_matchedSeedDict_dontReplace(self):
        self.setup(seed=71)
        with simulateInput("this is some random input"):
            architectureName, loggerPath, fitRunState, checkpointPath, isModelChanged = self.model._determineFitRunState(
                seed=self.seed, resume=False, seedSensitive=False)

        self.assertEqual(fitRunState, "don't run")

    def test_noResume_noSeedSensitive_matchedSeedDict_replace(self):
        self.setup(seed=71)
        with simulateInput("yes"):
            architectureName, loggerPath, fitRunState, checkpointPath, isModelChanged = self.model._determineFitRunState(
                seed=self.seed, resume=False, seedSensitive=False)

        self.assertEqual(fitRunState, "beginning")
        self.assertEqual_pathCompatibile(loggerPath, self.expectedLoggerPathSeed71)

    def test_noResume_noSeedSensitive_noMatchedSeedDict(self):
        self.setup(seed=81)
        architectureName, loggerPath, fitRunState, checkpointPath, isModelChanged = self.model._determineFitRunState(
            seed=self.seed, resume=False, seedSensitive=False)
        self.assertEqual(fitRunState, "beginning")
        self.assertEqual_pathCompatibile(loggerPath, self.expectedLoggerPathSeed81)  # note its 81

    def test_modelWithSameName_anotherArchitecture(self):
        model = NNDummy2(modelName='DetermineFitRunStateTests_mockSavedModels',
                         testPrints=True, seed=71,
                         lossFuncs=[nn.MSELoss(), nn.L1Loss()])
        architectureName, loggerPath, fitRunState, checkpointPath, isModelChanged = model._determineFitRunState(
            seed=71, resume=False, seedSensitive=False)
        self.assertEqual(fitRunState, "beginning")
        self.assertEqual(architectureName, 'arch2')

    def test_modelWithAnotherName(self):
        model = NNDummy1(modelName='NNDummy1',
                         testPrints=True, seed=71,
                         lossFuncs=[nn.MSELoss(), nn.L1Loss()])
        architectureName, loggerPath, fitRunState, checkpointPath, isModelChanged = model._determineFitRunState(
            seed=71)
        self.assertEqual(fitRunState, "beginning")

        self.assertEqual_pathCompatibile(loggerPath, os.path.join(getProjectDirectory(), 'tests',
                                                                  'NNDummy1', 'arch1',
                                                                  'mainRun_seed71'))
        self.assertEqual(architectureName, 'arch1')


# ---- run test
if __name__ == '__main__':
    unittest.main()
