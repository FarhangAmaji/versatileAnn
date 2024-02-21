import os.path
import unittest

from torch import nn

from brazingTorchFolder.brazingTorch import BrazingTorch
from commonDatasets.commonDatasetsPrep.epfFrBe import getEpfFrBeDataloaders, \
    dataInfo as epfFrBeDataInfo
from projectUtils.misc import getProjectDirectory
from tests.baseTest import BaseTestClass
from tests.utils import simulateInput


class NNDummy1(BrazingTorch):
    def __init__(self, **kwargs):
        # this just in order to see does it run of not so 1 neuron is enough
        self.l1 = nn.Linear(4, 1)
        self.l2 = nn.Linear(1, 4)

    def forward(self, inputs, targets):
        output = {}
        output['output'] = self.l2(self.l1(targets['output']))
        return output


class FitTests(BaseTestClass):
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
        # bugPotentialCheck1
        #  gives 'train is empty. the trainSeqLen seems to be high'; check is it giving warn
        #  with correct reason or not

        self.model = NNDummy1(testPrints=True, seed=self.seed,
                              lossFuncs=[nn.MSELoss(), nn.L1Loss()])


class DetermineFitRunStateTests(FitTests):
    expectedLoggerPathSeed71 = os.path.join(getProjectDirectory(), 'tests', 'brazingTorchTests',
                                            'NNDummy1', 'arch1', 'mainRun_seed71')
    expectedLoggerPathSeed81 = os.path.join(getProjectDirectory(), 'tests', 'brazingTorchTests',
                                            'NNDummy1', 'arch1', 'mainRun_seed81')

    # mustHave1
    #  for some of these cases also add matchedSeedDict (should )
    # cccDevStruct
    #  note test func which have _matchedSeedDict have seed=71 as there is a model saved with
    #  this seed

    def test_noResume_seedSensitive_matchedSeedDict_dontReplace(self):
        self.setup(seed=71)
        with simulateInput("this is some random input"):
            architectureName, loggerPath, fitRunState = self.model._determineFitRunState(
                seed=self.seed, resume=False, seedSensitive=True)

        self.assertEqual(fitRunState, "don't run")

    def test_noResume_seedSensitive_matchedSeedDict_replace(self):
        self.setup(seed=71)
        with simulateInput("yes"):
            architectureName, loggerPath, fitRunState = self.model._determineFitRunState(
                seed=self.seed, resume=False, seedSensitive=True)

        self.assertEqual(fitRunState, "beginning")
        self.assertEqual(loggerPath, self.expectedLoggerPathSeed71)

    def test_noResume_seedSensitive_noMatchedSeedDict(self):
        self.setup(seed=81)
        architectureName, loggerPath, fitRunState = self.model._determineFitRunState(
            seed=self.seed, resume=False, seedSensitive=True)
        self.assertEqual(fitRunState, "beginning")

    def test_noResume_noSeedSensitive_matchedSeedDict_dontReplace(self):
        self.setup(seed=71)
        with simulateInput("this is some random input"):
            architectureName, loggerPath, fitRunState = self.model._determineFitRunState(
                seed=self.seed, resume=False, seedSensitive=False)

        self.assertEqual(fitRunState, "don't run")

    def test_noResume_noSeedSensitive_matchedSeedDict_replace(self):
        self.setup(seed=71)
        with simulateInput("yes"):
            architectureName, loggerPath, fitRunState = self.model._determineFitRunState(
                seed=self.seed, resume=False, seedSensitive=False)

        self.assertEqual(fitRunState, "beginning")
        self.assertEqual(loggerPath, self.expectedLoggerPathSeed71)

    def test_noResume_noSeedSensitive_noMatchedSeedDict(self):
        self.setup(seed=81)
        architectureName, loggerPath, fitRunState = self.model._determineFitRunState(
            seed=self.seed, resume=False, seedSensitive=False)
        self.assertEqual(fitRunState, "beginning")
        self.assertEqual(loggerPath, self.expectedLoggerPathSeed71)


# ---- run test
if __name__ == '__main__':
    unittest.main()
