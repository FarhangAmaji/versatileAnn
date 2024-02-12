import unittest

import pytorch_lightning as pl
from torch import nn

from commonDatasets.commonDatasetsPrep.epfFrBe import getEpfFrBeDataloaders, \
    dataInfo as epfFrBeDataInfo
from tests.baseTest import BaseTestClass
from versatileAnn.newModule.brazingTorch import BrazingTorch


class preRunTests_Tests(BaseTestClass):
    def setup(self):
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

    def testTraining_step(self):
        # ccc it's just to see does it run or not
        self.setup()
        pl.seed_everything(71)

        class NNDummy(BrazingTorch):
            def __init__(self, **kwargs):
                # this just in order to see does it run of not so 1 neuron is enough
                self.l1 = nn.Linear(4, 1)
                self.l2 = nn.Linear(1, 4)

            def forward(self, inputs, targets):
                output = {}
                output['output'] = self.l2(self.l1(targets['output']))
                return output

        model = NNDummy(testPrints=True)
        model.preRunTests(trainDataloader=self.trainDataloader, valDataloader=self.valDataloader,
                          lossFuncs=[nn.MSELoss(), nn.L1Loss()], max_epochs=3)


# ---- run test
if __name__ == '__main__':
    unittest.main()
