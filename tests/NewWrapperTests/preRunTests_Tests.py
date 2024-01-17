import unittest

import pytorch_lightning as pl
from torch import nn

from dataBenchmarks.benchmarkPrep.stallion import getStallion_TftDataloaders, \
    dataInfo as stallionDataInfo
from tests.baseTest import BaseTestClass
from versatileAnn.newModule.newWrapper import NewWrapper


class preRunTests_Tests(BaseTestClass):
    def setup(self):
        batchSize = 4
        maxEncoderLength = 4
        maxPredictionLength = 3
        minEncoderLength = 2
        minPredictionLength = 1
        shuffle = False
        dataInfo = stallionDataInfo
        devTestMode = False
        self.trainDataloader, self.valDataloader, self.testDataloader, self.normalizer = getStallion_TftDataloaders(
            maxEncoderLength=maxEncoderLength, maxPredictionLength=maxPredictionLength,
            minEncoderLength=minEncoderLength, minPredictionLength=minPredictionLength,
            batchSize=batchSize, shuffle=shuffle, dataInfo=dataInfo, devTestMode=devTestMode)

    def testTraining_step(self):
        self.setup()
        pl.seed_everything(71)

        class NNDummy(NewWrapper):
            def __init__(self, **kwargs):
                self.l1 = nn.Linear(7, 20)
                self.l2 = nn.Linear(20, 1)

            def forward(self, inputs, targets):
                output = {}
                output['volume'] = self.l2(self.l1(targets['volume']))
                return output

        model = NNDummy(testPrints=True)
        model.preRunTests(trainDataloader=self.trainDataloader, valDataloader=self.valDataloader,
                          losses=[nn.MSELoss(), nn.L1Loss()], maxEpochs=1,
                          savePath='data\\bestModels\\newModuleTests\\test1', max_epochs=5)


# ---- run test
if __name__ == '__main__':
    unittest.main()
