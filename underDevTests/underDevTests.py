import unittest

import torch.nn as nn

from dataBenchmarks.benchmarkPrep.stallion import getStallion_TftDataloaders, \
    dataInfo as stallionDataInfo
from tests.baseTest import BaseTestClass
from versatileAnn.newModule import NewWrapper


# ----
class underDevTests(BaseTestClass):
    def test1(self):
        pass


class newModuleTests(BaseTestClass):
    def setup(self):
        batchSize = 4
        maxEncoderLength = 4
        maxPredictionLength = 3
        minEncoderLength = 2
        minPredictionLength = 1
        shuffle = False
        dataInfo = stallionDataInfo
        devTestMode = True  # kkk change to False
        self.trainDataloader, self.valDataloader, self.testDataloader, self.normalizer = getStallion_TftDataloaders(
            maxEncoderLength=maxEncoderLength, maxPredictionLength=maxPredictionLength,
            minEncoderLength=minEncoderLength, minPredictionLength=minPredictionLength,
            batchSize=batchSize, shuffle=shuffle, dataInfo=dataInfo, devTestMode=devTestMode)

    def testTraining_step(self):
        self.setup()
        model = NewWrapper(modelName='ModelKog2s')

        model.trainModel(trainDataloader=self.trainDataloader, valDataloader=self.valDataloader,
                         losses=[nn.MSELoss(), nn.L1Loss()], maxEpochs=1,
                         savePath='data\\bestModels\\newModuleTests\\test1', )

        x = 0


# ---- run test
if __name__ == '__main__':
    unittest.main()
