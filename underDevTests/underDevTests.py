import unittest

from dataPrep.commonDatasets.electricity import getElectricityProcessed
from tests.baseTest import BaseTestClass


# ----
class underDevTests(BaseTestClass):
    def test1(self):
        pass


class electricityTests(BaseTestClass):
    def setup(self):
        self.backcastLen = 3
        self.forecastLen = 2

    def processedSetup(self):
        self.setup()
        getElectricityProcessed(backcastLen=self.backcastLen, forecastLen=self.forecastLen)
        x = 0
        # self.mainDf, staticDf = getEpfFrBeProcessed_loadData(devTestMode=self.devTestMode,
        #                                                      backcastLen=self.backcastLen,
        #                                                      forecastLen=self.forecastLen)
        # self.trainDf, self.valDf, self.testDf, self.normalizer = getEpfFrBeProcessed(
        #     backcastLen=self.backcastLen, forecastLen=self.forecastLen,
        #     trainRatio=.6, valRatio=.2, rightPadTrain=True, aggColName=self.aggColName,
        #     devTestMode=True)

    # def testGetElectricityProcessed(self):
    #     self.processedSetup()


# ---- run test
if __name__ == '__main__':
    unittest.main()
