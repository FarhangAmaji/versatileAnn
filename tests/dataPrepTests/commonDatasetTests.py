# ---- imports

import os
import unittest

import pandas as pd
import torch

from commonDatasets.commonDatasetsPrep.electricity import getElectricity_processed, \
    getElectricity_data, \
    getElectricityDataloaders, dataInfo as electricityDataInfo
from commonDatasets.commonDatasetsPrep.epfFrBe import getEpfFrBe_data, getEpfFrBe_processed, \
    getEpfFrBeDataloaders, dataInfo as epfFrBeDataInfo
from commonDatasets.commonDatasetsPrep.stallion import getStallion_processed, \
    getStallion_TftDataloaders, \
    getStallion_data, dataInfo as stallionDataInfo
from commonDatasets.getData import getDatasetFiles, _getFilePathInDataStoreLocation
from dataPrep.utils import combineNSeries
from tests.baseTest import BaseTestClass
from utils.dataTypeUtils.tensor import toDevice, getTorchDevice


# ---- getDatasetFiles
class getDatasetFilesTests(BaseTestClass):
    def test(self):
        filepath = _getFilePathInDataStoreLocation("downloadDummy.csv")
        if os.path.exists(filepath):
            os.remove(filepath)
        getDatasetFiles("downloadDummy.csv")
        self.assertTrue(os.path.exists(filepath))
        os.remove(filepath)


# ---- epfFrBeTests
class epfFrBeTests(BaseTestClass):
    # bugPotentialCheck1
    #  recheck values are the ones supposed to be
    # cccDevStruct hard written values is chosen because other ways had to just follow the same procedure done int the getEpfFrBe_processed
    def setup(self):
        self.devTestMode = True
        self.backcastLen = 7
        self.forecastLen = 4
        self.aggColName = 'price'

    def processedSetup(self):
        self.setup()
        self.mainDf, staticDf = getEpfFrBe_data(devTestMode=self.devTestMode,
                                                backcastLen=self.backcastLen,
                                                forecastLen=self.forecastLen)
        self.trainDf, self.valDf, self.testDf, self.normalizer = \
            getEpfFrBe_processed(backcastLen=self.backcastLen, forecastLen=self.forecastLen,
                                 trainRatio=.6, valRatio=.2, rightPadTrain=True,
                                 aggColName=self.aggColName, dataInfo=epfFrBeDataInfo,
                                 devTestMode=True)

    def testGetEpfFrBeProcessed(self):
        # addTest2 could have had test for innerSteps also
        self.processedSetup()
        trainDf = self.trainDf.drop(columns='dateTime')
        valDf = self.valDf.drop(columns='dateTime')
        testDf = self.testDf.drop(columns='dateTime')

        trainDfCheck = pd.DataFrame({'genForecast': [-0.03080956797102298, 0.3186052798375584,
                                                     0.844584345427674, 1.0116957943796043,
                                                     1.4090496841097493, 1.0809035661677773,
                                                     0.1633097919226333, -0.24315926775034438,
                                                     -0.4396418198514017, -0.06760784662912478,
                                                     1.3229619679829974, 1.712888682204168,
                                                     1.00865740439866, 0.09106363015351601,
                                                     -0.23944567999585706, 0.3705955084003811,
                                                     -0.1763146881695723, -0.8545508516936688,
                                                     -1.4872111655036027] +
                                                    3 * [0.0] + [-0.03080956797102298,
                                                                 0.3186052798375584,
                                                                 0.844584345427674,
                                                                 1.0116957943796043,
                                                                 1.4090496841097493,
                                                                 1.0809035661677773,
                                                                 0.1633097919226333,
                                                                 -0.24315926775034438,
                                                                 -0.4396418198514017,
                                                                 -0.06760784662912478,
                                                                 1.3229619679829974,
                                                                 1.712888682204168,
                                                                 1.00865740439866,
                                                                 0.09106363015351601,
                                                                 -0.23944567999585706,
                                                                 0.3705955084003811,
                                                                 -0.1763146881695723,
                                                                 -0.8545508516936688,
                                                                 -1.4872111655036027] +
                                                    3 * [0.0],
                                     'systemLoad': [-0.6221650010346238, -0.2683097548411284,
                                                    0.4512716844203051, 0.6859123151938649,
                                                    1.0382521213608356, 1.4835389158213155,
                                                    0.3421600025105335, -0.34812292957142427,
                                                    -0.5208830925952294, -0.25896454134422664,
                                                    1.2377850581865752, 1.775008547589548,
                                                    1.0372418280098192, 0.27194461461489344,
                                                    -0.3569629963928178, 0.8018434772229972,
                                                    0.5939756202513721, -0.4365235977853596,
                                                    -0.5107801590850652] +
                                                   3 * [0.0] + [-0.6221650010346238,
                                                                -0.2683097548411284,
                                                                0.4512716844203051,
                                                                0.6859123151938649,
                                                                1.0382521213608356,
                                                                1.4835389158213155,
                                                                0.3421600025105335,
                                                                -0.34812292957142427,
                                                                -0.5208830925952294,
                                                                -0.25896454134422664,
                                                                1.2377850581865752,
                                                                1.775008547589548,
                                                                1.0372418280098192,
                                                                0.27194461461489344,
                                                                -0.3569629963928178,
                                                                0.8018434772229972,
                                                                0.5939756202513721,
                                                                -0.4365235977853596,
                                                                -0.5107801590850652]
                                                   + 3 * [0.0],
                                     'weekDay': 16 * [-0.3872983346207408] + 3 * [
                                         2.5819888974716125] +
                                                3 * [0.0] + 16 * [-0.3872983346207408] +
                                                3 * [2.5819888974716125] + 3 * [0.0],
                                     'mask': 19 * [True] + 3 * [False] + 19 * [True] + 3 * [False],
                                     '__startPoint__': 12 * [True] + 10 * [False] + 12 * [
                                         True] + 10 * [False],
                                     'price': [-0.8934245994612829, -0.797110549573975,
                                               -0.7089587073042353, -0.7040613827336943,
                                               -0.6044824497993591, -0.7889483419564064,
                                               -0.9521924943077759, -1.3496920052833608,
                                               -1.1203339712296865, -0.753850849200862,
                                               -0.27962658662013373, -0.010273735240374053,
                                               -0.27064815824080846, -0.7448724208215367,
                                               -0.16862056302120254, -0.016803501334428695,
                                               -0.0755713961809219, -0.29839966414054125,
                                               -0.06496052627808296] +
                                              3 * [0.0] + [0.36273915288250524, 0.9602127504885175,
                                                           1.5617674519033138, 1.8915206396530801,
                                                           2.016402416201878, 1.6923627737844096,
                                                           0.6435190949268605, 0.6769841461588916,
                                                           0.9846993733412227, 1.5642161141885844,
                                                           2.617140896854918, 1.1528408502631335,
                                                           0.615767589027128, 0.5814863170333403,
                                                           0.578221433986313, 0.41089617782615895,
                                                           -0.0755713961809219,
                                                           -0.29839966414054125,
                                                           -0.06496052627808296]
                                              + 3 * [0.0],
                                     'priceType': 22 * ['priceFr'] + 22 * ['priceBe'],
                                     'market0': 22 * [1.0] + 22 * [0.0],
                                     'market1': 22 * [0.0] + 22 * [1.0]})

        self.equalDfs(trainDf, trainDfCheck)
        valDfCheck = pd.DataFrame({'genForecast': [-1.7542518849399193, -1.686732107585604,
                                                   -0.9024898936152327, -0.03080956797102298,
                                                   0.3186052798375584, 0.844584345427674,
                                                   1.0116957943796043, 1.4090496841097493,
                                                   1.0809035661677773, 0.1633097919226333,
                                                   -0.24315926775034438, -0.4396418198514017,
                                                   -0.06760784662912478, -1.7542518849399193,
                                                   -1.686732107585604, -0.9024898936152327,
                                                   -0.03080956797102298, 0.3186052798375584,
                                                   0.844584345427674, 1.0116957943796043,
                                                   1.4090496841097493, 1.0809035661677773,
                                                   0.1633097919226333, -0.24315926775034438,
                                                   -0.4396418198514017, -0.06760784662912478],
                                   'systemLoad': [-1.937061797382473, -1.5145066033198622,
                                                  -1.1386774767417602, -0.6221650010346238,
                                                  -0.2683097548411284, 0.4512716844203051,
                                                  0.6859123151938649, 1.0382521213608356,
                                                  1.4835389158213155, 0.3421600025105335,
                                                  -0.34812292957142427, -0.5208830925952294,
                                                  -0.25896454134422664, -1.937061797382473,
                                                  -1.5145066033198622, -1.1386774767417602,
                                                  -0.6221650010346238, -0.2683097548411284,
                                                  0.4512716844203051, 0.6859123151938649,
                                                  1.0382521213608356, 1.4835389158213155,
                                                  0.3421600025105335, -0.34812292957142427,
                                                  -0.5208830925952294, -0.25896454134422664],
                                   'weekDay': 26 * [-0.3872983346207408],
                                   'mask': 26 * [True],
                                   '__startPoint__': 3 * [True] + 10 * [False] + 3 * [True] + 10 * [
                                       False],
                                   'price': [-1.5178334822052715, -1.3154107332895728,
                                             -0.9823926624927793, -0.8934245994612829,
                                             -0.797110549573975, -0.7089587073042353,
                                             -0.7040613827336943, -0.6044824497993591,
                                             -0.7889483419564064, -0.9521924943077759,
                                             -1.3496920052833608, -1.1203339712296865,
                                             -0.753850849200862, -0.5122495037208351,
                                             -1.3154107332895728, 0.12358646968774896,
                                             0.36273915288250524, 0.9602127504885175,
                                             1.5617674519033138, 1.8915206396530801,
                                             2.016402416201878, 1.6923627737844096,
                                             0.6435190949268605, 0.6769841461588916,
                                             0.9846993733412227, 1.5642161141885844],
                                   'priceType': 13 * ['priceFr'] + 13 * ['priceBe'],
                                   'market0': 13 * [1.0] + 13 * [0.0],
                                   'market1': 13 * [0.0] + 13 * [1.0]})

        self.equalDfs(valDf, valDfCheck)
        testDfCheck = pd.DataFrame({'genForecast': [-1.4521008812793588, -1.7542518849399193,
                                                    -1.686732107585604, -0.9024898936152327,
                                                    -0.03080956797102298, 0.3186052798375584,
                                                    0.844584345427674, 1.0116957943796043,
                                                    1.4090496841097493, 1.0809035661677773,
                                                    0.1633097919226333, -1.4521008812793588,
                                                    -1.7542518849399193, -1.686732107585604,
                                                    -0.9024898936152327, -0.03080956797102298,
                                                    0.3186052798375584, 0.844584345427674,
                                                    1.0116957943796043, 1.4090496841097493,
                                                    1.0809035661677773, 0.1633097919226333],
                                    'systemLoad': [-1.8059762350880948, -1.937061797382473,
                                                   -1.5145066033198622, -1.1386774767417602,
                                                   -0.6221650010346238, -0.2683097548411284,
                                                   0.4512716844203051, 0.6859123151938649,
                                                   1.0382521213608356, 1.4835389158213155,
                                                   0.3421600025105335, -1.8059762350880948,
                                                   -1.937061797382473, -1.5145066033198622,
                                                   -1.1386774767417602, -0.6221650010346238,
                                                   -0.2683097548411284, 0.4512716844203051,
                                                   0.6859123151938649, 1.0382521213608356,
                                                   1.4835389158213155, 0.3421600025105335],
                                    'weekDay': 22 * [-0.3872983346207408], 'mask': 22 * [True],
                                    '__startPoint__': [True] + 10 * [False] + [True] + 10 * [False],
                                    'price': [-1.534157897440408, -1.5178334822052715,
                                              -1.3154107332895728, -0.9823926624927793,
                                              -0.8934245994612829, -0.797110549573975,
                                              -0.7089587073042353, -0.7040613827336943,
                                              -0.6044824497993591, -0.7889483419564064,
                                              -0.9521924943077759, -0.2151451464413429,
                                              -0.5122495037208351, -1.3154107332895728,
                                              0.12358646968774896, 0.36273915288250524,
                                              0.9602127504885175, 1.5617674519033138,
                                              1.8915206396530801, 2.016402416201878,
                                              1.6923627737844096, 0.6435190949268605],
                                    'priceType': 11 * ['priceFr'] + 11 * ['priceBe'],
                                    'market0': 11 * [1.0] + 11 * [0.0],
                                    'market1': 11 * [0.0] + 11 * [1.0]})

        self.equalDfs(testDf, testDfCheck)

    def testInvNormalizer(self):
        # goodToHave3
        #  add comments
        self.processedSetup()
        for set1 in [self.trainDf, self.valDf, self.testDf]:
            set_ = combineNSeries(set1, self.aggColName)
            # cccAlgo
            #  in real use case the zero pad rows are not inverse transformed so for purpose of
            #  checking if invTransfrom work we can delete them
            setIndsWith0dateCond = set_[set_['dateTime'] == 0].index
            set_ = set_.drop(setIndsWith0dateCond)
            self.normalizer.inverseTransform(set_)

            mainDfIndexes_whichAreInSetDf = self.mainDf[
                self.mainDf['dateTime'].isin(set_['dateTime'].values)].index
            checkSet = self.mainDf.loc[mainDfIndexes_whichAreInSetDf]
            # we have to reset it's index as even checkIndex=False in equalDfs can handle it
            checkSet = checkSet.reset_index(drop=True)

            # removing cols added during getEpfFrBe_processed to be able to check
            set_ = set_.drop(columns=['__startPoint__', 'market0', 'market1', 'mask'])
            self.equalDfs(set_, checkSet, checkIndex=False, floatApprox=True)

    def testDataloader_data(self):
        device = getTorchDevice()
        self.setup()
        trainDataloader, valDataloader, testDataloader, normalizer = getEpfFrBeDataloaders(
            backcastLen=self.backcastLen, forecastLen=self.forecastLen,
            batchSize=64, trainRatio=.6, valRatio=.2,
            rightPadTrain=True, aggColName=self.aggColName, dataInfo=epfFrBeDataInfo,
            devTestMode=self.devTestMode)

        # these 2 are just here in order if they make error, get detected
        next(iter(trainDataloader))
        next(iter(valDataloader))

        testDataloader_inputs, testDataloader_outputs = next(iter(testDataloader))

        expectedInputs = {}
        expectedInputs['target'] = [[-1.5342, -1.5178, -1.3154, -0.9824, -0.8934, -0.7971, -0.7090],
                                    [-0.2151, -0.5122, -1.3154, 0.1236, 0.3627, 0.9602, 1.5618]]
        expectedInputs['mask'] = [[True, True, True, True, True, True, True],
                                  [True, True, True, True, True, True, True]]
        expectedInputs['historyExogenous'] = [
            [-1.8060, -1.9371, -1.5145, -1.1387, -0.6222, -0.2683, 0.4513],
            [-1.8060, -1.9371, -1.5145, -1.1387, -0.6222, -0.2683, 0.4513]]
        expectedInputs['staticExogenous'] = [[[1., 0.]], [[0., 1.]]]
        expectedInputs['futureExogenous'] = [
            [[-1.4521, -0.3873], [-1.7543, -0.3873], [-1.6867, -0.3873], [-0.9025, -0.3873],
             [-0.0308, -0.3873], [0.3186, -0.3873], [0.8446, -0.3873], [1.0117, -0.3873],
             [1.4090, -0.3873], [1.0809, -0.3873], [0.1633, -0.3873]],
            [[-1.4521, -0.3873], [-1.7543, -0.3873], [-1.6867, -0.3873], [-0.9025, -0.3873],
             [-0.0308, -0.3873], [0.3186, -0.3873], [0.8446, -0.3873], [1.0117, -0.3873],
             [1.4090, -0.3873], [1.0809, -0.3873], [0.1633, -0.3873]]]

        for key, value in testDataloader_inputs.items():
            self.equalTensors(value, toDevice(torch.tensor(expectedInputs[key]), device),
                              floatApprox=True)

        expectedOutputs = {}
        expectedOutputs['output'] = [[-0.7041, -0.6045, -0.7889, -0.9522],
                                     [1.8915, 2.0164, 1.6924, 0.6435]]
        expectedOutputs['outputMask'] = [[True, True, True, True], [True, True, True, True]]
        for key, value in testDataloader_outputs.items():
            self.equalTensors(value, toDevice(torch.tensor(expectedOutputs[key]), device),
                              floatApprox=True)


class electricityTests(BaseTestClass):
    # bugPotentialCheck1
    #  recheck values are the ones supposed to be
    def setup(self):
        self.backcastLen = 4
        self.forecastLen = 1

    def processedSetup(self):
        self.setup()
        self.trainDf, self.valDf, self.testDf, self.normalizer = getElectricity_processed(
            backcastLen=self.backcastLen, forecastLen=self.forecastLen,
            trainRatio=.6, valRatio=.2, dataInfo=electricityDataInfo, devTestMode=True)

    def testGetElectricity_processed(self):
        self.processedSetup()
        trainDf = self.trainDf.drop(columns='date')
        valDf = self.valDf.drop(columns='date')
        testDf = self.testDf.drop(columns='date')

        trainDfCheck = pd.DataFrame({'consumerId': 13 * [0] + 13 * [1],
                                     'hourOfDay': 2 * [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                                     'dayOfWeek': 26 * [2],
                                     'powerUsage': [-0.9244168595808495] + 3 * [
                                         -0.898671132847855] + 2 * [-0.9244168595808495] + [
                                                       -0.8729254061148606, -0.898671132847855,
                                                       -1.027399766512828, -1.1046369467118118,
                                                       -1.0788912199788172, -1.1046369467118118,
                                                       -1.0788912199788172, 0.8175763662820505,
                                                       0.7598590614012519, 0.860864344942648,
                                                       0.6444244516396631, 0.6732831040800602,
                                                       0.6155657991992657, 0.6299951254194637,
                                                       0.6444244516396631, 0.716571082740656,
                                                       0.9185816498234448, 1.1205922169062292,
                                                       1.4813253724112023, 1.5101840248516019],
                                     'daysFromStart': 26 * [0.0],
                                     'hoursFromStart': [-1.651445647689541, -1.4863010829205867,
                                                        -1.3211565181516327, -1.1560119533826787,
                                                        -0.9908673886137245, -0.8257228238447705,
                                                        -0.6605782590758164, -0.49543369430686224,
                                                        -0.3302891295379082, -0.1651445647689541,
                                                        0.0] +
                                                       [0.1651445647689541, 0.3302891295379082,
                                                        -1.651445647689541, -1.4863010829205867,
                                                        -1.3211565181516327, -1.1560119533826787,
                                                        -0.9908673886137245, -0.8257228238447705,
                                                        -0.6605782590758164, -0.49543369430686224,
                                                        -0.3302891295379082, -0.1651445647689541,
                                                        0.0, 0.1651445647689541,
                                                        0.3302891295379082],
                                     'dayOfMonth': 26 * [0.0], 'month': 26 * [1],
                                     'sequenceIdx': 2 * [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                                     '__startPoint__': 9 * [True] + 4 * [False] + 9 * [True] + 4 * [
                                         False]})
        self.equalDfs(trainDf, trainDfCheck, floatApprox=True)

        valDfCheck = pd.DataFrame({'consumerId': 8 * [0] + 8 * [1],
                                   'hourOfDay': 2 * [9, 10, 11, 12, 13, 14, 15, 16],
                                   'dayOfWeek': 16 * [2],
                                   'powerUsage': [-1.1046369467118118, -1.0788912199788172,
                                                  -1.1046369467118118, -1.0788912199788172] +
                                                 2 * [-1.1046369467118118] + [-1.0788912199788172,
                                                                              -0.9501625863138443,
                                                                              0.9185816498234448,
                                                                              1.1205922169062292,
                                                                              1.4813253724112023,
                                                                              1.5101840248516019,
                                                                              1.4091787413102055,
                                                                              1.2793148053284178,
                                                                              1.163880195566825,
                                                                              1.0484455858052322],
                                   'daysFromStart': 16 * [0.0],
                                   'hoursFromStart': 2 * [-0.1651445647689541, 0.0,
                                                          0.1651445647689541, 0.3302891295379082,
                                                          0.49543369430686224, 0.6605782590758164,
                                                          0.8257228238447705, 0.9908673886137245],
                                   'dayOfMonth': 16 * [0.0],
                                   'month': 16 * [1],
                                   'sequenceIdx': 2 * [9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                                                       16.0],
                                   '__startPoint__': 4 * [True] + 4 * [False] + 4 * [True] + 4 * [
                                       False]})
        self.equalDfs(valDf, valDfCheck, floatApprox=True)

        testDfCheck = pd.DataFrame({'consumerId': 8 * [0] + 8 * [1],
                                    'hourOfDay': 2 * [13, 14, 15, 16, 17, 18, 19, 20],
                                    'dayOfWeek': 16 * [2],
                                    'powerUsage': 2 * [-1.1046369467118118] + [
                                        -1.0788912199788172] + 2 * [-0.9501625863138443] +
                                                  2 * [-0.898671132847855] + [-0.9244168595808495,
                                                                              1.4091787413102055,
                                                                              1.2793148053284178,
                                                                              1.163880195566825,
                                                                              1.0484455858052322,
                                                                              1.0340162595850335,
                                                                              1.019586933364835,
                                                                              1.0484455858052322,
                                                                              1.1494508693466285],
                                    'daysFromStart': 16 * [0.0],
                                    'hoursFromStart': 2 * [0.49543369430686224, 0.6605782590758164,
                                                           0.8257228238447705, 0.9908673886137245,
                                                           1.1560119533826787, 1.3211565181516327,
                                                           1.4863010829205867, 1.651445647689541],
                                    'dayOfMonth': 16 * [0.0],
                                    'month': 16 * [1],
                                    'sequenceIdx': 2 * [13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0,
                                                        20.0],
                                    '__startPoint__': 4 * [True] + 4 * [False] + 4 * [True] + 4 * [
                                        False]})
        self.equalDfs(testDf, testDfCheck, floatApprox=True)

    def testInvNormalizer(self):
        self.processedSetup()
        df = getElectricity_data(backcastLen=self.backcastLen, forecastLen=self.forecastLen,
                                 devTestMode=True)
        for set_ in [self.trainDf, self.valDf, self.testDf]:
            self.normalizer.inverseTransform(set_)
            dfIndexes_whichAreInSetDf = df[df['date'].isin(set_['date'].values)].index
            checkSet = df.loc[dfIndexes_whichAreInSetDf]
            # we have to reset it's index as even checkIndex=False in equalDfs can handle it
            checkSet = checkSet.reset_index(drop=True)

            # removing cols added during getElectricity_processed to be able to check
            set_ = set_.drop(columns=['sequenceIdx', '__startPoint__'])
            self.equalDfs(set_, checkSet, checkIndex=False, floatApprox=True)

    def testDataloader_data(self):
        device = getTorchDevice()
        self.setup()
        trainDataloader, valDataloader, testDataloader, normalizer = getElectricityDataloaders(
            backcastLen=self.backcastLen, forecastLen=self.forecastLen,
            batchSize=64, trainRatio=.7, valRatio=.2,
            shuffle=False, dataInfo=electricityDataInfo, devTestMode=True)

        # these 2 are just here in order if they make error, get detected
        next(iter(trainDataloader))
        next(iter(valDataloader))

        testDataloader_inputs, testDataloader_outputs = next(iter(testDataloader))

        expectedInputs = {}
        expectedInputs['consumerId'] = [[0], [0], [1], [1]]
        expectedInputs['allReals'] = [[[15.0000, 2.0000, -1.0789, 0.0000, 0.8257, 0.0000, 1.0000],
                                       [16.0000, 2.0000, -0.9502, 0.0000, 0.9909, 0.0000, 1.0000],
                                       [17.0000, 2.0000, -0.9502, 0.0000, 1.1560, 0.0000, 1.0000],
                                       [18.0000, 2.0000, -0.8987, 0.0000, 1.3212, 0.0000, 1.0000]],

                                      [[16.0000, 2.0000, -0.9502, 0.0000, 0.9909, 0.0000, 1.0000],
                                       [17.0000, 2.0000, -0.9502, 0.0000, 1.1560, 0.0000, 1.0000],
                                       [18.0000, 2.0000, -0.8987, 0.0000, 1.3212, 0.0000, 1.0000],
                                       [19.0000, 2.0000, -0.8987, 0.0000, 1.4863, 0.0000, 1.0000]],

                                      [[15.0000, 2.0000, 1.1639, 0.0000, 0.8257, 0.0000, 1.0000],
                                       [16.0000, 2.0000, 1.0484, 0.0000, 0.9909, 0.0000, 1.0000],
                                       [17.0000, 2.0000, 1.0340, 0.0000, 1.1560, 0.0000, 1.0000],
                                       [18.0000, 2.0000, 1.0196, 0.0000, 1.3212, 0.0000, 1.0000]],

                                      [[16.0000, 2.0000, 1.0484, 0.0000, 0.9909, 0.0000, 1.0000],
                                       [17.0000, 2.0000, 1.0340, 0.0000, 1.1560, 0.0000, 1.0000],
                                       [18.0000, 2.0000, 1.0196, 0.0000, 1.3212, 0.0000, 1.0000],
                                       [19.0000, 2.0000, 1.0484, 0.0000, 1.4863, 0.0000, 1.0000]]]
        expectedInputs['target'] = [[-1.0789, -0.9502, -0.9502, -0.8987],
                                    [-0.9502, -0.9502, -0.8987, -0.8987],
                                    [1.1639, 1.0484, 1.0340, 1.0196],
                                    [1.0484, 1.0340, 1.0196, 1.0484]]
        for key, value in testDataloader_inputs.items():
            self.equalTensors(value, toDevice(torch.tensor(expectedInputs[key]), device),
                              floatApprox=True, checkType=False)

        expectedOutputs = [[-0.9502, -0.9502, -0.8987, -0.8987],
                           [-0.9502, -0.8987, -0.8987, -0.9244],
                           [1.0484, 1.0340, 1.0196, 1.0484],
                           [1.0340, 1.0196, 1.0484, 1.1495]]
        self.equalTensors(testDataloader_outputs, toDevice(torch.tensor(expectedOutputs), device),
                          floatApprox=True, checkType=False)


class stallionTests(BaseTestClass):
    # bugPotentialCheck1
    #  recheck values are the ones supposed to be
    def setup(self):
        self.maxEncoderLength = 4
        self.maxPredictionLength = 3
        self.minEncoderLength = 2
        self.minPredictionLength = 1

    def processedSetup(self):
        self.setup()
        self.trainDf, self.valDf, self.testDf, self.normalizer, self.dataInfo = getStallion_processed(
            maxEncoderLength=self.maxEncoderLength, maxPredictionLength=self.maxPredictionLength,
            minEncoderLength=self.minEncoderLength, minPredictionLength=self.minPredictionLength,
            trainRatio=.6, valRatio=.2, shuffle=False, dataInfo=stallionDataInfo, devTestMode=True)

    def testGetStallion_processed(self):
        self.processedSetup()
        trainDf = self.trainDf.drop(columns='date')
        valDf = self.valDf.drop(columns='date')
        testDf = self.testDf.drop(columns='date')

        trainDfCheck = pd.DataFrame({'agency': 30 * [0],
                                     'sku': 15 * [0] + 15 * [1],
                                     'volume': [-0.8625490915063628, -0.4061302167066971,
                                                0.5293867316404448, 0.8865841119184443,
                                                1.629327870909205, 1.7654030633960618,
                                                0.9517868083183961, 1.2494512918833958,
                                                0.8979237112923488, 0.9404472089444917,
                                                -0.47700271279360146, -0.2360362260981258,
                                                -0.34092752030674456, 0.26290614635368326,
                                                0.3054296440058258, -0.9605020235090816,
                                                -0.29367867625277666, 0.9224965477587231,
                                                0.8775809855082986, 1.9002737875179683,
                                                1.4476631217636888, 0.25221815725238533,
                                                -0.06910086500065282, -0.07255590825068563,
                                                -0.16238703275153527, -1.0088726290095391,
                                                -1.2092651375114338, -1.3060063485123485,
                                                -0.9397717640088857, -1.0503331480099314],
                                     'industryVolume': [-0.06286996471973942, -1.302157523349218,
                                                        0.27758903179402966, 0.7495849396802461,
                                                        1.1451101982961853, 0.7351482181593412,
                                                        0.25910544490171056, 0.7514163988145893,
                                                        -0.8463733008694923, -0.30631021792922125,
                                                        -1.2404062148178712, -1.4614495945513546,
                                                        -0.29260587172816455, -1.433049151137429,
                                                        1.2556315103024418, -0.06286996471973942,
                                                        -1.302157523349218, 0.27758903179402966,
                                                        0.7495849396802461, 1.1451101982961853,
                                                        0.7351482181593412, 0.25910544490171056,
                                                        0.7514163988145893, -0.8463733008694923,
                                                        -0.30631021792922125, -1.2404062148178712,
                                                        -1.4614495945513546, -0.29260587172816455,
                                                        -1.433049151137429, 1.2556315103024418],
                                     'sodaVolume': [-1.6800327760056237, -1.0458511375615682,
                                                    1.4208765800777308, 0.4557541730463729,
                                                    0.9253657367541211, 1.3732174286587804,
                                                    1.020801832768748, -0.9748739686074069,
                                                    -0.40914650915367307, -0.37284494300526544,
                                                    -0.11243853951571982, -0.482530766151648,
                                                    -0.8712635578209288, -1.3936727103490714,
                                                    0.28053011405802974, -1.6800327760056237,
                                                    -1.0458511375615682, 1.4208765800777308,
                                                    0.4557541730463729, 0.9253657367541211,
                                                    1.3732174286587804, 1.020801832768748,
                                                    -0.9748739686074069, -0.40914650915367307,
                                                    -0.37284494300526544, -0.11243853951571982,
                                                    -0.482530766151648, -0.8712635578209288,
                                                    -1.3936727103490714, 0.28053011405802974],
                                     'avgMaxTemp': [-1.8177710565184677, -1.2074263641967127,
                                                    -0.23992942059876715, 0.37460719955486704,
                                                    0.7652948734861, 1.6372158625171769,
                                                    0.7602645600878436, 0.7032543415742735,
                                                    0.2396271233683256, 0.08368740802238217,
                                                    -1.3177385568888917, -1.7233002471334486,
                                                    -1.39432185478484, -0.5083740296714047,
                                                    -0.5438441884100809, -1.8177710565184677,
                                                    -1.2074263641967127, -0.23992942059876715,
                                                    0.37460719955486704, 0.7652948734861,
                                                    1.6372158625171769, 0.7602645600878436,
                                                    0.7032543415742735, 0.2396271233683256,
                                                    0.08368740802238217, -1.3177385568888917,
                                                    -1.7233002471334486, -1.39432185478484,
                                                    -0.5083740296714047, -0.5438441884100809],
                                     'priceRegular': 2 * [-1.5773929659584842] + [
                                         -0.868765016705545, 0.017664121640219292,
                                         0.08588723400533319, 0.1701635438981933,
                                         0.17085189704521012, 0.1700984215674471,
                                         0.16960560392934446, 0.17647351952145468] + 5 * [
                                                         0.17681875400516697] +
                                                     [-2.8383133474645317, -2.8235766396075563,
                                                      -1.8116344882602144, -0.4518386922270582,
                                                      -0.4498819702804655, -0.4515520828202825,
                                                      -0.4476376091030059, -0.44929601909632305,
                                                      -0.44587576732272727, -0.4464659501476853,
                                                      -0.44954195981351136, -0.4514558785260367,
                                                      -0.45408870807498813, -0.4498542024781363,
                                                      -0.4493460498231102],
                                     'priceActual': [1033.432731, 1065.417195, 1101.133633,
                                                     1138.283357, 1148.969634, 1144.291368,
                                                     1157.884598, 1186.746188, 1152.156204,
                                                     1142.945273, 1133.686722, 1059.479533,
                                                     995.0062437, 1111.858392, 1108.242181,
                                                     969.1862085, 996.9507621, 1061.272227,
                                                     1113.636725, 1086.179255, 1093.956035,
                                                     1119.663707, 1166.178544, 1129.154946,
                                                     1093.258621, 1157.401598, 1122.145738,
                                                     1130.50629, 1082.039959, 1164.201429],
                                     'discount': [108.067269, 76.082805, 78.212187, 88.404143,
                                                  81.36147, 90.540696, 76.984229, 48.082398,
                                                  82.646062, 92.223789, 101.500778, 175.707967,
                                                  240.1812563, 123.329108, 126.945319, 104.9715905,
                                                  77.9940829, 67.717594, 87.975954, 115.537927,
                                                  107.671951, 82.17334, 35.569932, 72.776196,
                                                  108.641001, 44.333743, 79.487386, 70.986222,
                                                  119.678706, 37.544375],
                                     'avgPopulation2017': 30 * [0.0],
                                     'avgYearlyHouseholdIncome2017': 30 * [0.0],
                                     'easterDay': 2 * [0] + [1] + 14 * [0] + [1] + 12 * [0],
                                     'goodFriday': 2 * [0] + [1] + 14 * [0] + [1] + 12 * [0],
                                     'newYear': [1] + 11 * [0] + [1] + 2 * [0] + [1] + 11 * [0] + [
                                         1] + 2 * [0],
                                     'christmas': 11 * [0] + [1] + 14 * [0] + [1] + 3 * [0],
                                     'laborDay': 4 * [0] + [1] + 14 * [0] + [1] + 10 * [0],
                                     'independenceDay': 8 * [0] + [1] + 14 * [0] + [1] + 6 * [0],
                                     'revolutionDayMemorial': 10 * [0] + [1] + 14 * [0] + [
                                         1] + 4 * [
                                                                  0],
                                     'regionalGames': 30 * [0],
                                     'fifaU17WorldCup': 30 * [0],
                                     'footballGoldCup': 30 * [0],
                                     'beerCapital': 9 * [0] + [1] + 14 * [0] + [1] + 5 * [0],
                                     'musicFest': 2 * [0] + [1] + 11 * [0] + [1] + 2 * [0] + [
                                         1] + 11 * [0] + [1],
                                     'discountInPercent': [0.6098478870240189, -0.13053223529835317,
                                                           -0.13933980923547076,
                                                           0.012571713681743703,
                                                           -0.14432185502201866,
                                                           0.04573076440999029,
                                                           -0.24440657796164145,
                                                           -0.8628109804352438,
                                                           -0.12316002408626246,
                                                           0.08120803180917563, 0.2796352711891695,
                                                           1.8671021816161137, 3.2463380052801227,
                                                           0.7465947673133432, 0.8239540547874137,
                                                           0.6905262047136499, 0.025492499544592748,
                                                           -0.3068014852393213, 0.04289357761201891,
                                                           0.6487630143658267, 0.47598024295428476,
                                                           -0.08504379400596632, -1.109608161299191,
                                                           -0.29177455938201863, 0.4967495458580287,
                                                           -0.9169022557583258, -0.1438005709824514,
                                                           -0.3305561044501479, 0.7398079680454523,
                                                           -1.0661930448266814],
                                     'timeIdx': [-1.6583123951777, -1.507556722888818,
                                                 -1.3568010505999364, -1.2060453783110545,
                                                 -1.0552897060221726, -0.9045340337332909,
                                                 -0.753778361444409, -0.6030226891555273,
                                                 -0.45226701686664544, -0.30151134457776363,
                                                 -0.15075567228888181, 0.0, 0.15075567228888181,
                                                 0.30151134457776363, 0.45226701686664544,
                                                 -1.6583123951777, -1.507556722888818,
                                                 -1.3568010505999364, -1.2060453783110545,
                                                 -1.0552897060221726, -0.9045340337332909,
                                                 -0.753778361444409, -0.6030226891555273,
                                                 -0.45226701686664544, -0.30151134457776363,
                                                 -0.15075567228888181, 0.0, 0.15075567228888181,
                                                 0.30151134457776363, 0.45226701686664544],
                                     'month': [0, 4, 5, 6, 7, 8, 9, 10, 11, 1, 2, 3, 0, 4, 5, 0, 4,
                                               5, 6, 7, 8, 9, 10, 11, 1, 2, 3, 0, 4, 5],
                                     'logVolume': [-0.6995720042959636, -0.17370615540812895,
                                                   0.6615399404558162, 0.9226806039448114,
                                                   1.3960681173065124, 1.4744524693024348,
                                                   0.9677330949953845, 1.164345875919409,
                                                   0.9305701324708737, 0.959951798420098,
                                                   -0.24892801680148868, -0.00130585933448347,
                                                   -0.1063092126635116, 0.448783294548194,
                                                   0.48388069141942075, -0.7763996311220466,
                                                   -0.14126242158058952, 0.7324176652011533,
                                                   0.7047183015265613, 1.2740424899786063,
                                                   1.0367716014001136, 0.28617994798835505,
                                                   0.042863524405373055, 0.04012405079227184,
                                                   -0.032097875246977076, -0.8288627754187398,
                                                   -1.0577965815586712, -1.1756897914372764,
                                                   -0.7542244272070301, -0.8746588706227734],
                                     'avgVolumeBySku': [-0.8674550658709772, -0.3697889245865667,
                                                        0.6502721103442135, 1.0397499600450568,
                                                        1.849616599899191, 1.9979891140709405,
                                                        1.1108451230856864, 1.435409997836389,
                                                        1.0521143362260357, 1.0984807469047075,
                                                        -0.4470662757176861, -0.18432328187187938,
                                                        -0.29869376154593635, 0.35970927009120346,
                                                        0.406075680769875, -0.9323680408211178,
                                                        -0.335786890088874, 0.7522782138372913,
                                                        0.7120939912491091, 1.6270578286415658,
                                                        1.2221245087144987, 0.1526059690598026,
                                                        -0.13486577714796252, -0.13795687119320743,
                                                        -0.21832531636957211, -0.9756433574545448,
                                                        -1.1549268120787424, -1.2414774453455963,
                                                        -0.9138214765496492, -1.0127364859974823],
                                     'avgVolumeByAgency': [-1.1836587827608673, -0.464023940374445,
                                                           0.9223912079519402, 1.152105154815404,
                                                           2.2864448216279003, 2.1177168783565072,
                                                           0.8309122025638354, 0.8553066040006642,
                                                           0.6011982557003737, 0.5788367210499478,
                                                           -0.9356490348197839, -0.8807616315869212,
                                                           -1.012897972703072, -0.3644134678407306,
                                                           -0.39897220320957033,
                                                           -1.1836587827608673, -0.464023940374445,
                                                           0.9223912079519402, 1.152105154815404,
                                                           2.2864448216279003, 2.1177168783565072,
                                                           0.8309122025638354, 0.8553066040006642,
                                                           0.6011982557003737, 0.5788367210499478,
                                                           -0.9356490348197839, -0.8807616315869212,
                                                           -1.012897972703072, -0.3644134678407306,
                                                           -0.39897220320957033],
                                     'relativeTimeIdx': 30 * [0.0],
                                     'encoderLength': 30 * [4.0],
                                     'volumeMean': 15 * [1.0000000000000138] + 15 * [
                                         -0.9999999999999859],
                                     'volumeStd': 15 * [0.9999999999999971] + 15 * [
                                         -1.0000000000000033],
                                     'decoderLength': 30 * [3.0],
                                     'sequenceLength': 30 * [7.0],
                                     'fullLenConditions': 30 * [True],
                                     'notFullLen_ButGreaterThan_MinEncoderNPredictLen': 30 * [
                                         False],
                                     '__startPoint__': 9 * [True] + 6 * [False] + 9 * [True] + 6 * [
                                         False]})
        self.equalDfs(trainDf, trainDfCheck, floatApprox=True)

        valDfCheck = pd.DataFrame({'agency': 20 * [0],
                                   'sku': 10 * [0] + 10 * [1],
                                   'volume': [0.9404472089444917, -0.47700271279360146,
                                              -0.2360362260981258, -0.34092752030674456,
                                              0.26290614635368326, 0.3054296440058258,
                                              -1.1290296767931245, -2.4047346063574087,
                                              -1.228251171314791, -0.6414269037152205,
                                              -0.16238703275153527, -1.0088726290095391,
                                              -1.2092651375114338, -1.3060063485123485,
                                              -0.9397717640088857, -1.0503331480099314,
                                              0.7808397745073837, 1.019237758759638,
                                              2.014290214769046, 0.5251665740049661],
                                   'industryVolume': [-0.30631021792922125, -1.2404062148178712,
                                                      -1.4614495945513546, -0.29260587172816455,
                                                      -1.433049151137429, 1.2556315103024418,
                                                      0.38110099095036737, 1.613549785420319,
                                                      1.1876664188540598, 1.1224280095165373,
                                                      -0.30631021792922125, -1.2404062148178712,
                                                      -1.4614495945513546, -0.29260587172816455,
                                                      -1.433049151137429, 1.2556315103024418,
                                                      0.38110099095036737, 1.613549785420319,
                                                      1.1876664188540598, 1.1224280095165373],
                                   'sodaVolume': [-0.37284494300526544, -0.11243853951571982,
                                                  -0.482530766151648, -0.8712635578209288,
                                                  -1.3936727103490714, 0.28053011405802974,
                                                  1.0915596694718503, 1.3366716584495053,
                                                  0.9384294675145773, 0.4406523812451657,
                                                  -0.37284494300526544, -0.11243853951571982,
                                                  -0.482530766151648, -0.8712635578209288,
                                                  -1.3936727103490714, 0.28053011405802974,
                                                  1.0915596694718503, 1.3366716584495053,
                                                  0.9384294675145773, 0.4406523812451657],
                                   'avgMaxTemp': [0.08368740802238217, -1.3177385568888917,
                                                  -1.7233002471334486, -1.39432185478484,
                                                  -0.5083740296714047, -0.5438441884100809,
                                                  0.17281424300912465, 0.7725823786326019,
                                                  1.6029065450094901, 0.8983402135890076,
                                                  0.08368740802238217, -1.3177385568888917,
                                                  -1.7233002471334486, -1.39432185478484,
                                                  -0.5083740296714047, -0.5438441884100809,
                                                  0.17281424300912465, 0.7725823786326019,
                                                  1.6029065450094901, 0.8983402135890076],
                                   'priceRegular': [0.17647351952145468] +
                                                   6 * [0.17681875400516697] + [0.7214084578092269,
                                                                                0.8844786311636912,
                                                                                0.931283986342187,
                                                                                -0.4464659501476853,
                                                                                -0.44954195981351136,
                                                                                -0.4514558785260367,
                                                                                -0.45408870807498813,
                                                                                -0.4498542024781363,
                                                                                -0.4493460498231102,
                                                                                -0.45240910242945725,
                                                                                0.6948857128070935,
                                                                                0.7157884830515292,
                                                                                0.8109479739462481],
                                   'priceActual': [1142.945273, 1133.686722, 1059.479533,
                                                   995.0062437, 1111.858392, 1108.242181,
                                                   1109.019047, 1257.19, 1265.540441, 1271.486842,
                                                   1093.258621, 1157.401598, 1122.145738,
                                                   1130.50629, 1082.039959, 1164.201429,
                                                   1187.285874, 1232.161952, 1210.071952,
                                                   1132.122243],
                                   'discount': [92.223789, 101.500778, 175.707967, 240.1812563,
                                                123.329108, 126.945319, 126.168453, 7.0825,
                                                7.441177, 3.994518, 108.641001, 44.333743,
                                                79.487386, 70.986222, 119.678706, 37.544375,
                                                14.296341, 30.694043, 53.900401, 136.93231],
                                   'avgPopulation2017': 20 * [0.0],
                                   'avgYearlyHouseholdIncome2017': 20 * [0.0],
                                   'easterDay': 6 * [0] + [1] + 9 * [0] + [1] + 3 * [0],
                                   'goodFriday': 6 * [0] + [1] + 9 * [0] + [1] + 3 * [0],
                                   'newYear': 3 * [0] + [1] + 9 * [0] + [1] + 6 * [0],
                                   'christmas': 2 * [0] + [1] + 9 * [0] + [1] + 7 * [0],
                                   'laborDay': 7 * [0] + [1] + 9 * [0] + [1] + 2 * [0],
                                   'independenceDay': 20 * [0],
                                   'revolutionDayMemorial': [0, 1] + 9 * [0] + [1] + 8 * [0],
                                   'regionalGames': 20 * [0],
                                   'fifaU17WorldCup': 20 * [0],
                                   'footballGoldCup': 20 * [0],
                                   'beerCapital': [1] + 9 * [0] + [1] + 9 * [0],
                                   'musicFest': 5 * [0] + [1] + 9 * [0] + [1] + 4 * [0],
                                   'discountInPercent': [0.08120803180917563, 0.2796352711891695,
                                                         1.8671021816161137, 3.2463380052801227,
                                                         0.7465947673133432, 0.8239540547874137,
                                                         0.8073350577536106, -1.7436801030665439,
                                                         -1.7372476812076587, -1.808953285274126,
                                                         0.4967495458580287, -0.9169022557583258,
                                                         -0.1438005709824514, -0.3305561044501479,
                                                         0.7398079680454523, -1.0661930448266814,
                                                         -1.577320071122999, -1.2494740506198239,
                                                         -0.764908072497341, 0.9594254688076072],
                                   'timeIdx': [-0.30151134457776363, -0.15075567228888181, 0.0,
                                               0.15075567228888181, 0.30151134457776363,
                                               0.45226701686664544, 0.6030226891555273,
                                               0.753778361444409, 0.9045340337332909,
                                               1.0552897060221726, -0.30151134457776363,
                                               -0.15075567228888181, 0.0, 0.15075567228888181,
                                               0.30151134457776363, 0.45226701686664544,
                                               0.6030226891555273, 0.753778361444409,
                                               0.9045340337332909, 1.0552897060221726],
                                   'month': [1, 2, 3, 0, 4, 5, 6, 7, 8, 9, 1, 2, 3, 0, 4, 5, 6, 7,
                                             8, 9],
                                   'logVolume': [0.959951798420098, -0.24892801680148868,
                                                 -0.00130585933448347, -0.1063092126635116,
                                                 0.448783294548194, 0.48388069141942075,
                                                 -1.0619194164354484, -4.209829169107547,
                                                 -1.2103432860821035, -0.43198174912266035,
                                                 -0.032097875246977076, -0.8288627754187398,
                                                 -1.0577965815586712, -1.1756897914372764,
                                                 -0.7542244272070301, -0.8746588706227734,
                                                 0.6440729615063753, 0.7911278620575445,
                                                 1.3306577397418973, 0.4768976929539099],
                                   'avgVolumeBySku': [1.0984807469047075, -0.4470662757176861,
                                                      -0.18432328187187938, -0.29869376154593635,
                                                      0.35970927009120346, 0.406075680769875,
                                                      -1.1580179061239873, -2.549010226484141,
                                                      -1.266206197707555, -0.626349730341884,
                                                      -0.21832531636957211, -0.9756433574545448,
                                                      -1.1549268120787424, -1.2414774453455963,
                                                      -0.9138214765496492, -1.0127364859974823,
                                                      0.625543357982255, 0.8388288471041453,
                                                      1.7290639321346437, 0.3968023986341407],
                                   'avgVolumeByAgency': [0.5788367210499478, -0.9356490348197839,
                                                         -0.8807616315869212, -1.012897972703072,
                                                         -0.3644134678407306, -0.39897220320957033,
                                                         -0.3501834003359143, -1.1247056459551998,
                                                         0.3043997048856338, -0.15096245526848698,
                                                         0.5788367210499478, -0.9356490348197839,
                                                         -0.8807616315869212, -1.012897972703072,
                                                         -0.3644134678407306, -0.39897220320957033,
                                                         -0.3501834003359143, -1.1247056459551998,
                                                         0.3043997048856338, -0.15096245526848698],
                                   'relativeTimeIdx': 20 * [0.0],
                                   'encoderLength': 20 * [4.0],
                                   'volumeMean': 10 * [1.0000000000000138] + 10 * [
                                       -0.9999999999999859],
                                   'volumeStd': 10 * [0.9999999999999971] + 10 * [
                                       -1.0000000000000033],
                                   'decoderLength': 20 * [3.0],
                                   'sequenceLength': 8 * [7.0] + [6.0, 5.0] + 8 * [7.0] + [6.0,
                                                                                           5.0],
                                   'fullLenConditions': 8 * [True] + 2 * [False] + 8 * [
                                       True] + 2 * [False],
                                   'notFullLen_ButGreaterThan_MinEncoderNPredictLen': 8 * [
                                       False] + 2 * [True] + 8 * [False] + 2 * [True],
                                   '__startPoint__': 4 * [True] + 6 * [False] + 4 * [True] + 6 * [
                                       False]})
        self.equalDfs(valDf, valDfCheck, floatApprox=True)

        testDfCheck = pd.DataFrame({'agency': 20 * [0],
                                    'sku': 10 * [0] + 10 * [1],
                                    'volume': [0.26290614635368326, 0.3054296440058258,
                                               -1.1290296767931245, -2.4047346063574087,
                                               -1.228251171314791, -0.6414269037152205,
                                               -0.1623288301677448, -0.3324228207763161,
                                               -0.006409338776554596, -1.1913974733496007,
                                               -0.9397717640088857, -1.0503331480099314,
                                               0.7808397745073837, 1.019237758759638,
                                               2.014290214769046, 0.5251665740049661,
                                               -0.08637608125081638, -0.5217115307549328,
                                               -0.9605020235090816, -1.0987037535103887],
                                    'industryVolume': [-1.433049151137429, 1.2556315103024418,
                                                       0.38110099095036737, 1.613549785420319,
                                                       1.1876664188540598, 1.1224280095165373,
                                                       0.5521981385807955, -1.6878924780687503,
                                                       -1.1613507547302522, -0.23606401336912874,
                                                       -1.433049151137429, 1.2556315103024418,
                                                       0.38110099095036737, 1.613549785420319,
                                                       1.1876664188540598, 1.1224280095165373,
                                                       0.5521981385807955, -1.6878924780687503,
                                                       -1.1613507547302522, -0.23606401336912874],
                                    'sodaVolume': [-1.3936727103490714, 0.28053011405802974,
                                                   1.0915596694718503, 1.3366716584495053,
                                                   0.9384294675145773, 0.4406523812451657,
                                                   0.9783314511583215, -0.8059727197657723,
                                                   -0.38696442321334673, -1.7265984420531721,
                                                   -1.3936727103490714, 0.28053011405802974,
                                                   1.0915596694718503, 1.3366716584495053,
                                                   0.9384294675145773, 0.4406523812451657,
                                                   0.9783314511583215, -0.8059727197657723,
                                                   -0.38696442321334673, -1.7265984420531721],
                                    'avgMaxTemp': [-0.5083740296714047, -0.5438441884100809,
                                                   0.17281424300912465, 0.7725823786326019,
                                                   1.6029065450094901, 0.8983402135890076,
                                                   0.8368720412154541, 0.518110863031553,
                                                   0.4912395303997125, -1.1041114652953083,
                                                   -0.5083740296714047, -0.5438441884100809,
                                                   0.17281424300912465, 0.7725823786326019,
                                                   1.6029065450094901, 0.8983402135890076,
                                                   0.8368720412154541, 0.518110863031553,
                                                   0.4912395303997125, -1.1041114652953083],
                                    'priceRegular': 3 * [0.17681875400516697] + [0.7214084578092269,
                                                                                 0.8844786311636912,
                                                                                 0.931283986342187,
                                                                                 1.1204851034420502,
                                                                                 1.4584597579599226,
                                                                                 1.4382407347402055,
                                                                                 1.4202480788624585,
                                                                                 -0.4498542024781363,
                                                                                 -0.4493460498231102,
                                                                                 -0.45240910242945725,
                                                                                 0.6948857128070935,
                                                                                 0.7157884830515292,
                                                                                 0.8109479739462481,
                                                                                 0.985053517579264,
                                                                                 1.384996245273268,
                                                                                 1.3871908940332065,
                                                                                 1.3711949734490312],
                                    'priceActual': [1111.858392, 1108.242181, 1109.019047, 1257.19,
                                                    1265.540441, 1271.486842, 1239.516347,
                                                    1198.774971, 1248.267052, 1241.66266,
                                                    1082.039959, 1164.201429, 1187.285874,
                                                    1232.161952, 1210.071952, 1132.122243,
                                                    1157.897025, 1165.7119, 1098.897211,
                                                    1170.108509],
                                    'discount': [123.329108, 126.945319, 126.168453, 7.0825,
                                                 7.441177, 3.994518, 46.069711, 104.86136,
                                                 54.289438, 59.932893, 119.678706, 37.544375,
                                                 14.296341, 30.694043, 53.900401, 136.93231,
                                                 120.456014, 134.000952, 200.932851, 128.867256],
                                    'avgPopulation2017': 20 * [0.0],
                                    'avgYearlyHouseholdIncome2017': 20 * [0.0],
                                    'easterDay': 2 * [0] + [1] + 9 * [0] + [1] + 7 * [0],
                                    'goodFriday': 2 * [0] + [1] + 9 * [0] + [1] + 7 * [0],
                                    'newYear': 20 * [0],
                                    'christmas': 20 * [0],
                                    'laborDay': 3 * [0] + [1] + 9 * [0] + [1] + 6 * [0],
                                    'independenceDay': 7 * [0] + [1] + 9 * [0] + [1] + 2 * [0],
                                    'revolutionDayMemorial': 9 * [0] + [1] + 9 * [0] + [1],
                                    'regionalGames': 9 * [0] + [1] + 9 * [0] + [1],
                                    'fifaU17WorldCup': 20 * [0],
                                    'footballGoldCup': 20 * [0],
                                    'beerCapital': 8 * [0] + [1] + 9 * [0] + [1, 0],
                                    'musicFest': [0, 1] + 9 * [0] + [1] + 8 * [0],
                                    'discountInPercent': [0.7465947673133432, 0.8239540547874137,
                                                          0.8073350577536106, -1.7436801030665439,
                                                          -1.7372476812076587, -1.808953285274126,
                                                          -0.9448019862283222, 0.23374273811528343,
                                                          -0.7903939337446325, -0.6750135822840012,
                                                          0.7398079680454523, -1.0661930448266814,
                                                          -1.577320071122999, -1.2494740506198239,
                                                          -0.764908072497341, 0.9594254688076072,
                                                          0.5981216283182599, 0.832576064935554,
                                                          2.192954508180214, 0.7296930277130573],
                                    'timeIdx': [0.30151134457776363, 0.45226701686664544,
                                                0.6030226891555273, 0.753778361444409,
                                                0.9045340337332909, 1.0552897060221726,
                                                1.2060453783110545, 1.3568010505999364,
                                                1.507556722888818, 1.6583123951777,
                                                0.30151134457776363, 0.45226701686664544,
                                                0.6030226891555273, 0.753778361444409,
                                                0.9045340337332909, 1.0552897060221726,
                                                1.2060453783110545, 1.3568010505999364,
                                                1.507556722888818, 1.6583123951777],
                                    'month': [4, 5, 6, 7, 8, 9, 10, 11, 1, 2, 4, 5, 6, 7, 8, 9, 10,
                                              11, 1, 2],
                                    'logVolume': [0.448783294548194, 0.48388069141942075,
                                                  -1.0619194164354484, -4.209829169107547,
                                                  -1.2103432860821035, -0.43198174912266035,
                                                  0.07010614342666334, -0.09764130577819116,
                                                  0.21520724795730614, -1.1542555330744235,
                                                  -0.7542244272070301, -0.8746588706227734,
                                                  0.6440729615063753, 0.7911278620575445,
                                                  1.3306577397418973, 0.4768976929539099,
                                                  0.029138216748495677, -0.3420572904363609,
                                                  -0.7763996311220466, -0.9290904606111408],
                                    'avgVolumeBySku': [0.35970927009120346, 0.406075680769875,
                                                       -1.1580179061239873, -2.549010226484141,
                                                       -1.266206197707555, -0.626349730341884,
                                                       -0.10395483669551471, -0.28942047941020205,
                                                       0.06605533579294852, -1.2260219751193726,
                                                       -0.9138214765496492, -1.0127364859974823,
                                                       0.625543357982255, 0.8388288471041453,
                                                       1.7290639321346437, 0.3968023986341407,
                                                       -0.15032124737418665, -0.5397990970750299,
                                                       -0.9323680408211178, -1.0560118026309095],
                                    'avgVolumeByAgency': [-0.3644134678407306, -0.39897220320957033,
                                                          -0.3501834003359143, -1.1247056459551998,
                                                          0.3043997048856338, -0.15096245526848698,
                                                          -0.16722538955970512, -0.5453386118305379,
                                                          -0.5697330132673656, -1.5007860014396301,
                                                          -0.3644134678407306, -0.39897220320957033,
                                                          -0.3501834003359143, -1.1247056459551998,
                                                          0.3043997048856338, -0.15096245526848698,
                                                          -0.16722538955970512, -0.5453386118305379,
                                                          -0.5697330132673656, -1.5007860014396301],
                                    'relativeTimeIdx': 20 * [0.0],
                                    'encoderLength': 7 * [4.0] + [3.0, 2.0, 1.0] + 7 * [4.0] +
                                                     [3.0, 2.0, 1.0],
                                    'volumeMean': 10 * [1.0000000000000138] + 10 * [
                                        -0.9999999999999859],
                                    'volumeStd': 10 * [0.9999999999999971] + 10 * [
                                        -1.0000000000000033],
                                    'decoderLength': 8 * [3.0] + [2.0, 1.0] + 8 * [3.0] + [2.0,
                                                                                           1.0],
                                    'sequenceLength': 4 * [7.0] + [6.0, 5.0, 4.0, 3.0, 2.0,
                                                                   1.0] + 4 * [7.0] +
                                                      [6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
                                    'fullLenConditions': 4 * [True] + 6 * [False] + 4 * [
                                        True] + 6 * [False],
                                    'notFullLen_ButGreaterThan_MinEncoderNPredictLen': 4 * [
                                        False] + 5 * [True] + 5 * [False] + 5 * [True] + [False],
                                    '__startPoint__': 4 * [True] + 6 * [False] + 4 * [True] + 6 * [
                                        False]})
        self.equalDfs(testDf, testDfCheck, floatApprox=True)

    def testInvNormalizer(self):
        self.processedSetup()
        df = getStallion_data(maxEncoderLength=self.maxEncoderLength,
                              maxPredictionLength=self.maxPredictionLength, devTestMode=True)
        for set_ in [self.trainDf, self.valDf, self.testDf]:
            self.normalizer.inverseTransform(set_)
            dfIndexes_whichAreInSetDf = df[df['date'].isin(set_['date'].values)].index
            checkSet = df.loc[dfIndexes_whichAreInSetDf]
            # we have to reset it's index as even checkIndex=False in equalDfs can handle it
            checkSet = checkSet.reset_index(drop=True)

            # removing cols added during getStallion_processed to be able to check
            dropCols = set(set_.columns) - set(checkSet.columns)
            set_ = set_.drop(columns=dropCols)
            self.equalDfs(set_, checkSet, checkIndex=False, floatApprox=True)

    def testDataloader_data(self):
        device = getTorchDevice()
        self.setup()
        trainDataloader, valDataloader, testDataloader, normalizer = getStallion_TftDataloaders(
            maxEncoderLength=self.maxEncoderLength, maxPredictionLength=self.maxPredictionLength,
            minEncoderLength=self.minEncoderLength, minPredictionLength=self.minPredictionLength,
            batchSize=64, trainRatio=.7, valRatio=.2,
            shuffle=False, dataInfo=stallionDataInfo, devTestMode=True)

        # these 2 are just here in order if they make error, get detected
        next(iter(trainDataloader))
        next(iter(valDataloader))

        testDataloader_inputs, testDataloader_outputs = next(iter(testDataloader))

        expectedInputs = {'allReals': {}}
        expectedInputs['encoderLengths'] = [[4.0], [4.0], [4.0], [4.0], [4.0]]
        expectedInputs['decoderLengths'] = [[3.0], [3.0], [3.0], [3.0], [3.0]]
        self.equalTensors(testDataloader_inputs['encoderLengths'],
                          toDevice(torch.tensor(expectedInputs['encoderLengths']), device),
                          floatApprox=True, checkType=False)
        self.equalTensors(testDataloader_inputs['decoderLengths'],
                          toDevice(torch.tensor(expectedInputs['decoderLengths']), device),
                          floatApprox=True, checkType=False)

        expectedInputs['allReals']['avgVolumeByAgency'] = [
            [-0.3501833975315094, -1.1247056722640991, 0.304399698972702, -0.1509624570608139,
             -0.16722539067268372, -0.5453386306762695, -0.5697330236434937],
            [-1.1247056722640991, 0.304399698972702, -0.1509624570608139, -0.16722539067268372,
             -0.5453386306762695, -0.5697330236434937, -1.5007859468460083],
            [0.304399698972702, -0.1509624570608139, -0.16722539067268372, -0.5453386306762695,
             -0.5697330236434937, -1.5007859468460083, 0.0],
            [-0.3501833975315094, -1.1247056722640991, 0.304399698972702, -0.1509624570608139,
             -0.16722539067268372, -0.5453386306762695, -0.5697330236434937],
            [-1.1247056722640991, 0.304399698972702, -0.1509624570608139, -0.16722539067268372,
             -0.5453386306762695, -0.5697330236434937, -1.5007859468460083]]
        expectedInputs['allReals']['discountInPercent'] = [
            [0.8073350787162781, -1.7436801195144653, -1.7372477054595947, -1.8089532852172852,
             -0.9448019862174988, 0.23374274373054504, -0.7903939485549927],
            [-1.7436801195144653, -1.7372477054595947, -1.8089532852172852, -0.9448019862174988,
             0.23374274373054504, -0.7903939485549927, -0.6750136017799377],
            [-1.7372477054595947, -1.8089532852172852, -0.9448019862174988, 0.23374274373054504,
             -0.7903939485549927, -0.6750136017799377, 0.0],
            [-1.5773200988769531, -1.249474048614502, -0.7649080753326416, 0.9594254493713379,
             0.5981216430664062, 0.8325760364532471, 2.1929545402526855],
            [-1.249474048614502, -0.7649080753326416, 0.9594254493713379, 0.5981216430664062,
             0.8325760364532471, 2.1929545402526855, 0.7296930551528931]]
        expectedInputs['allReals']['volumeStd'] = [[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                                                   [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                                                   [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                                                   [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0],
                                                   [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0]]
        expectedInputs['allReals']['relativeTimeIdx'] = 5 * [
            [-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5]]
        expectedInputs['allReals']['avgMaxTemp'] = [
            [0.1728142499923706, 0.7725823521614075, 1.602906584739685, 0.8983402252197266,
             0.8368720412254333, 0.5181108713150024, 0.4912395179271698],
            [0.7725823521614075, 1.602906584739685, 0.8983402252197266, 0.8368720412254333,
             0.5181108713150024, 0.4912395179271698, -1.1041114330291748],
            [1.602906584739685, 0.8983402252197266, 0.8368720412254333, 0.5181108713150024,
             0.4912395179271698, -1.1041114330291748, 0.0],
            [0.1728142499923706, 0.7725823521614075, 1.602906584739685, 0.8983402252197266,
             0.8368720412254333, 0.5181108713150024, 0.4912395179271698],
            [0.7725823521614075, 1.602906584739685, 0.8983402252197266, 0.8368720412254333,
             0.5181108713150024, 0.4912395179271698, -1.1041114330291748]]
        expectedInputs['allReals']['timeIdx'] = [
            [0.6030226945877075, 0.753778338432312, 0.9045340418815613, 1.0552897453308105,
             1.206045389175415, 1.3568010330200195, 1.507556676864624],
            [0.753778338432312, 0.9045340418815613, 1.0552897453308105, 1.206045389175415,
             1.3568010330200195, 1.507556676864624, 1.658312439918518],
            [0.9045340418815613, 1.0552897453308105, 1.206045389175415, 1.3568010330200195,
             1.507556676864624, 1.658312439918518, 0.0],
            [0.6030226945877075, 0.753778338432312, 0.9045340418815613, 1.0552897453308105,
             1.206045389175415, 1.3568010330200195, 1.507556676864624],
            [0.753778338432312, 0.9045340418815613, 1.0552897453308105, 1.206045389175415,
             1.3568010330200195, 1.507556676864624, 1.658312439918518]]
        expectedInputs['allReals']['priceRegular'] = [
            [0.17681875824928284, 0.721408486366272, 0.8844786286354065, 0.9312840104103088,
             1.1204850673675537, 1.458459734916687, 1.4382407665252686],
            [0.721408486366272, 0.8844786286354065, 0.9312840104103088, 1.1204850673675537,
             1.458459734916687, 1.4382407665252686, 1.420248031616211],
            [0.8844786286354065, 0.9312840104103088, 1.1204850673675537, 1.458459734916687,
             1.4382407665252686, 1.420248031616211, 0.0],
            [-0.4524090886116028, 0.6948857307434082, 0.7157884836196899, 0.8109479546546936,
             0.985053539276123, 1.3849962949752808, 1.3871909379959106],
            [0.6948857307434082, 0.7157884836196899, 0.8109479546546936, 0.985053539276123,
             1.3849962949752808, 1.3871909379959106, 1.3711949586868286]]
        expectedInputs['allReals']['logVolume'] = [
            [-1.0619194507598877, -4.209829330444336, -1.2103432416915894, -0.43198174238204956,
             0.070106141269207, -0.09764130413532257, 0.21520724892616272],
            [-4.209829330444336, -1.2103432416915894, -0.43198174238204956, 0.070106141269207,
             -0.09764130413532257, 0.21520724892616272, -1.1542555093765259],
            [-1.2103432416915894, -0.43198174238204956, 0.070106141269207, -0.09764130413532257,
             0.21520724892616272, -1.1542555093765259, 0.0],
            [0.644072949886322, 0.7911278605461121, 1.330657720565796, 0.4768976867198944,
             0.029138216748833656, -0.3420572876930237, -0.7763996124267578],
            [0.7911278605461121, 1.330657720565796, 0.4768976867198944, 0.029138216748833656,
             -0.3420572876930237, -0.7763996124267578, -0.9290904402732849]]
        expectedInputs['allReals']['avgPopulation2017'] = 5 * ([7 * [0.0]])
        expectedInputs['allReals']['sodaVolume'] = [
            [1.091559648513794, 1.3366717100143433, 0.9384294748306274, 0.44065237045288086,
             0.978331446647644, -0.805972695350647, -0.3869644105434418],
            [1.3366717100143433, 0.9384294748306274, 0.44065237045288086, 0.978331446647644,
             -0.805972695350647, -0.3869644105434418, -1.7265985012054443],
            [0.9384294748306274, 0.44065237045288086, 0.978331446647644, -0.805972695350647,
             -0.3869644105434418, -1.7265985012054443, 0.0],
            [1.091559648513794, 1.3366717100143433, 0.9384294748306274, 0.44065237045288086,
             0.978331446647644, -0.805972695350647, -0.3869644105434418],
            [1.3366717100143433, 0.9384294748306274, 0.44065237045288086, 0.978331446647644,
             -0.805972695350647, -0.3869644105434418, -1.7265985012054443]]
        expectedInputs['allReals']['industryVolume'] = [
            [0.38110098242759705, 1.613549828529358, 1.187666416168213, 1.122428059577942,
             0.5521981120109558, -1.6878924369812012, -1.1613507270812988],
            [1.613549828529358, 1.187666416168213, 1.122428059577942, 0.5521981120109558,
             -1.6878924369812012, -1.1613507270812988, -0.23606401681900024],
            [1.187666416168213, 1.122428059577942, 0.5521981120109558, -1.6878924369812012,
             -1.1613507270812988, -0.23606401681900024, 0.0],
            [0.38110098242759705, 1.613549828529358, 1.187666416168213, 1.122428059577942,
             0.5521981120109558, -1.6878924369812012, -1.1613507270812988],
            [1.613549828529358, 1.187666416168213, 1.122428059577942, 0.5521981120109558,
             -1.6878924369812012, -1.1613507270812988, -0.23606401681900024]]
        expectedInputs['allReals']['avgVolumeBySku'] = [
            [-1.158017873764038, -2.5490102767944336, -1.26620614528656, -0.6263497471809387,
             -0.10395483672618866, -0.289420485496521, 0.06605533510446548],
            [-2.5490102767944336, -1.26620614528656, -0.6263497471809387, -0.10395483672618866,
             -0.289420485496521, 0.06605533510446548, -1.2260220050811768],
            [-1.26620614528656, -0.6263497471809387, -0.10395483672618866, -0.289420485496521,
             0.06605533510446548, -1.2260220050811768, 0.0],
            [0.6255433559417725, 0.8388288617134094, 1.7290639877319336, 0.3968023955821991,
             -0.15032124519348145, -0.5397990942001343, -0.9323680400848389],
            [0.8388288617134094, 1.7290639877319336, 0.3968023955821991, -0.15032124519348145,
             -0.5397990942001343, -0.9323680400848389, -1.0560117959976196]]
        expectedInputs['allReals']['encoderLength'] = 5 * ([7 * [0.25]])
        expectedInputs['allReals']['avgYearlyHouseholdIncome2017'] = 5 * ([7 * [0.0]])
        expectedInputs['allReals']['volumeMean'] = [[1., 1., 1., 1., 1., 1., 1.],
                                                    [1., 1., 1., 1., 1., 1., 1.],
                                                    [1., 1., 1., 1., 1., 1., 0.],
                                                    [-1., -1., -1., -1., -1., -1., -1.],
                                                    [-1., -1., -1., -1., -1., -1., -1.]]

        for key, value in testDataloader_inputs['allReals'].items():
            self.equalTensors(value,
                              toDevice(torch.tensor(expectedInputs['allReals'][key]), device),
                              floatApprox=True, checkType=False)

        expectedOutputs = {'volume': [[-0.1623, -0.3324, -0.0064, 0.0000, 0.0000, 0.0000, 0.0000],
                                      [-0.3324, -0.0064, -1.1914, 0.0000, 0.0000, 0.0000, 0.0000],
                                      [-0.0064, -1.1914, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                      [-0.0864, -0.5217, -0.9605, 0.0000, 0.0000, 0.0000, 0.0000],
                                      [-0.5217, -0.9605, -1.0987, 0.0000, 0.0000, 0.0000, 0.0000]]}
        self.equalTensors(testDataloader_outputs['volume'],
                          toDevice(torch.tensor(expectedOutputs['volume']), device),
                          floatApprox=True, checkType=False)


# ---- run test
if __name__ == '__main__':
    unittest.main()
