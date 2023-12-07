# ---- imports

import unittest

import pandas as pd
import torch

from dataPrep.commonDatasets.electricity import getElectricity_processed, getElectricity_data, \
    getElectricityDataloaders, dataInfo as electricityDataInfo
from dataPrep.commonDatasets.epfFrBe import getEpfFrBe_data, getEpfFrBe_processed, \
    getEpfFrBeDataloaders, dataInfo as epfFrBeDataInfo
from dataPrep.utils import combineNSeries
from tests.baseTest import BaseTestClass


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

        trainDfCheck = pd.DataFrame({'genForecast': [-0.03080956797102298, 0.3186052798375584, 0.844584345427674, 1.0116957943796043, 1.4090496841097493, 1.0809035661677773, 0.1633097919226333, -0.24315926775034438, -0.4396418198514017, -0.06760784662912478, 1.3229619679829974, 1.712888682204168, 1.00865740439866, 0.09106363015351601, -0.23944567999585706, 0.3705955084003811, -0.1763146881695723, -0.8545508516936688, -1.4872111655036027] +
                                                    3 * [0.0] + [-0.03080956797102298, 0.3186052798375584, 0.844584345427674, 1.0116957943796043, 1.4090496841097493, 1.0809035661677773, 0.1633097919226333, -0.24315926775034438, -0.4396418198514017, -0.06760784662912478, 1.3229619679829974, 1.712888682204168, 1.00865740439866, 0.09106363015351601, -0.23944567999585706, 0.3705955084003811, -0.1763146881695723, -0.8545508516936688, -1.4872111655036027] +
                                                    3 * [0.0],
                                     'systemLoad': [-0.6221650010346238, -0.2683097548411284, 0.4512716844203051, 0.6859123151938649, 1.0382521213608356, 1.4835389158213155, 0.3421600025105335, -0.34812292957142427, -0.5208830925952294, -0.25896454134422664, 1.2377850581865752, 1.775008547589548, 1.0372418280098192, 0.27194461461489344, -0.3569629963928178, 0.8018434772229972, 0.5939756202513721, -0.4365235977853596, -0.5107801590850652] +
                                                   3 * [0.0] + [-0.6221650010346238, -0.2683097548411284, 0.4512716844203051, 0.6859123151938649, 1.0382521213608356, 1.4835389158213155, 0.3421600025105335, -0.34812292957142427, -0.5208830925952294, -0.25896454134422664, 1.2377850581865752, 1.775008547589548, 1.0372418280098192, 0.27194461461489344, -0.3569629963928178, 0.8018434772229972, 0.5939756202513721, -0.4365235977853596, -0.5107801590850652]
                                                   + 3 * [0.0],
                                     'weekDay': 16 * [-0.3872983346207408] + 3 * [2.5819888974716125] +
                                                3 * [0.0] + 16 * [-0.3872983346207408] +
                                                3 * [2.5819888974716125] + 3 * [0.0],
                                     'mask': 19 * [True] + 3 * [False] + 19 * [True] + 3 * [False],
                                     '__startPoint__': 12 * [True] + 10 * [False] + 12 * [
                                         True] + 10 * [False],
                                     'price': [-0.8934245994612829, -0.797110549573975, -0.7089587073042353, -0.7040613827336943, -0.6044824497993591, -0.7889483419564064, -0.9521924943077759, -1.3496920052833608, -1.1203339712296865, -0.753850849200862, -0.27962658662013373, -0.010273735240374053, -0.27064815824080846, -0.7448724208215367, -0.16862056302120254, -0.016803501334428695, -0.0755713961809219, -0.29839966414054125, -0.06496052627808296] +
                                              3 * [0.0] + [0.36273915288250524, 0.9602127504885175, 1.5617674519033138, 1.8915206396530801, 2.016402416201878, 1.6923627737844096, 0.6435190949268605, 0.6769841461588916, 0.9846993733412227, 1.5642161141885844, 2.617140896854918, 1.1528408502631335, 0.615767589027128, 0.5814863170333403, 0.578221433986313, 0.41089617782615895, -0.0755713961809219, -0.29839966414054125, -0.06496052627808296]
                                              + 3 * [0.0],
                                     'priceType': 22 * ['priceFr'] + 22 * ['priceBe'],
                                     'market0': 22 * [1.0] + 22 * [0.0],
                                     'market1': 22 * [0.0] + 22 * [1.0]})

        self.equalDfs(trainDf, trainDfCheck)
        valDfCheck = pd.DataFrame({'genForecast': [-1.7542518849399193, -1.686732107585604, -0.9024898936152327, -0.03080956797102298, 0.3186052798375584, 0.844584345427674, 1.0116957943796043, 1.4090496841097493, 1.0809035661677773, 0.1633097919226333, -0.24315926775034438, -0.4396418198514017, -0.06760784662912478, -1.7542518849399193, -1.686732107585604, -0.9024898936152327, -0.03080956797102298, 0.3186052798375584, 0.844584345427674, 1.0116957943796043, 1.4090496841097493, 1.0809035661677773, 0.1633097919226333, -0.24315926775034438, -0.4396418198514017, -0.06760784662912478],
                                   'systemLoad': [-1.937061797382473, -1.5145066033198622, -1.1386774767417602, -0.6221650010346238, -0.2683097548411284, 0.4512716844203051, 0.6859123151938649, 1.0382521213608356, 1.4835389158213155, 0.3421600025105335, -0.34812292957142427, -0.5208830925952294, -0.25896454134422664, -1.937061797382473, -1.5145066033198622, -1.1386774767417602, -0.6221650010346238, -0.2683097548411284, 0.4512716844203051, 0.6859123151938649, 1.0382521213608356, 1.4835389158213155, 0.3421600025105335, -0.34812292957142427, -0.5208830925952294, -0.25896454134422664],
                                   'weekDay': 26 * [-0.3872983346207408],
                                   'mask': 26 * [True],
                                   '__startPoint__': 3 * [True] + 10 * [False] + 3 * [True] + 10 * [
                                       False],
                                   'price': [-1.5178334822052715, -1.3154107332895728, -0.9823926624927793, -0.8934245994612829, -0.797110549573975, -0.7089587073042353, -0.7040613827336943, -0.6044824497993591, -0.7889483419564064, -0.9521924943077759, -1.3496920052833608, -1.1203339712296865, -0.753850849200862, -0.5122495037208351, -1.3154107332895728, 0.12358646968774896, 0.36273915288250524, 0.9602127504885175, 1.5617674519033138, 1.8915206396530801, 2.016402416201878, 1.6923627737844096, 0.6435190949268605, 0.6769841461588916, 0.9846993733412227, 1.5642161141885844],
                                   'priceType': 13 * ['priceFr'] + 13 * ['priceBe'],
                                   'market0': 13 * [1.0] + 13 * [0.0],
                                   'market1': 13 * [0.0] + 13 * [1.0]})

        self.equalDfs(valDf, valDfCheck)
        testDfCheck = pd.DataFrame({'genForecast': [-1.4521008812793588, -1.7542518849399193, -1.686732107585604, -0.9024898936152327, -0.03080956797102298, 0.3186052798375584, 0.844584345427674, 1.0116957943796043, 1.4090496841097493, 1.0809035661677773, 0.1633097919226333, -1.4521008812793588, -1.7542518849399193, -1.686732107585604, -0.9024898936152327, -0.03080956797102298, 0.3186052798375584, 0.844584345427674, 1.0116957943796043, 1.4090496841097493, 1.0809035661677773, 0.1633097919226333],
                                    'systemLoad': [-1.8059762350880948, -1.937061797382473, -1.5145066033198622, -1.1386774767417602, -0.6221650010346238, -0.2683097548411284, 0.4512716844203051, 0.6859123151938649, 1.0382521213608356, 1.4835389158213155, 0.3421600025105335, -1.8059762350880948, -1.937061797382473, -1.5145066033198622, -1.1386774767417602, -0.6221650010346238, -0.2683097548411284, 0.4512716844203051, 0.6859123151938649, 1.0382521213608356, 1.4835389158213155, 0.3421600025105335],
                                    'weekDay': 22 * [-0.3872983346207408], 'mask': 22 * [True],
                                    '__startPoint__': [True] + 10 * [False] + [True] + 10 * [False],
                                    'price': [-1.534157897440408, -1.5178334822052715, -1.3154107332895728, -0.9823926624927793, -0.8934245994612829, -0.797110549573975, -0.7089587073042353, -0.7040613827336943, -0.6044824497993591, -0.7889483419564064, -0.9521924943077759, -0.2151451464413429, -0.5122495037208351, -1.3154107332895728, 0.12358646968774896, 0.36273915288250524, 0.9602127504885175, 1.5617674519033138, 1.8915206396530801, 2.016402416201878, 1.6923627737844096, 0.6435190949268605],
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

            # removing cols added during getEpfFrBe_processed to be able to check
            set_ = set_.drop(columns=['__startPoint__', 'market0', 'market1', 'mask'])
            self.equalDfs(set_, checkSet, checkIndex=False, floatApprox=True)

    def testDataloader_data(self):
        device = torch.device('cuda')
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
                                   [-0.2151, -0.5122, -1.3154,  0.1236,  0.3627,  0.9602,  1.5618]]
        expectedInputs['mask'] = [[True, True, True, True, True, True, True],
                                 [True, True, True, True, True, True, True]]
        expectedInputs['historyExogenous'] = [[-1.8060, -1.9371, -1.5145, -1.1387, -0.6222, -0.2683,  0.4513],
                                             [-1.8060, -1.9371, -1.5145, -1.1387, -0.6222, -0.2683,  0.4513]]
        expectedInputs['staticExogenous'] = [[[1., 0.]], [[0., 1.]]]
        expectedInputs['futureExogenous'] = [[[-1.4521, -0.3873], [-1.7543, -0.3873], [-1.6867, -0.3873], [-0.9025, -0.3873], [-0.0308, -0.3873], [ 0.3186, -0.3873], [ 0.8446, -0.3873], [ 1.0117, -0.3873], [ 1.4090, -0.3873], [ 1.0809, -0.3873], [ 0.1633, -0.3873]], [[-1.4521, -0.3873], [-1.7543, -0.3873], [-1.6867, -0.3873], [-0.9025, -0.3873], [-0.0308, -0.3873], [ 0.3186, -0.3873], [ 0.8446, -0.3873], [ 1.0117, -0.3873], [ 1.4090, -0.3873], [ 1.0809, -0.3873], [ 0.1633, -0.3873]]]


        for key, value in testDataloader_inputs.items():
            self.equalTensors(value, torch.tensor(expectedInputs[key]).to(device), floatApprox=True)

        expectedOutputs = {}
        expectedOutputs['output'] = [[-0.7041, -0.6045, -0.7889, -0.9522],
                                     [ 1.8915,  2.0164,  1.6924,  0.6435]]
        expectedOutputs['outputMask'] = [[True, True, True, True], [True, True, True, True]]
        for key, value in testDataloader_outputs.items():
            self.equalTensors(value, torch.tensor(expectedOutputs[key]).to(device), floatApprox=True)

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
                                     'hourOfDay': 2* [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                                     'dayOfWeek': 26 * [2],
                                     'powerUsage': [-0.9244168595808495] + 3 * [
                                         -0.898671132847855] + 2 * [-0.9244168595808495] + [-0.8729254061148606, -0.898671132847855, -1.027399766512828, -1.1046369467118118, -1.0788912199788172, -1.1046369467118118, -1.0788912199788172, 0.8175763662820505, 0.7598590614012519, 0.860864344942648, 0.6444244516396631, 0.6732831040800602, 0.6155657991992657, 0.6299951254194637, 0.6444244516396631, 0.716571082740656, 0.9185816498234448, 1.1205922169062292, 1.4813253724112023, 1.5101840248516019],
                                     'daysFromStart': 26 * [0.0],
                                     'hoursFromStart': [-1.651445647689541, -1.4863010829205867, -1.3211565181516327, -1.1560119533826787, -0.9908673886137245, -0.8257228238447705, -0.6605782590758164, -0.49543369430686224, -0.3302891295379082, -0.1651445647689541, 0.0] +
                                                       [0.1651445647689541, 0.3302891295379082, -1.651445647689541, -1.4863010829205867, -1.3211565181516327, -1.1560119533826787, -0.9908673886137245, -0.8257228238447705, -0.6605782590758164, -0.49543369430686224, -0.3302891295379082, -0.1651445647689541, 0.0, 0.1651445647689541, 0.3302891295379082],
                                     'dayOfMonth': 26 * [0.0], 'month': 26 * [1],
                                     'sequenceIdx': 2* [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                                     '__startPoint__': 9 * [True] + 4 * [False] + 9 * [True] + 4 * [
                                         False]})
        self.equalDfs(trainDf, trainDfCheck, floatApprox=True)

        valDfCheck = pd.DataFrame({'consumerId': 8 * [0] + 8 * [1],
                                   'hourOfDay': 2 * [9, 10, 11, 12, 13, 14, 15, 16],
                                   'dayOfWeek': 16 * [2],
                                   'powerUsage': [-1.1046369467118118, -1.0788912199788172, -1.1046369467118118, -1.0788912199788172] +
                                                 2 * [-1.1046369467118118] + [-1.0788912199788172, -0.9501625863138443, 0.9185816498234448, 1.1205922169062292, 1.4813253724112023, 1.5101840248516019, 1.4091787413102055, 1.2793148053284178, 1.163880195566825, 1.0484455858052322],
                                   'daysFromStart': 16 * [0.0],
                                   'hoursFromStart': 2 * [-0.1651445647689541, 0.0, 0.1651445647689541, 0.3302891295379082, 0.49543369430686224, 0.6605782590758164, 0.8257228238447705, 0.9908673886137245],
                                   'dayOfMonth': 16 * [0.0],
                                   'month': 16 * [1],
                                   'sequenceIdx': 2 * [9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0],
                                   '__startPoint__': 4 * [True] + 4 * [False] + 4 * [True] + 4 * [
                                       False]})
        self.equalDfs(valDf, valDfCheck, floatApprox=True)

        testDfCheck = pd.DataFrame({'consumerId': 8 * [0] + 8 * [1],
                                    'hourOfDay': 2 * [13, 14, 15, 16, 17, 18, 19, 20],
                                    'dayOfWeek': 16 * [2],
                                    'powerUsage': 2 * [-1.1046369467118118] + [
                                                -1.0788912199788172] + 2 * [-0.9501625863138443] +
                                                  2 * [-0.898671132847855] + [-0.9244168595808495, 1.4091787413102055, 1.2793148053284178, 1.163880195566825, 1.0484455858052322, 1.0340162595850335, 1.019586933364835, 1.0484455858052322, 1.1494508693466285],
                                    'daysFromStart': 16 * [0.0],
                                    'hoursFromStart': 2 * [0.49543369430686224, 0.6605782590758164, 0.8257228238447705, 0.9908673886137245, 1.1560119533826787, 1.3211565181516327, 1.4863010829205867, 1.651445647689541],
                                    'dayOfMonth': 16 * [0.0],
                                    'month': 16 * [1],
                                    'sequenceIdx': 2 * [13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0],
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

            # removing cols added during getElectricity_processed to be able to check
            set_ = set_.drop(columns=['sequenceIdx', '__startPoint__'])
            self.equalDfs(set_, checkSet, checkIndex=False, floatApprox=True)

    def testDataloader_data(self):
        device = torch.device('cuda')
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
        expectedInputs['allReals'] = [[[15.0000,  2.0000, -1.0789,  0.0000,  0.8257,  0.0000,  1.0000],
          [16.0000,  2.0000, -0.9502,  0.0000,  0.9909,  0.0000,  1.0000],
          [17.0000,  2.0000, -0.9502,  0.0000,  1.1560,  0.0000,  1.0000],
          [18.0000,  2.0000, -0.8987,  0.0000,  1.3212,  0.0000,  1.0000]],

         [[16.0000,  2.0000, -0.9502,  0.0000,  0.9909,  0.0000,  1.0000],
          [17.0000,  2.0000, -0.9502,  0.0000,  1.1560,  0.0000,  1.0000],
          [18.0000,  2.0000, -0.8987,  0.0000,  1.3212,  0.0000,  1.0000],
          [19.0000,  2.0000, -0.8987,  0.0000,  1.4863,  0.0000,  1.0000]],

         [[15.0000,  2.0000,  1.1639,  0.0000,  0.8257,  0.0000,  1.0000],
          [16.0000,  2.0000,  1.0484,  0.0000,  0.9909,  0.0000,  1.0000],
          [17.0000,  2.0000,  1.0340,  0.0000,  1.1560,  0.0000,  1.0000],
          [18.0000,  2.0000,  1.0196,  0.0000,  1.3212,  0.0000,  1.0000]],

         [[16.0000,  2.0000,  1.0484,  0.0000,  0.9909,  0.0000,  1.0000],
          [17.0000,  2.0000,  1.0340,  0.0000,  1.1560,  0.0000,  1.0000],
          [18.0000,  2.0000,  1.0196,  0.0000,  1.3212,  0.0000,  1.0000],
          [19.0000,  2.0000,  1.0484,  0.0000,  1.4863,  0.0000,  1.0000]]]
        expectedInputs['target'] = [[-1.0789, -0.9502, -0.9502, -0.8987],
                                    [-0.9502, -0.9502, -0.8987, -0.8987],
                                    [1.1639, 1.0484, 1.0340, 1.0196],
                                    [1.0484, 1.0340, 1.0196, 1.0484]]
        for key, value in testDataloader_inputs.items():
            self.equalTensors(value, torch.tensor(expectedInputs[key]).to(device), floatApprox=True,
                              checkType=False)

        expectedOutputs = [[-0.9502, -0.9502, -0.8987, -0.8987],
                           [-0.9502, -0.8987, -0.8987, -0.9244],
                           [1.0484, 1.0340, 1.0196, 1.0484],
                           [1.0340, 1.0196, 1.0484, 1.1495]]
        self.equalTensors(testDataloader_outputs, torch.tensor(expectedOutputs).to(device),
                          floatApprox=True, checkType=False)


# ---- run test
if __name__ == '__main__':
    unittest.main()
