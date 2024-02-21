"""
this is data preparation steps of hourly electricity price forecasts (EPF) for France and Belgium markets
the data exists in data\datasets EPF_FR_BE.csv, EPF_FR_BE_futr.csv and EPF_FR_BE_static.csv files
"""
# ---- imports
from typing import Union

from commonDatasets.commonDatasetsPrep.commonDatasetsPrep_innerStepNUtils import \
    _shuffleNRightpad_Compatibility, _normalizerFit_split, _rightPadTrain, \
    _splitNSeries_addStaticCorrespondentRows, _dataInfoAssert
from commonDatasets.getData import getDatasetFiles
from dataPrep.dataloader import VAnnTsDataloader
from dataPrep.dataset import VAnnTsDataset
from projectUtils.dataTypeUtils.dotDict_npDict import DotDict
from projectUtils.generalUtils import varPasser
from projectUtils.typeCheck import argValidator

# ----
dataInfo = DotDict({'futureExogenousCols': ['genForecast', 'weekDay'],
                    'historyExogenousCols': ['systemLoad'],
                    'staticExogenousCols': ['market0', 'market1'],
                    'targets': ['priceFr', 'priceBe'],
                    'unifiedTargets': ['price']})
necessaryKeys = dataInfo.keys()


# ---- getEpfFrBe_processed
@argValidator
def getEpfFrBe_processed(*, dataInfo: Union[DotDict, dict], backcastLen=110, forecastLen=22,
                         trainRatio=.7, valRatio=.2, rightPadTrain: bool = True,
                         aggColName: str = 'price', shuffle=False, shuffleSeed=None,
                         devTestMode=False):
    dataInfo = _dataInfoAssert(dataInfo, necessaryKeys)
    rightPadTrain, shuffle = _shuffleNRightpad_Compatibility(rightPadTrain, shuffle, shuffleSeed)

    mainDf, staticDf = getEpfFrBe_data(backcastLen=backcastLen, forecastLen=forecastLen,
                                       devTestMode=devTestMode)
    kwargs = varPasser(localArgNames=['mainDf', 'backcastLen', 'forecastLen', 'valRatio', 'shuffle',
                                      'shuffleSeed', 'dataInfo'])
    if rightPadTrain:
        # cccAlgo
        #  note if we have rightPadTrain==True, we want to get this order, 'testDf, valDf, trainDf',
        #  therefore have to pass trainRatio=1-(trainRatio+valRatio). so the last indexes of mainDf are dedicated
        #  to trainDf and the paddings are after them.
        #  note in splitTsTrainValTest_DfNNpDict the first data is usually train, next val and test
        testDf, valDf, trainDf, normalizer = _normalizerFit_split(
            trainRatio=1 - (trainRatio + valRatio), **kwargs)
    else:
        trainDf, valDf, testDf, normalizer = _normalizerFit_split(
            trainRatio=trainRatio, **kwargs)
        # bugPotentialCheck2
        #  this may be a bug for splitTsTrainValTest_DfNNpDict, with trainRatio=.7 and valRatio=.2
        #  with 13 points to have as startpoint returns empty testDf

    kwargs = varPasser(localArgNames=['rightPadTrain', 'trainDf', 'backcastLen', 'forecastLen'])
    trainDf = _rightPadTrain(**kwargs)

    kwargs = varPasser(
        localArgNames=['trainDf', 'valDf', 'testDf', 'staticDf', 'aggColName', 'dataInfo'])
    trainDf, valDf, testDf = _splitNSeries_addStaticCorrespondentRows(**kwargs)
    return trainDf, valDf, testDf, normalizer


# mustHave1 should have inverser also, take a look at epfFrBeTests.testInvNormalizer


def getEpfFrBe_data(*, backcastLen=110, forecastLen=22, devTestMode=False):
    mainDf, staticDf = getDatasetFiles('EPF_FR_BE.csv'), getDatasetFiles('EPF_FR_BE_static.csv')
    if devTestMode:  # goodToHave3 poor implementation
        mainDf = mainDf.loc[20 * (backcastLen + forecastLen):22 * (backcastLen + forecastLen)]
    return mainDf, staticDf


class EpfFrBeDataset(VAnnTsDataset):
    def __getitem__(self, idx):
        inputs = {}
        inputs['target'] = self.getBackForeCastData(idx, mode=self.castModes.backcast,
                                                    colsOrIndexes=self.dataInfo.unifiedTargets)
        inputs['mask'] = self.getBackForeCastData(idx, mode=self.castModes.backcast,
                                                  colsOrIndexes=['mask'])
        inputs['historyExogenous'] = self.getBackForeCastData(idx, mode=self.castModes.backcast,
                                                              colsOrIndexes=self.dataInfo.historyExogenousCols)
        inputs['staticExogenous'] = self.getBackForeCastData(idx, mode=self.castModes.singlePoint,
                                                             colsOrIndexes=self.dataInfo.staticExogenousCols)
        inputs['futureExogenous'] = self.getBackForeCastData(idx, mode=self.castModes.fullcast,
                                                             colsOrIndexes=self.dataInfo.futureExogenousCols)

        outputs = {}
        outputs['output'] = self.getBackForeCastData(idx, mode=self.castModes.forecast,
                                                     colsOrIndexes=self.dataInfo.unifiedTargets)
        outputs['outputMask'] = self.getBackForeCastData(idx, mode=self.castModes.forecast,
                                                         colsOrIndexes=['mask'])
        return inputs, outputs


# ---- dataloader
@argValidator
def getEpfFrBeDataloaders(*, dataInfo: Union[DotDict, dict], backcastLen=110, forecastLen=22,
                          batchSize=64, trainRatio=.7, valRatio=.2, rightPadTrain=True,
                          aggColName: str = 'price', shuffle=False, shuffleSeed=None,
                          devTestMode=False):
    # goodToHave2
    #  this is very slow, I should check which step is bottlenecking
    #  also find a way to make it faster
    dataInfo = _dataInfoAssert(dataInfo, necessaryKeys)
    rightPadTrain, shuffle = _shuffleNRightpad_Compatibility(rightPadTrain, shuffle, shuffleSeed)

    kwargs = varPasser(
        localArgNames=['backcastLen', 'forecastLen', 'trainRatio', 'valRatio', 'rightPadTrain',
                       'aggColName', 'devTestMode', 'shuffle', 'shuffleSeed', 'dataInfo'])
    trainDf, valDf, testDf, normalizer = getEpfFrBe_processed(**kwargs)

    kwargs = {'backcastLen': backcastLen, 'forecastLen': forecastLen,
              'mainGroups': [aggColName + 'Type'],
              'indexes': None, 'dataInfo': dataInfo}
    trainDataset = EpfFrBeDataset(trainDf, **kwargs)
    valDataset = EpfFrBeDataset(valDf, **kwargs)
    testDataset = EpfFrBeDataset(testDf, **kwargs)
    del trainDf, valDf, testDf

    trainDataloader = VAnnTsDataloader(trainDataset, batch_size=batchSize)
    valDataloader = VAnnTsDataloader(valDataset, batch_size=batchSize)
    testDataloader = VAnnTsDataloader(testDataset, batch_size=batchSize)
    return trainDataloader, valDataloader, testDataloader, normalizer
