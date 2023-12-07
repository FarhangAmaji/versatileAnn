"""
this is data preparation steps of hourly electricity price forecasts (EPF) for France and Belgium markets
the data exists in data\datasets EPF_FR_BE.csv, EPF_FR_BE_futr.csv and EPF_FR_BE_static.csv files
"""
from typing import Union

from dataPrep.commonDatasets.commonDatasetsUtils import _dataInfoAssert
# ---- imports
from dataPrep.dataloader import VAnnTsDataloader
from dataPrep.dataset import VAnnTsDataset
from dataPrep.normalizers_multiColNormalizer import MultiColStdNormalizer
from dataPrep.normalizers_normalizerStack import NormalizerStack
from dataPrep.normalizers_singleColsNormalizer import SingleColsStdNormalizer
from dataPrep.utils import getDatasetFiles, splitTsTrainValTest_DfNNpDict, addCorrespondentRow, \
    rightPadDf, splitToNSeries, regularizeTsStartPoints
from utils.globalVars import tsStartPointColName
from utils.typeCheck import argValidator
from utils.vAnnGeneralUtils import regularizeBoolCol, varPasser, DotDict
from utils.warnings import Warn

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
                                      'shuffleSeed'])
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

    kwargs = varPasser(localArgNames=['trainDf', 'valDf', 'testDf', 'staticDf', 'aggColName'])
    trainDf, valDf, testDf = _splitNSeries_addStaticCorrespondentRows(**kwargs)
    return trainDf, valDf, testDf, normalizer

# mustHave1 should have inverser also, take a look at epfFrBeTests.testInvNormalizer
# ---- getEpfFrBe_processedInnerSteps
def _shuffleNRightpad_Compatibility(rightPadTrain, shuffle, shuffleSeed):
    if shuffleSeed:
        shuffle = True
    if shuffle:
        rightPadTrain = False
        Warn.warn('with shuffle on, rightPadTrain is not gonna be applied')
    return rightPadTrain, shuffle


def getEpfFrBe_data(*, backcastLen=110, forecastLen=22, devTestMode=False):
    mainDf, staticDf = getDatasetFiles('EPF_FR_BE.csv'), getDatasetFiles('EPF_FR_BE_static.csv')
    if devTestMode:  # goodToHave3 poor implementation
        mainDf = mainDf.loc[20 * (backcastLen + forecastLen):22 * (backcastLen + forecastLen)]
    return mainDf, staticDf


def _normalizerFit_split(mainDf, backcastLen, forecastLen, trainRatio=.7,
                         valRatio=.2, shuffle=False, shuffleSeed=None):
    normalizer = NormalizerStack(
        SingleColsStdNormalizer([*dataInfo.futureExogenousCols,
                                 *dataInfo.historyExogenousCols]),
        MultiColStdNormalizer(dataInfo.targets))
    # cccUsage SingleColsStdNormalizer normalize data of each col separately
    # cccUsage MultiColStdNormalizer normalize data of multiple cols based on all of those cols
    # cccUsage here we use MultiColStdNormalizer for targets('priceFr', 'priceBe'), which have same unit(Euro€)
    mainDf['mask'] = True
    normalizer.fitNTransform(mainDf)
    setDfs = splitTsTrainValTest_DfNNpDict(mainDf, trainRatio=trainRatio,
                                           valRatio=valRatio,
                                           seqLen=backcastLen + forecastLen,
                                           shuffle=shuffle, shuffleSeed=shuffleSeed)
    trainDf, valDf, testDf = setDfs['train'], setDfs['val'], setDfs['test']
    return trainDf, valDf, testDf, normalizer


def _rightPadTrain(rightPadTrain, trainDf, backcastLen, forecastLen):
    if rightPadTrain:
        trainDf = rightPadDf(trainDf, forecastLen - 1)
        # because rightPadDf adds pad=0, but the mask and tsStartPointColName must have bool data
        regularizeBoolCol(trainDf, 'mask')
        trainDf[tsStartPointColName] = True
        trainDf.loc[len(trainDf) - (backcastLen + forecastLen - 1):, tsStartPointColName] = False
    return trainDf


def _splitNSeries_addStaticCorrespondentRows(trainDf, valDf, testDf, staticDf, aggColName):
    newSets = []
    for set_ in [trainDf, valDf, testDf]:
        set_ = splitToNSeries(set_, dataInfo.targets, aggColName)
        addCorrespondentRow(set_, staticDf, dataInfo.targets, aggColName)
        regularizeTsStartPoints(set_)  # this is just for safety
        newSets += [set_]
    trainDf, valDf, testDf = newSets
    return trainDf, valDf, testDf


# ----
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
