import numpy as np
import pandas as pd

from dataPrep.normalizers_multiColNormalizer import MultiColStdNormalizer
from dataPrep.normalizers_normalizerStack import NormalizerStack
from dataPrep.normalizers_singleColsNormalizer import SingleColsStdNormalizer
from dataPrep.utils import splitTrainValTest_mainGroup, _applyShuffleIfSeedExists, \
    splitTsTrainValTest_DfNNpDict, rightPadDf, splitToNSeries, addCorrespondentRow, \
    regularizeTsStartPoints
from models.temporalFusionTransformers_components import getFastAi_empericalEmbeddingSize
from utils.globalVars import tsStartPointColName
from utils.vAnnGeneralUtils import DotDict, regularizeBoolCol
from utils.warnings import Warn


# ---- EpfFrBe

def _shuffleNRightpad_Compatibility(rightPadTrain, shuffle, shuffleSeed):
    shuffle = _applyShuffleIfSeedExists(shuffle, shuffleSeed)
    if shuffle:
        rightPadTrain = False
        Warn.warn('with shuffle on, rightPadTrain is not gonna be applied')
    return rightPadTrain, shuffle


def _normalizerFit_split(mainDf, dataInfo, backcastLen, forecastLen, trainRatio=.7,
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


def _splitNSeries_addStaticCorrespondentRows(trainDf, valDf, testDf, staticDf, aggColName,
                                             dataInfo):
    newSets = []
    for set_ in [trainDf, valDf, testDf]:
        set_ = splitToNSeries(set_, dataInfo.targets, aggColName)
        addCorrespondentRow(set_, staticDf, dataInfo.targets, aggColName)
        regularizeTsStartPoints(set_)  # this is just for safety
        newSets += [set_]
    trainDf, valDf, testDf = newSets
    return trainDf, valDf, testDf


# ---- Electricity
def _devTestModeData(backcastLen, devTestMode, df, forecastLen):
    if devTestMode:  # goodToHave3 poor implementation
        consumer1data = df[df['consumerId'] == 'MT_001'].reset_index(drop=True).loc[
                        0:15 + backcastLen + forecastLen]
        consumer2data = df[df['consumerId'] == 'MT_002'].reset_index(drop=True).loc[
                        0:15 + backcastLen + forecastLen]
        df = pd.concat([consumer1data, consumer2data]).reset_index(drop=True)
    return df


# ---- Stallion

# ---- utils
def _dataInfoAssert(dataInfo, necessaryKeys):
    if isinstance(dataInfo, dict):
        dataInfo = DotDict(dataInfo)
    if not all([key in dataInfo.keys() for key in necessaryKeys]):
        raise ValueError(f"dataInfo should provided with {necessaryKeys}")
    dataInfo = dataInfo.copy()
    return dataInfo
