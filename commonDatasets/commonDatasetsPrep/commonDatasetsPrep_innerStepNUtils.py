import numpy as np
import pandas as pd

from dataPrep.normalizers_multiColNormalizer import MultiColStdNormalizer
from dataPrep.normalizers_normalizerStack import NormalizerStack
from dataPrep.normalizers_singleColsNormalizer import SingleColsStdNormalizer
from dataPrep.utils import splitTrainValTest_mainGroup, _applyShuffleIfSeedExists, \
    splitTsTrainValTest_DfNNpDict, rightPadDf, splitToNSeries, addCorrespondentRow, \
    regularizeTsStartPoints
from models.temporalFusionTransformers_components import getFastAi_empericalEmbeddingSize
from utils.dataTypeUtils.df_series import regularizeBoolCol
from utils.dataTypeUtils.dotDict_npDict import DotDict
from utils.globalVars import tsStartPointColName
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
def _devElectricityTestModeData(backcastLen, devTestMode, df, forecastLen):
    if devTestMode:  # goodToHave3 poor implementation
        consumer1data = df[df['consumerId'] == 'MT_001'].reset_index(drop=True).loc[
                        0:15 + backcastLen + forecastLen]
        consumer2data = df[df['consumerId'] == 'MT_002'].reset_index(drop=True).loc[
                        0:15 + backcastLen + forecastLen]
        df = pd.concat([consumer1data, consumer2data]).reset_index(drop=True)
    return df


# ---- Stallion
def _devStallionTestModeData(devTestMode, df, maxEncoderLength, maxPredictionLength):
    if devTestMode:  # goodToHave3 poor implementation
        consumer1data = df[(df['agency'] == 'Agency_01') & (df['sku'] == 'SKU_01')].reset_index(
            drop=True).loc[0:15 + maxEncoderLength + maxPredictionLength]
        consumer2data = df[(df['agency'] == 'Agency_01') & (df['sku'] == 'SKU_02')].reset_index(
            drop=True).loc[0:15 + maxEncoderLength + maxPredictionLength]
        df = pd.concat([consumer1data, consumer2data]).reset_index(drop=True)
    return df


def _makingTimeIdx(df):
    df["timeIdx"] = df["date"].dt.year * 12 + df["date"].dt.month
    df["timeIdx"] -= df["timeIdx"].min()


def _addingSomeOtherFeatures(dataInfo, df):
    df["month"] = df.date.dt.month.astype(str)
    df["logVolume"] = np.log(df.volume + 1e-8)
    df["avgVolumeBySku"] = df.groupby(["timeIdx", "sku"], observed=True).volume.transform("mean")
    df["avgVolumeByAgency"] = df.groupby(["timeIdx", "agency"], observed=True).volume.transform(
        "mean")
    # cccAlgo
    #  relativeTimeIdx in dataset creates "range(-encoderLength, decoderLength)" sequence for each point
    df['relativeTimeIdx'] = 0
    # cccAlgo
    #  encoderLength is not fixed and is different for each point
    df['encoderLength'] = 0
    dataInfo.timeVarying_knownReals += ['relativeTimeIdx']
    dataInfo.staticReals += ['encoderLength']


def _addTargetMeanNStd(dataInfo, df, normalizer):
    # cccAlgo
    #  setMeanNStd_ofMainGroups needs to have unTransformed mainGroups so we inverseTransform them
    #  and transform them again
    for col in dataInfo.mainGroups:
        df[col] = normalizer.inverseTransformCol(df, col)

    # add 'volumeMean' and 'volumeStd'
    normalizer.uniqueNormalizers[0].setMeanNStd_ofMainGroups(df)  # LStl2
    dataInfo.staticReals.extend(['volumeMean', 'volumeStd'])

    for col in dataInfo.mainGroups:
        df[col] = normalizer.transformCol(df, col)


def _addEmbeddingSizes(dataInfo, normalizer):
    categoricalClasses = normalizer.uniqueNormalizers[1].getClasses()
    embeddingSizes = {}
    for col in ['sku', 'month', 'agency']:
        embeddingSizes[col] = [len(categoricalClasses[col]),
                               getFastAi_empericalEmbeddingSize(len(categoricalClasses[col]))]
    classesLen = 0
    for col in dataInfo.categoricalGroupVariables['specialDays']:
        classesLen += len(categoricalClasses[col])
    embeddingSizes['specialDays'] = [classesLen, getFastAi_empericalEmbeddingSize(classesLen)]
    dataInfo['embeddingSizes'] = embeddingSizes


def _addTimeVarying_EncoderNDecoder(dataInfo):
    # timeVaryingEncoder = timeVarying knowns + timeVarying unkowns
    dataInfo['timeVarying_categoricalsEncoder'] = list(
        set(dataInfo.timeVarying_knownCategoricals + dataInfo.timeVarying_unknownCategoricals))
    dataInfo['timeVarying_realsEncoder'] = list(set(dataInfo.timeVarying_knownReals +
                                                    dataInfo.timeVarying_unknownReals))
    # timeVaryingDecoder = timeVarying knowns
    dataInfo['timeVarying_categoricalsDecoder'] = dataInfo.timeVarying_knownCategoricals[:]
    dataInfo['timeVarying_realsDecoder'] = dataInfo.timeVarying_knownReals[:]


def _addAllReals(dataInfo):
    dataInfo['allReals'] = list(set(dataInfo.staticReals + dataInfo.timeVarying_knownReals +
                                    dataInfo.timeVarying_unknownReals))
    dataInfo['allReals'] = list(set(dataInfo['allReals']) - set(dataInfo.targets))


def _normalizingAllReals(dataInfo, df, normalizer):
    # cccAlgo
    #  allReals is added to normalizers here(separate from line #LStl1 in stallion.py) because
    #  allReals needs to have staticReals and in line #LStl2 _addTargetMeanNStd 'volumeMean' and
    #  'volumeStd' get added to staticReals
    normalizer.addNormalizer(SingleColsStdNormalizer(dataInfo['allReals']))
    normalizer.uniqueNormalizers[-1].fitNTransform(df)


def _getFullOrNotConditions(dataInfo, df, maxEncoderLength, maxPredictionLength, minEncoderLength,
                            minPredictionLength, normalizer):
    df[dataInfo.timeIdx] = normalizer.inverseTransformCol(df, dataInfo.timeIdx)
    # cccAlgo
    #  timeIdxDiffWith_maxTimeIdx_ofMainGroup needs untransformed timeIdx so we inverseTransform
    #  it and transform it again
    maxTimeIdx_ofMainGroup = df.groupby(dataInfo.mainGroups)[dataInfo.timeIdx].transform('max')
    timeIdxDiffWith_maxTimeIdx_ofMainGroup = maxTimeIdx_ofMainGroup - df[dataInfo.timeIdx] + 1
    # goodToHave3 explain whats +1
    df[dataInfo.timeIdx] = normalizer.transformCol(df, dataInfo.timeIdx)

    for col, maxLength in zip(['encoderLength', 'decoderLength', 'sequenceLength'],
                              [maxEncoderLength, maxPredictionLength,
                               maxEncoderLength + maxPredictionLength]):
        df[col] = timeIdxDiffWith_maxTimeIdx_ofMainGroup
        df[col] = df[col].apply(lambda x: min(x, maxLength))

    fullLenConditions = (df['sequenceLength'] == maxEncoderLength + maxPredictionLength)
    df['fullLenConditions'] = np.where(fullLenConditions, True, False)

    notFullLen_ButGreaterThan_MinEncoderNPredictLen = (df['fullLenConditions'] == False) & (
            df['encoderLength'] >= minEncoderLength) & (df['decoderLength'] >= minPredictionLength)
    df['notFullLen_ButGreaterThan_MinEncoderNPredictLen'] = np.where(
        notFullLen_ButGreaterThan_MinEncoderNPredictLen, True, False)


def _splitFullN_nonFullEqually(dataInfo, df, maxEncoderLength, maxPredictionLength, trainRatio,
                               valRatio, shuffle, shuffleSeed):
    fullLenDfSets = splitTrainValTest_mainGroup(df, dataInfo.mainGroups,
                                                trainRatio=trainRatio, valRatio=valRatio,
                                                seqLen=maxEncoderLength + maxPredictionLength,
                                                shuffle=shuffle, shuffleSeed=shuffleSeed,
                                                conditions=['fullLenConditions==True'])
    nonFullLenDfSets = splitTrainValTest_mainGroup(df, dataInfo.mainGroups,
                                                   trainRatio=trainRatio, valRatio=valRatio,
                                                   seqLen=maxEncoderLength + maxPredictionLength,
                                                   shuffle=shuffle, shuffleSeed=shuffleSeed,
                                                   conditions=[
                                                       'notFullLen_ButGreaterThan_MinEncoderNPredictLen==True'],
                                                   tailIndexes_evenShorter=True)
    setsDf = {}
    for set_ in ['train', 'val', 'test']:
        concatDf = pd.concat([fullLenDfSets[set_], nonFullLenDfSets[set_]])
        # cccAlgo to understand lines below take look at devDocs\codeClarifier\commonDatasets #LStl3
        concatDf[tsStartPointColName] = concatDf.groupby(concatDf.index)[
            tsStartPointColName].transform('any')
        concatDf = concatDf[~concatDf.index.duplicated(keep='first')]
        setsDf[set_] = concatDf

    return setsDf


# ---- utils
def _dataInfoAssert(dataInfo, necessaryKeys):
    if isinstance(dataInfo, dict):
        dataInfo = DotDict(dataInfo)
    if not all([key in dataInfo.keys() for key in necessaryKeys]):
        raise ValueError(f"dataInfo should provided with {necessaryKeys}")
    dataInfo = dataInfo.copy()
    return dataInfo
