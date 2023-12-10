"""
https://www.kaggle.com/datasets/utathya/future-volume-prediction
this dataset is usually treated as multiple series(NSeries) with mainGroups of `agency` and
`sku`(stock keeping units (SKU))
it has also these features:
    date:'date'(months)
    production volumes:'sku'(volume that agency has stocked),'volume', 'industryVolume', 'sodaVolume'
    weather temperature:'avgMaxTemp'
    discounted price:'priceRegular', 'priceActual', 'discount', 'discountInPercent'
    population info:'avgPopulation2017', 'avgYearlyHouseholdIncome2017'
    specialDays:'easterDay',  'goodFriday', 'newYear', 'christmas', 'laborDay', 'independenceDay',
                'revolutionDayMemorial', 'regionalGames', 'fifaU17WorldCup', 'footballGoldCup',
                'beerCapital', 'musicFest'
note the encoder and decoder lengths are not fixed and differ for each point. therefore there are min
and max for them, also rightPadIfShorter=True is used in dataset
"""
from typing import Union

# ---- imports
import pandas as pd
import torch

from dataPrep.commonDatasets.commonDatasets_innerStepNUtils import _addTargetMeanNStd, \
    _addEmbeddingSizes, _addTimeVarying_EncoderNDecoder, _getFullOrNotConditions, \
    _addAllReals, _normalizingAllReals, _makingTimeIdx, \
    _addingSomeOtherFeatures, _dataInfoAssert, _devStallionTestModeData, _splitFullN_nonFullEqually
from dataPrep.dataloader import VAnnTsDataloader
from dataPrep.dataset import VAnnTsDataset
from dataPrep.normalizers_mainGroupNormalizers import MainGroupSingleColsStdNormalizer
from dataPrep.normalizers_normalizerStack import NormalizerStack
from dataPrep.normalizers_singleColsNormalizer import SingleColsLblEncoder
from dataPrep.utils import getDatasetFiles, _applyShuffleIfSeedExists
# kkk replace this from a particular model, to general embedding files
from dataPrep.utils import rightPadIfShorter_df, rightPadIfShorter_npArray
from utils.typeCheck import argValidator
from utils.vAnnGeneralUtils import DotDict, varPasser

# ----
# kkk explain timeVarying|static;real|categorical;known|unknown
# timeIdx = 'timeIdx'
# mainGroups = ['agency', 'sku']
# targets = ['volume']
# specialDays = ['easterDay',
#                'goodFriday', 'newYear', 'christmas', 'laborDay', 'independenceDay',
#                'revolutionDayMemorial', 'regionalGames', 'fifaU17WorldCup',
#                'footballGoldCup', 'beerCapital', 'musicFest']
# categoricalGroupVariables = {"specialDays": specialDays}
# categoricalSingularVariables = ["agency", "sku", "month"]

# staticCategoricals = ["agency", "sku"]
# staticReals = ["avgPopulation2017", "avgYearlyHouseholdIncome2017"]
# timeVarying_knownCategoricals = ["specialDays", "month"]
# timeVarying_knownReals = ["timeIdx", "priceRegular", "discountInPercent"]
# timeVarying_unknownCategoricals = []
# timeVarying_unknownReals = ["volume", "logVolume", "industryVolume", "sodaVolume", "avgMaxTemp",
#                            "avgVolumeByAgency", "avgVolumeBySku"]

dataInfo = DotDict({'timeIdx': 'timeIdx',
                    'mainGroups': ['agency', 'sku'],
                    'targets': ['volume'],
                    'categoricalGroupVariables': {
                        "specialDays": ['easterDay', 'goodFriday', 'newYear', 'christmas',
                                        'laborDay', 'independenceDay', 'revolutionDayMemorial',
                                        'regionalGames', 'fifaU17WorldCup', 'footballGoldCup',
                                        'beerCapital', 'musicFest']},
                    'categoricalSingularVariables': ["agency", "sku", "month"],
                    'staticCategoricals': ["agency", "sku"],
                    'staticReals': ["avgPopulation2017", "avgYearlyHouseholdIncome2017"],
                    'timeVarying_knownCategoricals': ["specialDays", "month"],
                    'timeVarying_knownReals': ["timeIdx", "priceRegular", "discountInPercent"],
                    'timeVarying_unknownCategoricals': [],
                    'timeVarying_unknownReals': ["volume", "logVolume", "industryVolume",
                                                 "sodaVolume", "avgMaxTemp", "avgVolumeByAgency",
                                                 "avgVolumeBySku"]})
necessaryKeys = dataInfo.keys()


@argValidator
def getStallion_processed(*, dataInfo: Union[DotDict, dict], maxEncoderLength=24,
                          maxPredictionLength=6, minEncoderLength=12, minPredictionLength=1,
                          trainRatio=.7, valRatio=.2, shuffle=False, shuffleSeed=None,
                          devTestMode=False):
    dataInfo = _dataInfoAssert(dataInfo, necessaryKeys)
    shuffle = _applyShuffleIfSeedExists(shuffle, shuffleSeed)
    # dataInfo = dataInfo['dataInfo']
    # mainGroups = dataInfo['mainGroups']
    # targets = dataInfo['targets']
    # categoricalGroupVariables = dataInfo['categoricalGroupVariables']
    # categoricalSingularVariables = dataInfo['categoricalSingularVariables']
    # staticCategoricals = dataInfo['staticCategoricals']
    # staticReals = dataInfo['staticReals']
    # timeVarying_knownCategoricals = dataInfo['timeVarying_knownCategoricals']
    # timeVarying_knownReals = dataInfo['timeVarying_knownReals']
    # timeVarying_unknownCategoricals = dataInfo['timeVarying_unknownCategoricals']
    # timeVarying_unknownReals = dataInfo['timeVarying_unknownReals']

    df = getStallion_data(devTestMode, maxEncoderLength, maxPredictionLength)
    _makingTimeIdx(df)
    _addingSomeOtherFeatures(dataInfo, df)

    df = df.sort_values(dataInfo.mainGroups + [dataInfo.timeIdx]).reset_index(drop=True)
    normalizer = NormalizerStack(  # LStl1
        MainGroupSingleColsStdNormalizer(df, dataInfo.mainGroups, dataInfo.targets),
        SingleColsLblEncoder(
            ['sku', 'agency', 'month', *dataInfo.categoricalGroupVariables['specialDays']]))
    normalizer.fitNTransform(df)
    # cccAlgo
    #  pay attention if the MainGroupSingleColsStdNormalizer was passed after
    #  SingleColsLblEncoder, because it sets up uniquecombos first and after SingleColsLblEncoder's
    #  fitNTransform those values would have changed,; we have to pass it before the SingleColsLblEncoder
    _addTargetMeanNStd(dataInfo, df, normalizer)
    _addEmbeddingSizes(dataInfo, normalizer)
    _addTimeVarying_EncoderNDecoder(dataInfo)

    _addAllReals(dataInfo)
    _normalizingAllReals(dataInfo, df, normalizer)

    _getFullOrNotConditions(dataInfo, df, maxEncoderLength, maxPredictionLength, minEncoderLength,
                            minPredictionLength, normalizer)
    # setsDf = _splitFullN_nonFullEqually(dataInfo, df, maxEncoderLength, maxPredictionLength)

    setsDf = _splitFullN_nonFullEqually(dataInfo, df, maxEncoderLength, maxPredictionLength,
                                        trainRatio, valRatio, shuffle, shuffleSeed)
    trainDf, valDf, testDf = setsDf['train'], setsDf['val'], setsDf['test']

    # adding predictLens to dataInfo
    predictLens = varPasser(
        localArgNames=['minPredictionLength', 'maxPredictionLength', 'maxEncoderLength',
                       'minEncoderLength'])
    dataInfo._data.update(predictLens)
    return trainDf, valDf, testDf, normalizer, dataInfo


def getStallion_data(devTestMode, maxEncoderLength, maxPredictionLength):
    df = getDatasetFiles('stallion.csv')
    df = _devStallionTestModeData(devTestMode, df, maxEncoderLength, maxPredictionLength)
    return df


# ----
class StallionTftDataset(VAnnTsDataset):
    def __getitem__(self, idx):
        mainGroupData, relIdx = self._IdxNdataToLook_WhileFetching(idx)
        encoderLength = int(mainGroupData['encoderLength'][relIdx])
        decoderLength = int(mainGroupData['decoderLength'][relIdx])

        inputs = {}
        inputs['encoderLengths'] = self.getBackForeCastData(idx, mode=self.castModes.singlePoint,
                                                            colsOrIndexes=['encoderLength'],
                                                            rightPadIfShorter=True)
        inputs['decoderLengths'] = self.getBackForeCastData(idx, mode=self.castModes.singlePoint,
                                                            colsOrIndexes=['decoderLength'],
                                                            rightPadIfShorter=True)

        inputs['allReals'] = {}
        for ar in self.dataInfo.allReals:
            inputs['allReals'][ar] = self.getBackForeCastData(idx, mode=self.castModes.fullcast,
                                                              colsOrIndexes=[ar],
                                                              rightPadIfShorter=True)

        fullcastLen = self.backcastLen + self.forecastLen
        relativeTimeIdx = pd.Series([i for i in range(-encoderLength, decoderLength)])
        relativeTimeIdx /= self.dataInfo.maxEncoderLength
        relativeTimeIdx = rightPadIfShorter_df(relativeTimeIdx, fullcastLen)
        inputs['allReals']['relativeTimeIdx'] = torch.tensor(relativeTimeIdx.values)

        encoderLengthRows = pd.Series(
            [(encoderLength - .5 * self.dataInfo.maxEncoderLength)
             for i in range(encoderLength + decoderLength)])
        encoderLengthRows /= self.dataInfo.maxEncoderLength * 2
        encoderLengthRows = rightPadIfShorter_df(encoderLengthRows, fullcastLen)
        inputs['allReals']['encoderLength'] = torch.tensor(encoderLengthRows.values)

        inputs['categorical'] = {}
        inputs['categorical']['singular'] = {}
        inputs['categorical']['groups'] = {}
        for sc in self.dataInfo.categoricalSingularVariables:
            inputs['categorical']['singular'][sc] = self.getBackForeCastData(idx,
                                                                     mode=self.castModes.fullcast,
                                                                     colsOrIndexes=[sc],
                                                                     rightPadIfShorter=True)

        for gc, gcVal in self.dataInfo.categoricalGroupVariables.items():
            inputs['categorical']['groups'][gc] = {}
            for gc1 in gcVal:
                inputs['categorical']['groups'][gc][gc1] = self.getBackForeCastData(idx,
                                                                    mode=self.castModes.fullcast,
                                                                    colsOrIndexes=[gc1],
                                                                    rightPadIfShorter=True)

        outputs = {}

        outputs['volume'] = mainGroupData['volume'][
                            relIdx + encoderLength:relIdx + encoderLength + decoderLength]
        outputs['volume'] = rightPadIfShorter_npArray(outputs['volume'], fullcastLen)

        return inputs, outputs


# ---- dataloader
@argValidator
def getStallion_TftDataloaders(*, dataInfo: Union[DotDict, dict], maxEncoderLength=24,
                               maxPredictionLength=6, minEncoderLength=12, minPredictionLength=1,
                               mainGroups=['agency', 'sku'], batchSize=64, trainRatio=.7,
                               valRatio=.2, shuffle=False, shuffleSeed=None, devTestMode=False):
    dataInfo = _dataInfoAssert(dataInfo, necessaryKeys)
    shuffle = _applyShuffleIfSeedExists(shuffle, shuffleSeed)
    kwargs = varPasser(localArgNames=['maxEncoderLength', 'maxPredictionLength', 'minEncoderLength',
                                      'minPredictionLength', 'dataInfo', 'trainRatio', 'valRatio',
                                      'shuffle', 'shuffleSeed', 'devTestMode'])
    trainDf, valDf, testDf, normalizer, dataInfo = getStallion_processed(**kwargs)

    kwargs = {'backcastLen': maxEncoderLength, 'forecastLen': maxPredictionLength,
              'mainGroups': mainGroups, 'dataInfo': dataInfo}
    trainDataset = StallionTftDataset(trainDf, **kwargs)
    valDataset = StallionTftDataset(valDf, **kwargs)
    testDataset = StallionTftDataset(testDf, **kwargs)
    del trainDf, valDf, testDf

    trainDataloader = VAnnTsDataloader(trainDataset, batch_size=batchSize)
    valDataloader = VAnnTsDataloader(valDataset, batch_size=batchSize)
    testDataloader = VAnnTsDataloader(testDataset, batch_size=batchSize)
    return trainDataloader, valDataloader, testDataloader, normalizer
