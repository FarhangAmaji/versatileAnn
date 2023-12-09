"""
this is data preparation steps of electricity of 369 consumers
data also can be found at https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014
this data has ['date', 'consumerId', 'hourOfDay', 'dayOfWeek', 'powerUsage','daysFromStart',
               'hoursFromStart', 'dayOfMonth', 'month'] columns
there are many date cols and consumerId and powerUsage cols
this dataset has different consumer data which are treated as separate data sequences(NSeries)
"""
from typing import Union

from dataPrep.commonDatasets.commonDatasets_innerStepNUtils import _dataInfoAssert, _devTestModeData
# ---- imports
from dataPrep.dataloader import VAnnTsDataloader
from dataPrep.dataset import VAnnTsDataset
from dataPrep.normalizers_normalizerStack import NormalizerStack
from dataPrep.normalizers_singleColsNormalizer import SingleColsStdNormalizer, SingleColsLblEncoder
from dataPrep.utils import getDatasetFiles, diffColValuesFromItsMin_mainGroups, \
    setExclusionFlag_seqEnd_mainGroups, splitTrainValTest_mainGroup, _applyShuffleIfSeedExists
from utils.globalVars import tsStartPointColName
from utils.typeCheck import argValidator
from utils.vAnnGeneralUtils import varPasser, DotDict

# ----
embedderInputSize = 369  # 369 different consumerIds #kkk where should I handle this, which is a bit more general
dataInfo = DotDict({'timeIdx': 'hoursFromStart',
                    'mainGroups': ['consumerId'],
                    'consumerId': ['consumerId'],
                    'targets': ['powerUsage'],
                    'allReals': ['hourOfDay', 'dayOfWeek', 'powerUsage', 'daysFromStart',
                                 'hoursFromStart', 'daysFromStart', 'month']})
dataInfo['covariatesNum'] = len(dataInfo.allReals)
necessaryKeys = dataInfo.keys()


# ----

# goodToHave1
#  do these commonDataset fetcher need to have a have in order to provide unified functionality
@argValidator
def getElectricity_processed(*, dataInfo: Union[DotDict, dict], backcastLen=192, forecastLen=1,
                             trainRatio=.7, valRatio=.2,
                             shuffle=False, shuffleSeed=None, devTestMode=False):
    dataInfo = _dataInfoAssert(dataInfo, necessaryKeys)
    shuffle = _applyShuffleIfSeedExists(shuffle, shuffleSeed)
    df = getElectricity_data(backcastLen=backcastLen, forecastLen=forecastLen,
                             devTestMode=devTestMode)
    # creating sequenceIdx
    diffColValuesFromItsMin_mainGroups(df, dataInfo.mainGroups, col=dataInfo.timeIdx,
                                       resultColName='sequenceIdx')
    # assigning start points by excluding last 'backcastLen + forecastLen-1' of each consumer
    setExclusionFlag_seqEnd_mainGroups(df, dataInfo.mainGroups, backcastLen + forecastLen - 1,
                                       col='sequenceIdx',
                                       resultColName=tsStartPointColName)

    normalizer = NormalizerStack(
        SingleColsStdNormalizer(['powerUsage', 'daysFromStart', 'hoursFromStart', 'dayOfMonth']),
        SingleColsLblEncoder(dataInfo.mainGroups))
    # cccUsage
    #  dont get SingleColsLblEncoder mixed up with MainGroupNormalizers,
    #  this line just wants to convert mainGroup to 'int categories'
    normalizer.fitNTransform(df)
    kwargs = varPasser(localArgNames=['trainRatio', 'valRatio', 'shuffle', 'shuffleSeed'])
    setDfs = splitTrainValTest_mainGroup(df, dataInfo.mainGroups, seqLen=backcastLen + forecastLen,
                                         **kwargs)
    trainDf, valDf, testDf = setDfs['train'], setDfs['val'], setDfs['test']
    return trainDf, valDf, testDf, normalizer


def getElectricity_data(*, backcastLen, forecastLen, devTestMode):
    df = getDatasetFiles('electricity.csv')
    df = _devTestModeData(backcastLen, devTestMode, df, forecastLen)
    return df


# ----
class Electricity_deepArDataset(VAnnTsDataset):
    def __getitem__(self, idx):
        # bugPotentialCheck2 check this part
        inputs = {}
        inputs['consumerId'] = self.getBackForeCastData(idx, mode=self.castModes.singlePoint,
                                                        colsOrIndexes=self.dataInfo.consumerId)
        inputs['allReals'] = self.getBackForeCastData(idx, mode=self.castModes.backcast,
                                                      colsOrIndexes=self.dataInfo.allReals)
        inputs['target'] = self.getBackForeCastData(idx, mode=self.castModes.backcast,
                                                    colsOrIndexes=self.dataInfo.targets,
                                                    shiftForward=0)

        outputs = self.getBackForeCastData(idx, mode=self.castModes.backcast,
                                           colsOrIndexes=self.dataInfo.targets,
                                           shiftForward=1, canBeOutOfStartIndex=True)

        return inputs, outputs


# ---- dataloader
@argValidator
def getElectricityDataloaders(*, dataInfo: Union[DotDict, dict], backcastLen=192, forecastLen=1,
                              batchSize=64, trainRatio=.7, valRatio=.2, shuffle=False,
                              shuffleSeed=None, devTestMode=False):
    dataInfo = _dataInfoAssert(dataInfo, necessaryKeys)
    shuffle = _applyShuffleIfSeedExists(shuffle, shuffleSeed)
    # cccAlgo forecastLen==1 is for shifting
    kwargs = varPasser(
        localArgNames=['backcastLen', 'forecastLen', 'trainRatio', 'valRatio', 'shuffle',
                       'shuffleSeed', 'devTestMode', 'dataInfo'])
    trainDf, valDf, testDf, normalizer = getElectricity_processed(**kwargs)

    kwargs = {'backcastLen': backcastLen, 'forecastLen': forecastLen, 'indexes': None,
              'dataInfo': dataInfo}
    trainDataset = Electricity_deepArDataset(trainDf, mainGroups=dataInfo.mainGroups, **kwargs)
    valDataset = Electricity_deepArDataset(valDf, mainGroups=dataInfo.mainGroups, **kwargs)
    testDataset = Electricity_deepArDataset(testDf, mainGroups=dataInfo.mainGroups, **kwargs)
    del trainDf, valDf, testDf

    trainDataloader = VAnnTsDataloader(trainDataset, batch_size=batchSize)
    valDataloader = VAnnTsDataloader(valDataset, batch_size=batchSize)
    testDataloader = VAnnTsDataloader(testDataset, batch_size=batchSize)
    return trainDataloader, valDataloader, testDataloader, normalizer
