"""
this is data preparation steps of electricity of 369 consumers
data also can be found at https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014
this data has ['date', 'consumerId', 'hourOfDay', 'dayOfWeek', 'powerUsage','daysFromStart',
               'hoursFromStart', 'dayOfMonth', 'month'] columns
there are many date cols and consumerId and powerUsage cols
this dataset has different consumer data which are treated as separate data sequences(NSeries)
"""
import pandas as pd

# ---- imports
from dataPrep.dataloader import VAnnTsDataloader
from dataPrep.dataset import VAnnTsDataset
from dataPrep.normalizers_normalizerStack import NormalizerStack
from dataPrep.normalizers_singleColsNormalizer import SingleColsStdNormalizer, SingleColsLblEncoder
from dataPrep.utils import getDatasetFiles, diffColValuesFromItsMin_mainGroups, \
    setExclusionFlag_seqEnd_mainGroups, splitTrainValTest_mainGroup
from utils.globalVars import tsStartPointColName
from utils.vAnnGeneralUtils import varPasser

# ----
embedderInputSize = 369  # 369 different consumerIds
allReals = ['hourOfDay', 'dayOfWeek', 'powerUsage', 'daysFromStart', 'hoursFromStart',
            'daysFromStart', 'month']
covariatesNum = len(allReals)

timeIdx = 'hoursFromStart'
mainGroups = ['consumerId']
target = ['powerUsage']
backcastLen = 192
forecastLen = 1  # ccc this is for the shift
datasetKwargs = {'mainGroups': mainGroups, 'consumerId': ['consumerId'], 'target': target,
                 'allReals': allReals}


# ----

# goodToHave1
#  do these commonDataset fetcher need to have a have in order to provide unified functionality
def getElectricity_processed(*, backcastLen=192, forecastLen=1,
                             trainRatio=.7, valRatio=.2,
                             shuffle=False, shuffleSeed=None, devTestMode=False):
    df = getElectricity_data(backcastLen=backcastLen, forecastLen=forecastLen,
                             devTestMode=devTestMode)
    # creating sequenceIdx
    diffColValuesFromItsMin_mainGroups(df, mainGroups, col=timeIdx, resultColName='sequenceIdx')
    # assigning start points by excluding last 'backcastLen + forecastLen-1' of each consumer
    setExclusionFlag_seqEnd_mainGroups(df, mainGroups, backcastLen + forecastLen - 1,
                                       col='sequenceIdx',
                                       resultColName=tsStartPointColName)

    normalizer = NormalizerStack(
        SingleColsStdNormalizer(['powerUsage', 'daysFromStart', 'hoursFromStart', 'dayOfMonth']),
        SingleColsLblEncoder(
            mainGroups))  # cccUsage dont get it mixed up to with MainGroupNormalizers, this line just wants to convert mainGroup to 'int categories'
    normalizer.fitNTransform(df)
    kwargs = varPasser(localArgNames=['trainRatio', 'valRatio', 'shuffle', 'shuffleSeed'])
    setDfs = splitTrainValTest_mainGroup(df, mainGroups, seqLen=backcastLen + forecastLen, **kwargs)
    trainDf, valDf, testDf = setDfs['train'], setDfs['val'], setDfs['test']
    return trainDf, valDf, testDf, normalizer


def getElectricity_data(*, backcastLen, forecastLen, devTestMode):
    df = getDatasetFiles('electricity.csv')
    df = _devTestModeData(backcastLen, devTestMode, df, forecastLen)
    return df


def _devTestModeData(backcastLen, devTestMode, df, forecastLen):
    if devTestMode:  # goodToHave3 poor implementation
        consumer1data = df[df['consumerId'] == 'MT_001'].reset_index(drop=True).loc[
                        0:15 + backcastLen + forecastLen]
        consumer2data = df[df['consumerId'] == 'MT_002'].reset_index(drop=True).loc[
                        0:15 + backcastLen + forecastLen]
        df = pd.concat([consumer1data, consumer2data]).reset_index(drop=True)
    return df


# ----
class Electricity_deepArDataset(VAnnTsDataset):
    def __getitem__(self, idx):
        # bugPotentialCheck2 check this part
        inputs = {}
        inputs['consumerId'] = self.getBackForeCastData(idx, mode=self.castModes.singlePoint,
                                                        colsOrIndexes=self.additionalInfo[
                                                            'consumerId'])
        inputs['allReals'] = self.getBackForeCastData(idx, mode=self.castModes.backcast,
                                                      colsOrIndexes=self.additionalInfo['allReals'])
        inputs['target'] = self.getBackForeCastData(idx, mode=self.castModes.backcast,
                                                    colsOrIndexes=self.additionalInfo['target'],
                                                    shiftForward=0)

        outputs = self.getBackForeCastData(idx, mode=self.castModes.backcast,
                                           colsOrIndexes=self.additionalInfo['target'],
                                           shiftForward=1, canBeOutOfStartIndex=True)

        return inputs, outputs


# ---- dataloader
def getElectricityDataloaders(*, backcastLen=192, forecastLen=1, batchSize=64,
                              trainRatio=.7, valRatio=.2,
                              shuffle=False, shuffleSeed=None, devTestMode=False):
    kwargs=varPasser(localArgNames=['backcastLen', 'forecastLen', 'trainRatio',
                                    'valRatio', 'shuffle', 'shuffleSeed', 'devTestMode'])
    trainDf, valDf, testDf, normalizer = getElectricity_processed(**kwargs)

    kwargs = {'backcastLen': backcastLen, 'forecastLen': forecastLen, 'indexes': None,
              'additionalInfo': datasetKwargs}
    trainDataset = Electricity_deepArDataset(trainDf, mainGroups=mainGroups, **kwargs)
    valDataset = Electricity_deepArDataset(valDf, mainGroups=mainGroups, **kwargs)
    testDataset = Electricity_deepArDataset(testDf, mainGroups=mainGroups, **kwargs)
    del trainDf, valDf, testDf

    trainDataloader = VAnnTsDataloader(trainDataset, batch_size=batchSize)
    valDataloader = VAnnTsDataloader(valDataset, batch_size=batchSize)
    testDataloader = VAnnTsDataloader(testDataset, batch_size=batchSize)
    return trainDataloader, valDataloader, testDataloader, normalizer
