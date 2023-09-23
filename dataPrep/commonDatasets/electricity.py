#%% imports
"""
this is data preparation steps of electricity of 369 consumers
data also can be found at https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014
this data has ['date', 'consumerId', 'hourOfDay', 'dayOfWeek', 'powerUsage','daysFromStart', 
               'hoursFromStart', 'dayOfMonth', 'month'] columns
there are many date cols and consumerId and powerUsage cols
this dataset has different consumer data which are treated as separate data sequences(NSeries)
"""
from dataPrep.utils import getDatasetFiles, calculateNSeriesMinDifference, excludeValuesFromEnd_NSeries, splitTrainValTest_NSeries
from dataPrep.normalizers import NormalizerStack, SingleColsStdNormalizer, SingleColsLblEncoder
from dataPrep.dataset import VAnnTsDataset
from dataPrep.dataloader import VAnnTsDataloader
from utils.globalVars import tsStartPointColName
#%%
embedderInputSize=369# 369 different consumerIds
allReals=['hourOfDay', 'dayOfWeek', 'powerUsage','daysFromStart', 'hoursFromStart', 'daysFromStartOfDf', 'month']
covariatesNum=len(allReals)
#%%
timeIdx='hoursFromStart'
mainGroups=['consumerId']
target=['powerUsage']
backcastLen=192
forecastLen=1 #ccc this is for the shift
datasetKwargs={'mainGroups':mainGroups, 'consumerId':['consumerId'],'target':target}
def getElectricityProcessed(backcastLen=192, forecastLen=1):
    df =getDatasetFiles('electricity.csv')
    #kkk that this file exists when pulled
    calculateNSeriesMinDifference(df, mainGroups, valueCol=timeIdx, resultCol='sequenceIdx')
    excludeValuesFromEnd_NSeries(df, mainGroups, backcastLen+forecastLen, valueCol='sequenceIdx', resultCol=tsStartPointColName)
    normalizer=NormalizerStack(SingleColsStdNormalizer(['powerUsage','daysFromStart', 'hoursFromStart', 'dayOfMonth']),
                    SingleColsLblEncoder(mainGroups))#ccc if the mainGroup had more than 1 element, we could have used MultiColLblEncoder or MainGroupSingleColsLblEncoder
    normalizer.fitNTransform(df)
    trainDf, valDf, testDf=splitTrainValTest_NSeries(df, mainGroups, trainRatio=.7, valRatio=.2, seqLen=backcastLen+forecastLen, shuffle=True)
    return trainDf, valDf, testDf, normalizer
#%% 
class ElectricityDeepArDataset(VAnnTsDataset):
    def __getitem__(self, idx):
        inputs={}
        inputs['consumerId']=self.getBackForeCastData(idx, mode=self.modes.singlePoint, colsOrIndexes=self.consumerId)
        inputs['allReals']=self.getBackForeCastData(idx, mode=self.modes.backcast, colsOrIndexes=self.allReals)
        inputs['target']=self.getBackForeCastData(idx, mode=self.modes.backcast, colsOrIndexes=self.target, shiftForward=0)

        outputs=self.getBackForeCastData(idx, mode=self.modes.backcast, colsOrIndexes=self.target, shiftForward=1)
        return inputs, outputs
#%% dataloader
def getElectricityDataloaders(backcastLen=192, forecastLen=1, batchSize=64):
    trainDf, valDf, testDf, normalizer=getElectricityProcessed(backcastLen=backcastLen, forecastLen=forecastLen)

    electricityDeepArTrainDataset=ElectricityDeepArDataset(trainDf, backcastLen=backcastLen, forecastLen=forecastLen, indexes=None, **datasetKwargs)
    electricityDeepArValDataset=ElectricityDeepArDataset(valDf, backcastLen=backcastLen, forecastLen=forecastLen, indexes=None, **datasetKwargs)
    electricityDeepArTestDataset=ElectricityDeepArDataset(testDf, backcastLen=backcastLen, forecastLen=forecastLen, indexes=None, **datasetKwargs)
    del trainDf, valDf, testDf

    electricityDeepArTrainDataloader=VAnnTsDataloader(electricityDeepArTrainDataset, batch_size=batchSize)
    electricityDeepArValDataloader=VAnnTsDataloader(electricityDeepArValDataset, batch_size=batchSize)
    electricityDeepArTestDataloader=VAnnTsDataloader(electricityDeepArTestDataset, batch_size=batchSize)
    return electricityDeepArTrainDataloader, electricityDeepArValDataloader, electricityDeepArTestDataloader, normalizer