#%% imports
"""
this is data preparation steps of hourly electricity price forecasts (EPF) for France and Belgium markets
the data exists in data\datasets EPF_FR_BE.csv, EPF_FR_BE_futr.csv and EPF_FR_BE_static.csv files
"""
#kkk EPF_FR_BE_futr has 24 nans
from dataPrep.utils import getDatasetFiles, splitTrainValTestDf, addCorrespondentRow, rightPadDf, splitToNSeries, nontsStartPointsFalse
from dataPrep.normalizers import NormalizerStack, SingleColsStdNormalizer, MultiColStdNormalizer
from dataPrep.dataset import VAnnTsDataset
from dataPrep.dataloader import VAnnTsDataloader
#%%
futureExogenousCols = ['genForecast', 'weekDay']
historyExogenousCols = ['systemLoad']
staticExogenousCols = ['market0', 'market1']
targets=['yFR', 'yBE']
datasetKwargs={'futureExogenousCols':futureExogenousCols, 'historyExogenousCols':historyExogenousCols,
'staticExogenousCols':staticExogenousCols, 'targets':['y']}
def getEpfFrBeProcessed(backcastLen=110, forecastLen=22, rightPadTrain=True, newColName='y'):
    mainDf, staticDf=getDatasetFiles('EPF_FR_BE.csv'), getDatasetFiles('EPF_FR_BE_static.csv')
    mainDf=mainDf.sort_values('dateTime').reset_index(drop=True)
    normalizer=NormalizerStack(SingleColsStdNormalizer([*futureExogenousCols, *historyExogenousCols]),
                    MultiColStdNormalizer(targets))
    mainDf['mask']=1
    normalizer.fitNTransform(mainDf)
    trainDf, valDf, testDf=splitTrainValTestDf(mainDf, trainRatio=.7, valRatio=.2, seqLen=backcastLen+forecastLen, shuffle=False)
    if rightPadTrain:
        trainDf=rightPadDf(trainDf, forecastLen-1)

    newSets=[]
    for set_ in [trainDf, valDf, testDf]:
        set_=splitToNSeries(set_, targets, newColName)
        addCorrespondentRow(set_, staticDf, targets, newColName, targetMapping={})
        set_=nontsStartPointsFalse(set_)
        newSets+=[set_]
    trainDf, valDf, testDf=newSets
    return trainDf, valDf, testDf, normalizer
#%% 
class EpfFrBeDataset(VAnnTsDataset):
    def __getitem__(self, idx):
        inputs={}
        inputs['target']=self.getBackForeCastData(self.data, idx, mode=self.modes.backcast, colsOrIndexes=self.targets)
        inputs['mask']=self.getBackForeCastData(self.data, idx, mode=self.modes.backcast, colsOrIndexes=['mask'])
        inputs['historyExogenous']=self.getBackForeCastData(self.data, idx, mode=self.modes.backcast, colsOrIndexes=self.historyExogenousCols)
        inputs['staticExogenous']=self.getBackForeCastData(self.data, idx, mode=self.modes.singlePoint, colsOrIndexes=self.staticExogenousCols)
        inputs['futureExogenous']=self.getBackForeCastData(self.data, idx, mode=self.modes.fullcast, colsOrIndexes=self.futureExogenousCols)

        outputs={}
        outputs['output']=self.getBackForeCastData(self.data, idx, mode=self.modes.forecast, colsOrIndexes=self.targets)
        outputs['outputMask']=self.getBackForeCastData(self.data, idx, mode=self.modes.forecast, colsOrIndexes=['mask'])
        return inputs, outputs
#%% dataloader
def getEpfFrBeDataloaders(backcastLen=110, forecastLen=22, batchSize=64, rightPadTrain=True, newColName='y'):
    trainDf, valDf, testDf, normalizer=getEpfFrBeProcessed(backcastLen=backcastLen, forecastLen=forecastLen, rightPadTrain=rightPadTrain,
                                                           newColName=newColName)

    epfFrBeTrainDataset=EpfFrBeDataset(trainDf, backcastLen=backcastLen, forecastLen=forecastLen, indexes=None, **datasetKwargs)
    epfFrBeValDataset=EpfFrBeDataset(valDf, backcastLen=backcastLen, forecastLen=forecastLen, indexes=None, **datasetKwargs)
    epfFrBeTestDataset=EpfFrBeDataset(testDf, backcastLen=backcastLen, forecastLen=forecastLen, indexes=None, **datasetKwargs)

    epfFrBeTrainDataloader=VAnnTsDataloader(epfFrBeTrainDataset, batch_size=batchSize)
    epfFrBeValDataloader=VAnnTsDataloader(epfFrBeValDataset, batch_size=batchSize)
    epfFrBeTestDataloader=VAnnTsDataloader(epfFrBeTestDataset, batch_size=batchSize)
    return epfFrBeTrainDataloader, epfFrBeValDataloader, epfFrBeTestDataloader, normalizer