#%% thing to check
'''
pass the blocks again like nbeats; here we have no stack because we only have 1 type of blocks
make common args of block in postInit(basis,h,input_size,futureInputSize,historyInputSize,staticInputSize)
postInit for blocks would be called by Model because it has all args needed for postInit
make changes for ann because this doesnt have val and best val stop(maybe dont make problem but check it)
check if with nn.ModuleList still we need to add parameters manually(check the named_params)


prep data for train:
    at least for train (probably for predict also) we know that the futureExogenous len is input+h and for historyExogenous len is input;#jjj where this is applied in orig code
prep data for predict
'''
#%% imports
"""
original code is from https://github.com/Nixtla/neuralforecast/blob/main/neuralforecast/models/nhits.py
adaptations were applied in order to make it compatible to our framework
"""
# models\nhitsMyTry.py
import sys
import os
parentFolder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parentFolder)
import pandas as pd
import numpy as np
from versatileAnn import ann
import torch
import torch.nn as nn
#kkk add own comments
# %% ../../nbs/models.nhits.ipynb 10
class nHits(ann):
    """NHITS:

    The Neural Hierarchical Interpolation for Time Series (NHITS), is an MLP-based deep
    neural architecture with backward and forward residual links. NHITS tackles volatility and
    memory complexity challenges, by locally specializing its sequential predictions into
    the signals frequencies with hierarchical interpolation and pooling.

    """
    def __init__(self, blocks, backcastLen, forecastLen, externalKwargs):
        super(nHits, self).__init__()
        assert isinstance(blocks, list),'blocks should be a list'
        self.blocks = nn.ModuleList(blocks)
        
        self.ExogenousColTypes = ['futureExogenousCols', 'historyExogenousCols', 'staticExogenousCols']
        [self.assertExogenousColNamesExistance(externalKwargs,colNames) for colNames in self.ExogenousColTypes]
        
        futureExogenousCols, historyExogenousCols, staticExogenousCols = [self.getExogenousColNames(externalKwargs,colNames) for colNames in self.ExogenousColTypes]
        self.futureInputSize = len(futureExogenousCols) if futureExogenousCols else 0
        self.historyInputSize = len(historyExogenousCols) if historyExogenousCols else 0
        self.staticInputSize = len(staticExogenousCols) if staticExogenousCols else 0
        
        self.postInitForBlocks(backcastLen, forecastLen, futureInputSize=self.futureInputSize, historyInputSize=self.historyInputSize, staticInputSize=self.staticInputSize)
        
        self.forecastLen = forecastLen
        self.backcastLen = backcastLen
        self.timeSeriesMode=True
    
    def postInitForBlocks(self, backcastLen, forecastLen, futureInputSize=0, historyInputSize=0, staticInputSize=0):
        for block in self.blocks:
            block.postInit(backcastLen, forecastLen, futureInputSize, historyInputSize, staticInputSize)
            block.futureInputSize = self.futureInputSize
            block.historyInputSize = self.historyInputSize
            block.staticInputSize = self.staticInputSize
    
    def preprocessTrainValTestData(self, dfPath, trainRatio, valRatio, ysCols, externalKwargs, staticDfPath=None):
        from sklearn.preprocessing import StandardScaler
        
        [self.assertExogenousColNamesExistance(externalKwargs,colNames) for colNames in self.ExogenousColTypes]
        futureExogenousCols, historyExogenousCols, staticExogenousCols = [self.getExogenousColNames(externalKwargs,colNames) for colNames in self.ExogenousColTypes]
        
        df=pd.read_csv(dfPath)
        df['dateTime'] = pd.to_datetime(df['dateTime'])
        df=df.sort_values('dateTime').reset_index(drop=True)
        
        if staticDfPath:
            staticDf = pd.read_csv(staticDfPath)
            assert staticExogenousCols,'the "staticExogenousCols" are not introduced while you have passed staticDfPath'
        else:
            assert not staticExogenousCols,'the "staticExogenousCols" are introduced while you have not passed staticDfPath'
            staticDf=pd.DataFrame({})
        
        dfNormalized=df.copy()
        "#ccc this model assumes that y data are in the same range so one scaler is applied to them"
        scalers={'ysScaler':StandardScaler(), 'futureHistoryScaler':StandardScaler()}
        dfNormalized[ysCols]=scalers['ysScaler'].fit_transform(pd.concat([df[col] for col in ysCols]).values.reshape(-1, 1)).reshape(-1, len(ysCols))
    
        futureHistoryCols=[*futureExogenousCols, *historyExogenousCols]
        dfNormalized[futureHistoryCols]=scalers['futureHistoryScaler'].fit_transform(df[futureHistoryCols])
    
        # add mask ==1 for existing data
        dfNormalized['mask']=1
        
        trainLen=int(trainRatio * len(dfNormalized))
        trainPlusValLen=int((trainRatio+valRatio) * len(dfNormalized))
        assert trainLen>self.backcastLen,'the trainLen should be bigger than backcastLen'
        trainData = dfNormalized.loc[:trainLen]
        valData = dfNormalized.loc[trainLen-self.backcastLen:trainPlusValLen].reset_index(drop=True)
        testData = dfNormalized.loc[trainPlusValLen-self.backcastLen:].reset_index(drop=True)
        
        # add padder with length of forecastLen-1 item for trainData
        "#ccc the part in original code is equivalent to add zero padding after data with length of forecastLen-1"
        trainData=pd.concat([trainData,pd.DataFrame(np.zeros((self.forecastLen-1, trainData.shape[1])), columns=trainData.columns)]).reset_index(drop=True)
        
        def putYColsAlongEachotherAndAddCorrespondantStatic(df, ysCols, staticDf, staticExogenousCols):
            processedData=pd.DataFrame({})
            for i in range(len(ysCols)):
                '#ccc just keep the columns of each y(target) with other shared exogenous columns'
                colsToAdd=df.columns.to_list()
                [colsToAdd.remove(j) for j in ysCols if j!=ysCols[i]]
                thisYDf=df[colsToAdd]
                thisYDf=thisYDf.rename(columns={col: 'y' for col in thisYDf.columns if col in ysCols})
                
                '#ccc the correspondant static rows are added'
                for sc in staticExogenousCols:
                    thisYDf[sc]=staticDf.loc[i, sc]
                
                processedData = pd.concat([processedData,thisYDf]).reset_index(drop=True).drop('dateTime', axis=1)
            return processedData
        
        trainDataProcessed = putYColsAlongEachotherAndAddCorrespondantStatic(trainData, ysCols, staticDf, staticExogenousCols)
        valDataProcessed = putYColsAlongEachotherAndAddCorrespondantStatic(valData, ysCols, staticDf, staticExogenousCols)
        testDataProcessed = putYColsAlongEachotherAndAddCorrespondantStatic(testData, ysCols, staticDf, staticExogenousCols)
        
        return trainDataProcessed, valDataProcessed, testDataProcessed
        
    def getExogenousColNames(self, externalKwargs, colNames):
        return externalKwargs['batchDatapreparation'][colNames]
    
    def assertExogenousColNamesExistance(self, externalKwargs,colNames):
        assert 'batchDatapreparation' in externalKwargs.keys() and  \
        isinstance(externalKwargs['batchDatapreparation'], dict) and \
        colNames in externalKwargs['batchDatapreparation'].keys(), \
        f"the list introducing '{colNames}' should exist in externalKwargs['batchDatapreparation'] keys; note u may pass it '[]' if your model doesn't have {colNames}"
    
    def batchDatapreparation(self, indexesIndex, indexes, inputs, outputs, batchSize, mode=None, identifier=None, externalKwargs=None):
        batchIndexes = indexes[indexesIndex*batchSize:indexesIndex*batchSize + batchSize]
        appliedBatchSize = len(batchIndexes)
        
        def stackListOfDfs(lodfs):
            tensorList=[]
            for df in lodfs:
                assert df.isnull().any().any()==False,'the data should be cleaned in order not to have nan or None data'
                tensorList.append(torch.tensor(df.values))
            
            tensor = torch.stack(tensorList).to(self.device).to(torch.float32)
            return tensor
        
        futureExogenousCols, historyExogenousCols, staticExogenousCols = [self.getExogenousColNames(externalKwargs,colNames) for colNames in ['futureExogenousCols', 'historyExogenousCols', 'staticExogenousCols']]
        
        "#ccc y(target), mask and historyExogenous need only backcastLen; historyExogenous needs backcastLen+forecastLen; for static only correspondant static is passed"
        targetDataForInput=[inputs.loc[idx:idx + self.backcastLen-1,'y'] for idx in batchIndexes]
        maskData=[inputs.loc[idx:idx + self.backcastLen-1,'mask'] for idx in batchIndexes]
        futureExogenousData=[inputs.loc[idx:idx + self.backcastLen+self.forecastLen-1, futureExogenousCols] for idx in batchIndexes]
        historyExogenousData=[inputs.loc[idx:idx + self.backcastLen-1, historyExogenousCols] for idx in batchIndexes]
        staticExogenousData=[inputs.loc[idx, staticExogenousCols] for idx in batchIndexes]
        
        targetDataForInput=stackListOfDfs(targetDataForInput)
        maskData=stackListOfDfs(maskData)
        futureExogenousData=stackListOfDfs(futureExogenousData)
        historyExogenousData=stackListOfDfs(historyExogenousData)
        staticExogenousData=stackListOfDfs(staticExogenousData)
        batchInputs = {'target':targetDataForInput, 'mask':maskData, 'futureExogenous':futureExogenousData, 'historyExogenous':historyExogenousData, 'staticExogenous':staticExogenousData}
        
        targetDataForOutput=[inputs.loc[idx+self.backcastLen:idx + self.backcastLen+self.forecastLen-1,'y'] for idx in batchIndexes]
        batchOutputs=stackListOfDfs(targetDataForOutput)
        
        outPutMask=[inputs.loc[idx+self.backcastLen:idx + self.backcastLen+self.forecastLen-1,'mask'] for idx in batchIndexes]
        outPutMask=stackListOfDfs(outPutMask)
        return batchInputs, batchOutputs, appliedBatchSize, outPutMask, identifier
    
    def forward(self, inputs):
        # Parse inputs
        targetY = inputs["target"]# shape: N * backcastLen
        mask = inputs["mask"]# shape: N * backcastLen#ccc except the sequences which have last data should the values are 1.0 and those have 0.0
        futureExogenous = inputs["futureExogenous"]# shape: N * (backcastLen+forecastLen) * futureInputSize
        historyExogenous = inputs["historyExogenous"]# shape: N * backcastLen * historyInputSize
        staticExogenous = inputs["staticExogenous"]# shape: N * backcastLen * staticInputSize

        residuals = targetY.flip(dims=(-1,))
        mask = mask.flip(dims=(-1,))
        forecast = targetY[:, -1:, None]  # Level with Naive1
        for i, block in enumerate(self.blocks):
            backcast, blockForecast = block(targetY = residuals,
                futureExogenous = futureExogenous,
                historyExogenous = historyExogenous,
                staticExogenous = staticExogenous)
            residuals = (residuals - backcast) * mask
            forecast = forecast + blockForecast
        return forecast.squeeze(-1)
#%%

#%%

#%%

#%%

#%%

