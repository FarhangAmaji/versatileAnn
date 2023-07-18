#%% imports
# trainnHits.py
import os
baseFolder = os.path.dirname(os.path.abspath(__file__))
os.chdir(baseFolder)
from models.nhits_blocks import nHitsBlock
from models.nhitsMyTry import nHits#kkk change name nhitsMyTry
import pandas as pd
import numpy as np
import torch
import torch.optim as optim
#%% redefine batchDatapreparation
class ModifiedNHits(nHits):#kkk _parse_windows
    def __init__(self, blocks, backcastLength, forecastLength, futureExogenousCols, historyExogenousCols, staticExogenousCols):#kkk correct order of backcastLen and forecastLen in all repo
        super(ModifiedNHits, self).__init__(blocks, backcastLength, forecastLength, futureExogenousCols, historyExogenousCols, staticExogenousCols)
    
    def batchDatapreparation(self, indexesIndex, indexes, inputs, outputs, batchSize, identifier=None, externalKwargs=None):#kkk move it to nHits
        batchIndexes = indexes[indexesIndex*batchSize:indexesIndex*batchSize + batchSize]
        appliedBatchSize = len(batchIndexes)
        
        stackListOfDfs= lambda lodfs: torch.stack([torch.tensor(df_values.values) for df_values in lodfs]).to(self.device).to(torch.float32)
        
        "#ccc y(target), mask and historyExogenous need only backcastLen; historyExogenous needs backcastLen+forecastLen; for static only correspondant static is passed"
        targetDataForInput=[inputs.loc[idx:idx + self.backcastLen-1,'y'] for idx in batchIndexes]
        maskData=[inputs.loc[idx:idx + self.backcastLen-1,'mask'] for idx in batchIndexes]
        futureExogenousData=[inputs.loc[idx:idx + self.backcastLen+self.forecastLen-1,externalKwargs['batchDatapreparation']['futureExogenousCols']] for idx in batchIndexes]
        historyExogenousData=[inputs.loc[idx:idx + self.backcastLen-1,externalKwargs['batchDatapreparation']['historyExogenousCols']] for idx in batchIndexes]
        staticExogenousData=[inputs.loc[idx,externalKwargs['batchDatapreparation']['staticExogenousCols']] for idx in batchIndexes]
        
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
#%%
'#ccc how to set optimizer manually'
# nHitsModel.lr=0.001
# nHitsModel.learningRate=0.001
# nHitsModel.changeLearningRate(0.001)
# nHitsModel.optimizer=optim.Adam(nHitsModel.parameters(), lr=0.4)
# nHitsModel.tensorboardWriter=newTensorboardPath
# nHitsModel.batchSize=32
# nHitsModel.evalBatchSize=1024
# nHitsModel.device=torch.device(type='cpu') or torch.device(type='cuda')
# nHitsModel.l1Reg=1e-3 or nHitsModel.l2Reg=1e-3 or nHitsModel.regularization=[None, None]

# nHitsModel.patience=10
# nHitsModel.saveOnDiskPeriod=1
# nHitsModel.lossMode='accuracy'
# nHitsModel.variationalAutoEncoderMode=True
#%% load data
#kkk move it to nHits
EPFDf=pd.read_csv(r'.\data\datasets\EPF_FR_BE2.csv')
EPFDf['ds'] = pd.to_datetime(EPFDf['ds'])
EPFDf=EPFDf.sort_values('ds').reset_index(drop=True)
EPFStatic=pd.read_csv(r'.\data\datasets\EPF_FR_BE_static.csv')
"#ccc this model supports Future, Historical and Static exogenous variables. so we should introduce them. also define y(target) columns"
ysCols=['yFR', 'yBE']
futureExogenousCols = ['genForecast', 'weekDay'] # <- Future exogenous variables
historyExogenousCols = ['systemLoad'] # <- Historical exogenous variables
staticExogenousCols = ['market0', 'market1'] # <- Static exogenous variables
#%% scale
from sklearn.preprocessing import StandardScaler
EPFDfNormalized=EPFDf.copy()
scalers={'ysScaler':StandardScaler(), 'futureHistoryScaler':StandardScaler()}
EPFDfNormalized[ysCols]=scalers['ysScaler'].fit_transform(pd.concat([EPFDf[col] for col in ysCols]).values.reshape(-1, 1)).reshape(-1, len(ysCols))

futureHistoryCols=[*futureExogenousCols, *historyExogenousCols]
EPFDfNormalized[futureHistoryCols]=scalers['futureHistoryScaler'].fit_transform(EPFDf[futureHistoryCols])

# add mask ==1 for existing data
EPFDfNormalized['mask']=1
#%% split train val
forecastLength=4
backcastLength=5*forecastLength
trainRatio=.7
valRatio=.15
trainData = EPFDfNormalized[:int(trainRatio * len(EPFDfNormalized))]
valData = EPFDfNormalized[int(trainRatio * len(EPFDfNormalized)):int((trainRatio+valRatio) * len(EPFDfNormalized))]#kkk
testData = EPFDfNormalized[int((trainRatio+valRatio) * len(EPFDfNormalized)):]#kkk

#kkk dont do valInputs preprations now
#kkk what are outputs
#kkk does the 'valInputs = EPFDfNormalized[int(trainRatio * len(EPFDfNormalized)):int((trainRatio+valRatio) * len(EPFDfNormalized))]' enough or we need to have forecastLength and backcastLength
trainOutputs = EPFDfNormalized#kkk
valOutputs = EPFDfNormalized#kkk
#%% add padder + remove last item
"#ccc the part in original code is equivalent to add zero padding after data with length of forecastLength-1"
trainData=pd.concat([trainData,pd.DataFrame(np.zeros((forecastLength-1, trainData.shape[1])), columns=trainData.columns)]).reset_index(drop=True)
#%% put together target data + correspondant static
trainDataProcessed=pd.DataFrame({})
for i in range(len(ysCols)):
    '#ccc just keep the columns of each y(target) with other shared exogenous columns'
    colsToAdd=trainData.columns.to_list()
    [colsToAdd.remove(j) for j in ysCols if j!=ysCols[i]]
    thisYDf=trainData[colsToAdd]
    thisYDf=thisYDf.rename(columns={col: 'y' for col in thisYDf.columns if col in ysCols})
    
    '#ccc the correspondant static rows are added'
    for sc in staticExogenousCols:
        thisYDf[sc]=EPFStatic.loc[i,sc]
    
    trainDataProcessed=pd.concat([trainDataProcessed,thisYDf]).reset_index(drop=True)
trainDataProcessed = trainDataProcessed.drop('ds', axis=1)
#%% define model
'''
#ccc the values of nFreqDownsample and nPoolKernelSize in later blocks are usually decreased
'''
# Set random seed for reproducibility
torch.manual_seed(42)
blocks=[nHitsBlock(nFreqDownsample=4, mlpUnits=3 * [[512, 512]], nPoolKernelSize=2, interpolationMode='cubic', pooling_mode='MaxPool1d', dropoutRate=0, activation='LeakyReLU'),
        nHitsBlock(nFreqDownsample=2, mlpUnits=3 * [[512, 512]], nPoolKernelSize=2, interpolationMode='linear', pooling_mode='MaxPool1d', dropoutRate=0, activation='LeakyReLU'),
        nHitsBlock(nFreqDownsample=1, mlpUnits=3 * [[512, 512]], nPoolKernelSize=1, interpolationMode='linear', pooling_mode='MaxPool1d', dropoutRate=0, activation='LeakyReLU'),
    ]
nHitsModel=ModifiedNHits(blocks, backcastLength=110, forecastLength=22, futureExogenousCols=futureExogenousCols, historyExogenousCols=historyExogenousCols, staticExogenousCols=staticExogenousCols)
len(list(nHitsModel.parameters()))
#%% shit


#%% 
runcell('imports', 'F:/projects/public github projects/private repos/versatileAnnModule/trainNhits.py')
runcell('redefine batchDatapreparation', 'F:/projects/public github projects/private repos/versatileAnnModule/trainNhits.py')
runcell('load data', 'F:/projects/public github projects/private repos/versatileAnnModule/trainNhits.py')
runcell('scale', 'F:/projects/public github projects/private repos/versatileAnnModule/trainNhits.py')
runcell('split train val', 'F:/projects/public github projects/private repos/versatileAnnModule/trainNhits.py')
runcell('add padder + remove last item', 'F:/projects/public github projects/private repos/versatileAnnModule/trainNhits.py')
runcell('put together target data + correspondant static', 'F:/projects/public github projects/private repos/versatileAnnModule/trainNhits.py')
runcell('define model', 'F:/projects/public github projects/private repos/versatileAnnModule/trainNhits.py')
#%%
externalKwargs={'batchDatapreparation':{'futureExogenousCols':futureExogenousCols, 'historyExogenousCols':historyExogenousCols, 'staticExogenousCols':staticExogenousCols}}
workerNum=9

criterion = torch.nn.MSELoss()

nHitsModel.trainModel(trainDataProcessed, None, None, None, criterion, numEpochs=30, savePath=r'data\bestModels\nHits1', workerNum=workerNum, externalKwargs=externalKwargs)#kkk rename savePath for all train examples
#%%
#%%
#%%
#%%
#%%

#%%

#%%

#%%

#%%


