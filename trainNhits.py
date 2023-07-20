#%% imports
# trainnHits.py
import os
baseFolder = os.path.dirname(os.path.abspath(__file__))
os.chdir(baseFolder)
from models.nhits_blocks import nHitsBlock
from models.nhits import nHits#kkk change name nhitsMyTry
import torch
import torch.optim as optim
#kkk the test which doesnt have the historyExogenousCols, check how it works then: they are assigned to None
#kkk append future data
#kkk predict should have at least backcastLen
#kkk recheck the val and test steps
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
externalKwargs={'batchDatapreparation':{'futureExogenousCols':['genForecast', 'weekDay'], 'historyExogenousCols':['systemLoad'], 'staticExogenousCols':['market0', 'market1']}}
nHitsModel=nHits(blocks, backcastLen=110, forecastLen=22, externalKwargs=externalKwargs)
len(list(nHitsModel.parameters()))
#%% preprocessTrainValTestData
trainDataProcessed, valDataProcessed, testDataProcessed = nHitsModel.preprocessTrainValTestData(dfPath=r'.\data\datasets\EPF_FR_BE2.csv', \
    trainRatio=.7, valRatio=.15, ysCols=['yFR', 'yBE'], externalKwargs=externalKwargs, staticDfPath=r'.\data\datasets\EPF_FR_BE_static.csv')
#%%
workerNum=8

criterion = torch.nn.MSELoss()

nHitsModel.trainModel(trainDataProcessed, None, valDataProcessed, None, criterion, numEpochs=30, savePath=r'data\bestModels\nHits1', workerNum=workerNum, externalKwargs=externalKwargs)#kkk rename savePath for all train examples
#%% test
nHitsModel.evaluateModel(testDataProcessed, None, criterion,stepNum=0, evalMode='test', workerNum=workerNum, externalKwargs=externalKwargs)
#%%
#%%
#%%
#%%

#%%

#%%

#%%

#%%


