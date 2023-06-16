#%% imports
# trainNbeats.py
import os
baseFolder = os.path.dirname(os.path.abspath(__file__))
os.chdir(baseFolder)
from models.nbeats_blocks import stack, stackWithSharedWeights, SeasonalityBlock, TrendBlock, GenericBlock
from models.nbeats import nBeats
import torch
import torch.optim as optim
#%% redefine batchDatapreparation
class ModifiedNBeats(nBeats):
    def __init__(self, stacks, backcastLength, forecastLength):
        super(ModifiedNBeats, self).__init__(stacks, backcastLength, forecastLength)
    
    def batchDatapreparation(self, indexesIndex, indexes, inputs, outputs, batchSize, identifier=None):
        batchIndexes = indexes[indexesIndex*batchSize:indexesIndex*batchSize + batchSize]
        appliedBatchSize = len(batchIndexes)

        inputsSlices = [inputs[idx:idx + self.tsInputWindow] for idx in batchIndexes]
        outputsSlices = [inputs[idx + self.tsInputWindow:idx + self.tsInputWindow+self.tsOutputWindow] for idx in batchIndexes]
        batchInputs = torch.stack(inputsSlices).to(self.device)
        batchOutputs = torch.stack(outputsSlices).to(self.device)
        return batchInputs, batchOutputs, appliedBatchSize, identifier
#%% define model
stacks=[
    stack([
    SeasonalityBlock(256, 1, True, 4),
    SeasonalityBlock(256, 1, True),
    TrendBlock(256,3,False),
    GenericBlock(256,10,False),
    ]),
    stack([
        GenericBlock(256,10,False),
        TrendBlock(256,3,False),
        SeasonalityBlock(256, 1, True),
        ]),
    stackWithSharedWeights(GenericBlock(256,10,True),4)
    ]
nBeatsModel=ModifiedNBeats(stacks, backcastLength=10, forecastLength=5)
len(list(nBeatsModel.parameters()))
#%%
'#ccc how to set optimizer manually'
# nBeatsModel.lr=0.001
# nBeatsModel.learningRate=0.001
# nBeatsModel.changeLearningRate(0.001)
# nBeatsModel.optimizer=optim.Adam(nBeatsModel.parameters(), lr=0.4)
# nBeatsModel.tensorboardWriter=newTensorboardPath
# nBeatsModel.batchSize=32
# nBeatsModel.evalBatchSize=1024
# nBeatsModel.device=torch.device(type='cpu') or torch.device(type='cuda')
# nBeatsModel.l1Reg=1e-3 or nBeatsModel.l2Reg=1e-3 or nBeatsModel.regularization=[None, None]

# nBeatsModel.patience=10
# nBeatsModel.saveOnDiskPeriod=1
# nBeatsModel.lossMode='accuracy'
# nBeatsModel.variationalAutoEncoderMode=True
#%% 
# Set random seed for reproducibility
torch.manual_seed(42)

workerNum=0

totalData=torch.rand(10000)
trainTowholeRatio=.7
trainInputs=totalData[:int(trainTowholeRatio*len(totalData))]
testInputs =totalData[int(trainTowholeRatio*len(totalData)):]

criterion = torch.nn.MSELoss()

nBeatsModel.trainModel(trainInputs, None, testInputs, None, criterion, numEpochs=30, savePath=r'data\bestModels\a1', workerNum=workerNum)
#%%
#%%
#%%
#%%
#%%

#%%

#%%

#%%

#%%


