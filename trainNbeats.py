# ---- imports
# trainNbeats.py

from models.nbeats_blocks import stack, stackWithSharedWeights, SeasonalityBlock, TrendBlock, GenericBlock
from models.nbeats import nBeats
import torch
import torch.optim as optim
# ---- redefine batchDatapreparation
class ModifiedNBeats(nBeats):
    def __init__(self, stacks, backcastLength, forecastLength):
        super(ModifiedNBeats, self).__init__(stacks, backcastLength, forecastLength)
    
    def batchDatapreparation(self, indexesIndex, indexes, inputs, outputs, batchSize, mode=None, identifier=None, externalKwargs=None):
        batchIndexes = indexes[indexesIndex*batchSize:indexesIndex*batchSize + batchSize]
        appliedBatchSize = len(batchIndexes)

        inputsSlices = [inputs[idx:idx + self.backcastLen] for idx in batchIndexes]
        batchInputs = torch.stack(inputsSlices).to(self.device)
        
        outputsSlices = [outputs[idx + self.backcastLen:idx + self.backcastLen+self.forecastLen] for idx in batchIndexes]
        batchOutputs = torch.stack(outputsSlices).to(self.device)
        outPutMask=None
        return batchInputs, batchOutputs, appliedBatchSize, outPutMask, identifier
# ---- define model
# Set random seed for reproducibility
torch.manual_seed(42)
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
# ----
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
# ---- 
workerNum=0

inputData=torch.rand(10000)
outputData=torch.rand(10000)
trainTowholeRatio=.7
trainInputs=inputData[:int(trainTowholeRatio*len(inputData))]
valInputs =inputData[int(trainTowholeRatio*len(inputData)):]

trainOutputs=outputData[:int(trainTowholeRatio*len(outputData))]
valOutputs =outputData[int(trainTowholeRatio*len(outputData)):]

criterion = torch.nn.MSELoss()

# nBeatsModel.trainModel(trainInputs, trainOutputs, valInputs, valOutputs, criterion, numEpochs=30, savePath=r'data\bestModels\a1', workerNum=workerNum)
nBeatsModel.trainModel(trainInputs, trainOutputs, None, None, criterion, numEpochs=30, savePath=r'data\bestModels\a1', workerNum=workerNum)
# ----
# ----
# ----
# ----
# ----

# ----

# ----

# ----

# ----


