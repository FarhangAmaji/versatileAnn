#%% imports
# trainUnivariateTransformers.py
import os
baseFolder = os.path.dirname(os.path.abspath(__file__))
os.chdir(baseFolder)
from models.univariateTransformers import univariateTransformer, TransformerInfo
import torch
import torch.optim as optim
#%% redefine batchDatapreparation
class ModifiedUnivariateTransformer(univariateTransformer):
    def __init__(self, transformerInfo):
        super(ModifiedUnivariateTransformer, self).__init__(transformerInfo)
    def batchDatapreparation(self, indexesIndex, indexes, inputs, outputs, batchSize, identifier=None):
        batchIndexes = indexes[indexesIndex*batchSize:indexesIndex*batchSize + batchSize]
        appliedBatchSize = len(batchIndexes)

        inputsSlices = [inputs[idx:idx + self.tsInputWindow] for idx in batchIndexes]
        outputsSlices = [inputs[idx + self.tsInputWindow-1:idx + self.tsInputWindow+self.tsOutputWindow] for idx in batchIndexes]
        batchInputs = torch.stack(inputsSlices).to(self.device)
        batchOutputs = torch.stack(outputsSlices).to(self.device)
        return batchInputs, batchOutputs, appliedBatchSize, identifier
#%%
# Set random seed for reproducibility
torch.manual_seed(42)

inpLen, outputLen= 12, 13
transformerInfo=TransformerInfo(embedSize=32, heads=8, forwardExpansion=4, encoderLayersNum=6, decoderLayersNum=6, dropoutRate=.6, inpLen=inpLen, outputLen=outputLen)
"""#ccc we dont have first prediction; so we add last temporal data from the input to output
pay attention to outputLen"""
model = ModifiedUnivariateTransformer(transformerInfo)

# x= torch.rand(2,inpLen).to(transformerInfo.device)
# trg = torch.rand(2,outputLen-1).to(transformerInfo.device)
# appendedTrg = torch.cat((x[:, -1].unsqueeze(1), trg), dim=1)
# out = model(x, appendedTrg)
#%%
workerNum=0

totalData=torch.rand(1000)
trainTowholeRatio=.7
trainInputs=totalData[:int(trainTowholeRatio*len(totalData))]
testInputs =totalData[int(trainTowholeRatio*len(totalData)):]

criterion = torch.nn.MSELoss()

model.trainModel(trainInputs, None, testInputs, None, criterion, numEpochs=30, savePath=r'data\bestModels\a1', workerNum=workerNum)
#%% model feedforward for unknown results
inputOfUnknown=torch.rand(inpLen)
# output=model.forwardForUnknown(inputOfUnknown, outputLen)
output=model.forwardForUnknownStraight(inputOfUnknown, outputLen)

#%%
#%%
#%%
#%%

#%%

#%%

#%%

#%%


