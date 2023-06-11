#%% imports
# trainMultivariateTransformers.py
import os
baseFolder = os.path.dirname(os.path.abspath(__file__))
os.chdir(baseFolder)
from models.multivariateTransformers import multivariateTransformer, TransformerInfo
import torch
import torch.optim as optim
#%% redefine batchDatapreparation
class ModifiedMultivariateTransformer(multivariateTransformer):
    def __init__(self, transformerInfo):
        super(ModifiedMultivariateTransformer, self).__init__(transformerInfo)
    def batchDatapreparation(self, indexesIndex, indexes, inputs, outputs, batchSize, identifier=None):
        batchIndexes = indexes[indexesIndex*batchSize:indexesIndex*batchSize + batchSize]
        appliedBatchSize = len(batchIndexes)

        inputsSlices = [inputs[idx:idx + self.tsInputWindow] for idx in batchIndexes]
        outputsSlices = [torch.cat((torch.zeros(1, self.transformerInfo.outputDim), outputs[idx:idx + self.tsOutputWindow]), dim=0) for idx in batchIndexes]#jjj so i dont need the indexes to be so much less but I dont change it
        '#ccc we append 0*outputDim to the beginning of each batchOutputs'
        batchInputs = torch.stack(inputsSlices).to(self.device)
        batchOutputs = torch.stack(outputsSlices).to(self.device)
        return batchInputs, batchOutputs, appliedBatchSize, identifier
#%%
# Set random seed for reproducibility
torch.manual_seed(42)

inpLen, outputLen= 12, 10
transformerInfo=TransformerInfo(embedSize=32, heads=8, forwardExpansion=4, encoderLayersNum=6, decoderLayersNum=6, dropoutRate=.6, inpLen=inpLen, outputLen=outputLen, inputDim=5, outputDim=2)
"""#ccc we dont have first prediction; so we add last temporal data from the input to output
pay attention to outputLen"""
model = ModifiedMultivariateTransformer(transformerInfo)

# x= torch.rand(2,inpLen).to(transformerInfo.device)
# trg = torch.rand(2,outputLen-1).to(transformerInfo.device)
# appendedTrg = torch.cat((x[:, -1].unsqueeze(1), trg), dim=1)
# out = model(x, appendedTrg)
#%%
workerNum=0

allSeqLen=1000
inputs=torch.rand(allSeqLen, transformerInfo.inputDim)
trainTowholeRatio=.7
trainInputs=inputs[:int(trainTowholeRatio*len(inputs))]
testInputs =inputs[int(trainTowholeRatio*len(inputs)):]

outputs=torch.rand(allSeqLen, transformerInfo.outputDim)
trainOutputs=outputs[:int(trainTowholeRatio*len(outputs))]
testOutputs=outputs[int(trainTowholeRatio*len(outputs)):]

criterion = torch.nn.MSELoss()

model.trainModel(trainInputs, trainOutputs, testInputs, testOutputs, criterion, numEpochs=200, savePath=r'data\bestModels\a1', workerNum=workerNum)
#%% model feedforward for unknown results
#kkk model feedforward for unknown results

#%%
#%%
#%%
#%%

#%%

#%%

#%%

#%%


