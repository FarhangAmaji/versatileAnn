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
    def batchDatapreparation(self, indexesIndex, indexes, inputs, outputs, batchSize, mode=None,identifier=None, externalKwargs=None):
        batchIndexes = indexes[indexesIndex*batchSize:indexesIndex*batchSize + batchSize]
        appliedBatchSize = len(batchIndexes)

        inputsSlices = [inputs[idx:idx + self.backcastLen] for idx in batchIndexes]
        outputsSlices = [torch.cat((torch.zeros(1, self.transformerInfo.outputDim), outputs[idx:idx + self.forecastLen]), dim=0) for idx in batchIndexes]
        '#ccc we append 0*outputDim to the beginning of each batchOutputs'
        batchInputs = torch.stack(inputsSlices).to(self.device)
        batchOutputs = torch.stack(outputsSlices).to(self.device)
        outPutMask=None
        return batchInputs, batchOutputs, appliedBatchSize, outPutMask, identifier
#%%
# Set random seed for reproducibility
torch.manual_seed(42)

inpLen, outputLen= 12, 13
transformerInfo=TransformerInfo(embedSize=32, heads=8, forwardExpansion=4, encoderLayersNum=6, decoderLayersNum=6, dropoutRate=.6, inpLen=inpLen, outputLen=outputLen, inputDim=5, outputDim=2)
"""#ccc we dont have first prediction; so we add last temporal data from the input to output
pay attention to outputLen"""
model = ModifiedMultivariateTransformer(transformerInfo)

# x= torch.rand(2,inpLen).to(transformerInfo.device)
# trg = torch.rand(2,outputLen-1).to(transformerInfo.device)
# appendedTrg = torch.cat((x[:, -1].unsqueeze(1), trg), dim=1)
# out = model(x, appendedTrg)
#%%
workerNum=8

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
inputOfUnknown=torch.rand(1,inpLen, transformerInfo.inputDim)
output=model.forwardForUnknown(inputOfUnknown, outputLen)
outputStraight=model.forwardForUnknownStraight(inputOfUnknown, outputLen)
#%%
z1=torch.tensor([[0.3152, 0.5333, 0.3562, 0.5580, 0.6787],
        [0.8185, 0.6405, 0.5854, 0.6332, 0.1547],
        [0.9564, 0.5171, 0.7604, 0.2390, 0.2469],
        [0.3129, 0.5263, 0.4841, 0.0554, 0.3992],
        [0.4123, 0.7898, 0.8287, 0.3734, 0.3171],
        [0.9313, 0.9074, 0.2239, 0.3096, 0.4186],
        [0.5491, 0.3172, 0.7427, 0.3214, 0.2458],
        [0.7938, 0.0624, 0.6982, 0.9243, 0.4545],
        [0.3574, 0.7641, 0.0716, 0.8256, 0.5401],
        [0.5540, 0.9970, 0.0403, 0.1630, 0.2039],
        [0.9093, 0.5066, 0.4663, 0.5369, 0.7438],
        [0.2120, 0.7883, 0.6041, 0.7700, 0.7930]], device='cuda:0').unsqueeze(0)
z2=torch.tensor([[0.0000, 0.0000],
        [0.4511, 0.6827],
        [0.0561, 0.4955],
        [0.0849, 0.2847],
        [0.0577, 0.8435],
        [0.6489, 0.8742],
        [0.5348, 0.7273],
        [0.1965, 0.9138],
        [0.0118, 0.8979],
        [0.6890, 0.2641],
        [0.2684, 0.9110],
        [0.1701, 0.1149],
        [0.2942, 0.2500],
        [0.5636, 0.6243]], device='cuda:0').unsqueeze(0)
model.forward(z1, z2)
model.forwardForUnknown(z1, outputLen)

encInps=model.encoder(z1)
model.decoder(torch.zeros_like(z2),encInps)
#%%
#%%
#%%

#%%

#%%

#%%

#%%


