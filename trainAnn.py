#%% imports
# trainAnn.py
import os
baseFolder = os.path.dirname(os.path.abspath(__file__))
os.chdir(baseFolder)
from versatileAnn import ann
from versatileAnn.layers import linLReluNormDropout, linLSigmoidNormDropout
import torch
import torch.optim as optim
#%% define model
import torch.nn as nn
class A:
    def __init__(self):
        self.x='aaaa'
class B(A):
    def __init__(self):
        super(B, self).__init__()
        self.x='bbbb'
class MyBaseClass:
    pass
class variationalEncoder(nn.Module):
    def __init__(self, inputSize, latentDim):
        super(variationalEncoder, self).__init__()
        self.linSig=linLSigmoidNormDropout(4,1)
        self.b11=B()
        self.a11=A()
        self.inputArgs = [inputSize, latentDim]
        self.latentDim = latentDim
        self.fc1 = nn.Linear(inputSize, 4*inputSize)
        self.fc2 = nn.Linear(4*inputSize, inputSize)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout()
        self.fcMean = nn.Linear(inputSize, latentDim)
        self.fcLogvar = nn.Linear(inputSize, latentDim)
    
    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mean + eps * std
        return z
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.sigmoid(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        x = self.dropout(x)
        mean = self.fcMean(x)
        logvar = self.fcLogvar(x)
        z = self.reparameterize(mean, logvar)
        return mean, logvar, z
class myAnn(ann):
    def __init__(self, inputSize, outputSize):
        super(myAnn, self).__init__()
        self.layer1 = linLReluNormDropout(inputSize, inputSize*4, dropoutRate=0.5, regularization=['l1',0.003])#
        self.layer2 = torch.nn.Linear(inputSize, outputSize)
        self.layer3 = linLReluNormDropout(inputSize*4, outputSize)
        self.l4= torch.nn.LayerNorm(outputSize)
        # self.l5= variationalEncoder(inputSize,2)
        self.l6= B()
        self.l7= MyBaseClass
        self.fcMean = torch.nn.Linear(inputSize, outputSize)
        self.fcLogvar = torch.nn.Linear(inputSize, outputSize)
        self.addLayerRegularization([[self.layer2,'l2',.004], [self.layer3,None,None]])
    
    def forward(self, x):
        mean = self.fcMean(x)
        logvar = self.fcLogvar(x)
        x = self.layer1(x)
        x = self.layer3(x)
        return x, mean, logvar
#%% make model instance
z1=myAnn(40,1)
#%%
'#ccc how to set optimizer manually'
# z1.lr=0.001
# z1.learningRate=0.001
# z1.changeLearningRate(0.001)
# z1.optimizer=optim.Adam(z1.parameters(), lr=0.4)
# z1.tensorboardWriter=newTensorboardPath
# z1.batchSize=32
# z1.evalBatchSize=1024
# z1.device=torch.device(type='cpu') or torch.device(type='cuda')
# z1.l1Reg=1e-3 or z1.l2Reg=1e-3 or z1.regularization=[None, None]

# z1.patience=10
# z1.saveOnDiskPeriod=1
# z1.lossMode='accuracy'
# z1.autoEncoderMode=True #kkk will it have problems of not saving these to saveModel
#%% regression test
z1.variationalAutoEncoderMode=True
workerNum=8
# Set random seed for reproducibility
torch.manual_seed(42)
import time
t0=time.time()
trainInputs = torch.randn(100, 40)  # Assuming 100 training samples with 40 features each
trainOutputs = torch.randn(100, 1)  # Assuming 100 training output values

testInputs = torch.randn(50, 40)  # Assuming 50 testing samples with 40 features each
testOutputs = torch.randn(50, 1)  # Assuming 50 testing output values

# Define the criterion (loss function)
criterion = torch.nn.MSELoss()  # Example: Mean Squared Error (MSE) loss

# Train the model
z1.trainModel(trainInputs, trainOutputs, testInputs, testOutputs, criterion, numEpochs=200, savePath=r'data\bestModels\a1', workerNum=workerNum)

# Evaluate the model
evalLoss = z1.evaluateModel(testInputs, testOutputs, criterion, workerNum=workerNum)
print("Evaluation Loss:", evalLoss)
print('time:',time.time()-t0)
'#ccc access to tensorboard with "tensorboard --logdir=data" from terminal'
#%% 
runcell=runcell
runcell('imports', 'F:/projects/public github projects/private repos/versatileAnnModule/trainAnn.py')
runcell('define model', 'F:/projects/public github projects/private repos/versatileAnnModule/trainAnn.py')
runcell('make model instance', 'F:/projects/public github projects/private repos/versatileAnnModule/trainAnn.py')
#%%
runcell('regression test', 'F:/projects/public github projects/private repos/versatileAnnModule/trainAnn.py')
#%%
#%% reload model
runcell('imports', 'F:/projects/public github projects/private repos/versatileAnnModule/trainAnn.py')
bestModel=ann.loadModel(r'data\bestModels\a1_UjNC')
# bestModel.evaluateModel(testInputs, testOutputs, criterion)
#%% 

#%%
#%%
#%%
#%%
#%%

#%%

#%%

#%%

#%%


