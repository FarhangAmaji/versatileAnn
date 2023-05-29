#%% imports
# trainAnn.py
import os
baseFolder = os.path.dirname(os.path.abspath(__file__))
os.chdir(baseFolder)
from annModule import ann
import torch
import torch.optim as optim
#%% define model
class myAnn(ann):
    def __init__(self):
        super(myAnn, self).__init__()
        self.layer1 = self.linLReluDropout(40, 160, dropoutRate=0.5)
        self.layer2 = self.linLReluDropout(160, 160, dropoutRate=0.8)
        self.layer3 = self.linLReluDropout(160, 1)
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x
#%% make model instance
z1=myAnn()
#%%
'#ccc how to set optimizer manually'
# z1.lr=0.001
# z1.learningRate=0.001
# z1.changeLearningRate(0.001)
# z1.optimizer=optim.Adam(z1.parameters(), lr=0.4)
# z1.batchSize=32
# z1.evalBatchSize=1024
# z1.device=torch.device(type='cpu')

# z1.patience=10
z1.saveOnDiskPeriod=1
#%% regression test
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
z1.trainModel(trainInputs, trainOutputs, testInputs, testOutputs, criterion, numEpochs=10, savePath=r'data\bestModels\a1')

# Evaluate the model
evalLoss = z1.evaluateModel(testInputs, testOutputs, criterion)
print("Evaluation Loss:", evalLoss)
print('time:',time.time()-t0)
#%% 
runcell('imports', 'F:/projects/public github projects/private repos/versatileAnnModule/trainAnn.py')
runcell('define model', 'F:/projects/public github projects/private repos/versatileAnnModule/trainAnn.py')
runcell('make model instance', 'F:/projects/public github projects/private repos/versatileAnnModule/trainAnn.py')
runcell('regression test', 'F:/projects/public github projects/private repos/versatileAnnModule/trainAnn.py')
#%%
#%%
#%%
#%%
#%%
#%%
