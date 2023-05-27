import torch
import torch.nn as nn
import torch.optim as optim
import inspect

class ann(nn.Module):
    def __init__(self, patience=10):
        super(ann, self).__init__()
        self.getInitInpArgs()
        self.device= torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        self.patience = patience
        
        
        # define model here
        # self.layer1 = self.linLReluDropout(40, 160, dropoutRate=0.5)
        # self.layer2 = self.linLSigmoidDropout(200, 300, dropoutRate=0.2)
        #self.layer3 = nn.Linear(inputSize, 4*inputSize)
        # Add more layers as needed
        
    
    def forward(self, x):
        # define forward step here
        pass
        # x = self.layer1(x)
        # x = self.layer2(x)
        # x = self.layer3(x)
        # return x
    def getInitInpArgs(self):
        args, _, _, values = inspect.getargvalues(inspect.currentframe().f_back)
        "#ccc patience and optimizer's change should not affect that doesnt let the model rerun"
        excludeInputArgs = ['patience', 'optimizer', 'device']
        for eia in excludeInputArgs:
            if eia in args:
                args.remove(eia)
        self.inputArgs = {arg: values[arg] for arg in args if arg != 'self'}
        
    def linLReluDropout(self, innerSize, outterSize, leakyReluNegSlope=0.05, dropoutRate=False):
        activation = nn.LeakyReLU(negative_slope=leakyReluNegSlope)
        return self.linActivationDropout(innerSize, outterSize, activation, dropoutRate)
    
    def linLSigmoidDropout(self, innerSize, outterSize, dropoutRate=False):
        activation = nn.Sigmoid()
        return self.linActivationDropout(innerSize, outterSize, activation, dropoutRate)
    
    def linActivationDropout(self, innerSize, outterSize, activation, dropoutRate=None):
        '#ccc instead of defining many times of leakyRelu and dropOuts I do them at once'
        layer = nn.Sequential(
            nn.Linear(innerSize, outterSize),
            activation
        )
        if dropoutRate:
            assert type(dropoutRate) in (int, float),f'dropoutRateType={type(dropoutRate)} is not int or float'
            assert 0 <= dropoutRate <= 1, f'dropoutRate={dropoutRate} is not between 0 and 1'
            drLayer = nn.Dropout(p=dropoutRate)
            return nn.Sequential(layer, drLayer)
        return layer
    
    def changeLearningRate(self, newLearningRate):
        for param_group in self.optimizer.param_groups:#kkk check does it change
            param_group['lr'] = newLearningRate
    
    def divideLearningRate(self,factor):
        self.changeLearningRate(self.optimizer.param_groups[0]['lr']/factor)
    
    def trainModel(self, trainInputs, trainOutputs, valInputs, valOutputs, optimizer, criterion, numEpochs, batchSize, savePath):#kkk add numSamples for ensemble
        self.train()
        bestValScore = float('inf') #kkk with if should be float('inf') if its loss and 0 if its accuracy
        patienceCounter = 0
        
        for epoch in range(numEpochs):
            trainLoss = 0.0
            
            # Create random indexes for sampling
            indexes = torch.randperm(trainInputs.shape[0])
            for i in range(0, len(trainInputs), batchSize):
                optimizer.zero_grad()
                # Create batch indexes
                batchIndexes = indexes[i:i+batchSize]
                
                batchTrainInputs = trainInputs[batchIndexes].to(device)
                batchTrainOutputs = trainOutputs[batchIndexes].to(device)
                
                batchTrainOutputsPred = model.forward(batchTrainInputs)
                loss = criterion(batchTrainOutputsPred, batchTrainOutputs)
                
                loss.backward()
                optimizer.step()
                
                runningLoss += loss.item()
            
            epochLoss = runningLoss / len(trainInputs)
            print(f"Epoch [{epoch+1}/{numEpochs}], aveItemLoss: {epochLoss:.6f}")
#%%
z1=ann()
z2=z1.linLReluDropout(4, 4,dropoutRate=.9)