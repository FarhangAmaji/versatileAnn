# annModule.py
import torch
import torch.nn as nn
import torch.optim as optim
import inspect

class PostInitCaller(type):
    def __call__(cls, *args, **kwargs):
        obj = type.__call__(cls, *args, **kwargs)
        obj.__post_init__()
        return obj

class ann(nn.Module, metaclass=PostInitCaller):
    def __init__(self, patience=10, batchSize=64):
        super(ann, self).__init__()
        self.getInitInpArgs()
        self.device= torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.patience = patience
        self.optimizer = None
        self.batchSize = batchSize
        self.evalBatchSize = 16*batchSize
        
        
        # define model here
        # self.layer1 = self.linLReluDropout(40, 160, dropoutRate=0.5)
        # self.layer2 = self.linLSigmoidDropout(200, 300, dropoutRate=0.2)
        #self.layer3 = nn.Linear(inputSize, 4*inputSize)
        # Add more layers as needed
    def __post_init__(self):
        if isinstance(self, ann) and not type(self) == ann:
            print("ann __post_init__",self.__class__.__name__)
            self.to(self.device)
            self.initOptimizer()

    def forward(self, x):
        # define forward step here
        # x = self.layer1(x)
        # x = self.layer2(x)
        # x = self.layer3(x)
        return x
    
    @property
    def device(self):
        return self._device
    
    @device.setter
    def device(self, value):
        assert value in ['cuda','cpu'],"device={value} must be 'cuda' or 'cpu'"
        self._device = value
        self.to(value)
    
    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, value):
        assert isinstance(value, (optim.Optimizer, type(None))),f'optimizerType={type(value)} is not correct'
        self._optimizer = value
    
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
    
    def initOptimizer(self):
        if list(self.parameters()):
            self.optimizer = optim.Adam(self.parameters(), lr=0.00001)
        else:
            self.optimizer = None
    
    
    def changeLearningRate(self, newLearningRate):
        for param_group in self.optimizer.param_groups:#kkk check does it change
            param_group['lr'] = newLearningRate
    
    def divideLearningRate(self,factor):
        self.changeLearningRate(self.optimizer.param_groups[0]['lr']/factor)
    
    def batchDatapreparation(self,indexesIndex, indexes, inputs, outputs, batchSize=None):
        if batchSize is None:
            batchSize = self.batchSize
        
        batchIndexes = indexes[indexesIndex:indexesIndex + batchSize]
        appliedBatchSize = len(batchIndexes)
        
        batchInputs = inputs[batchIndexes].to(self.device)
        batchOutputs = outputs[batchIndexes].to(self.device)
        return batchInputs, batchOutputs, appliedBatchSize
    
    def trainModel(self, trainInputs, trainOutputs, valInputs, valOutputs, criterion, numEpochs, batchSize, savePath):#kkk add numSamples for ensemble
        # self.havingOptimizerCheck()
        self.train()
        bestValScore = float('inf') #kkk with if should be float('inf') if its loss and 0 if its accuracy
        patienceCounter = 0
        
        for epoch in range(numEpochs):
            trainLoss = 0.0
            
            # Create random indexes for sampling
            indexes = torch.randperm(trainInputs.shape[0])
            for i in range(0, len(trainInputs), self.batchSize):
                self.optimizer.zero_grad()
                
                batchTrainInputs, batchTrainOutputs, appliedBatchSize = self.batchDatapreparation(i, indexes, trainInputs, trainOutputs)
                
                batchTrainOutputsPred = self.forward(batchTrainInputs)
                loss = criterion(batchTrainOutputsPred, batchTrainOutputs)
                
                loss.backward()
                self.optimizer.step()
                
                trainLoss += loss.item()
            
            epochLoss = trainLoss / len(trainInputs)
            print(f"Epoch [{epoch+1}/{numEpochs}], aveItemLoss: {epochLoss:.6f}")
    
    def evaluateModel(self, inputs, outputs, criterion):
        self.eval()
        evalLoss = 0.0
        
        with torch.no_grad():
            indexes = torch.arange(len(inputs))
            for i in range(0, len(inputs), self.evalBatchSize):
                batchEvalInputs, batchEvalOutputs, appliedBatchSize = self.batchDatapreparation(i, indexes, inputs, outputs, batchSize=self.evalBatchSize)
                
                batchEvalOutputsPred = self.forward(batchEvalInputs)
                loss = criterion(batchEvalOutputsPred, batchEvalOutputs)
                
                evalLoss += loss.item()
            
            evalLoss /= inputs.shape[0]
            return evalLoss