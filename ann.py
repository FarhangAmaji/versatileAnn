# dropoutNetModule.py
import torch
import torch.nn as nn
import inspect

class ann(nn.Module):
    def __init__(self):
        super(ann, self).__init__()
        self.getInitInpArgs()
        # define model here
        
        
        # self.layer1 = self.linLReluDropout(100, 200, dropoutRate=0.5)
        # self.layer2 = self.linLSigmoidDropout(200, 300, dropoutRate=0.2)
        #self.layer3 = nn.Linear(inputSize, 4*inputSize)
        # Add more layers as needed
        
    
    def forward(self, x):
        # define forward step here
        pass
        
    def getInitInpArgs(self):
        args, _, _, values = inspect.getargvalues(inspect.currentframe().f_back)
        self.inputArgs = {arg: values[arg] for arg in args if arg != 'self'}
        
    def linLReluDropout(self, innerSize, outterSize, leakyReluNegSlope=0.05, dropoutRate=False):
        activation = nn.LeakyReLU(negative_slope=leakyReluNegSlope)
        return self.linActivationDropout(innerSize, outterSize, activation, dropoutRate)
    
    def linLSigmoidDropout(self, innerSize, outterSize, dropoutRate=False):
        activation = nn.Sigmoid()
        return self.linActivationDropout(innerSize, outterSize, activation, dropoutRate)
    
    def linActivationDropout(self, innerSize, outterSize, activation, dropoutRate=False):
        '#ccc instead of defining many times of leakyRelu and dropOuts I do them at once'
        layer = nn.Sequential(
            nn.Linear(innerSize, outterSize),
            activation
        )
        if type(dropoutRate) in (int, float):
            assert 0 <= dropoutRate <= 1, f'dropoutRate={dropoutRate} is not between 0 and 1'
            drLayer = nn.Dropout(p=dropoutRate)
            return nn.Sequential(layer, drLayer)
        else:
            assert type(dropoutRate) in (bool),f'dropoutRate={dropoutRate} is not a int, float or bool'
        return layer
