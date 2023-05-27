# dropoutNetModule.py
import torch
import torch.nn as nn
import inspect

class ann(nn.Module):
    def __init__(self):
        super(ann, self).__init__()
        self.getInitInpArgs()
        # define model here
    
    def forward(self, x):
        # define forward step here
        pass
        
    def getInitInpArgs(self):
        args, _, _, values = inspect.getargvalues(inspect.currentframe().f_back)
        self.inputArgs = {arg: values[arg] for arg in args if arg != 'self'}
