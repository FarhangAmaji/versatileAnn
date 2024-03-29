# ---- imports
# models\nbeats.py
import sys
import os
parentFolder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parentFolder)
from brazingTorchFolder import ann
from models.nbeats_blocks import stack
import torch
# ---- define model
class nBeats(ann):
    def __init__(self, stacks, backcastLength=10, forecastLength=5):
        """#ccc the thetas are output size of backcast and forecast linear layers
        share_weights_in_stack,shareThetas  mean the blocks are the same like rnns
        harmonicsNum: in SeasonalityBlock if we have harmonicsNum, instead of forecastLength, harmonicsNum will be sent
        """
        super(nBeats, self).__init__()
        assert isinstance(stacks, list) and all(isinstance(element, stack) for element in stacks),'all elements in stacks argument should be from class stack'
        self.forecastLength = forecastLength
        self.backcastLength = backcastLength
        self.stacks= torch.nn.ModuleList(stacks)
        
        self.postInitForBlocks(backcastLength, forecastLength)
        
        self.backcastLen=backcastLength
        self.forecastLen=forecastLength
        self.timeSeriesMode=True
        self.isItUnivariate=False
    
    def postInitForBlocks(self, backcastLength, forecastLength):
        for stack_ in self.stacks:
            for block in stack_.blocks:
                block.postInit(backcastLength=backcastLength, forecastLength=forecastLength)
                block.device=self.device
        
    def isBackcastUnivariate(self, backcast):
        if self.isItUnivariate:
            return
        assert len(backcast.shape) == 2, 'this version of nBeats only works with univariate backcast(input data), if needed use ".squeeze(2)" on ur input data'
        self.isItUnivariate=True
    
    def forward(self, backcast):
        self.isBackcastUnivariate(backcast)
        forecast = torch.zeros(size=(backcast.size()[0], self.forecastLength,)).to(self.device)  # maybe batch size here.
        for stack_id in range(len(self.stacks)):
            for block_id in range(len(self.stacks[stack_id].blocks)):
                b, f = self.stacks[stack_id].blocks[block_id](backcast)
                backcast = backcast - b
                forecast = forecast + f
        return forecast
    
    def __str__(self):
        stack_str = ', '.join(str(stack_) for stack_ in self.stacks)
        return f'nBeats(stacks=[{stack_str}], backcastLength={self.backcastLength}, forecastLength={self.forecastLength})'
    
    def __repr__(self):
        return self.__str__()