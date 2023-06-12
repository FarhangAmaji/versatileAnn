#%% imports
# models\nbeats.py
import sys
import os
parentFolder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parentFolder)
from versatileAnn import ann
import torch
from torch import nn
import numpy as np
#%% define model
#kkk create blockInfo specifier
class NBeatsNet(ann):
    def __init__(self, stacks, forecastLength=5, backcastLength=10):
        """#ccc the thetas are output size of backcast and forecast linear layers
        share_weights_in_stack,shareThetas  mean the blocks are the same like rnns
        harmonicsNum: in SeasonalityBlock if we have harmonicsNum, instead of forecastLength, harmonicsNum will be sent
        """
        super(NBeatsNet, self).__init__()
        assert isinstance(stacks, list) and all(isinstance(element, stack) for element in stacks),'all elements in stacks argument should be from class stack'
        self.forecastLength = forecastLength
        self.backcastLength = backcastLength
        self.stacks = stacks
        [block.postInit(backcastLength=backcastLength, forecastLength=forecastLength) for block in stack.blocks for stack in self.stacks]

    def forward(self, backcast):
        backcast = squeeze_last_dim(backcast)#kkk add to check only univariate
        forecast = torch.zeros(size=(backcast.size()[0], self.forecastLength,))  # maybe batch size here.
        for stack_id in range(len(self.stacks)):
            for block_id in range(len(self.stacks[stack_id])):
                b, f = self.stacks[stack_id][block_id](backcast)
                backcast = backcast - b
                forecast = forecast + f
                block_type = self.stacks[stack_id][block_id].__class__.__name__
                layer_name = f'stack_{stack_id}-{block_type}_{block_id}'
        return backcast, forecast


