#%% imports
# models\nbeats_blocks.py
import sys
import os
parentFolder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parentFolder)
import torch
from torch import nn
import numpy as np
#%% define model
#kkk correct comments
class Block(nn.Module):#kkk check args
    def __init__(self, units, thetasDim, shareThetas=False, harmonicsNum=None):
        super(Block, self).__init__()
        if isinstance(self, TrendBlock):
            assert thetasDim <= 4, 'thetasDim for TrendBlock must be <=4'
        if harmonicsNum:
            assert isinstance(self, SeasonalityBlock),'only SeasonalityBlock can have harmonicsNum'
        self.units = units
        self.thetasDim = thetasDim
        self.shareThetas = shareThetas#yyy what does this do: like rnn there is only 1 layer for backcast and forecast final layer
        self.fc2 = nn.Linear(units, units)
        self.fc3 = nn.Linear(units, units)
        self.fc4 = nn.Linear(units, units)
        if self.shareThetas:
            self.thetaBackcastFc = nn.Linear(units, thetasDim, bias=False)
            self.thetaForecastFc = self.thetaBackcastFc
        else:
            self.thetaBackcastFc = nn.Linear(units, thetasDim, bias=False)
            self.thetaForecastFc = nn.Linear(units, thetasDim, bias=False)
    
    def postInit(self, backcastLength=10, forecastLength=5):
        self.fc1 = nn.Linear(backcastLength, self.units)#jjj this just takes univariate series
        self.backcastLength = backcastLength
        self.forecastLength = forecastLength
        self.backcastLinspace = self.linearSpace(backcastLength, forecastLength, is_forecast=False)
        self.forecastLinspace = self.linearSpace(backcastLength, forecastLength, is_forecast=True)
        self.lRelu = nn.LeakyReLU(negative_slope=0.05)

    def linearSpace(self, backcastLength, forecastLength, is_forecast=True):
        horizon = forecastLength if is_forecast else backcastLength
        return np.arange(0, horizon) / horizon
    
    def forward(self, x):
        x = self.lRelu(self.fc1(x))
        x = self.lRelu(self.fc2(x))
        x = self.lRelu(self.fc3(x))
        x = self.lRelu(self.fc4(x))
        return x

    def __str__(self):
        block_type = type(self).__name__
        return f'{block_type}(units={self.units}, thetasDim={self.thetasDim}, ' \
               f'backcastLength={self.backcastLength}, forecastLength={self.forecastLength}, ' \
               f'shareThetas={self.shareThetas}\nlayers={self._modules}) at @{id(self)}\n\n'
               
    def __repr__(self):
        return self.__str__()

class SeasonalityBlock(Block):
    def __init__(self, units, thetasDim, shareThetas=True, harmonicsNum=None):#yyy thetasDim==forecastLength or harmonicsNum
        self.harmonicsNum=harmonicsNum
        
        if not shareThetas:
            print('for SeasonalityBlock its recommended to make shareThetas=True')#kkk for SeasonalityBlock should pass forecastLength or harmonicsNum for thetasDim
        super(SeasonalityBlock, self).__init__(units, thetasDim, shareThetas, harmonicsNum)
    
    def postInit(self, backcastLength=10, forecastLength=5):
        thetasDim=self.harmonicsNum or forecastLength
        super(SeasonalityBlock, self).__init__(self.units, thetasDim, self.shareThetas, self.harmonicsNum)
        super().postInit(backcastLength=backcastLength, forecastLength=forecastLength)
    
    def seasonalityModel(self, thetas, linspace):
        p = thetas.size()[-1]
        assert p <= thetas.shape[1], 'thetasDim is too big.'#kkk do assertion sooner in forward
        p1, p2 = (p // 2, p // 2) if p % 2 == 0 else (p // 2, p // 2 + 1)
        s1 = torch.tensor(np.array([np.cos(2 * np.pi * i * linspace) for i in range(p1)])).float()#shape: forecastLen//2(+1) * forecastLen # H/2-1
        s2 = torch.tensor(np.array([np.sin(2 * np.pi * i * linspace) for i in range(p2)])).float()
        S = torch.cat([s1, s2])
        return thetas.mm(S)
    
    def forward(self, x):
        x = super(SeasonalityBlock, self).forward(x)
        backcast = self.seasonalityModel(self.thetaBackcastFc(x), self.backcastLinspace)
        forecast = self.seasonalityModel(self.thetaForecastFc(x), self.forecastLinspace)
        return backcast, forecast

class TrendBlock(Block):
    def __init__(self, units, thetasDim, shareThetas=True):
        if not shareThetas:
            print('for TrendBlock its recommended to make shareThetas=True')
        super(TrendBlock, self).__init__(units, thetasDim, shareThetas=shareThetas)

    def trendModel(self, thetas, linspace):
        p = thetas.size()[-1]
        T = torch.tensor(np.array([linspace ** i for i in range(p)])).float()
        return thetas.mm(T)
    
    def forward(self, x):
        x = super(TrendBlock, self).forward(x)
        backcast = self.trendModel(self.thetaBackcastFc(x), self.backcastLinspace)
        forecast = self.trendModel(self.thetaForecastFc(x), self.forecastLinspace)
        return backcast, forecast

class GenericBlock(Block):
    def __init__(self, units, thetasDim, shareThetas=False):
        if shareThetas:
            print('for GenericBlock its recommended to make shareThetas=False')
        super(GenericBlock, self).__init__(units, thetasDim, shareThetas=shareThetas)#yyy doesnt allow to shareThetas

    def postInit(self, backcastLength=10, forecastLength=5):
        super().postInit(backcastLength=backcastLength, forecastLength=forecastLength)
        self.backcastFc = nn.Linear(self.thetasDim, backcastLength)#yyy same as block except it has 2 more layers(backcastFc and forecastFc)
        self.forecastFc = nn.Linear(self.thetasDim, forecastLength)

    def forward(self, x):
        # no constraint for generic arch.
        x = super(GenericBlock, self).forward(x)

        theta_b = self.thetaBackcastFc(x)
        theta_f = self.thetaForecastFc(x)

        backcast = self.backcastFc(theta_b)  # generic. 3.3.
        forecast = self.forecastFc(theta_f)  # generic. 3.3.

        return backcast, forecast

class stack(nn.Module):
    def __init__(self, blocks):
        super(stack, self).__init__()
        for block in blocks:
            assert isinstance(block, Block),'blocks should be instance of Block class'
            assert not type(block) == Block,'blocks should be instance of children of Block class and not itself'
        self.blocks=blocks
        
    def __str__(self):
        blocks_str = ', '.join(str(block) for block in self.blocks)
        return f'stack(blocks=[{blocks_str}])'
    
    def __repr__(self):
        return self.__str__()
    
class stackWithSharedWeights(stack):
    def __init__(self, block, stackNumOfBlocks):
        super(stackWithSharedWeights, self).__init__([block for _ in range(stackNumOfBlocks)])
