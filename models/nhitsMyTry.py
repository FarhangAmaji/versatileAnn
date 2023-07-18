#%% thing to check
'''
pass the blocks again like nbeats; here we have no stack because we only have 1 type of blocks
make common args of block in postInit(basis,h,input_size,futureInputSize,historyInputSize,staticInputSize)
postInit for blocks would be called by Model because it has all args needed for postInit
make changes for ann because this doesnt have val and best val stop(maybe dont make problem but check it)
check if with nn.ModuleList still we need to add parameters manually(check the named_params)


prep data for train:
    at least for train (probably for predict also) we know that the futureExogenous len is input+h and for historyExogenous len is input;#jjj where this is applied in orig code
prep data for predict
'''
#%% imports
"""
original code is from https://github.com/Nixtla/neuralforecast/blob/main/neuralforecast/models/nhits.py
adaptations were applied in order to make it compatible to our framework
"""
# models\nhitsMyTry.py
import sys
import os
parentFolder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parentFolder)
from versatileAnn import ann
import torch.nn as nn
# %% ../../nbs/models.nhits.ipynb 10
class nHits(ann):
    """NHITS #kkk add own comments

    The Neural Hierarchical Interpolation for Time Series (NHITS), is an MLP-based deep
    neural architecture with backward and forward residual links. NHITS tackles volatility and
    memory complexity challenges, by locally specializing its sequential predictions into
    the signals frequencies with hierarchical interpolation and pooling.

    """
    def __init__(self, blocks, backcastLen, forecastLen, futureExogenousCols=None, historyExogenousCols=None, staticExogenousCols=None, flipTarget=False):
        super(nHits, self).__init__()
        assert isinstance(blocks, list),'blocks should be a list'
        self.blocks = nn.ModuleList(blocks)
        
        self.futureInputSize = len(futureExogenousCols) if futureExogenousCols else 0
        self.historyInputSize = len(historyExogenousCols) if historyExogenousCols else 0
        self.staticInputSize = len(staticExogenousCols) if staticExogenousCols else 0
        
        self.postInitForBlocks(backcastLen, forecastLen, futureInputSize=self.futureInputSize, historyInputSize=self.historyInputSize, staticInputSize=self.staticInputSize)
        
        self.forecastLen = forecastLen
        self.backcastLen = backcastLen
        self.timeSeriesMode=True
    
    def postInitForBlocks(self, backcastLen, forecastLen, futureInputSize=0, historyInputSize=0, staticInputSize=0):
        for block in self.blocks:
            block.postInit(backcastLen, forecastLen, futureInputSize, historyInputSize, staticInputSize)
            block.futureInputSize = self.futureInputSize
            block.historyInputSize = self.historyInputSize
            block.staticInputSize = self.staticInputSize

    def forward(self, inputs):
        # Parse inputs
        targetY = inputs["target"]# shape: N * backcastLen
        mask = inputs["mask"]# shape: N * backcastLen#ccc except the sequences which have last data should the values are 1.0 and those have 0.0
        futureExogenous = inputs["futureExogenous"]# shape: N * (backcastLen+forecastLen) * futureInputSize
        historyExogenous = inputs["historyExogenous"]# shape: N * backcastLen * historyInputSize
        staticExogenous = inputs["staticExogenous"]# shape: N * backcastLen * staticInputSize

        residuals = targetY.flip(dims=(-1,))
        mask = mask.flip(dims=(-1,))
        forecast = targetY[:, -1:, None]  # Level with Naive1
        for i, block in enumerate(self.blocks):
            backcast, blockForecast = block(targetY = residuals,
                futureExogenous = futureExogenous,
                historyExogenous = historyExogenous,
                staticExogenous = staticExogenous)
            residuals = (residuals - backcast) * mask
            forecast = forecast + blockForecast
        return forecast.squeeze(-1)
#%%

#%%

#%%

#%%

#%%

