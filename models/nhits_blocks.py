#%% imports
# models\nbeats.py
import torch
import torch.nn as nn
import numpy as np
#%%
class IdentityBasis(nn.Module):
    def __init__(self, forecastLen, backcastLen, interpolationMode):
        super().__init__()
        assert (interpolationMode in ["linear", "nearest"]) or ("cubic" in interpolationMode)
        self.backcastLen = backcastLen
        self.forecastLen = forecastLen
        self.interpolationMode = interpolationMode

    def forward(self, theta):
        """
        the last layer has size of "backcastLen + max(forecastLen // nFreqDownsample, 1)" so the forecast part(knots) 
        here will be interpolated to have the actual forecastLen
        """
        backcast = theta[:, : self.backcastLen]#ccc this is for backcast part
        knots = theta[:, self.backcastLen :]#ccc this is the forecast part which because of "max(forecastLen // nFreqDownsample, 1)" is usually less than the actual forecastLen

        knots = knots.reshape(len(knots), 1, -1)
        if self.interpolationMode in ["nearest", "linear"]:
            forecast = nn.functional.interpolate(knots, size=self.forecastLen, mode=self.interpolationMode)
        elif "cubic" in self.interpolationMode:
            batchSize = len(backcast)
            knots = knots[:, None, :, :]
            forecast = torch.zeros((len(knots), self.forecastLen)).to(knots.device)
            nBatches = int(np.ceil(len(knots) / batchSize))
            for i in range(nBatches):
                forecast_i = nn.functional.interpolate(knots[i * batchSize : (i + 1) * batchSize], size=self.forecastLen, mode="bicubic")# shape: N * 1 * forecastLen * forecastLen
                forecast[i * batchSize : (i + 1) * batchSize] += forecast_i[:, 0, 0, :]# shape: N * 1 * forecastLen * forecastLen -> N * forecastLen
            forecast = forecast[:, None, :]  # N * forecastLen -> N * 1 * forecastLen

        forecast = forecast.permute(0, 2, 1)# shape: N * 1 * forecastLen -> N * forecastLen * 1
        return backcast, forecast
#%%
ACTIVATIONS = ["ReLU", "Softplus", "Tanh", "SELU", "LeakyReLU", "PReLU", "Sigmoid"]

POOLING = ["MaxPool1d", "AvgPool1d"]

class nHitsBlock(nn.Module):
    def __init__(self, nFreqDownsample, mlpUnits, nPoolKernelSize, interpolationMode='linear', pooling_mode='MaxPool1d', dropoutRate=0, activation='LeakyReLU'):
        super().__init__()

        self.nPoolKernelSize = nPoolKernelSize
        self.nFreqDownsample = nFreqDownsample
        self.mlpUnits = mlpUnits
        self.interpolationMode = interpolationMode
        self.dropoutRate = dropoutRate

        assert activation in ACTIVATIONS, f"{activation} is not in {ACTIVATIONS}"
        assert pooling_mode in POOLING, f"{pooling_mode} is not in {POOLING}"

        self.activ = getattr(nn, activation)()

        self.pooling_layer = getattr(nn, pooling_mode)(kernel_size=nPoolKernelSize, stride=nPoolKernelSize, ceil_mode=True)

    def postInit(self, backcastLen, forecastLen, futureInputSize=0, historyInputSize=0, staticInputSize=0):
        """
        this is gonna be called from model because it has all args this func needs
        """
        pooledHistSize = int(np.ceil(backcastLen / self.nPoolKernelSize))
        pooledFutrSize = int(np.ceil((backcastLen + forecastLen) / self.nPoolKernelSize))
        nTheta = backcastLen + max(forecastLen // self.nFreqDownsample, 1)
        
        firstLayerInputSize = (
            pooledHistSize#ccc this one is for target column data and the reason its called pooledHistSize because hist exogenous has same input/nPoolKernelSize length
            + historyInputSize * pooledHistSize#ccc note gets to num of history exogenous cols
            + futureInputSize * pooledFutrSize#ccc note gets to num of future exogenous cols
            + staticInputSize#ccc we just put the static exogenous tensor here
        )
        
        # Block MLPs
        hiddenLayers = [nn.Linear(in_features=firstLayerInputSize, out_features=self.mlpUnits[0][0]), self.activ]
        for layer in self.mlpUnits:
            hiddenLayers.append(nn.Linear(in_features=layer[0], out_features=layer[1]))
            hiddenLayers.append(self.activ)

            if self.dropoutRate > 0:
                hiddenLayers.append(nn.Dropout(p=self.dropoutRate))
        
        outputLayer = [nn.Linear(in_features=self.mlpUnits[-1][1], out_features=nTheta)]
        layers = hiddenLayers + outputLayer
        self.layers = nn.Sequential(*layers)
        self.basis = IdentityBasis(forecastLen, backcastLen, self.interpolationMode)
        
    def forward(self, targetY, futureExogenous, historyExogenous, staticExogenous):
        '''
        first the pooled targetY, pooled historyExogenous, pooled futureExogenous and staticExogenous are put along the first dimension of each data(each one of batch data)
        then layers are applied then the basis is applied for interpolation
        '''
        # Pooling
        # Pool1d needs 3D input, (N,C,L), adding Channel dimension
        targetY = targetY.unsqueeze(1)
        targetY = self.pooling_layer(targetY)
        targetY = targetY.squeeze(1)

        batchSize = len(targetY)
        if self.historyInputSize > 0:
            historyExogenous = historyExogenous.permute(0, 2, 1)# shape: N * backcastLen * historyInputSize -> N * historyInputSize * backcastLen
            historyExogenous = self.pooling_layer(historyExogenous)
            historyExogenous = historyExogenous.permute(0, 2, 1)  # shape: N * historyInputSize * backcastLen -> N * backcastLen * historyInputSize
            targetY = torch.cat((targetY, historyExogenous.reshape(batchSize, -1)), dim=1)#flatten historyExogenous and added to targetY in dimesion of sequence
            '#ccc we append the pooled historyExogenous after the pooled targetY; same for futureExogenous'
            '#ccc it puts the historyExogenous after the targetY sequence so now targetY shape is: N * pooledHistSize + historyInputSize * pooledHistSize'

        if self.futureInputSize > 0:
            futureExogenous = futureExogenous.permute(0, 2, 1)# shape: N * (backcastLen+forecastLen) * futureInputSize -> N * futureInputSize * (backcastLen+forecastLen)
            futureExogenous = self.pooling_layer(futureExogenous)
            futureExogenous = futureExogenous.permute(0, 2, 1)# shape: N * futureInputSize * (backcastLen+forecastLen) -> N * (backcastLen+forecastLen) * futureInputSize
            targetY = torch.cat((targetY, futureExogenous.reshape(batchSize, -1)), dim=1)#flatten futureExogenous and added to targetY in dimesion of sequence

        if self.staticInputSize > 0:
            targetY = torch.cat((targetY, staticExogenous.reshape(batchSize, -1)), dim=1)#shape: N * firstLayerInputSize 
            #flatten staticExogenous and added to targetY in dimesion of sequence
            '#ccc we append the staticExogenous after the pooled targetY, pooled historyExogenous, pooled futureExogenous'

        theta = self.layers(targetY)
        backcast, forecast = self.basis(theta)
        return backcast, forecast
#%%

#%%

#%%

