#%% imports
# models\nbeats.py
import torch.nn as nn
#%%
class IdentityBasis(nn.Module):
    def __init__(self, tsOutputWindow, tsInputWindow, interpolationMode):
        super().__init__()
        assert (interpolationMode in ["linear", "nearest"]) or ("cubic" in interpolationMode)
        self.tsInputWindow = tsInputWindow
        self.tsOutputWindow = tsOutputWindow
        self.interpolationMode = interpolationMode

    def forward(self, theta):
        """
        the last layer has size of "tsInputWindow + max(tsOutputWindow // nFreqDownsample, 1)" so the predict part(knots) here will be interpolated to have the final predict size(self.tsInputWindow)
        """
        backcast = theta[:, : self.tsOutputWindow]#ccc this is the input part
        knots = theta[:, self.tsOutputWindow :]#ccc this is the predict part which is usually less than the final predict size("max(tsOutputWindow // nFreqDownsample, 1)")

        knots = knots.reshape(len(knots), 1, -1)#kkk changed self.outFeatures to 1
        if self.interpolationMode in ["nearest", "linear"]:
            forecast = nn.functional.interpolate(knots, size=self.tsInputWindow, mode=self.interpolationMode)
        elif "cubic" in self.interpolationMode:
            batchSize = len(backcast)
            knots = knots[:, None, :, :]
            forecast = torch.zeros((len(knots), self.tsInputWindow)).to(knots.device)
            nBatches = int(np.ceil(len(knots) / batchSize))
            for i in range(nBatches):
                forecast_i = nn.functional.interpolate(knots[i * batchSize : (i + 1) * batchSize], size=self.tsInputWindow, mode="bicubic")
                forecast[i * batchSize : (i + 1) * batchSize] += forecast_i[:, 0, 0, :]  # [B,None,H,H] -> [B,H]
            forecast = forecast[:, None, :]  # [B,H] -> [B,None,H]#kkk main doesnt have this

        forecast = forecast.permute(0, 2, 1)# shape: [B,Q,H] -> [B,H,Q]#kkk main doesnt have this
        return backcast, forecast
#%%
ACTIVATIONS = ["ReLU", "Softplus", "Tanh", "SELU", "LeakyReLU", "PReLU", "Sigmoid"]

POOLING = ["MaxPool1d", "AvgPool1d"]

class NHITSBlock(nn.Module):
    def __init__(self, nFreqDownsample, mlpUnits, nPoolKernelSize, interpolationMode='linear', pooling_mode='MaxPool1d', dropoutRate=1, activation='LeakyReLU'):
        super().__init__()

        self.nPoolKernelSize = nPoolKernelSize
        self.nFreqDownsample = nFreqDownsample
        self.interpolationMode = interpolationMode
        self.dropoutRate = dropoutRate
        self.futrInputSize = futrInputSize
        self.histInputSize = histInputSize
        self.statInputSize = statInputSize

        assert activation in ACTIVATIONS, f"{activation} is not in {ACTIVATIONS}"
        assert pooling_mode in POOLING, f"{pooling_mode} is not in {POOLING}"

        activ = getattr(nn, activation)()

        self.pooling_layer = getattr(nn, pooling_mode)(kernelSize=nPoolKernelSize, stride=nPoolKernelSize, ceil_mode=True)

    def postInit(self, basis, tsOutputWindow, tsInputWindow, futrInputSize=0, histInputSize=0, statInputSize=0):
        """
        this is gonna be called from model because it has all args this func needs
        """
        pooledHistSize = int(np.ceil(tsInputWindow / self.nPoolKernelSize))
        pooledFutrSize = int(np.ceil((tsInputWindow + tsOutputWindow) / self.nPoolKernelSize))
        nTheta = tsInputWindow + max(tsOutputWindow // self.nFreqDownsample, 1)
        
        firstLayerInputSize = (
            pooledHistSize#ccc this one is for target column data and the reason its called pooledHistSize because hist exogenous has same input/nPoolKernelSize length
            + histInputSize * pooledHistSize#ccc note gets to num of history exogenous cols
            + futrInputSize * pooledFutrSize#ccc note gets to num of future exogenous cols
            + statInputSize#ccc we just put the static exogenous matrix here
        )
        
        # Block MLPs
        hiddenLayers = [nn.Linear(in_features=firstLayerInputSize, out_features=mlpUnits[0][0])]
        for layer in mlpUnits:
            hiddenLayers.append(nn.Linear(in_features=layer[0], out_features=layer[1]))
            hiddenLayers.append(activ)

            if self.dropoutRate > 0:
                hiddenLayers.append(nn.Dropout(p=self.dropoutRate))
        
        outputLayer = [nn.Linear(in_features=mlpUnits[-1][1], out_features=nTheta)]
        layers = hiddenLayers + outputLayer
        self.layers = nn.Sequential(*layers)
        self.basis = IdentityBasis(tsOutputWindow, tsInputWindow, self.interpolationMode)
        
    def forward(self, targetY, futureExogenous, historyExogenous, staticExogenous):
        '''
        first the pooled targetY, pooled historyExogenous, pooled futureExogenous and staticExogenous are put along the first dimension of each data(each one of batch data)
        then layers are applied then the basis is applied for interpolation
        '''
        # Pooling
        # Pool1d needs 3D input, (B,C,L), adding C dimension
        targetY = targetY.unsqueeze(1)
        targetY = self.pooling_layer(targetY)
        targetY = targetY.squeeze(1)

        # Flatten MLP inputs [B, L+H, C] -> [B, (L+H)*C]
        # Contatenate [ Y_t, | X_{t-L},..., X_{t} | F_{t-L},..., F_{t+H} | S ]
        batchSize = len(targetY)
        if self.histInputSize > 0:
            historyExogenous = historyExogenous.permute(0, 2, 1)  # [B, L, C] -> [B, C, L]
            historyExogenous = self.pooling_layer(historyExogenous)
            historyExogenous = historyExogenous.permute(0, 2, 1)  # [B, C, L] -> [B, L, C]
            targetY = torch.cat((targetY, historyExogenous.reshape(batchSize, -1)), dim=1)
            '#ccc we append the pooled historyExogenous after the pooled targetY; same for futureExogenous'
            #yyy it puts the historyExogenous after the targetY sequence so now targetY shape is: N * tsInputWindow//nPoolKernelSize + historyExogenousLen * historyExogenousNum//nPoolKernelSize

        if self.futrInputSize > 0:
            futureExogenous = futureExogenous.permute(0, 2, 1)  # [B, L, C] -> [B, C, L]
            futureExogenous = self.pooling_layer(futureExogenous)
            futureExogenous = futureExogenous.permute(0, 2, 1)  # [B, C, L] -> [B, L, C]
            targetY = torch.cat((targetY, futureExogenous.reshape(batchSize, -1)), dim=1)

        if self.statInputSize > 0:
            targetY = torch.cat((targetY, staticExogenous.reshape(batchSize, -1)), dim=1)#shape: N * (len(pooled targetY) + len(pooled historyExogenous) + len(pooled futureExogenous) + statInputSize) 
            '#ccc we append the staticExogenous after the pooled targetY, pooled historyExogenous, pooled futureExogenous'

        # Compute local projection weights and projection
        theta = self.layers(targetY)
        backcast, forecast = self.basis(theta)
        return backcast, forecast
#%%

#%%

#%%

