# ---- imports
"""
original code is from https://github.com/zhykoties/TimeSeries
adaptations were applied in order to make it compatible to our framework
"""
# models\deepAr.py
import sys
import os
parentFolder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parentFolder)
import pandas as pd
import numpy as np
from brazingTorchFolder import ann
import torch
import torch.nn as nn
# ----
class deepArModel(ann):
    def __init__(
        self,
        backcastLen,
        forecastLen,
        embedderInputSize = 4,
        covariatesNum = 4,
        embeddingDim = 20,
        hiddenSize= 16,
        lstmLayers= 3,
        dropoutRate= 0.1):
        super(deepArModel, self).__init__()
        self.forecastLen = 1
        self.backcastLen = backcastLen
        self.timeSeriesMode=True
        self.lstmLayers = lstmLayers
        self.hiddenSize = hiddenSize
        self.embedding = nn.Embedding(embedderInputSize, embeddingDim)

        self.lstm = nn.LSTM(input_size=covariatesNum+embeddingDim,
                            hidden_size=hiddenSize,
                            num_layers=lstmLayers,
                            bias=True,
                            batch_first=False,
                            dropout=dropoutRate)

        self.relu = nn.ReLU()
        self.distributionMu = nn.Linear(hiddenSize * lstmLayers, 1)
        self.distributionPreSigma = nn.Linear(hiddenSize * lstmLayers, 1)
        self.distributionSigma = nn.Softplus()
        
    def getTrainBatchIndexes(self, trainInputs):
        trainIndexes=self.externalKwargs['trainIndexes']
        lenOfIndexes=len(trainIndexes)
        return trainIndexes

    def getEvalBatchIndexes(self, inputs):
        valIndexes=self.externalKwargs['valIndexes']
        lenOfIndexes=len(valIndexes)
        return valIndexes

    def batchDatapreparation(self, indexesIndex, indexes, inputs, outputs, batchSize, mode=None, identifier=None, externalKwargs=None):
        batchIndexes = indexes[indexesIndex*batchSize:indexesIndex*batchSize + batchSize]
        appliedBatchSize = len(batchIndexes)
        
        batchInputs={}
        batchInputs['consumerId']=torch.tensor(inputs.loc[batchIndexes,'consumerId'].values).to(self.device)
        
        batchInputs['allReals']=self.getBackForeCast(inputs, batchIndexes, mode='backcast', colsOrIndexes=externalKwargs['allReals'])
        batchInputs['allReals']=self.stackListOfDfs(batchInputs['allReals'])
        
        batchOutputs=self.getBackForeCast(inputs, batchIndexes, mode='backcast', colsOrIndexes='powerUsage')#kkk this is wrong but its just for now
        batchOutputs=self.stackListOfDfs(batchOutputs)
        batchInputs['outputs']=batchOutputs#kkk sometimes have problem with pass output
        
        outPutMask=None
        return batchInputs, batchOutputs, appliedBatchSize, outPutMask, identifier

    def oneLstmSigmaPass(self, allRealsOneStepVal, onehotEmbed, hidden, cell):
        lstmInput = torch.cat((allRealsOneStepVal, onehotEmbed), dim=2)# shape: seqLen, N, allRealsColDim+embeddingDim
        output, (hidden, cell) = self.lstm(lstmInput, (hidden, cell))# output shape: 1*N*hiddenSize;hidden and cell shape: lstmLayers*N*hiddenSize
        
        # use h from all three layers to calculate mu and sigma
        hiddenPermute = hidden.permute(1, 2, 0).contiguous().view(hidden.shape[1], -1)#shape: N,lstmLayers*hiddenSize
        preSigma = self.distributionPreSigma(hiddenPermute)# shape N,1
        mu = self.distributionMu(hiddenPermute)# shape N,1
        sigma = self.distributionSigma(preSigma)  #ccc softplus to make sure standard deviation is positive
        
        loss += self.externalKwargs['criterion'](mu, sigma, outputs[t])#kkk able to bring loss from args#kkk I need to be able return loss
        return mu, sigma, hidden, cell, loss

    def trainForward(self, x, seqLen):
        allReals = x['allReals'].permute(1, 0, 2)
        consumerId = x['consumerId'].unsqueeze(0)
        outputs = x['outputs'].permute(1, 0)
        
        batchSize = outputs.shape[1]
        hidden = torch.zeros(self.lstmLayers, batchSize, self.hiddenSize, device=self.device)
        cell = torch.zeros(self.lstmLayers, batchSize, self.hiddenSize, device=self.device)
        
        loss = torch.zeros(1, device=self.device)
        onehotEmbed = self.embedding(consumerId)# shape: 1*N*embeddingDim
        for t in range(seqLen):
            "#ccc if output of time t is missing, replace it by output mu from the last time step"
            zeroIndex = (allReals[t, :, 0] == 0)
            if t > 0 and torch.sum(zeroIndex) > 0:
                allReals[t, zeroIndex, 0] = mu[zeroIndex]
                
            mu, sigma, hidden, cell, loss= oneLstmSigmaPass(self, allReals[t].unsqueeze(0), onehotEmbed, hidden, cell)
        return mu, sigma, hidden, cell, loss
    
    def evalForward(self, x, seqLen1, seqLen2, samplesNum, ):#kkk this forward for eval, later after deciding the flow of the model this is gonna take that form
            #kkk we need another loss like mse to check the performance
            mu, sigma, hidden, cell, _=self.trainForward(x, seqLen1)

            allReals = x['allReals'].permute(1, 0, 2)
            consumerId = x['consumerId'].unsqueeze(0)
            outputs = x['outputs'].permute(1, 0)
            
            batchSize = outputs.shape[1]
            onehotEmbed = self.embedding(consumerId)# shape: 1*N*embeddingDim
            samples = torch.zeros(samplesNum, batchSize, seqLen2, device=self.device)

            for j in range(seqLen1):
                decoderHidden = hidden
                decoderCell = cell
                for t in range(self.params.predict_steps):
                    muDecoder, sigmaDecoder, decoderHidden, decoderCell, _ = self.oneLstmSigmaPass(allReals[seqLen1 + t].unsqueeze(0),
                                                                         onehotEmbed, decoderHidden, decoderCell)
                    gaussian = torch.distributions.normal.Normal(muDecoder, sigmaDecoder)
                    pred = gaussian.sample()
                    samples[j, :, t] = pred
                    if t < (seqLen2 - 1):
                        allReals[seqLen2+ t + 1, :, 0] = pred#kkk what does this do

            sampleMu = torch.median(samples, dim=0)[0]
            sampleSigma = samples.std(dim=0)
            return samples, sampleMu, sampleSigma
# ----

# ----

# ----

# ----

