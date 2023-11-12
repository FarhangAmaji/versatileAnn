import sys
import os
parentFolder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parentFolder)
from typing import Dict, List
import pandas as pd
import torch
from torch import nn
from versatileAnn import ann
from models.temporalFusionTransformers_components import addNorm, gatedLinearUnit, gateAddNorm, gatedResidualNetwork, variableSelectionNetwork, interpretableMultiHeadAttention, multiEmbedding
# ----
class temporalFusionTransformerModel(ann):
    def __init__(
        self,
        backcastLen,
        forecastLen,
        hiddenSize= 16,
        lstmLayers= 1,
        dropoutRate= 0.1,
        outputSize= 7,
        targetsNum= 1,
        attentionHeadSize = 4,
        maxEncoderLength = 10,
        staticCategoricals: List[str] = [],
        staticReals: List[str] = [],
        timeVaryingCategoricalsEncoder: List[str] = [],
        timeVaryingCategoricalsDecoder: List[str] = [],
        categoricalVariableGroups: Dict[str, List[str]] = {},
        timeVaryingRealsEncoder: List[str] = [],
        timeVaryingRealsDecoder: List[str] = [],
        allReals: List[str] = [],
        allCategoricalsNonGrouped: List[str] = [],
        embeddingSizes = {},#ccc the expected type is Dict[str, List[int, int]]
        causalAttention: bool = True):

        super(temporalFusionTransformerModel, self).__init__()
        self.forecastLen = forecastLen
        self.backcastLen = backcastLen
        self.timeSeriesMode=True
        
        self.causalAttention = causalAttention
        self.staticVariables=staticCategoricals+staticReals
        self.encoderVariables=timeVaryingCategoricalsEncoder+timeVaryingRealsEncoder
        self.decoderVariables=timeVaryingCategoricalsDecoder+timeVaryingRealsDecoder
        self.allReals= allReals
        self.hiddenSize= hiddenSize
        self.targetsNum= targetsNum
        self.lstmLayers= lstmLayers
        # processing inputs
        # embeddings
        self.inputEmbeddings = multiEmbedding(
            embeddingSizes=embeddingSizes,
            categoricalVariableGroups=categoricalVariableGroups,
            allCategoricalsNonGrouped=allCategoricalsNonGrouped,
            maxEmbeddingSize=hiddenSize)

        # continuous variable processing
        self.prescalers = nn.ModuleDict({name: nn.Linear(1, hiddenSize) for name in self.allReals})

        # variable selection
        # variable selection for static variables
        staticInputSizes = {name: self.inputEmbeddings.outputSize[name] for name in staticCategoricals}
        staticInputSizes.update({name: hiddenSize for name in staticReals})
        self.staticVariableSelection = variableSelectionNetwork(
            inputSizes=staticInputSizes,
            hiddenSize=hiddenSize,
            categoricals=staticCategoricals,
            dropoutRate=dropoutRate,
            prescalers=self.prescalers)

        # variable selection for encoder
        encoderInputSizes = {name: self.inputEmbeddings.outputSize[name] for name in timeVaryingCategoricalsEncoder}
        encoderInputSizes.update({name: hiddenSize for name in timeVaryingRealsEncoder})

        "#ccc this one has contextSize"
        self.encoderVariableSelection = variableSelectionNetwork(
            inputSizes=encoderInputSizes,
            hiddenSize=hiddenSize,
            categoricals=timeVaryingCategoricalsEncoder,
            dropoutRate=dropoutRate,
            contextSize=hiddenSize,
            prescalers=self.prescalers,
            singleVariableGrns={})

        # variable selection for decoder
        decoderInputSizes = {name: self.inputEmbeddings.outputSize[name] for name in timeVaryingCategoricalsDecoder}
        decoderInputSizes.update({name: hiddenSize for name in timeVaryingRealsDecoder})
        "#ccc this one has contextSize"
        self.decoderVariableSelection = variableSelectionNetwork(
            inputSizes=decoderInputSizes,
            hiddenSize=hiddenSize,
            categoricals=timeVaryingCategoricalsDecoder,
            dropoutRate=dropoutRate,
            contextSize=hiddenSize,
            prescalers=self.prescalers,
            singleVariableGrns={})

        # static encoders
        # for variable selection
        self.staticContextVariableSelection = gatedResidualNetwork(
            inputSize=hiddenSize,
            hiddenSize=hiddenSize,
            outputSize=hiddenSize,
            dropoutRate=dropoutRate)

        # hidden and cell state of the lstm
        self.staticContextInitialHiddenLstm = gatedResidualNetwork(
            inputSize=hiddenSize,
            hiddenSize=hiddenSize,
            outputSize=hiddenSize,
            dropoutRate=dropoutRate)

        self.staticContextInitialCellLstm = gatedResidualNetwork(
            inputSize=hiddenSize,
            hiddenSize=hiddenSize,
            outputSize=hiddenSize,
            dropoutRate=dropoutRate)

        # post lstm static enrichment
        self.staticContextEnrichment = gatedResidualNetwork(inputSize=hiddenSize,
        hiddenSize=hiddenSize,
        outputSize=hiddenSize,
        dropoutRate=dropoutRate)

        # lstm encoder (past) and decoder (future)
        self.lstmEncoder = nn.LSTM(
            input_size=hiddenSize,
            hidden_size=hiddenSize,
            num_layers=lstmLayers,
            dropout=dropoutRate if lstmLayers > 1 else 0,
            batch_first=True)

        self.lstmDecoder = nn.LSTM(
            input_size=hiddenSize,
            hidden_size=hiddenSize,
            num_layers=lstmLayers,
            dropout=dropoutRate if lstmLayers > 1 else 0,
            batch_first=True)

        # skip connection for lstm
        self.postLstmGateEncoder = gatedLinearUnit(inputSize=hiddenSize, hiddenSize=hiddenSize, dropoutRate=dropoutRate)
        self.postLstmGateDecoder = self.postLstmGateEncoder
        
        self.postLstmAddNormEncoder = addNorm(inputSize=hiddenSize, skipSize=hiddenSize)
        self.postLstmAddNormDecoder = self.postLstmAddNormEncoder

        # static enrichment and processing past LSTM
        "#ccc this one has contextSize"
        self.staticEnrichment = gatedResidualNetwork(
            inputSize=hiddenSize,
            hiddenSize=hiddenSize,
            outputSize=hiddenSize,
            dropoutRate=dropoutRate,
            contextSize=hiddenSize)

        # attention
        self.multiheadAttn = interpretableMultiHeadAttention(dModel=hiddenSize, nHead=attentionHeadSize, dropoutRate=dropoutRate)
        
        self.postAttnGateNorm = gateAddNorm(inputSize=hiddenSize, hiddenSize=hiddenSize, skipSize=hiddenSize, dropoutRate=dropoutRate)
        self.posWiseFf = gatedResidualNetwork(inputSize=hiddenSize,
        hiddenSize=hiddenSize,
        outputSize=hiddenSize,
        dropoutRate=dropoutRate)

        #ccc for final layer we dont use dropoutRate
        self.preOutputGateNorm = gateAddNorm(inputSize=hiddenSize, hiddenSize=hiddenSize, skipSize=hiddenSize, dropoutRate=None)

        if self.targetsNum > 1:
            self.outputLayer = nn.ModuleList([nn.Linear(hiddenSize, outputSize) for _ in range(self.targetsNum)])
        else:
            self.outputLayer = nn.Linear(hiddenSize, outputSize)
    
    def getTrainBatchIndexes(self, trainInputs):
        import random
        trainInputsIndexes = trainInputs[trainInputs['sequenceLength']>=self.externalKwargs['minEncoderLength']+self.externalKwargs['minPredictionLength']].index.tolist()
        indexes = random.sample(trainInputsIndexes, len(trainInputsIndexes))
        return indexes
    
    def getEvalBatchIndexes(self, inputs):
        import random
        inputsIndexes = inputs[inputs['sequenceLength']>=self.externalKwargs['minEncoderLength']+self.externalKwargs['minPredictionLength']].index.tolist()
        indexes = random.sample(inputsIndexes, len(inputsIndexes))
        return indexes
    
    def batchDatapreparation(self, indexesIndex, indexes, inputs, outputs, batchSize, mode=None, identifier=None, externalKwargs=None):
        batchIndexes = indexes[indexesIndex*batchSize:indexesIndex*batchSize + batchSize]
        appliedBatchSize = len(batchIndexes)
        
        allCols=externalKwargs['allCategoricalsNonGrouped']+externalKwargs['allReals']
        fullcastLen=externalKwargs['maxEncoderLength']+externalKwargs['maxPredictionLength']
        
        batchInputs={'encoderLengths':[], 'decoderLengths':[]}
        for idx in batchIndexes:
            batchInputs['encoderLengths'].append(int(inputs.loc[idx, 'encoderLength']))
            batchInputs['decoderLengths'].append(int(inputs.loc[idx, 'decoderLength']))
        
        for col1 in allCols:
            batchDfs=[]
            for idx1, idx in enumerate(batchIndexes):
                encoderLength = batchInputs['encoderLengths'][idx1]
                decoderLength = batchInputs['decoderLengths'][idx1]
                if col1=='relativeTimeIdx':
                    batchDfs.append(pd.Series([i / externalKwargs['maxEncoderLength'] for i in range(-encoderLength, decoderLength)]))
                elif col1=='encoderLength':
                    batchDfs.append(pd.Series([(encoderLength - .5*externalKwargs['maxEncoderLength'])/externalKwargs['maxEncoderLength']*2 for i in range(encoderLength+decoderLength)]))
                else:
                    batchDfs.append(inputs.loc[idx:idx + encoderLength+ decoderLength-1, col1])
            batchInputs[col1]=batchDfs
        
        batchInputs.update({col1:[self.rightPadSeriesIfShorter(df1,fullcastLen) for df1 in batchInputs[col1]] for col1 in allCols})#kkk is gonna be 
        batchInputs.update({col1:self.stackListOfDfs(batchInputs[col1], False) for col1 in allCols})
        batchInputs.update({col: val.to(torch.float32) if col in externalKwargs['allReals'] else val for col, val in batchInputs.items()})
        
        batchOutputs={}
        for col1 in externalKwargs['targets']:
            batchDfs=[]
            for idx1, idx in enumerate(batchIndexes):
                encoderLength = batchInputs['encoderLengths'][idx1]
                decoderLength = batchInputs['decoderLengths'][idx1]
                batchDfs.append(inputs.loc[idx+encoderLength:idx + encoderLength+ decoderLength-1, col1])
            batchOutputs[col1]=batchDfs
            
        batchOutputs = {col1:[self.rightPadSeriesIfShorter(df1,max(batchInputs['decoderLengths'])) for df1 in batchOutputs[col1]] for col1 in externalKwargs['targets']}
        batchOutputs = {col1:self.stackListOfDfs(batchOutputs[col1]) for col1 in externalKwargs['targets']}
        batchOutputs= batchOutputs['volume']#kkk correct this in general
        
        batchInputs['encoderLengths'] = torch.as_tensor(batchInputs['encoderLengths']).to(self.device)
        batchInputs['decoderLengths'] = torch.as_tensor(batchInputs['decoderLengths']).to(self.device)
        outPutMask=None
        return batchInputs, batchOutputs, appliedBatchSize, outPutMask, identifier

    def expandStaticContext(self, context, timesteps):
        return context[:, None].expand(-1, timesteps, -1)

    def forward(self, x):# x is a dictionary for each col used here and their shape is N*timesteps except encoderLengths and decoderLengths
        #kkk correct the input structure due to the new getStallionTftDataloaders
        encoderLengths = x["encoderLengths"]# shape: N
        decoderLengths = x["decoderLengths"]# shape: N
        batchSize= len(encoderLengths)
        #ccc note encoderLengths and decoderLengths are different in different samples
        maxEncoderLength = int(encoderLengths.max())
        maxDecoderLength= int(decoderLengths.max())
        timesteps = maxEncoderLength + maxDecoderLength
        inputVectors = self.inputEmbeddings(x)# inputVectors[categoricalCol].shape:N*timesteps*embeddingSizeOfCategoricalCol
        inputVectors.update({name: x[name].unsqueeze(-1) for name in self.allReals})#shape of reals :N*timesteps*1

        # Embedding and variable selection
        if len(self.staticVariables) > 0:
            staticEmbedding = {name: inputVectors[name][:, 0] for name in self.staticVariables}
            "#ccc it only uses first item of time sequence for staticVariableSelection"
            staticEmbedding, staticVariableSelection = self.staticVariableSelection(staticEmbedding)# staticEmbedding shape: N * hiddenSize; staticVariableSelection shape: N * 1 * len(staticEmbedding)
        else:
            staticEmbedding = torch.zeros((batchSize, self.hiddenSize), dtype=self.dtype, device=self.device)

        staticContextVariableSelection = self.expandStaticContext(# shape: N*timesteps*hiddenSize
            self.staticContextVariableSelection(staticEmbedding),#N*hiddenSize
            timesteps)

        embeddingsVaryingEncoder = {name: inputVectors[name][:, :maxEncoderLength] for name in self.encoderVariables}
        embeddingsVaryingEncoder, encoderSparseWeights = self.encoderVariableSelection(# embeddingsVaryingEncoder:N*maxEncoderLength*hiddenSize;encoderSparseWeights: N*maxEncoderLength* 1* len(encoderVariables)
            embeddingsVaryingEncoder,
            staticContextVariableSelection[:, :maxEncoderLength])

        embeddingsVaryingDecoder = {name: inputVectors[name][:, maxEncoderLength:] for name in self.decoderVariables}
        embeddingsVaryingDecoder, decoderSparseWeights = self.decoderVariableSelection(#N*(timesteps-maxEncoderLength)*hiddenSize,decoderSparseWeights:N*(timesteps-maxEncoderLength)*1*len(decoderVariables)
            embeddingsVaryingDecoder,
            staticContextVariableSelection[:, maxEncoderLength:])

        # LSTM
        # calculate initial state
        inputHidden = self.staticContextInitialHiddenLstm(staticEmbedding).expand(self.lstmLayers, -1, -1)#lstmLayers*N*hiddenSize
        inputCell = self.staticContextInitialCellLstm(staticEmbedding).expand(self.lstmLayers, -1, -1)#lstmLayers*N*hiddenSize

        encoderOutput, (hidden, cell) = self.lstmEncoder(#N*maxEncoderLength*hiddenSize;lstmLayers*N*hiddenSize;lstmLayers*N*hiddenSize
            embeddingsVaryingEncoder, (inputHidden, inputCell))

        decoderOutput, _ = self.lstmDecoder(embeddingsVaryingDecoder,(hidden, cell))#decoderOutput:N*6*hiddenSize

        # skip connection over lstm
        lstmOutputEncoder = self.postLstmGateEncoder(encoderOutput)#N*maxEncoderLength*hiddenSize
        lstmOutputEncoder = self.postLstmAddNormEncoder(lstmOutputEncoder, embeddingsVaryingEncoder)#N*maxEncoderLength*hiddenSize

        lstmOutputDecoder = self.postLstmGateDecoder(decoderOutput)#N*(timesteps-maxEncoderLength)*hiddenSize
        lstmOutputDecoder = self.postLstmAddNormDecoder(lstmOutputDecoder, embeddingsVaryingDecoder)#N*(timesteps-maxEncoderLength)*hiddenSize

        lstmOutput = torch.cat([lstmOutputEncoder, lstmOutputDecoder], dim=1)#N*timesteps*hiddenSize

        # static enrichment
        staticContextEnrichment = self.staticContextEnrichment(staticEmbedding)#N*hiddenSize
        attnInput = self.staticEnrichment(#N*timesteps*hiddenSize
            lstmOutput,
            self.expandStaticContext(staticContextEnrichment, timesteps))#N*timesteps*hiddenSize

        # Attention
        attnOutput, attnOutputWeights = self.multiheadAttn(#N*(timesteps-maxEncoderLength)*hiddenSize;N*(timesteps-maxEncoderLength)*attentionHeadSize*timesteps
            q=attnInput[:, maxEncoderLength:],  #ccc query is only for predictions
            k=attnInput,
            v=attnInput,
            mask=self.getAttentionMask(encoderLengths=encoderLengths, decoderLengths=decoderLengths),#N*(timesteps-maxEncoderLength)*timesteps
        )

        # skip connection over attention
        attnOutput = self.postAttnGateNorm(attnOutput, attnInput[:, maxEncoderLength:])#N*(timesteps-maxEncoderLength)*hiddenSize

        output = self.posWiseFf(attnOutput)#N*(timesteps-maxEncoderLength)*hiddenSize
        output = self.preOutputGateNorm(output, lstmOutput[:, maxEncoderLength:])#N*(timesteps-maxEncoderLength)*hiddenSize
        "#ccc skip using lstmOutput"
        if self.targetsNum > 1:
            output = [outputLayer(output) for outputLayer in self.outputLayer]#kkk
        else:
            output = self.outputLayer(output)#N*6*numOfQuantilesLevelsForLoss
        return output

    def getAttentionMask(self, encoderLengths: torch.LongTensor, decoderLengths: torch.LongTensor):
        def createMask(size, lengths, inverse: bool = False) -> torch.BoolTensor:
            if inverse:  #ccc return where values are
                return torch.arange(size, device=lengths.device).unsqueeze(0) < lengths.unsqueeze(-1)
            else:  #ccc return where no values are
                output = torch.arange(size, device=lengths.device).unsqueeze(0) >= lengths.unsqueeze(-1)# shape len(lengths) * size
                '#ccc this is a mask with "len(lengths)" rows, so for row i there is mask if torch.arange(size) is bigger or equal to lengths[i]'
                return output
            
        decoderLength = decoderLengths.max()
        if self.causalAttention:
            attendStep = torch.arange(decoderLength, device=self.device)
            predictStep = torch.arange(0, decoderLength, device=self.device)[:, None]#shape: decoderLength*1
            decoderMask = (attendStep >= predictStep)#ccc is upper triangular matrix with shape decoderLength*decoderLength
            decoderMask = decoderMask.unsqueeze(0).expand(encoderLengths.size(0), -1, -1)#shape:N*decoderLength*decoderLength
        else:
            decoderMask = createMask(decoderLength, decoderLengths).unsqueeze(1).expand(-1, decoderLength, -1)
        encoderMask = createMask(encoderLengths.max(), encoderLengths).unsqueeze(1).expand(-1, decoderLength, -1)#shape: len(encoderLengths)* decoderLength * encoderLengths.max()
        combinedMask = torch.cat((encoderMask, decoderMask),dim=2)
        return combinedMask