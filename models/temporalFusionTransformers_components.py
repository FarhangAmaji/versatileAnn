
from typing import Dict, List
import pandas as pd
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
# ----
#kkk change camelCases
#kkk add comments
#kkk add comments for multiEmbedding
#kkk add comments main tft init
#kkk add comments main tft forward
#kkk delete arg type definitions
def getFastAi_empericalEmbeddingSize(n, maxSize = 100) -> int:
    """
    Determine empirically good embedding sizes (formula taken from fastai).
    """
    if n > 2:
        return min(round(1.6 * n**0.56), maxSize)
    else:
        return 1

class timeDistributedInterpolation(nn.Module):
    "interpolates last dimension of input tensor so it would have outputSize"
    def __init__(self, outputSize):
        super().__init__()
        self.outputSize = outputSize

    def interpolate(self, x):
        upsampled = F.interpolate(x.unsqueeze(1), self.outputSize, mode="linear", align_corners=True).squeeze(1)
        return upsampled

    def forward(self, x):
        if len(x.size()) <= 2:
            return self.interpolate(x)

        # Squash samples and timesteps into a single axis
        xReshape = x.contiguous().view(-1, x.size(-1)) # shape: N * timesteps, xLastDimSize

        y = self.interpolate(xReshape)

        # We have to reshape Y
        y = y.contiguous().view(x.size(0), -1, y.size(-1)) # shape: N, timesteps, outputSize
        return y

class addNorm(nn.Module):
    def __init__(self, inputSize, skipSize):
        super().__init__()

        self.inputSize = inputSize
        self.skipSize = skipSize
        
        if self.inputSize != self.skipSize:
            self.resample = timeDistributedInterpolation(self.inputSize)

        self.norm = nn.LayerNorm(self.inputSize)

    def forward(self, x: torch.Tensor, skip: torch.Tensor):
        "#ccc if last dimesion of skip is not equal to last dimension of input, we would interpolate"
        if self.inputSize != self.skipSize:
            skip = self.resample(skip)

        output = self.norm(x + skip)
        return output

class gatedLinearUnit(nn.Module):
    def __init__(self, inputSize, hiddenSize, dropoutRate: float = None):
        super().__init__()

        if dropoutRate is not None:
            self.dropout = nn.Dropout(dropoutRate)
        else:
            self.dropout = None
        self.hiddenSize = hiddenSize
        self.fc = nn.Linear(inputSize, self.hiddenSize * 2)

        self.initWeights()

    def initWeights(self):
        for n, p in self.named_parameters():
            if "bias" in n:
                torch.nn.init.zeros_(p)
            elif "fc" in n:
                torch.nn.init.xavier_uniform_(p)

    def forward(self, x):
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.fc(x)
        x = F.glu(x, dim=-1)#last dimension size is hiddenSize
        return x

class gateAddNorm(nn.Module):
    "this is gatedLinearUnit + addNorm"
    def __init__(
        self,
        inputSize,
        hiddenSize,
        skipSize,
        dropoutRate: float = None):
        super().__init__()

        self.inputSize = inputSize
        self.hiddenSize = hiddenSize
        self.skipSize = skipSize

        self.glu = gatedLinearUnit(self.inputSize, hiddenSize=self.hiddenSize, dropoutRate=dropoutRate)
        self.addNorm = addNorm(inputSize=self.hiddenSize, skipSize=self.skipSize)

    def forward(self, x, skip):
        output = self.glu(x)
        output = self.addNorm(output, skip)
        return output

class resampleNorm(nn.Module):
    'this is timeDistributedInterpolation but prevents going through it if inputSize and outputSize are the same; also applies layerNorm'
    def __init__(self, inputSize, outputSize):
        super().__init__()
        self.inputSize = inputSize
        self.outputSize = outputSize

        if self.inputSize != self.outputSize:
            self.resample = timeDistributedInterpolation(self.outputSize)

        self.norm = nn.LayerNorm(self.outputSize)

    def forward(self, x):
        if self.inputSize != self.outputSize:
            x = self.resample(x)

        output = self.norm(x)
        return output

class gatedResidualNetwork(nn.Module):
    def __init__(
        self,
        inputSize,
        hiddenSize,
        outputSize,
        dropoutRate: float = 0.1,
        contextSize = None,
        residual: bool = False):
        super().__init__()
        self.inputSize = inputSize
        self.outputSize = outputSize
        self.contextSize = contextSize
        self.hiddenSize = hiddenSize
        self.residual = residual
        
        if self.residual:
            residualSize = self.outputSize
        else:
            if self.inputSize != self.outputSize:
                residualSize = self.inputSize
            else:
                residualSize = self.outputSize

        if self.outputSize != residualSize:
            self.resampleNorm = resampleNorm(inputSize=residualSize, outputSize=self.outputSize)

        self.fc1 = nn.Linear(self.inputSize, self.hiddenSize)
        self.elu = nn.ELU()

        if self.contextSize is not None:
            self.context = nn.Linear(self.contextSize, self.hiddenSize, bias=False)

        self.fc2 = nn.Linear(self.hiddenSize, self.hiddenSize)
        self.initWeights()

        self.gateNorm = gateAddNorm(
            inputSize=self.hiddenSize,
            skipSize=self.outputSize,
            hiddenSize=self.outputSize,
            dropoutRate=dropoutRate)

    def initWeights(self):
        for name, p in self.named_parameters():
            if "bias" in name:
                torch.nn.init.zeros_(p)
            elif "fc1" in name or "fc2" in name:
                torch.nn.init.kaiming_normal_(p, a=0, mode="fan_in", nonlinearity="leaky_relu")
            elif "context" in name:
                torch.nn.init.xavier_uniform_(p)

    def forward(self, x, context=None, residual=None):
        if residual is None:
            residual = x# note changing x wouldnt change residual

        if self.inputSize != self.outputSize and not self.residual:
            residual = self.resampleNorm(residual)

        x = self.fc1(x)
        if context is not None:
            context = self.context(context)
            x = x + context
        x = self.elu(x)
        x = self.fc2(x)
        x = self.gateNorm(x, residual)
        return x

class variableSelectionNetwork(nn.Module):
    def __init__(
        self,
        inputSizes: Dict[str, int],
        hiddenSize,
        categoricals: list = [],
        dropoutRate: float = 0.1,
        contextSize = None,
        singleVariableGrns: Dict[str, gatedResidualNetwork] = {},
        prescalers: Dict[str, nn.Linear] = {},
    ):
        super().__init__()

        self.hiddenSize = hiddenSize
        self.inputSizes = inputSizes
        self.categoricals = categoricals
        self.contextSize = contextSize

        if len(self.inputSizes) > 1:
            if self.contextSize is not None:
                self.flattenedGrn = gatedResidualNetwork(
                    inputSize=self.inputSizeTotal,#ccc if inputSize!=outputSize it may go through resampleNorm to get interpolated to have proper sizes
                    hiddenSize=min(self.hiddenSize, len(self.inputSizes)),#ccc this is arbitrary to choose min of self.hiddenSize, len(self.inputSizes)
                    outputSize=len(self.inputSizes),
                    dropoutRate=dropoutRate,
                    contextSize=self.contextSize,
                    residual=False)
            else:
                self.flattenedGrn = gatedResidualNetwork(
                    inputSize=self.inputSizeTotal,
                    hiddenSize=min(self.hiddenSize, len(self.inputSizes)),
                    outputSize=len(self.inputSizes),
                    dropoutRate=dropoutRate,
                    residual=False)

        self.singleVariableGrns = nn.ModuleDict()
        self.prescalers = nn.ModuleDict()
        for name, inputSize in self.inputSizes.items():
            # defining singleVariableGrns
            if name in singleVariableGrns:
                self.singleVariableGrns[name] = singleVariableGrns[name]
            elif name in self.categoricals:#ccc if its categorical
                self.singleVariableGrns[name] = resampleNorm(inputSize=inputSize, outputSize=self.hiddenSize)#yyy ResampleNorm if inputSize, self.hiddenSize are same just takes norm, but if not same: interpolates len input to hidden
            else:#ccc if its real
                self.singleVariableGrns[name] = gatedResidualNetwork(
                    inputSize=inputSize,
                    hiddenSize=min(inputSize, self.hiddenSize),
                    outputSize=self.hiddenSize,
                    dropoutRate=dropoutRate)
            # defining prescalers
            if name in prescalers: #ccc only reals prescalers
                self.prescalers[name] = prescalers[name]
            elif name not in self.categoricals:#ccc if some real is not in allReals which in tft is always in allReals, its gone have local prescaler here
                self.prescalers[name] = nn.Linear(1, inputSize)

        self.softmax = nn.Softmax(dim=-1)

    @property
    def inputSizeTotal(self):
        "#ccc inputEmbeddings would have some different embedded sizes for categoricals; sum of all embedded sizes for categoricals is calculated here"
        return sum(size if name in self.categoricals else size for name, size in self.inputSizes.items())

    def forward(self, x: Dict[str, torch.Tensor], context = None):
        if len(self.inputSizes) > 1:
            singleVariableGrnsOutputs = []
            weightInputs = []#ccc for collecting variableEmbeddings input of flattenedGrn
            
            # transform single variables
            for name in self.inputSizes.keys():
                variableEmbedding = x[name]
                if name in self.prescalers:
                    variableEmbedding = self.prescalers[name](variableEmbedding)
                weightInputs.append(variableEmbedding)
                singleVariableGrnsOutputs.append(self.singleVariableGrns[name](variableEmbedding))
            singleVariableGrnsOutputs = torch.stack(singleVariableGrnsOutputs, dim=-1)

            # calculate variable weights
            flatEmbedding = torch.cat(weightInputs, dim=-1)
            sparseWeights = self.flattenedGrn(flatEmbedding, context)
            sparseWeights = self.softmax(sparseWeights).unsqueeze(-2)

            outputs = singleVariableGrnsOutputs * sparseWeights
            outputs = outputs.sum(dim=-1)
        else:  
            "#ccc for one input, do not perform variable selection but just encoding"
            '#ccc if it has single input, we only prescale and pass it through singleVariableGrns'
            name = next(iter(self.singleVariableGrns.keys()))
            variableEmbedding = x[name]
            if name in self.prescalers:
                variableEmbedding = self.prescalers[name](variableEmbedding)
            outputs = self.singleVariableGrns[name](variableEmbedding)
            if outputs.ndim == 3:  # shape: N, timeSteps, hidden size
                sparseWeights = torch.ones(outputs.size(0), outputs.size(1), 1, 1, device=outputs.device)
            else:  # ndim == 2 shape: batch size, hidden size
                sparseWeights = torch.ones(outputs.size(0), 1, 1, device=outputs.device)
        return outputs, sparseWeights

class timeDistributedEmbeddingBag(nn.EmbeddingBag):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        if len(x.size()) <= 2:
            return super().forward(x)

        # Squash samples and timesteps into a single axis
        xReshape = x.contiguous().view(-1, x.size(-1))  # shape: N * timesteps, xLastDimSize

        y = super().forward(xReshape)

        # We have to reshape Y
        y = y.contiguous().view(x.size(0), -1, y.size(-1))  # shape: N, timesteps, outputSize
        return y

class multiEmbedding(nn.Module):
    def __init__(
        self,
        embeddingSizes,#ccc the expected type is Dict[str, List[int, int]]
        allCategoricalsNonGrouped: List[str] = None,
        categoricalGroupVariables: Dict[str, List[str]] = {},
        maxEmbeddingSize = None):
        super().__init__()
        
        # input data checks
        assert allCategoricalsNonGrouped is not None, "allCategoricalsNonGrouped must be provided."
        categoricalGroupVariables = [name for names in categoricalGroupVariables.values() for name in names]
        if len(categoricalGroupVariables) > 0:
            assert all(name in embeddingSizes for name in categoricalGroupVariables), "categoricalGroupVariables must be in embeddingSizes."
            assert not any(name in embeddingSizes for name in categoricalGroupVariables), "group variables in categoricalGroupVariables must not be in embeddingSizes."
            assert all(name in allCategoricalsNonGrouped for name in categoricalGroupVariables), "group variables in categoricalGroupVariables must be in allCategoricalsNonGrouped."
        assert all(name in embeddingSizes for name in embeddingSizes if name not in categoricalGroupVariables), ("all variables in embeddingSizes must be in allCategoricalsNonGrouped - but only if not already in categoricalGroupVariables.")

        self.embeddingSizes = embeddingSizes
        self.categoricalGroupVariables = categoricalGroupVariables
        self.maxEmbeddingSize = maxEmbeddingSize
        self.allCategoricalsNonGrouped = allCategoricalsNonGrouped
        self.initEmbeddings()

    def initEmbeddings(self):
        self.embeddings = nn.ModuleDict()
        for name in self.embeddingSizes.keys():
            embeddingSize = self.embeddingSizes[name][1]
            if self.maxEmbeddingSize is not None:
                embeddingSize = min(embeddingSize, self.maxEmbeddingSize)
            self.embeddingSizes[name][1] = embeddingSize
        
        for name in self.embeddingSizes.keys():
            if name in self.categoricalGroupVariables:
                #ccc embeddingBag for group categoricals like specialDays
                self.embeddings[name] = timeDistributedEmbeddingBag(self.embeddingSizes[name][0], self.embeddingSizes[name][1], mode="sum")
            else:
                #ccc normal embeddings for non-group categoricals
                self.embeddings[name] = nn.Embedding(self.embeddingSizes[name][0],self.embeddingSizes[name][1],padding_idx=None)

    @property
    def outputSize(self):
        return {name: s[1] for name, s in self.embeddingSizes.items()}

    def forward(self, x) -> Dict[str, torch.Tensor]:
        inputVectors = {}
        for name, emb in self.embeddings.items():
            if name in self.categoricalGroupVariables:
                categoricalVariableGroup=[x[col] for col in self.categoricalGroupVariables[name]]
                categoricalVariableGroup=torch.stack(categoricalVariableGroup).to(categoricalVariableGroup[0].device).permute(1,2,0)
                inputVectors[name] = emb(categoricalVariableGroup)
            else:
                inputVectors[name] = emb(x[name])
        return inputVectors

class scaledDotProductAttention(nn.Module):
    def __init__(self, dropoutRate: float = None, scale: bool = True):
        super(scaledDotProductAttention, self).__init__()
        if dropoutRate is not None:
            self.dropout = nn.Dropout(p=dropoutRate)
        else:
            self.dropout = None
        self.softmax = nn.Softmax(dim=2)
        self.scale = scale

    def forward(self, q, k, v, mask=None):
        attn = torch.bmm(q, k.permute(0, 2, 1))

        if self.scale:
            dimension = torch.as_tensor(k.size(-1), dtype=attn.dtype, device=attn.device).sqrt()
            attn = attn / dimension

        if mask is not None:
            attn = attn.masked_fill(mask, -1e9)
        attn = self.softmax(attn)

        if self.dropout is not None:
            attn = self.dropout(attn)
        output = torch.bmm(attn, v)
        return output, attn


class interpretableMultiHeadAttention(nn.Module):#kkk how its interpretable
    def __init__(self, nHead, dModel, dropoutRate: float = 0.0):
        super(interpretableMultiHeadAttention, self).__init__()

        self.nHead = nHead
        self.dModel = dModel
        self.dK = self.dQ = self.dV = dModel // nHead
        self.dropout = nn.Dropout(p=dropoutRate)

        self.vLayer = nn.Linear(self.dModel, self.dV)#jjj why q and k are a list of nHead but v is not
        self.qLayers = nn.ModuleList([nn.Linear(self.dModel, self.dQ) for _ in range(self.nHead)])
        self.kLayers = nn.ModuleList([nn.Linear(self.dModel, self.dK) for _ in range(self.nHead)])
        self.attention = scaledDotProductAttention()
        self.wH = nn.Linear(self.dV, self.dModel, bias=False)

        self.initWeights()

    def initWeights(self):
        for name, p in self.named_parameters():
            if "bias" not in name:
                torch.nn.init.xavier_uniform_(p)
            else:
                torch.nn.init.zeros_(p)

    def forward(self, q, k, v, mask=None):
        heads = []
        attns = []
        vs = self.vLayer(v)
        for i in range(self.nHead):
            qs = self.qLayers[i](q)
            ks = self.kLayers[i](k)
            head, attn = self.attention(qs, ks, vs, mask)
            headDropout = self.dropout(head)
            heads.append(headDropout)
            attns.append(attn)

        head = torch.stack(heads, dim=2) if self.nHead > 1 else heads[0]
        attn = torch.stack(attns, dim=2)

        outputs = torch.mean(head, dim=2) if self.nHead > 1 else head
        outputs = self.wH(outputs)
        outputs = self.dropout(outputs)

        return outputs, attn
# ----
def preprocessTemporalFusionTransformerTrainValTestData(data, trainRatio, valRatio, minPredictionLength, maxPredictionLength, maxEncoderLength, minEncoderLength,
                            mainGroups, categoricalGroupVariables, timeIdx, targets, staticCategoricals,
                            staticReals, timeVarying_knownCategoricals, timeVarying_knownReals, timeVarying_unknownCategoricals,
                            timeVarying_unknownReals):
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from functools import reduce
    def addSequenceEncoderNDecoderLength(df, mainGroupCombinations, minEncoderLength, maxEncoderLength, minPredictionLength, maxPredictionLength):
        for i in range(len(df)):
            indexCombination=tuple(data.loc[i,mainGroups])
            maxTimeIdx=mainGroupCombinations.loc[indexCombination,'maxTime']
            if df.loc[i,timeIdx+'_']<maxTimeIdx+1-maxPredictionLength-maxEncoderLength+1:
                df.loc[i,'sequenceLength']=maxEncoderLength+maxPredictionLength
                df.loc[i,'encoderLength']=maxEncoderLength
            else:
                df.loc[i,'encoderLength']=max(maxEncoderLength-df.loc[i,timeIdx+'_']+(maxTimeIdx+1-maxPredictionLength-maxEncoderLength),minEncoderLength)
                df.loc[i,'sequenceLength']=maxEncoderLength+maxPredictionLength-df.loc[i,timeIdx+'_']+(maxTimeIdx+1-maxPredictionLength-maxEncoderLength)
            df.loc[i,'decoderLength']=max(min(df.loc[i,'sequenceLength']-df.loc[i,'encoderLength'],maxPredictionLength),minPredictionLength)
        return df
    
    testRatio=round(1-trainRatio-valRatio,3)
    allcategoricalGroupVariables = {vg1: vg for vg in categoricalGroupVariables.keys() for vg1 in categoricalGroupVariables[vg]}

    data.loc[:,'relativeTimeIdx'] = 0
    timeVarying_knownReals += ['relativeTimeIdx']
    data.loc[:,'encoderLength'] = 0
    staticReals += ['encoderLength']

    data = data.sort_values(mainGroups + [timeIdx]).reset_index(drop=True)
    allCategoricals = list(set(staticCategoricals + timeVarying_knownCategoricals + timeVarying_unknownCategoricals))
    allCategoricalsNonGrouped = [ac for ac in allCategoricals if ac not in categoricalGroupVariables.keys()]
    allCategoricalsNonGrouped += list(allcategoricalGroupVariables.keys())

    categoricalEncoders = {}
    for c1 in allCategoricals:
        if c1 not in categoricalGroupVariables.keys() and c1 not in targets:
            categoricalEncoders[c1] = LabelEncoder().fit(data[c1].to_numpy().reshape(-1))
        elif c1 in categoricalGroupVariables.keys():
            cols = categoricalGroupVariables[c1]
            categoricalEncoders[c1] = LabelEncoder().fit(data[cols].to_numpy().reshape(-1))

    embeddingSizes = {name: [len(encoder.classes_), getFastAi_empericalEmbeddingSize(len(encoder.classes_))]
        for name, encoder in categoricalEncoders.items()}

    for ce in allCategoricalsNonGrouped:
        if ce not in allcategoricalGroupVariables.keys():
            data[ce] = categoricalEncoders[ce].transform(data[ce])
        elif ce in allcategoricalGroupVariables.keys():
            data[ce] = categoricalEncoders[allcategoricalGroupVariables[ce]].transform(data[ce])

    eps = np.finfo(np.float16).eps
    targetsCenterNStd = pd.DataFrame()
    for tg in targets:
        targetsCenterNStdInstance = data[mainGroups + [tg]].groupby(mainGroups, observed=True).agg(center=(tg, "mean"), scale=(tg, "std")).assign(center=lambda x: x["center"], scale=lambda x: x.scale + eps)
        targetsCenterNStdInstance.rename(columns={'center': f'{tg}Center', 'scale': f'{tg}Scale'}, inplace=True)
        staticReals.extend([f'{tg}Center', f'{tg}Scale'])
        targetsCenterNStd = pd.concat([targetsCenterNStd, targetsCenterNStdInstance])
    
    data = data.merge(targetsCenterNStd, left_on=mainGroups, right_index=True)
    for tg in targets:
        data[tg] = (data[tg] - data[f'{tg}Center']) / data[f'{tg}Scale']
    
    "time Varying Encoder= time Varying known + time Varying unkown"
    "time Varying Decoder= time Varying known"
    timeVarying_categoricalsEncoder = list(set(timeVarying_knownCategoricals + timeVarying_unknownCategoricals))
    timeVarying_realsEncoder = list(set(timeVarying_knownReals + timeVarying_unknownReals))
    timeVarying_categoricalsDecoder = timeVarying_knownCategoricals[:]
    timeVarying_realsDecoder = timeVarying_knownReals[:]
    
    #scaling real data
    allReals=list(set(staticReals+timeVarying_knownReals+timeVarying_unknownReals))
    data[timeIdx+'_']=data[timeIdx]
    realScalers={}
    for ar in allReals:
        if ar in targets:
            continue
        realScalers[ar]=StandardScaler().fit(data[ar].to_numpy().reshape(-1,1))
        data[ar]=realScalers[ar].transform(data[ar].to_numpy().reshape(-1,1))

    dfByUpperAndLowerBound = lambda df, lowerBound, upperBound: df[(df[timeIdx+'_'] > lowerBound) & (df[timeIdx+'_'] <= upperBound)]
    mainGroupCombinations = pd.DataFrame(index=targetsCenterNStd.index)
    trainData= pd.DataFrame()
    valData= pd.DataFrame()
    testData= pd.DataFrame()
    for tcnsi in targetsCenterNStd.index:
        conditions = [data[group] == value for group, value in zip(mainGroups, tcnsi)]
        conditionsResult=data[reduce(lambda x, y: x & y, conditions)]
        mainGroupCombinations.loc[tcnsi,'minTime']=conditionsResult[timeIdx+'_'].min()
        mainGroupCombinations.loc[tcnsi,'maxTime']=conditionsResult[timeIdx+'_'].max()
        timeMaxMinDiff=mainGroupCombinations.loc[tcnsi,'maxTime']-mainGroupCombinations.loc[tcnsi,'minTime']+1
        if all([trainRatio, valRatio, testRatio]):
            eachSetAdd=timeMaxMinDiff-(3*maxPredictionLength+maxEncoderLength)
            
            trainPredictionEndIdx=mainGroupCombinations.loc[tcnsi,'minTime']-1+maxEncoderLength+maxPredictionLength+int(trainRatio*eachSetAdd)
            trainData = pd.concat([trainData, dfByUpperAndLowerBound(conditionsResult, mainGroupCombinations.loc[tcnsi,'minTime']-1, trainPredictionEndIdx)])
            
            valPredictionEndIdx=trainPredictionEndIdx+maxPredictionLength-int(trainRatio*eachSetAdd)+int((trainRatio+valRatio)*eachSetAdd)
            valData = pd.concat([valData, dfByUpperAndLowerBound(conditionsResult, trainPredictionEndIdx-maxEncoderLength, valPredictionEndIdx)])
            
            testData = pd.concat([testData, dfByUpperAndLowerBound(conditionsResult, valPredictionEndIdx-maxEncoderLength, mainGroupCombinations.loc[tcnsi,'maxTime'])])
        else:
            trainPredictionEndIdx=int(timeMaxMinDiff*trainRatio)
            trainData = pd.concat([trainData, dfByUpperAndLowerBound(conditionsResult, mainGroupCombinations.loc[tcnsi,'minTime']-1, trainPredictionEndIdx)])
            
            valPredictionEndIdx=trainPredictionEndIdx+int(valRatio*eachSetAdd)
            valData = pd.concat([valData, dfByUpperAndLowerBound(conditionsResult, trainPredictionEndIdx-maxEncoderLength, valPredictionEndIdx)])
            
            testData = pd.concat([testData, dfByUpperAndLowerBound(conditionsResult, valPredictionEndIdx-maxEncoderLength, mainGroupCombinations.loc[tcnsi,'maxTime'])])

    trainData= trainData.reset_index(drop=True)
    valData= valData.reset_index(drop=True)
    testData= testData.reset_index(drop=True)

    trainData=addSequenceEncoderNDecoderLength(trainData, mainGroupCombinations, minEncoderLength,maxEncoderLength,minPredictionLength,maxPredictionLength)
    valData = addSequenceEncoderNDecoderLength(valData, mainGroupCombinations, minEncoderLength, maxEncoderLength, minPredictionLength, maxPredictionLength)
    testData = addSequenceEncoderNDecoderLength(testData, mainGroupCombinations, minEncoderLength, maxEncoderLength, minPredictionLength, maxPredictionLength)
    return data, trainData, valData, testData, allCategoricalsNonGrouped, categoricalEncoders, embeddingSizes, targetsCenterNStd, \
         timeVarying_categoricalsEncoder, timeVarying_realsEncoder, timeVarying_categoricalsDecoder, timeVarying_realsDecoder, allReals, realScalers
