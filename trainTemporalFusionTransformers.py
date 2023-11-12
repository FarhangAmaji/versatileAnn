# ---- imports
# trainTemporalFusionTransformers.py

import pandas as pd
import numpy as np
from models.temporalFusionTransformers import temporalFusionTransformerModel
from models.temporalFusionTransformers_components import preprocessTemporalFusionTransformerTrainValTestData
from metrics.loss import quantileLoss
import torch.optim as optim
# ----
'#ccc how to set optimizer manually'
# nHitsModel.lr=0.001
# nHitsModel.learningRate=0.001
# nHitsModel.changeLearningRate(0.001)
# nHitsModel.optimizer=optim.Adam(nHitsModel.parameters(), lr=0.4)
# nHitsModel.tensorboardWriter=newTensorboardPath
# nHitsModel.batchSize=32
# nHitsModel.evalBatchSize=1024
# nHitsModel.device=torch.device(type='cpu') or torch.device(type='cuda')
# nHitsModel.l1Reg=1e-3 or nHitsModel.l2Reg=1e-3 or nHitsModel.regularization=[None, None]

# nHitsModel.patience=10
# nHitsModel.saveOnDiskPeriod=1
# nHitsModel.lossMode='accuracy'
# nHitsModel.variationalAutoEncoderMode=True
# ----
data= pd.read_csv(r'F:\projects\public github projects\private repos\versatileAnnModule\data\datasets\stallion.csv')
# data = pd.read_csv(r'.\data\datasets\stallion.csv')
data['date']=pd.to_datetime(data['date'])
# add time index
data["timeIdx"] = data["date"].dt.year * 12 + data["date"].dt.month
data["timeIdx"] -= data["timeIdx"].min()

# add additional features
data["month"] = data.date.dt.month.astype(str).astype("category")  # categories have be strings
data["logVolume"] = np.log(data.volume + 1e-8)
data["avgVolumeBySku"] = data.groupby(["timeIdx", "sku"], observed=True).volume.transform("mean")
data["avgVolumeByAgency"] = data.groupby(["timeIdx", "agency"], observed=True).volume.transform("mean")

# we want to encode special days as one variable and thus need to first reverse one-hot encoding
specialDays = ['easterDay',
'goodFriday', 'newYear', 'christmas', 'laborDay', 'independenceDay',
'revolutionDayMemorial', 'regionalGames', 'fifaU17WorldCup',
'footballGoldCup', 'beerCapital', 'musicFest'
]
data[specialDays] = data[specialDays].apply(lambda x: x.map({0: "-", 1: x.name})).astype("category")

minPredictionLength = 1
maxPredictionLength = 10
maxEncoderLength = 16
minEncoderLength = maxEncoderLength // 2
# ----
mainGroups=["agency", "sku"]
categoricalVariableGroups={"specialDays": specialDays}

timeIdx="timeIdx"
targets=['volume']
staticCategoricals=["agency", "sku"]
staticReals=["avgPopulation2017", "avgYearlyHouseholdIncome2017"]
timeVaryingknownCategoricals=["specialDays", "month"]
timeVaryingknownReals=["timeIdx", "priceRegular", "discountInPercent"]
timeVaryingUnknownCategoricals=[]
timeVaryingUnknownReals=["volume","logVolume","industryVolume","sodaVolume","avgMaxTemp","avgVolumeByAgency","avgVolumeBySku"]
# ---- 
data, trainData, valData, testData, allCategoricalsNonGrouped, categoricalEncoders, embeddingSizes, targetsCenterNStd, timeVaryingCategoricalsEncoder, \
    timeVaryingRealsEncoder, timeVaryingCategoricalsDecoder, timeVaryingRealsDecoder, allReals, realScalers = preprocessTemporalFusionTransformerTrainValTestData(
    data=data,
    trainRatio=.7,
    valRatio=.3,#kkk
    minPredictionLength=minPredictionLength,
    maxPredictionLength=maxPredictionLength,
    maxEncoderLength=maxEncoderLength,
    minEncoderLength=minEncoderLength,
    mainGroups=mainGroups,
    categoricalVariableGroups=categoricalVariableGroups,
    timeIdx=timeIdx,
    targets=targets,
    staticCategoricals=staticCategoricals,
    staticReals=staticReals,
    timeVaryingknownCategoricals=timeVaryingknownCategoricals,
    timeVaryingknownReals=timeVaryingknownReals,
    timeVaryingUnknownCategoricals=timeVaryingUnknownCategoricals,
    timeVaryingUnknownReals=timeVaryingUnknownReals
)
# ----
from sklearn.preprocessing import LabelEncoder, StandardScaler
from models.temporalFusionTransformers_components import getEmbeddingSize
allCategoricalVariableGroups = {vg1: vg for vg in categoricalVariableGroups.keys() for vg1 in categoricalVariableGroups[vg]}

data['relativeTimeIdx'] = 0
timeVaryingknownReals+=['relativeTimeIdx']
data['encoderLength'] = 0
staticReals+=['encoderLength']

data = data.sort_values(mainGroups + [timeIdx]).reset_index(drop=True)
allCategoricals=list(set(staticCategoricals + timeVaryingknownCategoricals + timeVaryingUnknownCategoricals))
allCategoricalsNonGrouped=[ac for ac in allCategoricals if ac not in categoricalVariableGroups.keys()]
allCategoricalsNonGrouped+=list(allCategoricalVariableGroups.keys())

categoricalEncoders={}
for c1 in allCategoricals:
    if c1 not in categoricalVariableGroups.keys() and c1 not in targets:
        categoricalEncoders[c1]=LabelEncoder().fit(data[c1].to_numpy().reshape(-1))
    elif c1 in categoricalVariableGroups.keys():
        cols=categoricalVariableGroups[c1]
        categoricalEncoders[c1]=LabelEncoder().fit(data[cols].to_numpy().reshape(-1))

embeddingSizes={name: [len(encoder.classes_), getEmbeddingSize(len(encoder.classes_))]
    for name, encoder in categoricalEncoders.items()}

for ce in allCategoricalsNonGrouped:
    if ce not in allCategoricalVariableGroups.keys():
        data[ce] = categoricalEncoders[ce].transform(data[ce])
    elif ce in allCategoricalVariableGroups.keys():
        data[ce]=categoricalEncoders[allCategoricalVariableGroups[ce]].transform(data[ce])
# ---- scaling
eps = np.finfo(np.float16).eps
targetsCenterNStd=pd.DataFrame()
for tg in targets:
    targetsCenterNStdInstance=data[mainGroups+[tg]].groupby(mainGroups, observed=True).agg(center=(tg, "mean"), scale=(tg, "std")).assign(center=lambda x: x["center"], scale=lambda x: x.scale + eps)
    targetsCenterNStdInstance.rename(columns={'center': f'{tg}Center', 'scale': f'{tg}Scale'}, inplace=True)
    staticReals.extend([f'{tg}Center',f'{tg}Scale'])
    targetsCenterNStd=pd.concat([targetsCenterNStd,targetsCenterNStdInstance])

for tg in targets:
    for i in range(len(data)):
        indexCombination=tuple(data.loc[i,mainGroups])
        center=targetsCenterNStd.loc[indexCombination,f'{tg}Center']
        scale=targetsCenterNStd.loc[indexCombination,f'{tg}Scale']
        data.loc[i,tg]=(data.loc[i,tg]-center)/scale
        data.loc[i,f'{tg}Center'], data.loc[i,f'{tg}Scale']=center, scale
# ----
"time Varying Encoder= time Varying known + time Varying unkown"
"time Varying Decoder= time Varying known"
timeVaryingCategoricalsEncoder=list(set(timeVaryingknownCategoricals+timeVaryingUnknownCategoricals))
timeVaryingRealsEncoder=list(set(timeVaryingknownReals+timeVaryingUnknownReals))
timeVaryingCategoricalsDecoder=timeVaryingknownCategoricals[:]
timeVaryingRealsDecoder=timeVaryingknownReals[:]
# ---- split train and val
#kkk split with respect to mainGroups
#kkk at least maxPredictionLength in val
#kkk 
aveEachMainGroupCombinations=0
for ii in targetsCenterNStd.index:
    aveEachMainGroupCombinations+=len(data[(data['agency']==ii[0]) & (data['sku']==ii[1])])
aveEachMainGroupCombinations/=len(targetsCenterNStd.index)
valPredictionRatio=.2
'each mainGroup combinations has 60 rows'
"here we choose 20% of all length of each mainGroup combinations to be at least in the val data for prediction so 60-12=48"
trainData=data[data[timeIdx]<aveEachMainGroupCombinations*(1-valPredictionRatio)].reset_index(drop=True)
def addSequenceEncoderNDecoderLength(df,minEncoderLength,maxEncoderLength,minPredictionLength,maxPredictionLength):
    maxTimeIdx=df["timeIdx"].max()
    for i in range(len(df)):
        if df.loc[i,timeIdx]<maxTimeIdx+1-maxPredictionLength-maxEncoderLength+1:
            df.loc[i,'sequenceLength']=maxEncoderLength+maxPredictionLength
            df.loc[i,'encoderLength']=maxEncoderLength
        else:
            df.loc[i,'encoderLength']=max(maxEncoderLength-df.loc[i,timeIdx]+(maxTimeIdx+1-maxPredictionLength-maxEncoderLength),minEncoderLength)
            df.loc[i,'sequenceLength']=maxEncoderLength+maxPredictionLength-df.loc[i,timeIdx]+(maxTimeIdx+1-maxPredictionLength-maxEncoderLength)
        df.loc[i,'decoderLength']=max(min(df.loc[i,'sequenceLength']-df.loc[i,'encoderLength'],maxPredictionLength),minPredictionLength)
    return df
trainData=addSequenceEncoderNDecoderLength(trainData,minEncoderLength,maxEncoderLength,minPredictionLength,maxPredictionLength)

"valData for each mainGroup combinations has 12 prediction and max"
valData=data[aveEachMainGroupCombinations*(1-valPredictionRatio)-maxEncoderLength-1<data[timeIdx]].reset_index(drop=True)
valData=addSequenceEncoderNDecoderLength(valData,minEncoderLength,maxEncoderLength,minPredictionLength,maxPredictionLength)
# ----
allReals=list(set(staticReals+timeVaryingknownReals+timeVaryingUnknownReals))
realScalers={}
for ar in allReals:
    if ar in targets:
        continue
    realScalers[ar]=StandardScaler().fit(data[ar].to_numpy().reshape(-1,1))
    data[ar]=realScalers[ar].transform(data[ar].to_numpy().reshape(-1,1))
# ----
quantiles=[.02,.05,.25,.5,.75,.95,.98]
model=temporalFusionTransformerModel(hiddenSize= 8, outputSize = len(quantiles), lstmLayers = 1, maxEncoderLength = 10,
targetsNum=len(targets), attentionHeadSize = 4, dropoutRate = 0.1,
staticCategoricals = staticCategoricals, staticReals = staticReals,
timeVaryingCategoricalsEncoder = timeVaryingCategoricalsEncoder,
timeVaryingCategoricalsDecoder = timeVaryingCategoricalsDecoder,
categoricalVariableGroups = categoricalVariableGroups,
timeVaryingRealsEncoder = timeVaryingRealsEncoder,
timeVaryingRealsDecoder = timeVaryingRealsDecoder,
allReals = allReals,
allCategoricalsNonGrouped = allCategoricalsNonGrouped,
embeddingSizes = embeddingSizes,
backcastLen=maxEncoderLength,
forecastLen=maxPredictionLength)#kkk may pass with col names with externalKwargs
# ---- 
runcell('imports', 'F:/projects/public github projects/private repos/versatileAnnModule/trainTemporalFusionTransformers.py')
runcell(3, 'F:/projects/public github projects/private repos/versatileAnnModule/trainTemporalFusionTransformers.py')
runcell(4, 'F:/projects/public github projects/private repos/versatileAnnModule/trainTemporalFusionTransformers.py')
runcell('scaling', 'F:/projects/public github projects/private repos/versatileAnnModule/trainTemporalFusionTransformers.py')
runcell(6, 'F:/projects/public github projects/private repos/versatileAnnModule/trainTemporalFusionTransformers.py')
runcell('split train and val', 'F:/projects/public github projects/private repos/versatileAnnModule/trainTemporalFusionTransformers.py')
runcell(8, 'F:/projects/public github projects/private repos/versatileAnnModule/trainTemporalFusionTransformers.py')
runcell(9, 'F:/projects/public github projects/private repos/versatileAnnModule/trainTemporalFusionTransformers.py')
# ----
workerNum=8
#kkk am I correctly doing timeseries which only is from one combination of 'agency' and 'sku': I must not to do trainData=trainData[trainData['sequenceLength']>=minEncoderLength+minPredictionLength] and valData=valData[valData['sequenceLength']>=minEncoderLength+minPredictionLength]; then change where batchIndexes are defined to do it properly
#kkk main code for relativeTimeIdx in getItem had seqLen and forwad had 30 len but seemed to be 0rightpadded encoder and 0rightpadded decoder
#kkk does the forward have tensors of different lengths?: no if they dont have maxLen they are 0 padded
criterion = quantileLoss(quantiles)
# trainData=trainData[trainData['sequenceLength']>=minEncoderLength+minPredictionLength].reset_index(drop=True)#kkk
# valData=valData[valData['sequenceLength']>=minEncoderLength+minPredictionLength].reset_index(drop=True)#kkk
externalKwargs={#kkk I just need allReals allCategoricalsNonGrouped targets maxEncoderLength maxPredictionLength
    'staticCategoricals' : staticCategoricals, 'staticReals' : staticReals,
    'timeVaryingCategoricalsEncoder' : timeVaryingCategoricalsEncoder,
    'timeVaryingCategoricalsDecoder' : timeVaryingCategoricalsDecoder,
    'categoricalVariableGroups' : categoricalVariableGroups,
    'timeVaryingRealsEncoder' : timeVaryingRealsEncoder,
    'timeVaryingRealsDecoder' : timeVaryingRealsDecoder,
    'allReals' : allReals, 'targets': targets,
    'allCategoricalsNonGrouped' : allCategoricalsNonGrouped,
    'embeddingSizes' : embeddingSizes,
    'maxEncoderLength':maxEncoderLength, 'maxPredictionLength':maxPredictionLength,
    'minEncoderLength':minEncoderLength, 'minPredictionLength':minPredictionLength,
    }
model.trainModel(trainData, None, valData, None, criterion, numEpochs=30, savePath=r'data\bestModels\tft1', workerNum=workerNum, externalKwargs=externalKwargs)
# ----

# ----

# ----

# ----

