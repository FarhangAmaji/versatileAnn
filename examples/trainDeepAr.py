# ---- imports
# trainDeepAr.py
# bugPotentialCheck1 check this file and correct kkks

from models.deepAr import deepArModel
import numpy as np
import pandas as pd
import torch
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
# ---- load UCI electricity dataset
data = pd.read_csv(r'../data/datasets/electricity.csv')
"""
#ccc this data has ['date', 'consumerId', 'hourOfDay', 'dayOfWeek', 'powerUsage','daysFromStart', 
               'hoursFromStart', 'daysFromStartOfDf', 'month'] columns
there are many date cols and consumerId and powerUsage cols
this dataset has different consumer data which are treated as separate data sequences
"""

'#ccc note we need to make sure that data is sorted'
timeIdx='hoursFromStart'
mainGroups=['consumerId']
data=data.sort_values(by=[*mainGroups,timeIdx], ascending=[True, True]).reset_index(drop=True)

"also some index(here 'sequenceIdx') is created to show the index in each consumer sequence"
min_hours = data.groupby('consumerId')['hoursFromStart'].transform('min')

# Create a new column 'sequenceIdx' with the desired calculation
data['sequenceIdx'] = data['hoursFromStart'] - min_hours
# ---- 
backcastLen=192
forecastLen=0#ccc this is not gonna used

'#ccc data point which are possible to be chosen for sequence starts are selected'
uniqueConsumerMaxDataLen = data.groupby('consumerId')['sequenceIdx'].transform('max')

# Set 'possibleStartPoint' to 1 for rows meeting the condition
data.loc[data['sequenceIdx'] <= uniqueConsumerMaxDataLen - backcastLen, 'possibleStartPoint'] = 1

# Set 'possibleStartPoint' to 0 when it's not 1
data.loc[data['possibleStartPoint'] != 1, 'possibleStartPoint'] = 0
# ---- 
from sklearn.preprocessing import LabelEncoder, StandardScaler
allReals=['hourOfDay', 'dayOfWeek', 'powerUsage','daysFromStart', 'hoursFromStart', 'daysFromStartOfDf', 'month']
realScalers={}
for ar in allReals:
    realScalers[ar]=StandardScaler().fit(data[ar].to_numpy().reshape(-1,1))
    data[ar]=realScalers[ar].transform(data[ar].to_numpy().reshape(-1,1))

allCategoricals=['consumerId']
categoricalEncoders = {}
for c1 in allCategoricals:
    categoricalEncoders[c1] = LabelEncoder().fit(data[c1].to_numpy().reshape(-1))
    data[c1] = categoricalEncoders[c1].transform(data[c1])
z=data[:20000]#kkk
# ---- split train val test
trainRatio=.7
valRatio=.2
indexes=np.array(data[data['possibleStartPoint']==1].index)
np.random.shuffle(indexes)
data = data.drop(['possibleStartPoint'], axis=1)

trainIndexes=indexes[:int(trainRatio*len(indexes))]
valIndexes=indexes[int(trainRatio*len(indexes)):int((trainRatio+valRatio)*len(indexes))]
testIndexes=indexes[int((trainRatio+valRatio)*len(indexes)):]

#kkk this is unnecessary because by having the whole data and indexes we would create batches
def addSequentIndexes(indexes, nextSequentsToBeAdded):
    newIndexes = set()
    for num in indexes:
        newIndexes.update(range(num + 1, num + nextSequentsToBeAdded))
    newIndexes.difference_update(indexes)  # Remove existing elements from the newIndexes set
    indexes = np.concatenate((indexes, np.array(list(newIndexes))))
    indexes.sort()
    return indexes

trainIndexes2=addSequentIndexes(trainIndexes, backcastLen)
valIndexes2=addSequentIndexes(valIndexes, backcastLen)
testIndexes2=addSequentIndexes(testIndexes, backcastLen)


train=data.loc[trainIndexes2]
val=data.loc[valIndexes2]
test=data.loc[testIndexes2]
# ---- 
embedderInputSize=len(data['consumerId'].unique())
covariatesNum=len(allReals)
model=deepArModel(192, 0, embedderInputSize = embedderInputSize, covariatesNum = covariatesNum, embeddingDim = 20,
                  hiddenSize= 16, lstmLayers= 3, dropoutRate= 0.1)
# ---- 
runcell('imports', 'F:/projects/public github projects/private repos/versatileAnnModule/trainDeepAr.py')
runcell('load UCI electricity dataset', 'F:/projects/public github projects/private repos/versatileAnnModule/trainDeepAr.py')
runcell(4, 'F:/projects/public github projects/private repos/versatileAnnModule/trainDeepAr.py')
runcell(5, 'F:/projects/public github projects/private repos/versatileAnnModule/trainDeepAr.py')
runcell('split train val test', 'F:/projects/public github projects/private repos/versatileAnnModule/trainDeepAr.py')
runcell(7, 'F:/projects/public github projects/private repos/versatileAnnModule/trainDeepAr.py')
# ---- 
class gaussianLogLikehoodLoss(torch.nn.Module):
    def __call__(self, mu, sigma, labels):
        '''
        Compute using gaussian the log-likehood which needs to be maximized. Ignore time steps where labels are missing.
        Args:
            mu: (Variable) dimension [batch_size] - estimated mean at time step t
            sigma: (Variable) dimension [batch_size] - estimated standard deviation at time step t
            labels: (Variable) dimension [batch_size] z_t
        Returns:
            loss: (Variable) average log-likelihood loss across the batch
        '''
        zero_index = (labels != 0)
        distribution = torch.distributions.normal.Normal(mu[zero_index], sigma[zero_index])
        likelihood = distribution.log_prob(labels[zero_index])
        return -torch.mean(likelihood)
criterion=gaussianLogLikehoodLoss()
workerNum=0
externalKwargs={'trainIndexes': trainIndexes, 'valIndexes': valIndexes, 'testIndexes':testIndexes,
                'allReals':allReals, 'criterion':criterion}#kkk criterion
model.trainModel(train, None, val, None, criterion, numEpochs=30, savePath=r'data\bestModels\tft1', workerNum=workerNum, externalKwargs=externalKwargs)
# ---- 

# ---- 

# ---- 

# ---- 



# ---- 

# ---- 

