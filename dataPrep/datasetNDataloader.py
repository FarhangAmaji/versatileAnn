import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pandas as pd
from utils.globalVars import tsStartPointColName
#%%
class VAnnTsDataset(Dataset):
    def __init__(self, data, backcastLen, forecastLen, indexes=None, **kwargs):
        self.data = data#kkk make sure its compatible with lists and np arrays
        self.backcastLen = backcastLen
        self.forecastLen = forecastLen
        if indexes is None:
            assert not (backcastLen==0 and backcastLen==0 and tsStartPointColName not in data.columns),"u can't have timeseries data, without passing indexes or __startPoint__ column" #kkk supposes data only is df
            if tsStartPointColName in data.columns:
                indexes=data[data[tsStartPointColName]==True].index
        self.indexes = indexes
        assert data.loc[indexes].isnull().any().any()==False,'the data should be cleaned in order not to have nan or None data'
        self.device= torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __len__(self):
        if self.indexes is None:
            return len(self.data)
        return len(self.indexes)

    def getDfRows(self, df, idx, lowerBoundGap, upperBoundGap, cols):#kkk move it to self?#kkk batchIndexes from self.indexes?
        assert '___all___' not in df.columns,'df shouldnt have a column named "___all___", use other manuall methods of obtaining cols'
        if cols=='___all___':
            return df.loc[idx + lowerBoundGap:idx + upperBoundGap-1]
        else:
            return df.loc[idx + lowerBoundGap:idx + upperBoundGap-1, cols]

    def getTensorRows(self, tensor, idx, lowerBoundGap, upperBoundGap, colIndexes):
        if colIndexes=='___all___':
            return tensor[idx + lowerBoundGap:idx + upperBoundGap,:]
        else:
            return tensor[idx + lowerBoundGap:idx + upperBoundGap, colIndexes]

    #kkk we dont have batches here
    def getBackForeCastData(self, dfOrTensor, idx, mode='backcast', colsOrIndexes='___all___', toTensor=True, dtypeChange=True):#kkk may add query taking ability to df part
        assert mode in ['backcast', 'forecast', 'fullcast','singlePoint'], "mode should be either 'backcast', 'forecast' or 'fullcast'"#kkk if query is added, these modes have to be more flexible
        #kkk add singlePoint mode
        def getCastByMode(typeFunc, dfOrTensor, idx, mode='backcast', colsOrIndexes='___all___'):
            if mode=='backcast':
                return typeFunc(dfOrTensor, idx, 0, self.backcastLen, colsOrIndexes)
            elif mode=='forecast':
                return typeFunc(dfOrTensor, idx, self.backcastLen, self.backcastLen+self.forecastLen, colsOrIndexes)
            elif mode=='fullcast':
                return typeFunc(dfOrTensor, idx, 0, self.backcastLen+self.forecastLen, colsOrIndexes)
            elif mode=='fullcast':
                return typeFunc(dfOrTensor, idx, 0, 0, colsOrIndexes)

        if isinstance(dfOrTensor, pd.DataFrame):
            #kkk add NpDict
            res=getCastByMode(self.getDfRows, dfOrTensor, idx=idx, mode=mode, colsOrIndexes=colsOrIndexes)
            if toTensor:
                res=self.stackListOfDfsTensor(res, dtypeChange=dtypeChange)
            return res
        elif isinstance(dfOrTensor, torch.Tensor):
            res=getCastByMode(self.getTensorRows, dfOrTensor, idx=idx, mode=mode, colsOrIndexes=colsOrIndexes)
            if toTensor:
                res=self.stackTensors(res, dtypeChange=dtypeChange)
            return res
        else:
            assert False, 'dfOrTensor type should be pandas.DataFrame or torch.Tensor'

    def stackListOfDfsTensor(self, listOfDfs, dtypeChange=True):#kkk to dataloader
        tensorList=[torch.tensor(df.values) for df in listOfDfs]
        return self.stackTensors(tensorList, dtypeChange=dtypeChange)

    def stackTensors(self, list_, dtypeChange=True):#kkk to dataloader
        stackTensor=torch.stack(list_).to(self.device)
        if dtypeChange:#kkk check if its float, then change it to float32
            stackTensor = stackTensor.to(torch.float32)#kkk make it compatible to global precision
        return stackTensor

    def __getitem__(self, idx):
        if self.indexes is None:
            return self.data.loc[idx]
        return self.data[self.indexes[idx]]

class VAnnTsDataloader(DataLoader):
    def __init__(self, dataset, *args, **kwargs):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        super().__init__(dataset, *args, **kwargs)
    #kkk seed everything
    def __iter__(self):
        for batch in super().__iter__():
            # Move the batch to GPU before returning it
            yield [item.to(self.device) for item in batch]#kkk make it compatible to self.device of vAnn