import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pandas as pd
#%%
class VAnnTsDataset(Dataset):
    def __init__(self, data, backcastLen, forecastLen, indexes=None, **kwargs):
        self.data = data
        self.backcastLen = backcastLen
        self.forecastLen = forecastLen
        self.indexes = indexes#kkk would it give error if no indexes are available#kkk point to start then is needed
        assert data.isnull().any().any()==False,'the data should be cleaned in order not to have nan or None data'#kkk assert here
        self.device= torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __len__(self):
        if self.indexes is None:
            return len(self.data)
        return len(self.indexes)

    #kkk we dont have batches here
    def getBackForeCastData(self, dfOrTensor, batchIndexes, mode='backcast', colsOrIndexes='___all___', toTensor=True, dtypeChange=True):#kkk may add query taking ability to df part
        assert mode in ['backcast', 'forecast', 'fullcast'], "mode should be either 'backcast', 'forecast' or 'fullcast'"#kkk if query is added, these modes have to be more flexible
        def getDfRows(df, lowerBoundGap, upperBoundGap, cols, batchIndexes):#kkk move it to self?#kkk batchIndexes from self.indexes?
        #kkk add singlePoint mode
            assert '___all___' not in df.columns,'df shouldnt have a column named "___all___", use other manuall methods of obtaining cols'
            if cols=='___all___':
                return [df.loc[idx + lowerBoundGap:idx + upperBoundGap-1] for idx in batchIndexes]
            else:
                return [df.loc[idx + lowerBoundGap:idx + upperBoundGap-1, cols] for idx in batchIndexes]
        
        def getTensorRows(tensor, lowerBoundGap, upperBoundGap, colIndexes, batchIndexes):
            if colIndexes=='___all___':
                return [tensor[idx + lowerBoundGap:idx + upperBoundGap,:] for idx in batchIndexes]
            else:
                return [tensor[idx + lowerBoundGap:idx + upperBoundGap, colIndexes] for idx in batchIndexes]

        def getCastByMode(typeFunc, dfOrTensor, batchIndexes, mode='backcast', colsOrIndexes='___all___'):
            if mode=='backcast':
                return typeFunc(dfOrTensor, 0, self.backcastLen, colsOrIndexes, batchIndexes)
            elif mode=='forecast':
                return typeFunc(dfOrTensor, self.backcastLen, self.backcastLen+self.forecastLen, colsOrIndexes, batchIndexes)
            elif mode=='fullcast':
                return typeFunc(dfOrTensor, 0, self.backcastLen+self.forecastLen, colsOrIndexes, batchIndexes)

        if isinstance(dfOrTensor, pd.DataFrame):
            #kkk add NpDict
            res=getCastByMode(getDfRows, dfOrTensor, mode=mode, colsOrIndexes=colsOrIndexes, batchIndexes=batchIndexes)
            if toTensor:
                res=self.stackListOfDfsTensor(res, dtypeChange=dtypeChange)
            return res
        elif isinstance(dfOrTensor, torch.Tensor):
            res=getCastByMode(getTensorRows, dfOrTensor, mode=mode, colsOrIndexes=colsOrIndexes, batchIndexes=batchIndexes)
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