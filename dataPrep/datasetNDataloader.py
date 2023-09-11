import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from utils.vAnnGeneralUtils import NpDict, DotDict
import warnings
import pandas as pd
import numpy as np
from dataPrep.dataCleaning import noNanOrNoneData
from utils.globalVars import tsStartPointColName
#%%
class VAnnTsDataset(Dataset):#kkk needs tests
    def __init__(self, data, backcastLen, forecastLen, indexes=None, useNpDictForDfs=True, **kwargs):
        self.data = data#kkk make sure its compatible with lists and np arrays
        self.backcastLen = backcastLen
        self.forecastLen = forecastLen
        if indexes is None:
            assert not ((backcastLen==0 and forecastLen==0) or (isinstance(data,pd.DataFrame) and tsStartPointColName not in data.columns)),"u have to pass indexes unless both backcastLen and forecastLen are 0, or u have passed a pd df with __startPoint__ columns"
            if tsStartPointColName in data.columns:
                indexes=data[data[tsStartPointColName]==True].index
        self.indexes = indexes
        self.shapeWarning()
        self.noNanOrNoneData()
        if useNpDictForDfs:
            self.data=NpDict(self.data)
        self.device= torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.pointTypes=DotDict({'backcast':'backcast', 'forecast':'forecast', 'fullcast':'fullcast','singlePoint':'singlePoint'})

    def shapeWarning(self):
        if isinstance(self.data, (torch.Tensor, np.ndarray)):
            shape = self.data.shape
            if shape[0] < shape[1]:
                warnings.warn("The data shape suggests that different features may be along shape[1]. "
                              "Consider transposing the data to have features along shape[0].")

    def noNanOrNoneData(self):
        if isinstance(self.data, (pd.DataFrame, pd.Series)):
            noNanOrNoneData(self.data.loc[self.indexes])
        elif isinstance(self.data, np.ndarray):
            noNanOrNoneData(self.data[:,self.indexes])
        elif isinstance(self.data, NpDict):
            noNanOrNoneData(self.data[:][self.indexes])
        elif isinstance(self.data, torch.Tensor):
            noNanOrNoneData(self.data[:,self.indexes])

    def __len__(self):
        if self.indexes is None:
            return len(self.data)
        return len(self.indexes)

    def getDfRows(self, df, idx, lowerBoundGap, upperBoundGap, cols):
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

    def getNpDictRows(self, npDict, idx, lowerBoundGap, upperBoundGap, colIndexes):
        if colIndexes=='___all___':
            return npDict[:][idx + lowerBoundGap:idx + upperBoundGap]
        else:
            return npDict[colIndexes][idx + lowerBoundGap:idx + upperBoundGap]

    def getNpArrayRows(self, npArray, idx, lowerBoundGap, upperBoundGap, colIndexes):
        if colIndexes=='___all___':
            return npArray[idx + lowerBoundGap:idx + upperBoundGap,:]
        else:
            return npArray[idx + lowerBoundGap:idx + upperBoundGap,colIndexes]

    def getBackForeCastData(self, data, idx, mode='backcast', colsOrIndexes='___all___'):#kkk may add query taking ability to df part
        assert mode in self.pointTypes.keys(), "mode should be either 'backcast', 'forecast' or 'fullcast'"#kkk if query is added, these modes have to be more flexible
        def getCastByMode(typeFunc, data, idx, mode=self.pointTypes.backcast, colsOrIndexes='___all___'):
            if mode==self.pointTypes.backcast:
                return typeFunc(data, idx, 0, self.backcastLen, colsOrIndexes)
            elif mode==self.pointTypes.forecast:
                return typeFunc(data, idx, self.backcastLen, self.backcastLen+self.forecastLen, colsOrIndexes)
            elif mode==self.pointTypes.fullcast:
                return typeFunc(data, idx, 0, self.backcastLen+self.forecastLen, colsOrIndexes)
            elif mode==self.pointTypes.singlePoint:
                return typeFunc(data, idx, 0, 0, colsOrIndexes)

        if isinstance(data, NpDict):
            return getCastByMode(self.getNpDictRows, data, idx=idx, mode=mode, colsOrIndexes=colsOrIndexes)
        elif isinstance(data, pd.DataFrame):
            return getCastByMode(self.getDfRows, data, idx=idx, mode=mode, colsOrIndexes=colsOrIndexes)
        elif isinstance(data, np.ndarray):#kkk do I need single col np.array
            return getCastByMode(self.getNpArrayRows, data, idx=idx, mode=mode, colsOrIndexes=colsOrIndexes)
        elif isinstance(data, torch.Tensor):
            return getCastByMode(self.getTensorRows, data, idx=idx, mode=mode, colsOrIndexes=colsOrIndexes)
        else:
            assert False, 'data type should be pandas.DataFrame or torch.Tensor or np ndarray or NpDict'

    def __getitem__(self, idx):#kkk give warning if the idx is not in tsStartpoints#kkk other thing is that we should be able to turn the warnings off by type for i.e. we can turn off this type of warning
        if self.indexes is None:
            return self.data.loc[idx]
        return self.data[self.indexes[idx]]

class VAnnTsDataloader(DataLoader):
    #kkk seed everything
    def __init__(self, dataset, *args, **kwargs):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        super().__init__(dataset, *args, **kwargs)

    def stackListOfDfsTensor(self, listOfDfs):
        tensorList=[torch.tensor(df.values) for df in listOfDfs]
        return self.stackTensors(tensorList)

    def stackTensors(self, list_):
        stackTensor=torch.stack(list_).to(self.device)
        if stackTensor.dtype == torch.float16 or stackTensor.dtype == torch.float64:
            stackTensor = stackTensor.to(torch.float32)#kkk make it compatible to global precision
        return stackTensor

    def __iter__(self):
        for batch in super().__iter__():
            # Move the batch to GPU before returning it
            yield [item.to(self.device) for item in batch]#kkk make it compatible to self.device of vAnn