import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from torch.utils.data import Dataset
from utils.vAnnGeneralUtils import NpDict, DotDict, floatDtypeChange
import warnings
import pandas as pd
import numpy as np
from dataPrep.dataCleaning import noNanOrNoneData
from utils.globalVars import tsStartPointColName
#%% VAnnTsDataset
class TsRowFetcher:
    #kkk needs tests
    def __init__(self):
        self.modes=DotDict({key: key for key in ['backcast', 'forecast', 'fullcast','singlePoint']})

    def singleFeatureShapeCorrection(self, data):
        if len(data.shape)==2 and data.shape[1]==1:
            return data.squeeze(1)
        return data

    def getDfRows(self, df, idx, lowerBoundGap, upperBoundGap, cols):#kkk move these to class getTsRows
        assert '___all___' not in df.columns,'df shouldnt have a column named "___all___", use other manuall methods of obtaining cols'
        if cols=='___all___':
            return df[idx + lowerBoundGap:idx + upperBoundGap]
        else:
            return df[cols][idx + lowerBoundGap:idx + upperBoundGap]

    def getTensorRows(self, tensor, idx, lowerBoundGap, upperBoundGap, colIndexes):
        if colIndexes=='___all___':
            res = tensor[idx + lowerBoundGap:idx + upperBoundGap,:]
        else:
            res = tensor[idx + lowerBoundGap:idx + upperBoundGap, colIndexes]
        return self.singleFeatureShapeCorrection(res)
        
    def getNpDictRows(self, npDict, idx, lowerBoundGap, upperBoundGap, colIndexes):
        if colIndexes=='___all___':
            res =  npDict[:][idx + lowerBoundGap:idx + upperBoundGap]
        else:
            res =  npDict[colIndexes][idx + lowerBoundGap:idx + upperBoundGap]
        return self.singleFeatureShapeCorrection(res)

    def getNpArrayRows(self, npArray, idx, lowerBoundGap, upperBoundGap, colIndexes):
        if colIndexes=='___all___':
            res =  npArray[idx + lowerBoundGap:idx + upperBoundGap,:]
        else:
            res =  npArray[idx + lowerBoundGap:idx + upperBoundGap,colIndexes]
        return self.singleFeatureShapeCorrection(res)

    def makeTensor(self,input_):
        tensor = torch.tensor(input_)
        tensor = floatDtypeChange(tensor)
        return tensor

    def getBackForeCastData(self, data, idx, mode='backcast', colsOrIndexes='___all___', makeTensor=True):#kkk may add query taking ability to df part
        assert mode in self.modes.keys(), "mode should be either 'backcast', 'forecast','fullcast' or 'singlePoint'"#kkk if query is added, these modes have to be more flexible
        assert colsOrIndexes=='___all___' or isinstance(colsOrIndexes, list),"u should either pass '___all___' for all feature cols or a list of their columns or indexes"

        def getCastByMode(typeFunc, data, idx, mode=self.modes.backcast, colsOrIndexes='___all___'):
            if mode==self.modes.backcast:
                return typeFunc(data, idx, 0, self.backcastLen, colsOrIndexes)
            elif mode==self.modes.forecast:
                return typeFunc(data, idx, self.backcastLen, self.backcastLen+self.forecastLen, colsOrIndexes)
            elif mode==self.modes.fullcast:
                return typeFunc(data, idx, 0, self.backcastLen+self.forecastLen, colsOrIndexes)
            elif mode==self.modes.singlePoint:
                return typeFunc(data, idx, 0, 1, colsOrIndexes)

        if isinstance(data, NpDict):
            res = getCastByMode(self.getNpDictRows, data, idx=idx, mode=mode, colsOrIndexes=colsOrIndexes)
        elif isinstance(data, pd.DataFrame):
            res = getCastByMode(self.getDfRows, data, idx=idx, mode=mode, colsOrIndexes=colsOrIndexes)
        elif isinstance(data, np.ndarray):
            res = getCastByMode(self.getNpArrayRows, data, idx=idx, mode=mode, colsOrIndexes=colsOrIndexes)
        elif isinstance(data, torch.Tensor):
            res = getCastByMode(self.getTensorRows, data, idx=idx, mode=mode, colsOrIndexes=colsOrIndexes)
        else:
            assert False, 'to use "getBackForeCastData" data type should be pandas.DataFrame or torch.Tensor or np ndarray or NpDict'

        if makeTensor:
            res = self.makeTensor(res)
        return res

class VAnnTsDataset(Dataset, TsRowFetcher):
    #kkk needs tests
    def __init__(self, data, backcastLen, forecastLen, indexes=None, useNpDictForDfs=True, **kwargs):
        super().__init__()
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
        for key, value in kwargs.items():
            setattr(self, key, value)

    def noNanOrNoneData(self):
        if isinstance(self.data, (pd.DataFrame, pd.Series)):
            noNanOrNoneData(self.data.loc[self.indexes])
        elif isinstance(self.data, np.ndarray):
            noNanOrNoneData(self.data[:,self.indexes])
        elif isinstance(self.data, NpDict):
            noNanOrNoneData(self.data[:][self.indexes])
        elif isinstance(self.data, torch.Tensor):
            noNanOrNoneData(self.data[:,self.indexes])
    
    def shapeWarning(self):
        if isinstance(self.data, (torch.Tensor, np.ndarray)):
            shape = self.data.shape
            if shape[0] < shape[1]:
                warnings.warn("The data shape suggests that different features may be along shape[1]. "
                              "Consider transposing the data to have features along shape[0].")
    def __len__(self):
        if self.indexes is None:
            return len(self.data)
        return len(self.indexes)

    def __getitem__(self, idx):#kkk give warning if the idx is not in tsStartpoints#kkk other thing is that we should be able to turn the warnings off by type for i.e. we can turn off this type of warning
        if self.indexes is None:
            return self.data.loc[idx]
        return self.data[self.indexes[idx]]