import torch
from torch.utils.data import Dataset
from utils.vAnnGeneralUtils import NpDict, DotDict, floatDtypeChange
import warnings
import pandas as pd
import numpy as np
from dataPrep.dataCleaning import noNanOrNoneData
from utils.globalVars import tsStartPointColName
#%% TsRowFetcher
class TsRowFetcher:
    def __init__(self, backcastLen, forecastLen):
        self.modes=DotDict({key: key for key in ['backcast', 'forecast', 'fullcast','singlePoint']})
        self.backcastLen = backcastLen
        self.forecastLen = forecastLen
        self.indexes=None

    def assertIdxInIndexes(self, idx):
        if not self.indexes is None:
            assert idx in self.indexes,f'{idx} is not indexes'

    def singleFeatureShapeCorrection(self, data):
        if len(data.shape)==2 and data.shape[1]==1:
            return data.squeeze(1)
        return data

    def getDfRows(self, df, idx, lowerBoundGap, upperBoundGap, cols):#kkk does this idx match with getItem of dataset
        assert '___all___' not in df.columns,'df shouldnt have a column named "___all___", use other manuall methods of obtaining cols'
        if cols=='___all___':
            return df.loc[idx + lowerBoundGap:idx + upperBoundGap-1]
        else:
            return df.loc[idx + lowerBoundGap:idx + upperBoundGap-1,cols]

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
        if isinstance(input_, pd.DataFrame):
            input_=input_.values
        tensor = torch.tensor(input_)
        tensor = floatDtypeChange(tensor)
        return tensor

    def getBackForeCastData(self, data, idx, mode='backcast', colsOrIndexes='___all___', makeTensor=True, canBeOutStartIndex=False):#kkk may add query taking ability to df part
        assert mode in self.modes.keys(), "mode should be either 'backcast', 'forecast','fullcast' or 'singlePoint'"#kkk if query is added, these modes have to be more flexible
        assert colsOrIndexes=='___all___' or isinstance(colsOrIndexes, list),"u should either pass '___all___' for all feature cols or a list of their columns or indexes"
        if canBeOutStartIndex:
            self.assertIdxInIndexes(idx)

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
            assert False, 'to use "getBackForeCastData" data type should be pandas.DataFrame or torch.Tensor or np.ndarray or NpDict'

        if makeTensor:
            res = self.makeTensor(res)
        return res
#%% VAnnTsDataset
class VAnnTsDataset(Dataset, TsRowFetcher):
    noIndexesAssertionMsg="u have to pass indexes unless both backcastLen and forecastLen are 0, or u have passed a pd df or NpDict with __startPoint__ column"
    #kkk needs tests
    #kkk model should check device, backcastLen, forecastLen with this
    def __init__(self, data, backcastLen, forecastLen, indexes=None, useNpDictForDfs=True, **kwargs):
        super().__init__(backcastLen=backcastLen, forecastLen=forecastLen)
        if useNpDictForDfs and isinstance(data,pd.DataFrame):
            self.data=NpDict(data)
        else:
            self.data = data
        if indexes is None:
            assert (backcastLen==0 and forecastLen==0) or (isinstance(data,pd.DataFrame) and tsStartPointColName  in data.columns)\
                or (isinstance(data, NpDict) and tsStartPointColName  in data.cols()),\
                VAnnTsDataset.noIndexesAssertionMsg
            if isinstance(data,pd.DataFrame) and tsStartPointColName  in data.columns:
                indexes=data[data[tsStartPointColName]==True].index
                "#ccc note indexes has kept their values"
            if isinstance(data, NpDict) and tsStartPointColName  in data.cols():
                indexes=data.__index__[data['__startPoint__']==True]
                indexes=[list(data.__index__).index(i) for i in indexes]
                "#ccc note indexes for NpDict are according to their order"
        self.indexes = indexes
        assert len(self)>=backcastLen + forecastLen,'the data provided should have a length greater equal than (backcastLen+forecastLen)'
        self.shapeWarning()
        self.noNanOrNoneDataAssertion()
        for key, value in kwargs.items():
            setattr(self, key, value)

    def noNanOrNoneDataAssertion(self):
        noNanOrNoneData(self.data)
    
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

    def __getitem__(self, idx):
        self.assertIdxInIndexes(idx)
        if isinstance(self.data, (pd.DataFrame, pd.Series)):
            return self.data.loc[idx]
        elif isinstance(self.data, NpDict):
            return self.data[:][idx]
        elif isinstance(self.data, (np.ndarray, torch.Tensor)):
            return self.data[idx]