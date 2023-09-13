import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from torch.utils.data import Dataset, DataLoader
from utils.vAnnGeneralUtils import NpDict, DotDict
import warnings
import pandas as pd
import numpy as np
from dataPrep.dataCleaning import noNanOrNoneData
from utils.globalVars import tsStartPointColName
#%% VAnnTsDataset
class VAnnTsDataset(Dataset):
    #kkk needs tests
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
        self.pointTypes=DotDict({key: key for key in ['backcast', 'forecast', 'fullcast','singlePoint']})

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

    def getDfRows(self, df, idx, lowerBoundGap, upperBoundGap, cols):#kkk move these to class getTsRows
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
        elif isinstance(data, np.ndarray):
            return getCastByMode(self.getNpArrayRows, data, idx=idx, mode=mode, colsOrIndexes=colsOrIndexes)
        elif isinstance(data, torch.Tensor):
            return getCastByMode(self.getTensorRows, data, idx=idx, mode=mode, colsOrIndexes=colsOrIndexes)
        else:
            assert False, 'to use "getBackForeCastData" data type should be pandas.DataFrame or torch.Tensor or np ndarray or NpDict'

    def __getitem__(self, idx):#kkk give warning if the idx is not in tsStartpoints#kkk other thing is that we should be able to turn the warnings off by type for i.e. we can turn off this type of warning
        if self.indexes is None:
            return self.data.loc[idx]
        return self.data[self.indexes[idx]]
#%% dataset output for batch structure detection
def isTensorable(obj):
    try:
        torch.tensor(obj)
        return True
    except:
        return False

"#ccc eventhough npArray, tuple, df,series may contain str, which cant converted to tensor, but still we keep it this way"
knownTypesToBeTensored=DotDict({
    'directTensorables':DotDict({
        'int':"<class 'int'>", 'float':"<class 'float'>", 'complex':"<class 'complex'>", 
        'tuple':"<class 'tuple'>", 'npArray':"<class 'numpy.ndarray'>", 
        'pdSeries':"<class 'pandas.core.series.Series'>", 
        'bool':"<class 'bool'>", 'bytearray': "<class 'bytearray'>"}),

    'tensor':
        DotDict({'tensor':"<class 'torch.Tensor'>"}),

    'errorPrones':
        DotDict({'list':"<class 'list'>"}),# depending on items ok and not

    'df':#indirectTensorables
        DotDict({'df':"<class 'pandas.core.frame.DataFrame'>"}),# cant directly be changed to tensor

    'notTensorables':DotDict({#these below can't be changed to tensor
        'set':"<class 'set'>", 'dict':"<class 'dict'>",'str':"<class 'str'>",
        'none':"<class 'NoneType'>", 'bytes':"<class 'bytes'>"})
    })

class returnDictStruct_Non_ReturnDictStruct_Objects:
    def __init__(self, obj):
        self.values=[]
        self.type=str(type(obj))

        #kkk add npDict
        if self.type in knownTypesToBeTensored.tensor.values():
            self.funcToUse='stackTensors'
        elif self.type in knownTypesToBeTensored.directTensorables.values():
            self.funcToUse='stackListOfDirectTensorablesToTensor'
        elif self.type in knownTypesToBeTensored.df.values():
            self.funcToUse='stackListOfDfsToTensor'
        elif self.type in knownTypesToBeTensored.notTensorables.values():
            self.funcToUse='notTensorables'
        else:#includes knownTypesToBeTensored.errorPrones
            if isTensorable(obj):
                self.funcToUse='stackListOfErrorPronesToTensor'#kkk if had taken prudencyFactor, this could have been notTensorables or stackListOfDirectTensorablesToTensor
            else:
                self.funcToUse='notTensorables'

    def __repr__(self):
        return '{'+f'values:{self.values}, type:{self.type}, funcToUse:{self.funcToUse}'+'}'

class returnDictStruct:
    def __init__(self, inputDict):
        self.ObjsFunc=returnDictStruct_Non_ReturnDictStruct_Objects
        self.dictStruct=self.returnDictStructFunc(inputDict)

    def returnDictStructFunc(self, inputDict):
        if not isinstance(inputDict, dict):
            return self.ObjsFunc(inputDict)
        returnDict={}
        if len(inputDict)==0:
            return self.ObjsFunc(inputDict)
        for key, value in inputDict.items():
            if isinstance(value, dict):
                returnDict[key] = self.returnDictStructFunc(value)
            else:
                returnDict[key] = self.ObjsFunc(value)
        return returnDict

    def fillDataWithDictStruct(self, itemToAdd, path=[]):
        if not isinstance(self.dictStruct, dict) and path==[]:#this is for the case we have made returnDictStruct of non dictionary object
            self.dictStruct.values.append(itemToAdd)
            return
        path=path[:]
        if len(itemToAdd)==0:#this is for the case we are retrieving an empty object, somewhere in the .dictStruct dictionaries' items
            appendValueToNestedDictPath(self, path, {})
        for key, value in itemToAdd.items():
            path2=path+[key]
            if isinstance(value, dict):
                self.fillDataWithDictStruct(value, path2)
            else:
                appendValueToNestedDictPath(self, path2, value)

    def getDictStructDictionaryValues(self, dictionary):
        returnDict={}
        if len(dictionary)==0:
            return {}
        for key, value in dictionary.items():
            if isinstance(value, dict):
                returnDict[key] = self.getDictStructDictionaryValues(value)
            else:
                returnDict[key] = value.values
        return returnDict

    def getDictStructValues(self):
        if isinstance(self.dictStruct, returnDictStruct_Non_ReturnDictStruct_Objects):#this is for the case we have made returnDictStruct of non dictionary object
            return self.dictStruct.values
        return self.getDictStructDictionaryValues(self.dictStruct)

    def __repr__(self):
        return str(self.dictStruct)

#%% VAnnTsDataloader
class TensorStacker:
    def __init(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def stackTensors(self, list_):
        stackTensor=torch.stack(list_).to(self.device)
        if stackTensor.dtype == torch.float16 or stackTensor.dtype == torch.float64:
            stackTensor = stackTensor.to(torch.float32)#kkk make it compatible to global precision
        return stackTensor

    def __iter__(self):
        for batch in super().__iter__():
            yield [item.to(self.device) for item in batch]#kkk make it compatible to self.device of vAnn