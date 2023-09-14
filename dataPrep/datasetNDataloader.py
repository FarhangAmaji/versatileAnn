import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from torch.utils.data import Dataset, DataLoader
from utils.vAnnGeneralUtils import NpDict, DotDict, isListTupleOrSet
import warnings
import pandas as pd
import numpy as np
from dataPrep.dataCleaning import noNanOrNoneData
from utils.globalVars import tsStartPointColName
from torch.utils.data.dataloader import default_collate
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
    #kkk add npDict and dotDict
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

class BatchStructTemplate_Non_BatchStructTemplate_Objects:
    def __init__(self, obj):
        self.values=[]
        self.type=str(type(obj))

        #kkk add npDict and dotDict
        if self.type in knownTypesToBeTensored.tensor.values():
            self.toTensorFunc='stackTensors'
        elif self.type in knownTypesToBeTensored.directTensorables.values():
            self.toTensorFunc='stackListOfDirectTensorablesToTensor'
        elif self.type in knownTypesToBeTensored.df.values():
            self.toTensorFunc='stackListOfDfsToTensor'
        elif self.type in knownTypesToBeTensored.notTensorables.values():
            self.toTensorFunc='notTensorables'
        else:#includes knownTypesToBeTensored.errorPrones
            if isTensorable(obj):
                self.toTensorFunc='stackListOfErrorPronesToTensor'#kkk if had taken prudencyFactor, this could have been notTensorables or stackListOfDirectTensorablesToTensor
            else:
                self.toTensorFunc='notTensorables'

    def __repr__(self):
        return '{'+f'values:{self.values}, type:{self.type}, toTensorFunc:{self.toTensorFunc}'+'}'

def appendValueToNestedDictPath(inputDictStyle, path, value):
    assert isinstance(inputDictStyle, (dict, NpDict, BatchStructTemplate)),'inputDictStyle must be in one of dict, NpDict, BatchStructTemplate types'
    current = inputDictStyle
    if isinstance(current, BatchStructTemplate):
        current = current.dictStruct
    for i, key in enumerate(path[:-1]):
        assert isinstance(current, (dict, NpDict)), f'{path[:i+1]} is not a dict or NpDict'
        assert key in current.keys(), f'{key} is not in {path[:i]}'
        current = current[key]
    last_key = path[-1]
    assert last_key in current.keys(), f'{last_key} is not in {path}'

    if isinstance(inputDictStyle, BatchStructTemplate):
        assert isinstance(current[last_key].values, list), f'{path} doesn\'t lead to a list'
        current[last_key].values.append(value)
    else:
        assert isinstance(current[last_key], list), f'{path} doesn\'t lead to a list'
        current[last_key].append(value)

class TensorStacker:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def stackTensors(self, list_):
        stackTensor=torch.stack(list_).to(self.device)
        if stackTensor.dtype == torch.float16 or stackTensor.dtype == torch.float64:
            stackTensor = stackTensor.to(torch.float32)#kkk make it compatible to global precision
        return stackTensor

    def stackListOfErrorPronesToTensor(self, listOfErrorPrones):
        try:
            tensorList=[torch.tensor(na) for na in listOfErrorPrones]
        except:
            return listOfErrorPrones
        return self.stackTensors(tensorList)

    def stackListOfDirectTensorablesToTensor(self, listOfDirectTensorables):
        tensorList=[torch.tensor(na) for na in listOfDirectTensorables]
        return self.stackTensors(tensorList)

    def stackListOfDfsToTensor(self, listOfDfs):
        listOfNpArrays=[df.values for df in listOfDfs]
        return self.stackListOfDirectTensorablesToTensor(listOfNpArrays)

    def notTensorables(self, listOfNotTensorables):
        return listOfNotTensorables

class BatchStructTemplate(TensorStacker):#kkk move to collateUtils#kkk rename to collateStruct
    def __init__(self, inputDict):
        super().__init__()
        self.ObjsFunc=BatchStructTemplate_Non_BatchStructTemplate_Objects
        self.dictStruct=self.batchStructTemplateFunc(inputDict)

    def batchStructTemplateFunc(self, inputDict):
        if not isinstance(inputDict, dict):
            return self.ObjsFunc(inputDict)
        returnDict={}
        if len(inputDict)==0:
            return self.ObjsFunc(inputDict)
        for key, value in inputDict.items():
            if isinstance(value, dict):
                returnDict[key] = self.batchStructTemplateFunc(value)
            else:
                returnDict[key] = self.ObjsFunc(value)
        return returnDict

    def fillBatchStructWithData(self, itemToAdd, path=[]):
        if not isinstance(self.dictStruct, dict) and path==[]:#this is for the case we have made BatchStructTemplate of non dictionary object
            self.dictStruct.values.append(itemToAdd)
            return
        path=path[:]
        if len(itemToAdd)==0:#this is for the case we are retrieving an empty object, somewhere in the .dictStruct dictionaries' items
            appendValueToNestedDictPath(self, path, {})
        for key, value in itemToAdd.items():
            path2=path+[key]
            if isinstance(value, dict):
                self.fillBatchStructWithData(value, path2)
            else:
                appendValueToNestedDictPath(self, path2, value)

    def getBatchStructDictionaryValues(self, dictionary, toTensor=False):
        returnDict={}
        if len(dictionary)==0:
            return {}
        for key, value in dictionary.items():
            if isinstance(value, dict):
                returnDict[key] = self.getBatchStructDictionaryValues(value, toTensor=toTensor)
            else:
                if toTensor:
                    toTensorFunc = getattr(self,value.toTensorFunc)
                    returnDict[key] = toTensorFunc(value.values)
                else:
                    returnDict[key] = value.values
        return returnDict

    def getBatchStructValues(self, toTensor=False):
        if isinstance(self.dictStruct, BatchStructTemplate_Non_BatchStructTemplate_Objects):#this is for the case we have made BatchStructTemplate of non dictionary object
            if toTensor:
                toTensorFunc = getattr(self,self.dictStruct.toTensorFunc)
                return toTensorFunc (self.dictStruct.values)
            return self.dictStruct.values
        return self.getBatchStructDictionaryValues(self.dictStruct, toTensor=toTensor)

    def getBatchStructTensors(self):
        return self.getBatchStructValues(toTensor=True)

    def __repr__(self):
        return str(self.dictStruct)
#%% VAnnTsDataloader
class VAnnTsDataloader(DataLoader):
    #kkk needs tests
    #kkk seed everything
    #kkk can later take modes, 'speed', 'gpuMemory'. for i.e. pin_memory occupies the gpuMemory but speeds up
    def __init__(self, dataset, batch_size=64, collate_fn=None, doBatchStructureCheckOnAllData=False, *args, **kwargs):
        if 'batchSize' in kwargs.keys():
            batch_size=kwargs['batchSize']

        if collate_fn is None:
            collate_fn=self.commonCollate_fn
        super().__init__(dataset, batch_size=batch_size, collate_fn=collate_fn, *args, **kwargs)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')#kkk make it compatible to self.device of vAnn
        if doBatchStructureCheckOnAllData:
            self.doBatchStructureCheckOnAllData()

    def doBatchStructureCheckOnAllData(self):#kkk do I need this
        pass
        #kkk this useful to check some custom collate_fns
        #kkk implement it later
        #kkk find dictStruct which is sure works on all items of 1epoch, and in case of type incompatibility change the explicit
        #...type for i.e. "<class 'list'>" to 'MIX'
        #kkk do it in parallel

    def bestNumWorkerFinder(self):
        pass
        #kkk implement it later

    def findBatchStruct(self, batch):
        if isListTupleOrSet(batch):
            batchStructOfBatch=[]
            for item in batch:
                batchStructOfBatch.append(BatchStructTemplate(item))
            self.batchStruct = batchStructOfBatch
        else:
            self.batchStruct = BatchStructTemplate(batch)

    def commonCollate_fn(self, batch):
        defaultCollateRes=default_collate(batch)
        if not hasattr(self, 'batchStruct'):
            self.findBatchStruct(defaultCollateRes)
        batchStructCopy=self.batchStruct.copy()
        batchStructCopy.fillBatchStructWithData(defaultCollateRes)
        return batchStructCopy.getBatchStructTensors()