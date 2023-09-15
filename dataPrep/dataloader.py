import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from torch.utils.data import DataLoader
from utils.vAnnGeneralUtils import NpDict, DotDict, isListTupleOrSet, floatDtypeChange, isIterable
from torch.utils.data.dataloader import default_collate
import copy
#%% batch structure detection
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

    'NpDict':#indirectTensorables
        DotDict({'NpDict':"<class 'utils.vAnnGeneralUtils.NpDict'>"}),# cant directly be changed to tensor

    'df':#indirectTensorables
        DotDict({'df':"<class 'pandas.core.frame.DataFrame'>"}),# cant directly be changed to tensor

    'notTensorables':DotDict({#these below can't be changed to tensor
        'set':"<class 'set'>", 'dict':"<class 'dict'>",'str':"<class 'str'>",
        'none':"<class 'NoneType'>", 'bytes':"<class 'bytes'>",
        'DotDict':"<class 'utils.vAnnGeneralUtils.DotDict'>"})
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
        elif self.type in knownTypesToBeTensored.NpDict.values():
            self.toTensorFunc='stackListOfNpDictsToTensor'
        elif self.type in knownTypesToBeTensored.notTensorables.values():
            self.toTensorFunc='notTensorables'
        else:#includes knownTypesToBeTensored.errorPrones
            if isTensorable(obj):
                self.toTensorFunc='stackListOfErrorPronesToTensor'#kkk if had taken prudencyFactor, this could have been notTensorables or stackListOfDirectTensorablesToTensor
            else:
                self.toTensorFunc='notTensorables'

    def __repr__(self):
        return '{'+f'values:{self.values}, type:{self.type}, toTensorFunc:{self.toTensorFunc}'+'}'

def updateNestedDictPath(inputDictStyle, path, value, extendIfPossible=False):
    assert isinstance(inputDictStyle, (dict, NpDict, BatchStructTemplate)), 'inputDictStyle must be in one of dict, NpDict, BatchStructTemplate types'
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
        if extendIfPossible and isIterable(value):
            current[last_key].values.extend(value)
        else:
            current[last_key].values.append(value)
    else:
        assert isinstance(current[last_key], list), f'{path} doesn\'t lead to a list'
        if extendIfPossible and isIterable(value):
            current[last_key].extend(value)
        else:
            current[last_key].append(value)

def appendValueToNestedDictPath(inputDictStyle, path, value):
    updateNestedDictPath(inputDictStyle, path, value, extendIfPossible=False)

def extendIfPossibleValueToNestedDictPath(inputDictStyle, path, value):
    updateNestedDictPath(inputDictStyle, path, value, extendIfPossible=True)


class TensorStacker:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def stackTensors(self, list_):
        stackTensor=torch.stack(list_).to(self.device)
        stackTensor = floatDtypeChange(stackTensor)
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
    
    def stackListOfNpDictsToTensor(self, listOfNpDicts):
        listOfDfs=[npDict.df for npDict in listOfNpDicts]
        return self.stackListOfDirectTensorablesToTensor(listOfDfs)

    def notTensorables(self, listOfNotTensorables):
        return listOfNotTensorables

class BatchStructTemplate(TensorStacker):#kkk move to collateUtils#kkk rename to collateStruct
    def __init__(self, inputDict):
        super().__init__()
        self.objsType=BatchStructTemplate_Non_BatchStructTemplate_Objects
        self.dictStruct=self.batchStructTemplateFunc(inputDict)

    def batchStructTemplateFunc(self, inputDict):
        if not isinstance(inputDict, dict):
            return self.objsType(inputDict)
        returnDict={}
        if len(inputDict)==0:
            return self.objsType(inputDict)
        for key, value in inputDict.items():
            if isinstance(value, dict):
                returnDict[key] = self.batchStructTemplateFunc(value)
            else:
                returnDict[key] = self.objsType(value)
        return returnDict

    def fillWithData(self, itemToAdd, path=[]):
        if not isinstance(self.dictStruct, dict) and path==[]:#this is for the case we have made BatchStructTemplate of non dictionary object
            self.dictStruct.values.append(itemToAdd)
            return
        path=path[:]
        if len(itemToAdd)==0:#this is for the case we are retrieving an empty object, somewhere in the .dictStruct dictionaries' items
            appendValueToNestedDictPath(self, path, {})
        for key, value in itemToAdd.items():
            path2=path+[key]
            if isinstance(value, dict):
                self.fillWithData(value, path2)
            else:
                appendValueToNestedDictPath(self, path2, value)

    def assertIsBatchStructTemplateOrListOfBatchStructTemplates(obj):
        assert isinstance(obj, BatchStructTemplate) or \
        (isinstance(obj, list) and all([isinstance(it, BatchStructTemplate) for it in obj])),\
            'this is not, list of BatchStructTemplates or BatchStructTemplate type'

    def fillSingleOrMultipleWithData(batchStructTemplates, itemsToAdd):
        BatchStructTemplate.assertIsBatchStructTemplateOrListOfBatchStructTemplates(batchStructTemplates)
        if isinstance(batchStructTemplates, list):
            assert len(batchStructTemplates)==len(itemsToAdd),'batchStructTemplates and itemsToAdd dont have the same length'
            for i, batchStructTemplate in enumerate(batchStructTemplates):
                batchStructTemplate.fillWithData(itemsToAdd[i])
        else:
            batchStructTemplates.fillWithData(itemsToAdd)

    def squeezeshape0of1(self, tensor):
        if not isinstance(tensor, torch.Tensor):
            return tensor
        while tensor.shape[0]==1:
            tensor = tensor.squeeze(0)
        return tensor

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
                    returnDict[key] = self.squeezeshape0of1(toTensorFunc(value.values))#kkk maybe could have found better but more complex solution than squeezeshape0of1; this one has problem with batchSize 1!!!
                else:
                    returnDict[key] = value.values
        return returnDict

    def getBatchStructValues(self, toTensor=False):
        if isinstance(self.dictStruct, BatchStructTemplate_Non_BatchStructTemplate_Objects):#this is for the case we have made BatchStructTemplate of non dictionary object
            if toTensor:
                toTensorFunc = getattr(self,self.dictStruct.toTensorFunc)
                return self.squeezeshape0of1(toTensorFunc(self.dictStruct.values))
            return self.dictStruct.values
        return self.getBatchStructDictionaryValues(self.dictStruct, toTensor=toTensor)

    def getBatchStructTensors(self):
        return self.getBatchStructValues(toTensor=True)
    
    def getSingleOrMultipleBatchStructValues(batchStructTemplates, toTensor=False):
        BatchStructTemplate.assertIsBatchStructTemplateOrListOfBatchStructTemplates(batchStructTemplates)
        if isinstance(batchStructTemplates, list):
            res=[]
            for batchStructTemplate in batchStructTemplates:
                res.append(batchStructTemplate.getBatchStructValues(toTensor))
        else:
            res=batchStructTemplates.getBatchStructValues(toTensor)
        return res

    def getSingleOrMultipleBatchStructTensors(batchStructTemplates):
        return BatchStructTemplate.getSingleOrMultipleBatchStructValues(batchStructTemplates, toTensor=True)

    def __repr__(self):
        return str(self.dictStruct)
#%% VAnnTsDataloader
class VAnnTsDataloader(DataLoader):
    #kkk needs tests
    #kkk num_workers>0 problem
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
        batchStructCopy=copy.deepcopy(self.batchStruct)
        BatchStructTemplate.fillSingleOrMultipleWithData(batchStructCopy, defaultCollateRes)
        return BatchStructTemplate.getSingleOrMultipleBatchStructTensors(batchStructCopy)