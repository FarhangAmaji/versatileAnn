import copy
import math
from typing import Union

import torch
from torch.utils.data import DataLoader, Sampler
from torch.utils.data.dataloader import default_collate

from dataPrep.dataset import VAnnTsDataset
from utils.typeCheck import argValidator
from utils.vAnnGeneralUtils import DotDict, isListTupleOrSet, \
    tensor_floatDtypeChangeIfNeeded, isIterable, validate_IsObjOfTypeX_orAListOfTypeX, shuffleData
from utils.warnings import Warn


# cccDevAlgo
#  the code below was designed to serve these purposes:
#   1. making gpu memory usage more efficient:
#       - so instead of common thing which is moving tensors to gpu in dataset, which wastes gpu
#       memory especially dealing with big batchSize or big data in general, this code attempts to:
#           a. detect the structure of 'output of dataset'|or 'created batch'. with emphasizing on
#           the nested dict structure
#           b. detect its tensors or detect which parts of that output, can be tensored if forgotten
#           to be tensored in dataset
#           c. provide ability to move tensors to gpu in complex structures, when wanted
#   2. sampler:
#       a. create a custom sampler to fix, dataloader to respect VAnnTSDataset.indexes
#       b. pass sampler as default sampler
#   3. potentially create other versions of 'pytorch default_collate':
#       'pytorch default_collate' is used to take control of how batch structure should be, and
#       shape the structure of batch built from individual dataset outputs. this code has not
#       implemented this functionality so later if this would be needed, note that u may utilize
#       existing funcs. also may see devDocs/codeClarifier/dataLoader.py to understand how the
#       things work without this module.


# ---- batch structure detection
def isTensorable(obj):
    try:
        torch.tensor(obj)
        return True
    except:
        return False


knownTypesToBeTensored = DotDict({
    'directTensorables': DotDict({
        'int': "<class 'int'>", 'float': "<class 'float'>",
        'complex': "<class 'complex'>",
        'tuple': "<class 'tuple'>", 'npArray': "<class 'numpy.ndarray'>",
        'pdSeries': "<class 'pandas.core.series.Series'>",
        'bool': "<class 'bool'>", 'bytearray': "<class 'bytearray'>"}),
    # cccDevAlgo
    #  even though npArray, tuple, df, series may contain data which may be `str` and cant be converted to tensor
    #  but still we keep them in the format below, so in the case of having str, the error would be raised by pytorch

    'tensor':
        DotDict({'tensor': "<class 'torch.Tensor'>"}),

    'errorPrones':
        DotDict({'list': "<class 'list'>"}),  # depending on items ok and not

    'NpDict':  # indirectTensorables
        DotDict({'NpDict': "<class 'utils.vAnnGeneralUtils.NpDict'>"}),
    # can't directly be changed to tensor

    'df':  # indirectTensorables
        DotDict({'df': "<class 'pandas.core.frame.DataFrame'>"}),
    # can't directly be changed to tensor

    'notTensorables': DotDict({  # these below can't be changed to tensor
        'set': "<class 'set'>", 'dict': "<class 'dict'>",
        'str': "<class 'str'>",
        'none': "<class 'NoneType'>", 'bytes': "<class 'bytes'>",
        'DotDict': "<class 'utils.vAnnGeneralUtils.DotDict'>"})
})


class TensorStacker:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def stackListToTensor_withDeviceNDtypeMatch(self, list_):
        stackTensor = torch.stack(list_).to(self.device)
        stackTensor = tensor_floatDtypeChangeIfNeeded(stackTensor)
        return stackTensor

    def stackListOfErrorPrones(self, listOfErrorPrones):
        try:
            tensorList = [torch.tensor(na) for na in listOfErrorPrones]
        except:
            return listOfErrorPrones
        return self.stackListToTensor_withDeviceNDtypeMatch(tensorList)

    def stackListOfDirectTensorables(self, listOfDirectTensorables):
        tensorList = [torch.tensor(na) for na in listOfDirectTensorables]
        return self.stackListToTensor_withDeviceNDtypeMatch(tensorList)

    def stackListOfDfs(self, listOfDfs):
        listOfNpArrays = [df.values for df in listOfDfs]
        return self.stackListOfDirectTensorables(listOfNpArrays)

    def stackListOfNpDicts(self, listOfNpDicts):
        listOfDfs = [npDict.df for npDict in listOfNpDicts]
        return self.stackListOfDfs(listOfDfs)

    def notTensorables(self, listOfNotTensorables):
        return listOfNotTensorables


class _ObjectToBeTensored(TensorStacker):
    # cccDevAlgo
    #  this is gonna be used in _NestedDictStruct.
    #  this is gonna wrap non dict objects and contains value, type
    #  and "func which can be used to convert that object to tensor"
    #  so later if its possible the object can be converted to tensor and moved to gpu
    def __init__(self, obj):
        super().__init__()
        self.values = []
        self.type = str(type(obj))

        if self.type in knownTypesToBeTensored.tensor.values():  # tensor
            self.toTensorFunc = 'stackListToTensor_withDeviceNDtypeMatch'
        elif self.type in knownTypesToBeTensored.directTensorables.values():  # directTensorables
            # like ('int', 'float', 'complex', 'tuple', 'npArray', 'pdSeries', 'bool', 'bytearray')
            self.toTensorFunc = 'stackListOfDirectTensorables'
        elif self.type in knownTypesToBeTensored.df.values():  # df
            self.toTensorFunc = 'stackListOfDfs'
        elif self.type in knownTypesToBeTensored.NpDict.values():  # NpDict
            self.toTensorFunc = 'stackListOfNpDicts'
        elif self.type in knownTypesToBeTensored.notTensorables.values():  # notTensorables
            # like 'set', 'dict', 'str', 'none', 'bytes', 'DotDict'
            self.toTensorFunc = 'notTensorables'
        else:  # includes knownTypesToBeTensored.errorPrones
            if isTensorable(obj):
                self.toTensorFunc = 'stackListOfErrorPrones'
                # goodToHave2 if had taken prudencyFactor, this could have been notTensorables or stackListOfDirectTensorables
            else:
                self.toTensorFunc = 'notTensorables'

    def __repr__(self):
        return '{' + f'values:{self.values}, type:{self.type}, toTensorFunc:{self.toTensorFunc}' + '}'


class _NestedDictStruct:
    # addTest2 has been tested but no for every single detail
    def __init__(self, wrappedObj):
        # cccAlgo
        #  an object of this class can get wrapped around other objects(wrappedObj)
        #  it differentiates between if wrappedObj is `dict` or from other types of data
        #  for dicts it maps their structure.
        #  non dicts type wrappingObjs or inner objects within dict, would be wrapped with _ObjectToBeTensored
        #  ----
        #  functionalities:
        #   1. creating empty structure as above
        #   2. filling empty structure with data of wrappedObj(when wanted)
        #   3. get data like wrappedObj(as this class was not wrapped around it), but adding ability
        #   to convert wrappedObj to tensors in gpu
        #   ----
        #   so main purpose is to move inner parts of complex structures to gpu, when needed
        self.struct = self.giveEmptyStruct(wrappedObj)  # can be a dict or _ObjectToBeTensored

    def giveEmptyStruct(self, wrappedObj):
        if not isinstance(wrappedObj, dict):
            return _ObjectToBeTensored(wrappedObj)
        returnDict = {}
        # the case of empty dict
        if wrappedObj == {}:
            return _ObjectToBeTensored({})
        # if the dict not empty
        for key, value in wrappedObj.items():
            if isinstance(value, dict):
                returnDict[key] = self.giveEmptyStruct(value)
            else:
                returnDict[key] = _ObjectToBeTensored(
                    value)  # inner parts of dicts if are not a dict, would be wrapped with _ObjectToBeTensored
        return returnDict

    # ---- fill with data
    def fillWithData(self, itemToAdd, path=None):
        # cccAlgo
        #  in giveEmptyStruct the structure has been created but without values and here is filled up with data
        if path is None:
            path = []
        # note this is a recursive func and this condition is for the case if the first itemToAdd is not a dict
        if not isinstance(self.struct, dict) and path == []:
            self.struct.values.append(itemToAdd)
            return
        path = path[:]  # to preserve path from being messed up in recursive calls
        if len(itemToAdd) == 0:  # this is for the case of setting an empty dict
            appendValue_ToNestedDictPath(self, path, {})
        for key, value in itemToAdd.items():
            path2 = path + [key]
            if isinstance(value, dict):
                self.fillWithData(value, path2)
            else:
                appendValue_ToNestedDictPath(self, path2, value)

    def fillSingleOrMultiple_WithItemData(nestedDictStructs, itemsToAdd):
        # cccAlgo
        #  if nestedDictStructs is type of `_NestedDictStruct` is called `single`
        #  and if is `a list of _NestedDictStructs` is called `multiple`
        validate_IsObjOfTypeX_orAListOfTypeX(_NestedDictStruct)(nestedDictStructs)
        if isinstance(nestedDictStructs, list):
            assert len(nestedDictStructs) == len(itemsToAdd), \
                'nestedDictStructs and itemsToAdd dont have the same length'
            for i, nestedDictStruct in enumerate(nestedDictStructs):
                nestedDictStruct.fillWithData(itemsToAdd[i])
        else:
            nestedDictStructs.fillWithData(itemsToAdd)

    # ---- extract(unwrap) data from _NestedDictStruct
    def _squeezeDim0sEqual1(self, tensor):
        # goodToHave2
        #  maybe could have found better but more complex solution; this one may have problems
        if not isinstance(tensor, torch.Tensor):
            return tensor
        while len(tensor.shape) > 1 and tensor.shape[0] == 1:
            tensor = tensor.squeeze(0)
        return tensor

    def getValuesOfDictTypeStruct(self, dictionary, toGpuTensor=False):
        returnDict = {}
        if len(dictionary) == 0:
            return {}
        for key, value in dictionary.items():
            if isinstance(value, dict):
                returnDict[key] = self.getValuesOfDictTypeStruct(value, toGpuTensor=toGpuTensor)
            else:
                if toGpuTensor:
                    toTensorFunc = getattr(value, value.toTensorFunc)
                    returnDict[key] = self._squeezeDim0sEqual1(toTensorFunc(value.values))
                else:
                    returnDict[key] = value.values
        return returnDict

    def getData_single(self, toGpuTensor=False):
        # goodToHave3 it was good if it was possible to take data only as tensor and not solely gpuTensor
        if isinstance(self.struct, _ObjectToBeTensored):
            if toGpuTensor:
                toTensorFunc = getattr(self.struct,
                                       self.struct.toTensorFunc)  # addtest2 because of refactor this part has given error
                return self._squeezeDim0sEqual1(toTensorFunc(self.struct.values))
            return self.struct.values
        elif isinstance(self.struct, dict):
            return self.getValuesOfDictTypeStruct(self.struct, toGpuTensor=toGpuTensor)
        else:
            assert False, '.struct is not dict or _ObjectToBeTensored'

    def getDataAsGpuTensors_single(self):
        return self.getData_single(toGpuTensor=True)

    def getData_singleNMultiple(nestedDictStructs, toGpuTensor=False):
        validate_IsObjOfTypeX_orAListOfTypeX(_NestedDictStruct)(nestedDictStructs)
        if isinstance(nestedDictStructs, list):  # multiple
            res = []
            for nestedDictStruct in nestedDictStructs:
                res.append(nestedDictStruct.getData_single(toGpuTensor))
        else:  # single
            res = nestedDictStructs.getData_single(toGpuTensor)
        return res

    def getDataAsGpuTensors_singleNMultiple(nestedDictStructs):
        return _NestedDictStruct.getData_singleNMultiple(
            nestedDictStructs, toGpuTensor=True)

    def __repr__(self):
        return str(self.struct)


# ---- util funcs
@argValidator
def alterAValue_InANestedDictPath(inputDictStyle: Union[dict, DotDict, _NestedDictStruct],
                                  path, value, extendIfPossible=False):
    # cccAlgo
    #  for i.e. in a nested dict like {'a':{'b':{'c':[]}},'a2'} with path like ['a','b','c'] goes
    #  through the nested path and append/extends the value to that list
    current = inputDictStyle
    if isinstance(current, _NestedDictStruct):
        current = current.struct
    # iterate through the path to access the nested dictionaries
    for i, key in enumerate(path[:-1]):
        # note the path is for nested dicts; so inner paths should be path to a correct key of dict
        assert isinstance(current, (dict, DotDict)), f'{path[:i + 1]} is not a dict or DotDict'
        assert key in current.keys(), f'{key} is not in {path[:i]}'
        current = current[key]
    # same as above, but checking correctness of last path item
    lastKey = path[-1]
    assert lastKey in current.keys(), f'{lastKey} is not in {path}'

    listToUpdate = current[lastKey]
    if isinstance(inputDictStyle, _NestedDictStruct):
        # in the _NestedDictStruct, the list is in .values
        listToUpdate = current[lastKey].values
    assert isinstance(listToUpdate, list), f"{path} doesn't lead to a list"

    if extendIfPossible and isIterable(value):
        listToUpdate.extend(value)
    else:
        listToUpdate.append(value)


@argValidator
def appendValue_ToNestedDictPath(inputDictStyle: Union[dict, DotDict, _NestedDictStruct],
                                 path, value):
    alterAValue_InANestedDictPath(inputDictStyle, path, value, extendIfPossible=False)


@argValidator
def extendValueIfPossible_ToNestedDictPath(
        inputDictStyle: Union[dict, DotDict, _NestedDictStruct],
        path, value):
    alterAValue_InANestedDictPath(inputDictStyle, path, value, extendIfPossible=True)


# ---- SamplerFor_vAnnTsDataset
class SamplerFor_vAnnTsDataset(Sampler):
    # cccDevAlgo
    #  this is created, because neither default dataloader or vAnnDataloader didnt respect indexes of the vAnnDataset

    @argValidator
    def __init__(self, dataset: VAnnTsDataset, batchSize=None, shuffle=False, seed=None):
        # goodToHave2 super().init adds dataSource, which I dont know what it is, so later may add it
        if seed:
            shuffle=True
        if shuffle and not batchSize:
            raise ValueError('batchSize must be passed with shuffle True')
        self.indexes = dataset.indexes
        self._iterLen = None
        if batchSize:
            self._iterLen= math.ceil(len(self.indexes)/batchSize)
        self.shuffle = shuffle
        self.seed = seed
        self._shuffleNumerator = 0

    def __iter__(self):
        if self.shuffle:
            return self._iterShuffleLogic()
        else:
            return iter(self.indexes)

    def _iterShuffleLogic(self):
        assert self._shuffleNumerator < self._iterLen, 'logical error'
        if self._shuffleNumerator == 0:
            self.indexes = shuffleData(self.indexes, self.seed)
            # cccAlgo
            #  note indexes by getting shuffled result get changed inplace
            #  and there is way back even by making shuffle False
        self._shuffleNumerator += 1
        if self._shuffleNumerator == self._iterLen:
            self._shuffleNumerator = 0
        return iter(self.indexes)

    def __len__(self):
        return len(self.indexes)

    @property
    def shuffle(self):
        return self._shuffle

    @shuffle.setter
    @argValidator
    def shuffle(self, value: bool):
        self._shuffle = value


# ---- VAnnTsDataloader
class VAnnTsDataloader(DataLoader):
    # addTest1 needs tests
    # mustHave2 num_workers>0 problem?!!? its not stable in `windows os`
    # goodToHave2 seed everything
    # goodToHave2 can later take modes, 'speed', 'gpuMemory'. for i.e. pin_memory occupies the gpuMemory but speeds up
    @argValidator
    def __init__(self, dataset: VAnnTsDataset, batch_size=64, collate_fn=None, sampler=None,
                 createBatchStructEverytime=False, shuffle=False, randomSeed=None, *args, **kwargs):
        """
        createBatchStructEverytime: on some rare cases the structure of output of nn model may differ.
                for efficiency the code only creates that structure once, but in this case,
                this options enables remaking that structure everytime
        """
        # cccDevAlgo
        #  the naming format is in library is camelCase; but the pytorch uses snake_case,
        #  so if the user has passed `batchSize` it would work correctly. also `batchSize` has priority over 'batch_size'
        if 'batchSize' in kwargs.keys():
            batch_size = kwargs['batchSize']

        # goodToHave1 make it compatible to self.device of vAnn
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.createBatchStructEverytime = createBatchStructEverytime
        if collate_fn is None:
            collate_fn = self.commonCollate_fn

        if sampler is None:
            sampler = SamplerFor_vAnnTsDataset(dataset, shuffle=shuffle, seed=randomSeed, batchSize=batch_size)
        else:
            Warn.warn('make sure you have set, VAnnTsDataset.indexes to .indexes of sampler')
        super().__init__(dataset=dataset, batch_size=batch_size,
                         collate_fn=collate_fn, sampler=sampler,
                         shuffle=False, *args, **kwargs)

    @property
    def shuffle(self):
        return self.sampler.shuffle

    @shuffle.setter
    @argValidator
    def shuffle(self, value: bool):
        self.sampler.shuffle = value

    def bestNumWorkerFinder(self):
        pass
        # mustHave2
        #  implement it later: there is a code in `stash` but because it had sometimes errors while working
        #  with having any num_workers, its not complete; note num_workers for sure is `super unstable` in `windows os`

    def findBatchStruct(self, batch):
        # cccAlgo
        #  .batchStruct is a `_NestedDictStruct` object(by convention, called singleType of _NestedDictStruct) or `a list of '_NestedDictStruct's`(multiple type)
        if isListTupleOrSet(batch):
            batchStructOfBatch = []
            for item in batch:
                batchStructOfBatch.append(_NestedDictStruct(item))
            self.batchStruct = batchStructOfBatch
        else:
            self.batchStruct = _NestedDictStruct(batch)

    def commonCollate_fn(self, batch):
        # mustHave1
        #  I am not sure that, moving tensors in collate_fn to gpu, is the right place for gpu efficiency
        # bugPotentialCheck1
        #  Im not sure how the default_collate is what I want.
        #  from other hand I have used it in commonCollate_fn which VAnnTsDataloader uses by default
        #  therefore double check it on commonDatasetExamples
        # bugPotentialCheck1
        #  dicts can't directly passed to the default_collate
        # bugPotentialCheck1
        #  TypeError: default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found object
        # cccDevAlgo
        #  (how default_collate works):
        #   example1:
        #       converts 32lists of 2tuples which each one is a dict of 5 keys, each key has tensor of shape(7)
        #       -> 2lists which each one is a dict of 5 keys with tensor (32,7)
        #       and output of this func has exactly same structure as defaultCollateRes
        #       and just tries to convert inner data to tensor, which btw probably has been already converted to tensor by default_collate
        #       and move tensors to device(gpu)
        #   example2:
        #       if the output of nn model was for item1={'a':[1,2,3],'b':[88,97,103]} and item2={'a':[6,7,8],'b':[89,98,104]}.
        #       passing them as batch=[item1,item2], to default_collate
        #       the result would be {'a': [tensor([1, 6]), tensor([2, 7]), tensor([3, 8])],
        #                            'b': [tensor([88, 89]), tensor([97, 98]), tensor([103, 104])]}
        defaultCollateRes = default_collate(batch)
        # this is to create the batchStruct once at the beginning and not everytime unless self.createBatchStructEverytime
        if not hasattr(self, 'batchStruct') or self.createBatchStructEverytime:
            self.findBatchStruct(defaultCollateRes)
        batchStructCopy = copy.deepcopy(self.batchStruct)
        _NestedDictStruct.fillSingleOrMultiple_WithItemData(batchStructCopy, defaultCollateRes)
        return _NestedDictStruct.getDataAsGpuTensors_singleNMultiple(batchStructCopy)
