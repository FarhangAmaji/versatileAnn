import copy
import math
from typing import Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Sampler
from torch.utils.data.dataloader import default_collate

from dataPrep.dataset import VAnnTsDataset
from projectUtils.dataTypeUtils.dotDict_npDict import DotDict, NpDict
from projectUtils.dataTypeUtils.list import isListTupleOrSet, isIterable
from projectUtils.dataTypeUtils.tensor import tensor_floatDtypeChangeIfNeeded, getTorchDevice, toDevice
from projectUtils.misc import validate_IsObjOfTypeX_orAListOfTypeX, shuffleData
from projectUtils.typeCheck import argValidator
from projectUtils.warnings import Warn


# ccc1
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
        'int': str(int), 'float': str(float),
        'complex': str(complex),
        'tuple': str(tuple), 'npArray': str(np.ndarray),
        'pdSeries': str(pd.Series),
        'bool': str(bool), 'bytearray': str(bytearray)}),
    # ccc1
    #  even though npArray, tuple, df, series may contain data which may be `str` and cant be converted to tensor
    #  but still we keep them in the format below, so in the case of having str, the error would be raised by pytorch

    'tensor':
        DotDict({'tensor': str(torch.Tensor)}),

    'errorPrones':
        DotDict({'list': str(list)}),  # depending on items ok and not

    'NpDict':  # indirectTensorables
        DotDict({'NpDict': str(NpDict)}),
    # can't directly be changed to tensor

    'df':  # indirectTensorables
        DotDict({'df': str(pd.DataFrame)}),
    # can't directly be changed to tensor

    'notTensorables': DotDict({  # these below can't be changed to tensor
        'set': str(set), 'dict': str(dict),
        'str': str(str),
        'none': str(type(None)), 'bytes': str(bytes),
        'DotDict': str(DotDict)})
})


class TensorStacker:
    def __init__(self):
        self.device = getTorchDevice()

    def stackListToTensor_withDeviceNDtypeMatch(self, list_):
        stackTensor = toDevice(torch.stack(list_), self.device)
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
    # ccc1
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
                # goodToHave2
                #  if had taken prudencyFactor(if this option would added later),
                #  this could have been notTensorables or stackListOfDirectTensorables(it would have
                #  some if conditions to assign type depending on prudencyFactor)
            else:
                self.toTensorFunc = 'notTensorables'

    def __repr__(self):
        return '{' + f'values:{self.values}, type:{self.type}, toTensorFunc:{self.toTensorFunc}' + '}'


class _NestedDictStruct:
    # addTest2 has been tested but no for every single detail
    def __init__(self, wrappedObj, giveFilledStruct=False):
        # goodToHave3
        #  later may make giveFilledStruct=True by default
        # ccc1
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
        self.emptyStruct = self.giveEmptyStruct(wrappedObj)  # can be a dict or _ObjectToBeTensored
        self.struct = copy.deepcopy(self.emptyStruct)
        if giveFilledStruct:
            self.fillWithData(wrappedObj)

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
                # inner parts of dicts if are not a dict, would be wrapped with _ObjectToBeTensored
                returnDict[key] = _ObjectToBeTensored(value)
        return returnDict

    # ---- fill with data
    def fillWithData(self, itemToAdd, path=None):
        # ccc1
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

    @staticmethod
    def fillSingleOrMultiple_WithItemData(nestedDictStructs, itemsToAdd):
        # ccc1
        #  if nestedDictStructs is type of `_NestedDictStruct` is called `single`
        #  and if is `a list of _NestedDictStructs` is called `multiple`
        validate_IsObjOfTypeX_orAListOfTypeX(_NestedDictStruct)(nestedDictStructs)
        if isinstance(nestedDictStructs, list):
            if len(nestedDictStructs) != len(itemsToAdd):
                raise ValueError('nestedDictStructs and itemsToAdd dont have the same length')
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

    def getValuesOfDictTypeStruct(self, dictionary=None, toGpuTensor=False):
        dictionary = dictionary or self.struct
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

    def toList(self):
        # ccc1
        #  Loops through self.struct and adds the values to a list
        result = []

        def recursiveToList(struct, result):
            if isinstance(struct, dict):
                for key, value in struct.items():
                    recursiveToList(value, result)
            else:  # value is _ObjectToBeTensored
                result.append(struct.values[0])
            return result

        return recursiveToList(self.struct, result)

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
            raise ValueError('.struct is not dict or _ObjectToBeTensored')

    def getDataAsGpuTensors_single(self):
        return self.getData_single(toGpuTensor=True)

    @staticmethod
    def getData_singleNMultiple(nestedDictStructs, toGpuTensor=False):
        validate_IsObjOfTypeX_orAListOfTypeX(_NestedDictStruct)(nestedDictStructs)
        if isinstance(nestedDictStructs, list):  # multiple
            res = []
            for nestedDictStruct in nestedDictStructs:
                res.append(nestedDictStruct.getData_single(toGpuTensor))
        else:  # single
            res = nestedDictStructs.getData_single(toGpuTensor)
        return res

    @staticmethod
    def getDataAsGpuTensors_singleNMultiple(nestedDictStructs):
        # bugPotn1(same for fillSingleOrMultiple_WithItemData and getData_singleNMultiple)
        #  in the past(how its tests are written), it hadn't @staticmethod and probably
        #  nestedDictStructs was assumed to be `self` but after adding @staticmethod worked as before
        return _NestedDictStruct.getData_singleNMultiple(
            nestedDictStructs, toGpuTensor=True)

    def __repr__(self):
        return str(self.struct)


# ---- util funcs
@argValidator
def alterAValue_InANestedDictPath(inputDictStyle: Union[dict, DotDict, _NestedDictStruct],
                                  path, value, extendIfPossible=False):
    # ccc1
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
    # ccc1
    #  this is created, because neither default dataloader or vAnnDataloader didnt respect indexes of the vAnnDataset

    @argValidator
    def __init__(self, *, dataset: Union[VAnnTsDataset, None] = None, indexes: list = None,
                 batchSize,
                 shuffle=False, seed=None, shuffleFirst=True):
        # goodToHave2 super().init adds dataSource, which I dont know what it is, so later may add it
        if dataset is not None:
            super().__init__(dataset)
        if seed:
            shuffle = True

        if indexes:
            self.indexes = indexes
        else:
            if dataset is None:
                raise ValueError('either dataset or indexes must be provided.')
            else:
                self.indexes = dataset.indexes
        self.seed = seed
        self.shuffle = shuffle
        self.batchSize = batchSize
        self._shuffleNumerator_initial = 0
        self._shuffleNumerator = self._shuffleNumerator_initial

        if shuffle and shuffleFirst:
            self._shuffleIndexes()

    def __iter__(self):
        if self.shuffle:
            return self._iterShuffleLogic()
        else:
            return iter(self.indexes)

    def _iterShuffleLogic(self):
        if self._shuffleNumerator == self._iterLen:
            self._shuffleNumerator = self._shuffleNumerator_initial  # reset self._shuffleNumerator

            # shuffle indexes
            self._shuffleIndexes()
        self._shuffleNumerator += 1
        return iter(self.indexes)

    def _shuffleIndexes(self):
        self.indexes = shuffleData(self.indexes, self.seed)
        # ccc1
        #  note indexes by getting shuffled result get changed inplace
        #  and there is way back even by making shuffle False
        #  note this is gonna work only when shuffleFirst=False is applied

    def __len__(self):
        return len(self.indexes)

    def changeBatchSize(self, newBatchSize):
        return type(self)(indexes=self.indexes, batchSize=newBatchSize, shuffle=self.shuffle,
                          seed=self.seed, shuffleFirst=False)

    @property
    def shuffle(self):
        return self._shuffle

    @shuffle.setter
    @argValidator
    def shuffle(self, value: bool):
        self._shuffle = value

    @property
    def batchSize(self):
        return self._batchSize

    @batchSize.setter
    @argValidator
    def batchSize(self, value: int):
        if value < 1:
            raise ValueError('batchSize must be positive.')
        if value % 2 != 0:
            Warn.warn('batchSize is not even, which is not recommended.')

        self._batchSize = value
        self._iterLen = math.ceil(len(self.indexes) / value)


# ---- VAnnTsDataloader
class VAnnTsDataloader(DataLoader):
    # addTest1 needs tests
    # mustHave2 num_workers>0 problem?!!? its not stable in `windows os`
    # goodToHave2 can later take modes, 'speed', 'gpuMemory'. for i.e. pin_memory occupies the gpuMemory but speeds up
    # goodToHave3 detect is it from predefined dataset or not
    # goodToHave3 to store the place that dataset and dataloader are defined
    @argValidator
    def __init__(self, dataset: VAnnTsDataset, *, name='', phase='unKnown',
                 batch_size=64, collate_fn=None, sampler=None,
                 createBatchStructEverytime=False, shuffle=False,
                 shuffleFirst=True, randomSeed=None, **kwargs):
        """
        createBatchStructEverytime: on some rare cases the structure of output of nn model may differ.
                for efficiency the code only creates that structure once, but in this case,
                this options enables remaking that structure everytime
        """
        # dataloaderName
        self._setNameNPhase(dataset, name, phase)

        # ccc1
        #  the naming format is in library is camelCase; but the pytorch uses snake_case,
        #  so if the user has passed `batchSize` it would work correctly. also `batchSize` has priority over 'batch_size'
        if 'batchSize' in kwargs.keys():
            batch_size = kwargs['batchSize']
            if batch_size % 2 != 0:
                Warn.warn('batchSize is not even, which is not recommended.')

        # mustHave3 make it compatible to self.device of vAnn
        self.device = getTorchDevice()
        self.createBatchStructEverytime = createBatchStructEverytime
        if collate_fn is None:
            collate_fn = self.commonCollate_fn

        if sampler is None:
            sampler = SamplerFor_vAnnTsDataset(dataset=dataset, shuffle=shuffle, seed=randomSeed,
                                               batchSize=batch_size, shuffleFirst=shuffleFirst)
        else:
            Warn.warn('make sure you have set, VAnnTsDataset.indexes to .indexes of sampler')

        DataLoader.__init__(self, dataset=dataset, batch_size=batch_size,
                            collate_fn=collate_fn, sampler=sampler,
                            shuffle=False, **kwargs)

        # self._initArgs is used in changeBatchSize
        self._initArgs = {'batch_size': batch_size, 'sampler': self.sampler,
                          'collate_fn': collate_fn, 'randomSeed': randomSeed, 'kwargs': kwargs}

    def findBatchStruct(self, batch):
        # ccc1
        #  .batchStruct is a `_NestedDictStruct` object(by convention, called singleType of _NestedDictStruct) or `a list of '_NestedDictStruct's`(multiple type)
        if isListTupleOrSet(batch):
            batchStructOfBatch = []
            for item in batch:
                batchStructOfBatch.append(_NestedDictStruct(item))
            self.batchStruct = batchStructOfBatch
        else:
            self.batchStruct = _NestedDictStruct(batch)

    def commonCollate_fn(self, batch):
        # ccc1
        #  for understanding gpu memory efficiency provided here take a look at devDocs\codeClarifier\gpuMemoryEfficiencyDataloader
        # bugPotn1
        #  dicts can't directly passed to the default_collate
        # bugPotn1
        #  TypeError: default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found object(note this was caused with dtype being object)
        # ccc1
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

    def changeBatchSize(self, newBatchSize):
        initArgs = {key: val for key, val in self._initArgs.items()}
        kwargs = self._initArgs['kwargs']
        del initArgs['kwargs']
        del initArgs['sampler']  # sampler is gonna recreated so we don't pass it

        initArgs['batch_size'] = newBatchSize
        initArgs['name'] = self.name
        initArgs['phase'] = self.phase
        initArgs['shuffle'] = self.shuffle
        initArgs['createBatchStructEverytime'] = self.createBatchStructEverytime
        initArgs['dataset'] = self.dataset
        newInstance = type(self)(**initArgs, **kwargs)
        return newInstance

    def bestNumWorkerFinder(self):
        pass
        # mustHave2
        #  implement it later: there is a code in `devDocs\halfCompleteCode` but because it had sometimes errors while working
        #  with having any num_workers, its not complete; note num_workers for sure is `super unstable` in `windows os`

    def _setNameNPhase(self, dataset, name, phase):
        self.__possiblePhaseNames = ['train', 'val', 'validation', 'test', 'predict', 'unKnown']
        self.name = ''
        if name:
            self.name = name
        else:
            if type(self).__name__ != 'VAnnTsDataloader':
                self.name = type(self).__name__
            else:
                if type(dataset).__name__ != 'VAnnTsDataset':
                    datasetName = type(dataset).__name__
                    if 'dataset' in datasetName:
                        datasetName = datasetName.replace('dataset', 'Dataloader')
                    if 'Dataset' in datasetName:
                        datasetName = datasetName.replace('Dataset', 'Dataloader')
                    self.name = datasetName
        if not self.name:
            raise ValueError('provide a name for Dataloader.')

        self.phase = phase
        self._addPhaseToNameIfItsIsnt()

    @property
    def shuffle(self):
        return self.sampler.shuffle

    @shuffle.setter
    @argValidator
    def shuffle(self, value: bool):
        self.sampler.shuffle = value

    @property
    def phase(self):
        return self._phase

    @phase.setter
    @argValidator
    def phase(self, value: str):
        if value not in self.__possiblePhaseNames:
            raise ValueError(f"phase must be one of these: {', '.join(self.__possiblePhaseNames)}.")

        # remove last phase suffix from self.name
        if hasattr(self, '_phase'):  # in order not to give error for the setting first time
            if self.phase.capitalize() in self.name:
                self.name = self.name.replace(self.phase.capitalize(), '')

        self._phase = value
        # change phase suffix for self.name
        self._addPhaseToNameIfItsIsnt()

    def _addPhaseToNameIfItsIsnt(self):
        if self.phase != 'unKnown':
            if self.phase.capitalize() not in self.name:
                self.name += self.phase.capitalize()
