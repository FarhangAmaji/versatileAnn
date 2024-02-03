from typing import Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from dataPrep.dataCleaning import noNanOrNoneData
from dataPrep.utils import rightPadIfShorter_df, rightPadIfShorter_npArray, \
    rightPadIfShorter_tensor
from utils.customErrors import InternalLogicError
from utils.globalVars import tsStartPointColName
from utils.typeCheck import argValidator
from utils.vAnnGeneralUtils import NpDict, DotDict, tensor_floatDtypeChangeIfNeeded, \
    varPasser
from utils.warnings import Warn


# ---- _TsRowFetcher
class _TsRowFetcher:
    errMsgs = {}
    errMsgs['shorterLen'] = "this output is shorter than requested"
    for it in ['Df', 'Tensor', 'NpDict', 'NpArray']:
        errMsgs[f'non-negStartingPoint{it}'] = f'the starting point for {it} should be non-negative'

    errMsgs = DotDict(errMsgs)

    def __init__(self, *, backcastLen, forecastLen):
        self.castModes = DotDict({key: key for key in
                                  ['backcast', 'forecast', 'fullcast',
                                   'singlePoint']})
        self.backcastLen = backcastLen
        self.forecastLen = forecastLen
        self.indexes = None

    @staticmethod
    def singleFeatureShapeCorrection(data):
        # goodToHave3 find other way to shape single feature, therefore it wont have last dim equal to 1
        if len(data.shape) >= 2 and data.shape[-1] == 1:
            return data.squeeze(-1)
        return data

    @argValidator
    def getRows_df(self, df: Union[pd.DataFrame, pd.Series], idx, *, lowerBoundGap,
                   upperBoundGap, colsOrIndexes, shiftForward=0,
                   canBeOutOfStartIndex=False, canHaveShorterLength=False,
                   rightPadIfShorter=False):

        # note this also works with series
        # qqq does this idx match with getItem of dataset

        self._assertIdx_NShiftInIndexes(idx, shiftForward, canBeOutOfStartIndex)
        # bugPotentialCheck2
        #  in past I had 'if idx + shiftForward not in df.index:' then raise ValueError but found it
        #  not needed and self._assertIdx_NShiftInIndexes
        #  recheck later and if not found any it needed again remove this msg
        if idx + shiftForward < 0:
            raise ValueError(_TsRowFetcher.errMsgs['non-negStartingPointDf'])

        slice_ = slice(idx + lowerBoundGap + shiftForward,
                       idx + upperBoundGap - 1 + shiftForward)

        if colsOrIndexes == '___all___':
            res = df.loc[slice_]
        else:
            res = df.loc[slice_, colsOrIndexes]

        res = self._rightPadShorterIfAllowed(canHaveShorterLength,
                                             rightPadIfShorter,
                                             res, slice_, isItDfLen=True)
        return res

    @argValidator
    def getRows_tensor(self, tensor: torch.Tensor, idx,
                       *, lowerBoundGap, upperBoundGap,
                       colsOrIndexes, shiftForward=0,
                       canBeOutOfStartIndex=False, canHaveShorterLength=False,
                       rightPadIfShorter=False):

        self._assertIdx_NShiftInIndexes(idx, shiftForward, canBeOutOfStartIndex)
        if idx + shiftForward < 0:
            raise ValueError(_TsRowFetcher.errMsgs['non-negStartingPointTensor'])

        slice_ = slice(idx + lowerBoundGap + shiftForward,
                       idx + upperBoundGap + shiftForward)

        if colsOrIndexes == '___all___':
            res = tensor[slice_, :]
        else:
            res = tensor[slice_, colsOrIndexes]

        res = self._rightPadShorterIfAllowed(canHaveShorterLength,
                                             rightPadIfShorter, res, slice_)
        return self.singleFeatureShapeCorrection(res)

    @argValidator
    def getRows_npDict(self, npDict: NpDict, idx,
                       *, lowerBoundGap, upperBoundGap,
                       colsOrIndexes, shiftForward=0,
                       canBeOutOfStartIndex=False, canHaveShorterLength=False,
                       rightPadIfShorter=False):

        self._assertIdx_NShiftInIndexes(idx, shiftForward, canBeOutOfStartIndex)
        if idx + shiftForward < 0:
            raise ValueError(_TsRowFetcher.errMsgs['non-negStartingPointNpDict'])

        slice_ = slice(idx + lowerBoundGap + shiftForward,
                       idx + upperBoundGap + shiftForward)

        if colsOrIndexes == '___all___':
            res = npDict[:][slice_]
        else:
            res = npDict[colsOrIndexes][slice_]

        res = self._rightPadShorterIfAllowed(canHaveShorterLength,
                                             rightPadIfShorter, res, slice_)
        return self.singleFeatureShapeCorrection(res)

    @argValidator
    def getRows_npArray(self, npArray: np.ndarray, idx,
                        *, lowerBoundGap, upperBoundGap,
                        colsOrIndexes, shiftForward=0,
                        canBeOutOfStartIndex=False, canHaveShorterLength=False,
                        rightPadIfShorter=False):

        self._assertIdx_NShiftInIndexes(idx, shiftForward, canBeOutOfStartIndex)
        if idx + shiftForward < 0:
            raise ValueError(_TsRowFetcher.errMsgs['non-negStartingPointNpArray'])

        # cccDevAlgo
        #  for np arrays [-1]results a value so we have to make assertion;
        #  no matter it wont give [-1:1] values,
        #  but then again even in this case it doesnt assert

        slice_ = slice(idx + lowerBoundGap + shiftForward,
                       idx + upperBoundGap + shiftForward)

        if colsOrIndexes == '___all___':
            res = npArray[slice_, :]
        else:
            res = npArray[slice_, colsOrIndexes]

        res = self._rightPadShorterIfAllowed(canHaveShorterLength,
                                             rightPadIfShorter, res, slice_)
        return self.singleFeatureShapeCorrection(res)

    @staticmethod
    def convertToTensor(input_):
        if isinstance(input_, pd.DataFrame):
            input_ = input_.values
        tensor = torch.tensor(input_)
        tensor = tensor_floatDtypeChangeIfNeeded(tensor)
        return tensor

    def getBackForeCastData_general(self, data, idx, mode='backcast',
                                    colsOrIndexes='___all___', shiftForward=0,
                                    outputTensor=True, canBeOutOfStartIndex=False,
                                    canHaveShorterLength=False,
                                    rightPadIfShorter=False):

        # cccDevAlgo
        #  obviously in timeseries data we want to get next rows of data.
        #  for i.e. with data=[1, 2, 3, 4, 5, 6] and with seqLen=4, only 1, 2, 3 can provide data with seqLen with want.
        #  otherwise data will be shorter than seqLen, so having shorter data is banned by default unless you allow
        #  to have shorter data, or rightPadding data.
        #  canBeOutOfStartIndex refers to indexes of VAnnTsDataset

        # goodToHave2
        #  may add query taking ability to df part; plus to castModes,
        #  like the sequence can have upto 10 len or till have reached 'zValueCol <20';
        #  maybe not needed and the query is better used at other places in data preparation or split
        #  - if query is added, these castModes have to be more flexible

        if mode not in self.castModes.keys():
            raise ValueError(
                f"{mode} should be either 'backcast', 'forecast','fullcast' or 'singlePoint'")
        if not (colsOrIndexes == '___all___' or isinstance(colsOrIndexes, list)):
            raise ValueError(
                "u should either pass '___all___' for all feature cols or a list of their columns or indexes")

        self._assertIdx_NShiftInIndexes(idx, shiftForward, canBeOutOfStartIndex)
        # cccAlgo idx+shiftForward also should be in data indexes
        varDicts = varPasser(localArgNames=['data', 'idx', 'mode', 'colsOrIndexes', 'shiftForward',
                                            'canHaveShorterLength', 'rightPadIfShorter'])
        res = self._getBackForeCastData_general_byDataType_NCastMode(**varDicts)

        if outputTensor:
            res = self.convertToTensor(res)
        return res

    @argValidator
    def _getBackForeCastData_general_byDataType_NCastMode(self,
                                                          data: Union[
                                                              pd.DataFrame, np.ndarray, NpDict, torch.Tensor],
                                                          idx,
                                                          mode, colsOrIndexes,
                                                          shiftForward,
                                                          canHaveShorterLength,
                                                          rightPadIfShorter):
        varDicts = varPasser(localArgNames=['data', 'idx', 'mode', 'colsOrIndexes', 'shiftForward',
                                            'canHaveShorterLength', 'rightPadIfShorter'])
        # send to _getCastByMode depending on datatype
        if isinstance(data, NpDict):  # NpDict
            res = self._getCastByMode(self.getRows_npDict, **varDicts)

        elif isinstance(data, (pd.DataFrame, pd.Series)):  # pd.df
            res = self._getCastByMode(self.getRows_df, **varDicts)

        elif isinstance(data, np.ndarray):  # np.array
            res = self._getCastByMode(self.getRows_npArray, **varDicts)

        elif isinstance(data, torch.Tensor):  # tensor
            res = self._getCastByMode(self.getRows_tensor, **varDicts)
        else:
            raise ValueError('to use "getBackForeCastData" data type should be ' + \
                             'pandas.DataFrame or torch.Tensor or np.ndarray or NpDict')
        return res

    def _getCastByMode(self, dataTypeFunc, data, idx,
                       mode, colsOrIndexes,
                       shiftForward, canHaveShorterLength,
                       rightPadIfShorter):
        canBeOutOfStartIndex = True
        # cccDevStruct canBeOutOfStartIndex=True is in order not to check it again
        varDicts = varPasser(
            localArgNames=['idx', 'colsOrIndexes', 'shiftForward', 'canHaveShorterLength',
                           'rightPadIfShorter', 'canBeOutOfStartIndex'])

        if mode == self.castModes.backcast:  # backcast mode
            return dataTypeFunc(data, lowerBoundGap=0,
                                upperBoundGap=self.backcastLen, **varDicts)

        elif mode == self.castModes.forecast:  # forecast mode
            return dataTypeFunc(data, lowerBoundGap=self.backcastLen,
                                upperBoundGap=self.backcastLen + self.forecastLen, **varDicts)

        elif mode == self.castModes.fullcast:  # fullcast mode
            return dataTypeFunc(data, lowerBoundGap=0,
                                upperBoundGap=self.backcastLen + self.forecastLen, **varDicts)

        elif mode == self.castModes.singlePoint:  # singlePoint mode
            return dataTypeFunc(data, lowerBoundGap=0,
                                upperBoundGap=1, **varDicts)
        else:
            raise ValueError("_getCastByMode is only works one of 'backcast', 'forecast', " + \
                             "'fullcast','singlePoint' castModes")

    def _assertIdxInIndexes(self, idx, isAllowed=False):
        if not isAllowed:
            if self.indexes is not None:
                if idx not in self.indexes:
                    raise ValueError(f'{idx} is not in indexes')
                # goodToHave3 this should have been IndexError, but changing it, requires changing some tests

    def _assertIdx_NShiftInIndexes(self, idx, shiftForward, isAllowed):
        self._assertIdxInIndexes(idx, isAllowed)
        self._assertIdxInIndexes(idx + shiftForward, isAllowed)

    def _hasShorterLen(self, len_, slice_, isItDfLen=False):
        normalSliceLen = slice_.stop - slice_.start

        if isItDfLen:
            sliceLen = normalSliceLen + 1
        else:
            sliceLen = normalSliceLen

        if not sliceLen >= len_:
            raise InternalLogicError("_hasShorterLen: internal logic error, " + \
                                     "Length is greater than expected")

        return sliceLen > len_

    def _assertCanHaveShorterLength_dependingOnAllowance(self, isAllowed, len_,
                                                         slice_, isItDfLen=False):
        if not isAllowed:
            if self._hasShorterLen(len_, slice_, isItDfLen=isItDfLen):
                raise ValueError(_TsRowFetcher.errMsgs['shorterLen'])

    @argValidator
    def _rightPadShorterIfAllowed(self, shorterLenAllowance,
                                  rightPadAllowance,
                                  resData: Union[pd.DataFrame, pd.Series, np.ndarray, torch.Tensor],
                                  slice_, pad=0, isItDfLen=False):

        dataLen = len(resData)
        shorterLen = self._hasShorterLen(dataLen, slice_, isItDfLen=isItDfLen)

        sliceLen = slice_.stop - slice_.start
        if shorterLen:
            if rightPadAllowance:

                if isinstance(resData, (pd.DataFrame, pd.Series)):
                    return rightPadIfShorter_df(resData, sliceLen + 1, pad=pad)

                elif isinstance(resData, np.ndarray):
                    return rightPadIfShorter_npArray(resData, sliceLen, pad=pad)

                elif isinstance(resData, torch.Tensor):
                    return rightPadIfShorter_tensor(resData, sliceLen, pad=pad)
                else:
                    raise ValueError('only pd.DataFrame,pd.Series, Np array and tensor are allowed')
            else:
                # cccAlgo
                #  this part returns result which is shorter than regular(depending on back and forecastLens);
                #  ofc if its allowed
                self._assertCanHaveShorterLength_dependingOnAllowance(
                    shorterLenAllowance, dataLen,
                    slice_, isItDfLen=isItDfLen)
                return resData
        else:
            return resData


# ---- VAnnTsDataset
class VAnnTsDataset(Dataset, _TsRowFetcher):
    noIndexesAssertionMsg = "u have to pass indexes unless both backcastLen and forecastLen are 0," + \
                            " or u have passed a pd.df or NpDict with __startPoint__ column"

    # goodToHave1
    #  having data splitted in train,val,test datasets, makes many self.data duplicated with only
    #  tsStartPointColName different for each set. so its not memory efficient.
    #  so to allocate less memory there should some things changed in
    #  splits(splitTsTrainValTest_DfNNpDict and splitTrainValTest_mainGroup) and datasets and
    #  dataloader
    # mustHave2 model should check device, backcastLen, forecastLen with this

    # cccAlgo
    #  VAnnTsDataset provides datachecking with some flexibilities.
    #  its also works with grouped(Nseries) data, to prevent data scrambling between groups data.
    #  ____
    #  types of data allowed:
    #     the data passed is either:
    #         1. NpDict(type of object, wrapped arround pd.dfs to act like df and a dictionary of np.arrays)
    #         2. pd.df
    #             a. with useNpDictForDfs=False
    #             b. with useNpDictForDfs=True, which gonna be converted to NpDict
    #         3. other types if:
    #             a. indexes is passed
    #             b. both of backcastLen and forecastLen are 0
    #     note 1, 2.a and 2.b are called "mainTypes"
    #   ____
    #   in the process of fetching timeSeries sequences '.indexes' is used to determine
    #   whether a point should be involved as starting point of the sequence or not.
    #   this should be provided either with `indexes` passed to dataset
    #   or with having `__startPoint__` in cols of df or NpDict.

    def __init__(self, data, mainGroups=None, indexes=None,
                 *, backcastLen, forecastLen,
                 useNpDictForDfs=True, dataInfo=None, **kwargs):
        Dataset.__init__(self)
        _TsRowFetcher.__init__(self, backcastLen=backcastLen,
                               forecastLen=forecastLen)
        self.dataInfo = dataInfo
        self._setIndexes(data, indexes, useNpDictForDfs,
                         backcastLen, forecastLen)

        self.mainGroups = mainGroups or []
        # cccAlgo these Idxs are explained at _makeMainGroupsIndexes
        self.mainGroupsGeneralIdxs = {}
        self.mainGroupsRelIdxs = {}
        self._assignData_NMainGroupsIdxs(data, mainGroups, useNpDictForDfs)
        # goodToHave3
        #  if splitNSeries is used, could add __hasMainGroups__ to the data,
        #  gets detected here. therefore prevents forgetting to assign mainGroups manually

        self._shapeWarning()
        self._noNanOrNoneDataAssertion()
        # for key, value in kwargs.items():# bugPotentialCheck1 this seems to be wrong
        #     setattr(self, key, value)

    def getBackForeCastData(self, idx, mode='backcast',
                            colsOrIndexes='___all___',
                            *, shiftForward=0, outputTensor=True,
                            canHaveShorterLength=False, rightPadIfShorter=False,
                            canShiftedIndex_BeOutOfStartIndexes=False):

        self._assertIdxInIndexes(idx, False)
        self._assertIdxInIndexes(idx + shiftForward, canShiftedIndex_BeOutOfStartIndexes)
        # note _IdxNdataToLook_WhileFetching works only idx is in indexes
        dataToLook, idx = self._IdxNdataToLook_WhileFetching(idx)

        varDicts = varPasser(
            localArgNames=['idx', 'mode', 'colsOrIndexes', 'shiftForward', 'outputTensor',
                           'canHaveShorterLength', 'rightPadIfShorter',
                           'canShiftedIndex_BeOutOfStartIndexes'],
            exclude=['dataToLook'],
            rename={'canShiftedIndex_BeOutOfStartIndexes': 'canBeOutOfStartIndex'})

        return self.getBackForeCastData_general(dataToLook, **varDicts)

    def __len__(self):
        return len(self.indexes)

    def __getitem__(self, idx):
        # bugPotentialCheck1
        #  these has problem with dataloader commonCollate_fn:
        #           TypeError: default_collate: batch must contain tensors, numpy arrays, numbers,
        #           dicts or lists; found object
        self._assertIdxInIndexes(idx, False)
        if self.mainGroups:
            dataToLook, idx = self._IdxNdataToLook_WhileFetching(idx)
            if isinstance(dataToLook, NpDict):
                return dataToLook[:][idx]

            elif isinstance(dataToLook, pd.DataFrame):
                return dataToLook.loc[idx]
            else:
                raise InternalLogicError('__getitem__: internal logic error')

        # no mainGroups
        if isinstance(self.data, NpDict):
            dataToLook, idx = self._IdxNdataToLook_WhileFetching(idx)
            return dataToLook[:][idx]

        elif isinstance(self.data, pd.DataFrame):
            dataToLook, idx = self._IdxNdataToLook_WhileFetching(idx)
            return dataToLook.loc[idx]

        elif isinstance(self.data, (np.ndarray, torch.Tensor)):
            return self.data[idx]
        else:
            raise ValueError(
                'only datasets with pd.DataFrame, NpDict, np.array and torch.Tensor data can use __getitem__')

    # ---- Private methods
    def _IdxNdataToLook_WhileFetching(self, idx):
        self._assertIdxInIndexes(idx, False)
        # goodToHave3 its was better a cccAlgo was written, to know which if parts handles what situations
        if self.mainGroups:
            groupName = self._findIdxIn_mainGroupsRelIdxs(idx)
            dataToLook = self.data[groupName]

            if isinstance(dataToLook, NpDict):
                relIdx = self.mainGroupsGeneralIdxs[groupName].index(idx)
                relIdx = self.mainGroupsRelIdxs[groupName][relIdx]
                idx = relIdx
        else:
            dataToLook = self.data
        return dataToLook, idx

    # ---- Private methods for __init__
    def _setIndexes(self, data, indexes, useNpDictForDfs,
                    backcastLen, forecastLen):

        # cccAlgo
        #   - indexes serves 3 purposes:
        #   1. showing only allowed indexes to sampler and therefore dataloader
        #   2. indicator to fetch rows from data: so we either need df.index or df.loc
        #   or when the data is NpDict and was originally pd.DataFrame which was converted to NpDict
        #   3. ability to disallow getting data through getBackForeCastData and __getitem__

        # cccAlgo note the NpDict is used by default to speed up data fetching process, because the df.loc is so much slow.

        if indexes is None:
            noBackNForeLenCond = backcastLen == 0 and forecastLen == 0
            dfDataWith_tsStartPointColNameInCols_cond = isinstance(data, pd.DataFrame) and \
                                                        tsStartPointColName in data.columns
            npDictData_tsStartPointColNameInCols_cond = isinstance(data, NpDict) and \
                                                        tsStartPointColName in data.cols()
            dfWithSP = dfDataWith_tsStartPointColNameInCols_cond
            ndWithSP = npDictData_tsStartPointColNameInCols_cond

            if not (noBackNForeLenCond or dfWithSP or ndWithSP):
                raise ValueError(VAnnTsDataset.noIndexesAssertionMsg)

            if noBackNForeLenCond and not (dfWithSP or ndWithSP):
                # cccAlgo this is no npArray or tensors which have not provided indexes but dont want TS(backLen=foreLen=0)
                indexes = [i for i in range(len(data))]

            elif dfWithSP and not useNpDictForDfs:
                indexes = list(data[data[tsStartPointColName] == True].index)
                # cccUsage note indexes are same as df.index
            elif (dfWithSP and useNpDictForDfs) or ndWithSP:
                if isinstance(data, pd.DataFrame):
                    npDict = NpDict(data)
                if isinstance(data, NpDict):
                    npDict = data

                indexes = npDict.__index__[npDict['__startPoint__'] == True]
                indexes = [list(npDict.__index__).index(i) for i in indexes]
                # cccAlgo note the indexes are relative df.indexes. for i.e. if the df.indexes was [130,131,132,...]
                # and 130 and 132 have __startPoint__==True, indexes would be [0,2,...]
            else:
                raise InternalLogicError('_setIndexes: internal logic error')

        self.indexes = list(indexes)

    def _assignMainGroupsIdxs(self, data, mainGroups, useNpDictForDfs):
        if not self.mainGroups:
            raise InternalLogicError('no mainGroups to assign idxs')
        if not (isinstance(data, pd.DataFrame) or isinstance(data, NpDict)):
            raise ValueError('only pd.DataFrame or NpDict can have mainGroups defined')

        self.data = {}
        if isinstance(data, NpDict):
            self._makeMainGroupsIndexes(data, mainGroups,
                                        npDictData=True,
                                        convGroupData_ToNpDict=True)
        elif isinstance(data, pd.DataFrame):
            if useNpDictForDfs:
                self._makeMainGroupsIndexes(data, mainGroups,
                                            npDictData=False,
                                            convGroupData_ToNpDict=True)
            else:
                self._makeMainGroupsIndexes(data, mainGroups,
                                            npDictData=False,
                                            convGroupData_ToNpDict=False)
        else:
            raise InternalLogicError('only pd.DataFrame and NpDicts can have mainGroups defined')

    def _assignData_NMainGroupsIdxs(self, data, mainGroups, useNpDictForDfs):
        if mainGroups:
            self._assignMainGroupsIdxs(data, mainGroups, useNpDictForDfs)
        else:
            if useNpDictForDfs and isinstance(data, pd.DataFrame):
                self.data = NpDict(data)
            else:
                self.data = data

    def _makeMainGroupsIndexes(self, data, mainGroups,
                               npDictData=False, convGroupData_ToNpDict=False):
        # cccAlgo
        #  if the data passed to init
        #     1. is pd.df, mainGroupsGeneralIdxs and mainGroupsRelIdxs are indexes of df.index
        #     2. if pd.df with useNpDictForDfs==True, assume we have df.index==[130,131,...,135],
        #     first 3 in group 'A' and next 3 in group 'B'.
        #     also only [130,132,133,135] have '__startPoint__'==True
        #     for "npDict" and "df with useNpDictForDfs":
        #         self.indexes gonna be:[0,2,3,5]
        #         mainGroupsGeneralIdxs for Group 'B' gonna be:[3,5]
        #         mainGroupsRelIdxs for Group 'B' gonna be:[0,2]
        #         to see how the data at idx==5 is gonna be fetched, take look at _IdxNdataToLook_WhileFetching
        #     for "df with no useNpDictForDfs":
        #         self.indexes gonna be:[130, 132, 133, 135]
        #         mainGroupsGeneralIdxs for Group 'B' gonna be:[133, 135]
        #         mainGroupsRelIdxs for Group 'B' gonna be:[]

        df = data.df if npDictData else data
        for groupName, groupDf in df.groupby(mainGroups):
            if convGroupData_ToNpDict:  # this accounts for npDict and (df with useNpDictForDfs)
                # cccUsage note all indexes like self.indexes or mainGroupsGeneralIdxs are the indexes of df
                self.data[groupName] = NpDict(groupDf)
                generalIdxs = [list(df.index).index(idx) \
                               for idx in groupDf.index]

                self.mainGroupsGeneralIdxs[groupName] = \
                    [idx for idx in generalIdxs if idx in self.indexes]

                self.mainGroupsRelIdxs[groupName] = \
                    [generalIdxs.index(idx) for idx in generalIdxs \
                     if idx in self.mainGroupsGeneralIdxs[groupName]]

            else:  # this accounts for df with useNpDictForDfs=False
                # cccUsage note all indexes like self.indexes or mainGroupsGeneralIdxs are the indexes of df
                self.data[groupName] = groupDf
                self.mainGroupsGeneralIdxs[groupName] = \
                    [idx for idx in groupDf.index \
                     if idx in self.indexes]

                self.mainGroupsRelIdxs[groupName] = []

    def _findIdxIn_mainGroupsRelIdxs(self, idx):
        if not self.mainGroups:
            raise InternalLogicError('dataset doesnt have mainGroups')
        for groupName in self.mainGroupsGeneralIdxs.keys():
            if idx in self.mainGroupsGeneralIdxs[groupName]:
                return groupName
        raise IndexError(f'{idx} is not in any of groups')

    def _noNanOrNoneDataAssertion(self):
        # cccwhy
        #  note self.data can be anything but main types check are tailored to be used are  df, NpDict or a dict of dfs or NpDicts
        if isinstance(self.data, dict):
            for key in self.data.keys():
                if isinstance(self.data[key],
                              (torch.Tensor, NpDict, np.ndarray, pd.DataFrame, pd.Series)):
                    # these types are the ones which can be checked by noNanOrNoneData
                    # and if self.data[key] is not from these types is ignored; as simple dicts
                    # are also allowed for self.data
                    noNanOrNoneData(self.data[key])
        else:
            noNanOrNoneData(self.data)
    def _shapeWarning(self):
        if isinstance(self.data, (torch.Tensor, np.ndarray)):
            shape = self.data.shape
            if shape[0] < shape[1]:
                Warn.warn(
                    "The data shape suggests that different features may be along shape[1]. "
                    "Consider transposing the data to have features along shape[0].")
