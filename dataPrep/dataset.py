import torch
from torch.utils.data import Dataset
from utils.vAnnGeneralUtils import NpDict, DotDict, Tensor_floatDtypeChange
from dataPrep.utils import rightPadDfIfShorter, rightPadNpArrayIfShorter, rightPadTensorIfShorter
import warnings
import pandas as pd
import numpy as np
from dataPrep.dataCleaning import noNanOrNoneData
from utils.globalVars import tsStartPointColName
#%% TsRowFetcher
class TsRowFetcher:
    errMsgs={}
    #kkk change it to dotdict
    errMsgs['shorterLen']="this output is shorter than requested"
    errMsgs['non-negStartingPointDf']='the starting point is not in df'
    errMsgs['non-negStartingPointTensor']='the starting point for tensor should be non-negative'
    errMsgs['non-negStartingPointNpDict']='the starting point for NpDict should be non-negative'
    errMsgs['non-negStartingPointNpArray']='the starting point for NpArray should be non-negative'
    def __init__(self, backcastLen, forecastLen):
        self.modes=DotDict({key: key for key in ['backcast', 'forecast', 'fullcast','singlePoint']})
        self.backcastLen = backcastLen
        self.forecastLen = forecastLen
        self.indexes = None

    def assertIdxInIndexes(self, idx):
        if not self.indexes is None:
            assert idx in self.indexes,f'{idx} is not in indexes'

    def assertIdxInIndexes_dependingOnAllowance(self, allowance, idx):
        if not allowance:
            self.assertIdxInIndexes(idx)

    def hasShorterLen(self, len_, slice_, isItDfLen=False):
        normalSliceLen=slice_.stop-slice_.start
        if isItDfLen:
            sliceLen=normalSliceLen+1
        else:
            sliceLen=normalSliceLen
        assert sliceLen>=len_,"Length is greater than expected"
        #kkk sliceLen<len_ may not happen, unless internal bug
        if sliceLen>len_:
            return True
        if sliceLen==len_:
            return False

    def assertCanHaveShorterLength_dependingOnAllowance(self, allowance, len_, slice_, isItDfLen=False):
        if not allowance:
            assert not self.hasShorterLen(len_, slice_, isItDfLen=isItDfLen),TsRowFetcher.errMsgs['shorterLen']

    def rightPadShorterIfAllowed(self, shorterLenAllowance, rightPadAllowance, resData, slice_, pad=0, isItDfLen=False):
        dataLen=len(resData)
        shorterLen=self.hasShorterLen(dataLen, slice_, isItDfLen=isItDfLen)
        sliceLen=slice_.stop-slice_.start
        if shorterLen:
            if rightPadAllowance:
                if isinstance(resData, (pd.DataFrame,pd.Series)):
                    return rightPadDfIfShorter(resData, sliceLen+1, pad=pad)
                elif isinstance(resData, np.ndarray):
                    return rightPadNpArrayIfShorter(resData, sliceLen, pad=pad)
                elif isinstance(resData, torch.Tensor):
                    return rightPadTensorIfShorter(resData, sliceLen, pad=pad)
                else:
                    assert False,'only pd.DataFrame,pd.Series, Np array and tensor are allowed'
            else:
                self.assertCanHaveShorterLength_dependingOnAllowance(shorterLenAllowance, dataLen, slice_, isItDfLen=isItDfLen)
                return resData
        else:
            return resData

    def singleFeatureShapeCorrection(self, data):
        if len(data.shape)>=2 and data.shape[-1]==1:
            return data.squeeze(-1)
        return data

    def getDfRows(self, df, idx, lowerBoundGap, upperBoundGap, cols, shiftForward=0,
                  canBeOutOfStartIndex=False, canHaveShorterLength=False, rightPadIfShorter=False):
        #kkk does this idx match with getItem of dataset
        self.assertIdxInIndexes_dependingOnAllowance(canBeOutOfStartIndex, idx)
        self.assertIdxInIndexes_dependingOnAllowance(canBeOutOfStartIndex, idx+shiftForward)
        #kkk does it work with series
        assert '___all___' not in df.columns,'df shouldnt have a column named "___all___", use other manuall methods of obtaining cols'
        #kkk this is not the case and for addressing a col named '___all___', users should have provide it with ['___all___']
        assert idx + shiftForward in df.index, TsRowFetcher.errMsgs['non-negStartingPointDf']
        slice_=slice(idx + lowerBoundGap + shiftForward,idx + upperBoundGap-1 + shiftForward)
        if cols=='___all___':
            res = df.loc[slice_]
        else:
            res = df.loc[slice_,cols]
        res= self.rightPadShorterIfAllowed(canHaveShorterLength, rightPadIfShorter,res, slice_, isItDfLen=True)
        return res

    def getTensorRows(self, tensor, idx, lowerBoundGap, upperBoundGap, colIndexes, shiftForward=0,
                      canBeOutOfStartIndex=False, canHaveShorterLength=False, rightPadIfShorter=False):
        self.assertIdxInIndexes_dependingOnAllowance(canBeOutOfStartIndex, idx)
        self.assertIdxInIndexes_dependingOnAllowance(canBeOutOfStartIndex, idx+shiftForward)
        assert idx + shiftForward >=0, TsRowFetcher.errMsgs['non-negStartingPointTensor']
        slice_=slice(idx + lowerBoundGap + shiftForward,idx + upperBoundGap + shiftForward)
        if colIndexes=='___all___':
            res = tensor[slice_,:]
        else:
            res = tensor[slice_, colIndexes]
        res= self.rightPadShorterIfAllowed(canHaveShorterLength, rightPadIfShorter,res, slice_)
        return self.singleFeatureShapeCorrection(res)

    def getNpDictRows(self, npDict, idx, lowerBoundGap, upperBoundGap, colIndexes, shiftForward=0,
                      canBeOutOfStartIndex=False, canHaveShorterLength=False, rightPadIfShorter=False):
        #kkk may get reduced with using getNpArrayRows
        self.assertIdxInIndexes_dependingOnAllowance(canBeOutOfStartIndex, idx)
        self.assertIdxInIndexes_dependingOnAllowance(canBeOutOfStartIndex, idx+shiftForward)
        assert idx + shiftForward >=0, TsRowFetcher.errMsgs['non-negStartingPointNpDict']
        slice_=slice(idx + lowerBoundGap + shiftForward,idx + upperBoundGap + shiftForward)
        if colIndexes=='___all___':
            res =  npDict[:][slice_]
        else:
            res =  npDict[colIndexes][slice_]
        res= self.rightPadShorterIfAllowed(canHaveShorterLength, rightPadIfShorter,res, slice_)
        return self.singleFeatureShapeCorrection(res)

    def getNpArrayRows(self, npArray, idx, lowerBoundGap, upperBoundGap, colIndexes, shiftForward=0,
                       canBeOutOfStartIndex=False, canHaveShorterLength=False, rightPadIfShorter=False):
        self.assertIdxInIndexes_dependingOnAllowance(canBeOutOfStartIndex, idx)
        self.assertIdxInIndexes_dependingOnAllowance(canBeOutOfStartIndex, idx+shiftForward)
        assert idx + shiftForward >=0, TsRowFetcher.errMsgs['non-negStartingPointNpArray']
        "#ccc for np arrays [-1]results a value so we have to make assertion; no matter it wont give [-1:1] values, but then again even in this case it doesnt assert"
        slice_=slice(idx + lowerBoundGap + shiftForward,idx + upperBoundGap + shiftForward)
        if colIndexes=='___all___':
            res =  npArray[slice_,:]
        else:
            res =  npArray[slice_,colIndexes]
        res= self.rightPadShorterIfAllowed(canHaveShorterLength, rightPadIfShorter,res, slice_)
        return self.singleFeatureShapeCorrection(res)

    def makeTensor(self,input_):
        if isinstance(input_, pd.DataFrame):
            input_=input_.values
        tensor = torch.tensor(input_)
        tensor = Tensor_floatDtypeChange(tensor)
        return tensor

    def getBackForeCastData_general(self, data, idx, mode='backcast', colsOrIndexes='___all___', shiftForward=0, makeTensor=True,
                            canBeOutOfStartIndex=False, canHaveShorterLength=False, rightPadIfShorter=False):
        #kkk may add query taking ability to df part; plus to modes, like the sequence can have upto 10 len or till have reached 'zValueCol <20'; maybe not needed and the query is better used at other places in data prepration or split
        #kkk if query is added, these modes have to be more flexible
        assert mode in self.modes.keys(), "mode should be either 'backcast', 'forecast','fullcast' or 'singlePoint'"
        assert colsOrIndexes=='___all___' or isinstance(colsOrIndexes, list),"u should either pass '___all___' for all feature cols or a list of their columns or indexes"
        self.assertIdxInIndexes_dependingOnAllowance(canBeOutOfStartIndex, idx)
        self.assertIdxInIndexes_dependingOnAllowance(canBeOutOfStartIndex, idx+shiftForward)
        "#ccc idx+shiftForward also should be in data indexes"

        def getCastByMode(typeFunc):
            if mode==self.modes.backcast:
                return typeFunc(data, idx, 0, self.backcastLen, colsOrIndexes, shiftForward, canBeOutOfStartIndex=True,
                                canHaveShorterLength=canHaveShorterLength, rightPadIfShorter=rightPadIfShorter)#ccc canBeOutOfStartIndex=True is in order not to check it again
            elif mode==self.modes.forecast:
                return typeFunc(data, idx, self.backcastLen, self.backcastLen+self.forecastLen, colsOrIndexes, shiftForward,
                                canBeOutOfStartIndex=True, canHaveShorterLength=canHaveShorterLength, rightPadIfShorter=rightPadIfShorter)
            elif mode==self.modes.fullcast:
                return typeFunc(data, idx, 0, self.backcastLen+self.forecastLen, colsOrIndexes, shiftForward,
                                canBeOutOfStartIndex=True, canHaveShorterLength=canHaveShorterLength, rightPadIfShorter=rightPadIfShorter)
            elif mode==self.modes.singlePoint:
                return typeFunc(data, idx, 0, 1, colsOrIndexes, shiftForward, canBeOutOfStartIndex=True,
                                canHaveShorterLength=canHaveShorterLength, rightPadIfShorter=rightPadIfShorter)

        if isinstance(data, NpDict):
            res = getCastByMode(self.getNpDictRows)
        elif isinstance(data, pd.DataFrame):
            res = getCastByMode(self.getDfRows)
        elif isinstance(data, np.ndarray):
            res = getCastByMode(self.getNpArrayRows)
        elif isinstance(data, torch.Tensor):
            res = getCastByMode(self.getTensorRows)
        else:
            assert False, 'to use "getBackForeCastData" data type should be pandas.DataFrame or torch.Tensor or np.ndarray or NpDict'

        if makeTensor:
            res = self.makeTensor(res)
        return res
#%% VAnnTsDataset
class VAnnTsDataset(Dataset, TsRowFetcher):
    noIndexesAssertionMsg = "u have to pass indexes unless both backcastLen and forecastLen are 0," + \
                            " or u have passed a pd df or NpDict with __startPoint__ column"
    #kkk needs tests
    #kkk model should check device, backcastLen, forecastLen with this
    #kkk may take trainIndexes, valIndexes, testIndexes; this way we would have only 1 dataset and less memory occupied
    def __init__(self, data, backcastLen, forecastLen, mainGroups=[], indexes=None, useNpDictForDfs=True, **kwargs):
        Dataset.__init__(self)
        TsRowFetcher.__init__(self, backcastLen=backcastLen, forecastLen=forecastLen)
        """#ccc
        VAnnTsDataset provides datachecking based on allowance. its also works with grouped(Nseries) data,
        to prevent data scrambling between groups data.
        
        types of data allowed:
            the data passed is either:
                1. NpDict(type of object, wrapped arround pd.dfs to act like df and a dictionary of np.arrays)
                2. pd.df
                    a. with useNpDictForDfs=False
                    b. with useNpDictForDfs=True, which gonna be converted to NpDict
                3. other types if:
                    a. indexes is passed
                    b. both of backcastLen and forecastLen are 0
            note 1, 2.a and 2.b are called "mainTypes"

        allowance:
            in timeseries data we want to get next rows of data.
            so (except in rare cases of allowing incomplete sequence lengths, for zeropad if its gonna be incomplete),
            we need to dont allow some point which can't provide the next complete row.
            note allowance comes from the fact the backcast and forecast lens should not be allowed as tsStartingPoints.
            for i.e. if the seq len is 4, with data=[1, 2, 3, 4, 5, 6], only 1,2,3 are allowed.
            
            note the allowance is not determined here, and its should be provides either with `indexes` passed 
            to dataset or with having `__startPoint__` in cols of df or NpDict.
        """
        
        self._setIndexes(data, indexes, mainGroups, useNpDictForDfs, backcastLen, forecastLen)

        #kkk if splitNSeries is used, could add __hasMainGroups__ to the data, gets detected here
        #... therefore prevents forgetting to assign mainGroups manually
        self.mainGroups = mainGroups
        self.mainGroupsGeneralIdxs={}
        self.mainGroupsRelIdxs = {}
        self._assignData_NMainGroupsIdxs(data, mainGroups, useNpDictForDfs)

        self._shapeWarning()
        self._noNanOrNoneDataAssertion()
        for key, value in kwargs.items():
            setattr(self, key, value)

    def getBackForeCastData(self, idx, mode='backcast', colsOrIndexes='___all___',
                            shiftForward=0, makeTensor=True,
                            canHaveShorterLength=False, rightPadIfShorter=False):

        self.assertIdxInIndexes_dependingOnAllowance(canBeOutOfStartIndex, idx)
        self.assertIdxInIndexes_dependingOnAllowance(canBeOutOfStartIndex, idx+shiftForward)

        dataToLook, idx = self._IdxNdataToLook_WhileFetching(idx)

        return self.getBackForeCastData_general(dataToLook, idx=idx, mode=mode,
                           colsOrIndexes=colsOrIndexes, shiftForward=shiftForward,
                           makeTensor=makeTensor,canBeOutOfStartIndex=False,# note _IdxNdataToLook_WhileFetching works only idx is in indexes
                           canHaveShorterLength=canHaveShorterLength, rightPadIfShorter=rightPadIfShorter)

    def __len__(self):
        return len(self.indexes)

    def __getitem__(self, idx):
        self.assertIdxInIndexes(idx)
        if self.mainGroups:
            dataToLook, idx = self._IdxNdataToLook_WhileFetching(idx)
            if isinstance(dataToLook, NpDict):
                return dataToLook[:][idx]
            elif isinstance(dataToLook, pd.DataFrame):
                return dataToLook.loc[idx]
            else:
                assert False, '__getitem__: internal logic error'

        if isinstance(self.data, NpDict):
            dataToLook, idx = self._IdxNdataToLook_WhileFetching(idx)
            return dataToLook[:][idx]
        elif isinstance(self.data, pd.DataFrame):
            dataToLook, idx = self._IdxNdataToLook_WhileFetching(idx)
            return dataToLook.loc[idx]
        elif isinstance(self.data, (np.ndarray, torch.Tensor)):
            return self.data[idx]
        else:
            assert False,'only datasets with pd.DataFrame, NpDict, np.array and torch.Tensor data can use __getitem__'

    #---- Private methods
    def _IdxNdataToLook_WhileFetching(self, idx):
        self.assertIdxInIndexes(idx)
        if self.mainGroups:
            groupName = self._findIdxInmainGroupsRelIdxs(idx)
            dataToLook=self.data[groupName]

            if isinstance(self.data[groupName], NpDict):
                relIdx = self.mainGroupsGeneralIdxs[groupName].index(idx)
                relIdx = self.mainGroupsRelIdxs[groupName][relIdx]
                idx = relIdx
        else:
            dataToLook = self.data
        return dataToLook, idx

    #---- Private methods for __init__
    def _setIndexes(self, data, indexes, mainGroups, useNpDictForDfs, backcastLen, forecastLen):
        """
        (dev)indexes serves 3 purposes:
            1. showing only allowed indexes to sampler and therefore dataloader
            2. indicator to fetch rows from data: so we either need df.index df.loc or 
                when the data is NpDict and was originally pd.DataFrame which was converted to NpDict
            3. abilitiy to disallow getting data through getBackForeCastData and __getitem__

        note the NpDict is used by default to speed up data fetching process, because the df.loc is so much slow.
        """
        if indexes is None:
            noBackNForeLenCond = backcastLen==0 and forecastLen==0
            dfDataWith_tsStartPointColNameInCols = isinstance(data,pd.DataFrame) and tsStartPointColName  in data.columns
            npDictData_tsStartPointColNameInColsCond = isinstance(data, NpDict) and tsStartPointColName  in data.cols() 
            dfTs = dfDataWith_tsStartPointColNameInCols
            ndTs = npDictData_tsStartPointColNameInColsCond

            assert  noBackNForeLenCond or dfTs or ndTs, VAnnTsDataset.noIndexesAssertionMsg


            if noBackNForeLenCond and not (dfTs or ndTs):
                indexes = [i for i in range(len(data))]

            elif dfTs and not useNpDictForDfs:
                indexes = list(data[data[tsStartPointColName]==True].index)
                "#ccc note indexes are same as df.index"
            elif (dfTs and useNpDictForDfs) or ndTs:
                if isinstance(data, pd.DataFrame):
                    npDict = NpDict(data)
                if isinstance(data, NpDict):
                    npDict = data

                indexes = npDict.__index__[npDict['__startPoint__']==True]
                indexes = [list(npDict.__index__).index(i) for i in indexes]
                "#ccc note the indexes are relative df.indexes. for i.e. if the df.indexes was [130,131,132,...]"
                "... and 130 and 132 have __startPoint__==True, indexes would be [0,2,...]"
            else:
                assert False, '_setIndexes: internal logic error'

        self.indexes = list(indexes)

    def _assignData_NMainGroupsIdxs(self, data, mainGroups, useNpDictForDfs):
        if mainGroups:
            assert isinstance(data, pd.DataFrame) or isinstance(data, NpDict), \
                'only pd.DataFrame or NpDict can have mainGroups defined'

            self.data={}
            if isinstance(data, NpDict):
                self._makeMainGroupsIndexes(data, mainGroups, npDictData=True, convGroupData_ToNpDict=True)
            elif isinstance(data, pd.DataFrame):
                if useNpDictForDfs:
                    self._makeMainGroupsIndexes(data, mainGroups, npDictData=False, convGroupData_ToNpDict=True)
                else:
                    self._makeMainGroupsIndexes(data, mainGroups, npDictData=False, convGroupData_ToNpDict=False)
            else:
                assert False, 'only pd.DataFrame and NpDicts can have mainGroups defined'
        else:
            if useNpDictForDfs and isinstance(data,pd.DataFrame):
                self.data = NpDict(data)
            else:
                self.data = data

    def _makeMainGroupsIndexes(self, data, mainGroups, npDictData=False, convGroupData_ToNpDict=False):
        """#ccc (dev)
        if the data passed to init
            1. is pd.df, mainGroupsGeneralIdxs and mainGroupsRelIdxs are indexes of df.index
            2. if pd.df with useNpDictForDfs==True, assume we have df.index==[130,131,...,135],
            first 3 in group 'A' and next 3 in group 'B'.
            also only [130,132,133,135] have '__startPoint__'==True
            self.indexes gonna be:[0,2,3,5]
            mainGroupsGeneralIdxs for Group 'B' gonna be:[3,5]
            mainGroupsRelIdxs for Group 'B' gonna be:[0,2]
            to see data at idx==5 is gonna be reached, take look at _dataNIdxWhileFetching
        """
        df= data.df if npDictData else data
        for groupName, groupDf in df.groupby(mainGroups):
            if convGroupData_ToNpDict:
                self.data[groupName] = NpDict(groupDf)
            else:
                self.data[groupName] = groupDf
            self.mainGroupsGeneralIdxs[groupName] = [list(df.index).index(idx) for idx in groupDf.index]
            self.mainGroupsGeneralIdxs[groupName] = [idx for idx in self.mainGroupsGeneralIdxs[groupName] if idx in self.indexes]
            self.mainGroupsRelIdxs[groupName]=[list(groupDf.index).index(idx) for idx in groupDf.index]

    def _findIdxInmainGroupsRelIdxs(self, idx):
        assert self.mainGroups,'dataset doesnt have mainGroups'
        for groupName in self.mainGroupsGeneralIdxs.keys():
            if idx in self.mainGroupsGeneralIdxs[groupName]:
                return groupName
        raise IndexError(f'{idx} is not in any of groups')

    def _noNanOrNoneDataAssertion(self):
        noNanOrNoneData(self.data)

    def _shapeWarning(self):
        if isinstance(self.data, (torch.Tensor, np.ndarray)):
            shape = self.data.shape
            if shape[0] < shape[1]:
                warnings.warn("The data shape suggests that different features may be along shape[1]. "
                              "Consider transposing the data to have features along shape[0].")