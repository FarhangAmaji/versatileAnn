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

    def assertIdxInIndexesDependingOnAllowance(self, allowance, idx):
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

    def assertCanHaveShorterLengthDependingOnAllowance(self, allowance, len_, slice_, isItDfLen=False):
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
                self.assertCanHaveShorterLengthDependingOnAllowance(shorterLenAllowance, dataLen, slice_, isItDfLen=isItDfLen)
                return resData
        else:
            return resData

    def singleFeatureShapeCorrection(self, data):
        if len(data.shape)>=2 and data.shape[-1]==1:
            return data.squeeze(-1)
        return data

    def getDfRows(self, df, idx, lowerBoundGap, upperBoundGap, cols, shiftForward=0,
                  canBeOutStartIndex=False, canHaveShorterLength=False, rightPadIfShorter=False):
        #kkk does this idx match with getItem of dataset
        self.assertIdxInIndexesDependingOnAllowance(canBeOutStartIndex, idx)
        self.assertIdxInIndexesDependingOnAllowance(canBeOutStartIndex, idx+shiftForward)
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
                      canBeOutStartIndex=False, canHaveShorterLength=False, rightPadIfShorter=False):
        self.assertIdxInIndexesDependingOnAllowance(canBeOutStartIndex, idx)
        self.assertIdxInIndexesDependingOnAllowance(canBeOutStartIndex, idx+shiftForward)
        assert idx + shiftForward >=0, TsRowFetcher.errMsgs['non-negStartingPointTensor']
        slice_=slice(idx + lowerBoundGap + shiftForward,idx + upperBoundGap + shiftForward)
        if colIndexes=='___all___':
            res = tensor[slice_,:]
        else:
            res = tensor[slice_, colIndexes]
        res= self.rightPadShorterIfAllowed(canHaveShorterLength, rightPadIfShorter,res, slice_)
        return self.singleFeatureShapeCorrection(res)

    def getNpDictRows(self, npDict, idx, lowerBoundGap, upperBoundGap, colIndexes, shiftForward=0,
                      canBeOutStartIndex=False, canHaveShorterLength=False, rightPadIfShorter=False):
        #kkk may get reduced with using getNpArrayRows
        self.assertIdxInIndexesDependingOnAllowance(canBeOutStartIndex, idx)
        self.assertIdxInIndexesDependingOnAllowance(canBeOutStartIndex, idx+shiftForward)
        assert idx + shiftForward >=0, TsRowFetcher.errMsgs['non-negStartingPointNpDict']
        slice_=slice(idx + lowerBoundGap + shiftForward,idx + upperBoundGap + shiftForward)
        if colIndexes=='___all___':
            res =  npDict[:][slice_]
        else:
            res =  npDict[colIndexes][slice_]
        res= self.rightPadShorterIfAllowed(canHaveShorterLength, rightPadIfShorter,res, slice_)
        return self.singleFeatureShapeCorrection(res)

    def getNpArrayRows(self, npArray, idx, lowerBoundGap, upperBoundGap, colIndexes, shiftForward=0,
                       canBeOutStartIndex=False, canHaveShorterLength=False, rightPadIfShorter=False):
        self.assertIdxInIndexesDependingOnAllowance(canBeOutStartIndex, idx)
        self.assertIdxInIndexesDependingOnAllowance(canBeOutStartIndex, idx+shiftForward)
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

    def getBackForeCastDataGeneral(self, data, idx, mode='backcast', colsOrIndexes='___all___', shiftForward=0, makeTensor=True,
                            canBeOutStartIndex=False, canHaveShorterLength=False, rightPadIfShorter=False):
        #kkk may add query taking ability to df part; plus to modes, like the sequence can have upto 10 len or till have reached 'zValueCol <20'; maybe not needed and the query is better used at other places in data prepration or split
        #kkk if query is added, these modes have to be more flexible
        assert mode in self.modes.keys(), "mode should be either 'backcast', 'forecast','fullcast' or 'singlePoint'"
        assert colsOrIndexes=='___all___' or isinstance(colsOrIndexes, list),"u should either pass '___all___' for all feature cols or a list of their columns or indexes"
        self.assertIdxInIndexesDependingOnAllowance(canBeOutStartIndex, idx)
        self.assertIdxInIndexesDependingOnAllowance(canBeOutStartIndex, idx+shiftForward)
        "#ccc idx+shiftForward also should be in data indexes"

        def getCastByMode(typeFunc):
            if mode==self.modes.backcast:
                return typeFunc(data, idx, 0, self.backcastLen, colsOrIndexes, shiftForward, canBeOutStartIndex=True,
                                canHaveShorterLength=canHaveShorterLength, rightPadIfShorter=rightPadIfShorter)#ccc canBeOutStartIndex=True is in order not to check it again
            elif mode==self.modes.forecast:
                return typeFunc(data, idx, self.backcastLen, self.backcastLen+self.forecastLen, colsOrIndexes, shiftForward,
                                canBeOutStartIndex=True, canHaveShorterLength=canHaveShorterLength, rightPadIfShorter=rightPadIfShorter)
            elif mode==self.modes.fullcast:
                return typeFunc(data, idx, 0, self.backcastLen+self.forecastLen, colsOrIndexes, shiftForward,
                                canBeOutStartIndex=True, canHaveShorterLength=canHaveShorterLength, rightPadIfShorter=rightPadIfShorter)
            elif mode==self.modes.singlePoint:
                return typeFunc(data, idx, 0, 1, colsOrIndexes, shiftForward, canBeOutStartIndex=True,
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
    noIndexesAssertionMsg="u have to pass indexes unless both backcastLen and forecastLen are 0, or u have passed a pd df or NpDict with __startPoint__ column"
    #kkk needs tests
    #kkk model should check device, backcastLen, forecastLen with this
    #kkk may take trainIndexes, valIndexes, testIndexes; this way we would have only 1 dataset and less memory occupied
    def __init__(self, data, backcastLen, forecastLen, mainGroups=[], indexes=None, useNpDictForDfs=True, **kwargs):
        Dataset.__init__(self)
        TsRowFetcher.__init__(self, backcastLen=backcastLen, forecastLen=forecastLen)
        self.usedDfToNpInds = False
        if indexes is None:
            noBackNForeLenCond = backcastLen==0 and forecastLen==0
            dfDataWith_tsStartPointColNameInCols = isinstance(data,pd.DataFrame) and tsStartPointColName  in data.columns
            npDictData_tsStartPointColNameInColsCond = isinstance(data, NpDict) and tsStartPointColName  in data.cols() 
        
            assert noBackNForeLenCond or dfDataWith_tsStartPointColNameInCols \
                or npDictData_tsStartPointColNameInColsCond, VAnnTsDataset.noIndexesAssertionMsg
        
            if  dfDataWith_tsStartPointColNameInCols:
                indexes = data[data[tsStartPointColName]==True].index
                "#ccc note indexes has kept their values"
                if useNpDictForDfs:
                    self.usedDfToNpInds = True
        
            elif npDictData_tsStartPointColNameInColsCond:
                indexes=data.__index__[data['__startPoint__']==True]
                indexes=[i for i in range(indexes)]
                "#ccc note indexes for NpDict are according to their order"
        self.indexes = list(indexes)

        #kkk if splitNSeries is used, could add __hasMainGroups__ to the data, gets detected here
        #... therefore prevents forgetting to assign mainGroups manually
        self.mainGroups = mainGroups
        self.mainGroupsIndexes = {}
        self.assignData(data, mainGroups, useNpDictForDfs)

        self.shapeWarning()
        self.noNanOrNoneDataAssertion()
        for key, value in kwargs.items():
            setattr(self, key, value)

    def assignData(self, data, mainGroups, useNpDictForDfs):
        if mainGroups:
            assert isinstance(data, pd.DataFrame) or isinstance(data, NpDict), \
                'only pd.DataFrame or NpDict can have mainGroups defined'

            self.data={}
            if isinstance(data, NpDict):
                self.doMainGroup_dataIndexes(data.df, mainGroups, convGroupData_ToNpDict=True)
            elif isinstance(data, pd.DataFrame):
                if useNpDictForDfs:
                    self.doMainGroup_dataIndexes(data, mainGroups, convGroupData_ToNpDict=True)
                    self.usedDfToNpInds = True
                else:
                    self.doMainGroup_dataIndexes(data, mainGroups)
            else:
                assert False,'only pd.DataFrame and NpDicts can have mainGroups defined'
        else:
            if useNpDictForDfs and isinstance(data,pd.DataFrame):
                self.data = NpDict(data)
                self.usedDfToNpInds = True
            else:
                self.data = data

    def doMainGroup_dataIndexes(self, df, mainGroups, convGroupData_ToNpDict=False):
        for groupName, groupDf in df.groupby(mainGroups):
            if convGroupData_ToNpDict:
                self.data[groupName]=NpDict(groupDf)
            else:
                self.data[groupName]=groupDf
            self.mainGroupsIndexes[groupName]=[list(groupDf.index)]

    def findIdxInMainGroupsIndexes(self, idx):
        assert self.mainGroups,'dataset doesnt have mainGroups'
        for groupName in self.mainGroupsIndexes.keys():
            if idx in self.mainGroupsIndexes[groupName]:
                return groupName
        raise IndexError(f'{idx} is not in any of groups')

    def noNanOrNoneDataAssertion(self):
        noNanOrNoneData(self.data)
    
    def shapeWarning(self):
        if isinstance(self.data, (torch.Tensor, np.ndarray)):
            shape = self.data.shape
            if shape[0] < shape[1]:
                warnings.warn("The data shape suggests that different features may be along shape[1]. "
                              "Consider transposing the data to have features along shape[0].")


    def getBackForeCastData(self, idx, mode='backcast', colsOrIndexes='___all___', shiftForward=0, makeTensor=True,
                            canBeOutStartIndex=False, canHaveShorterLength=False, rightPadIfShorter=False):

        if self.mainGroups:
            groupName, relIdx = self.findIdxInMainGroupsIndexes(idx)
            dataToSendTo_getBackForeCastDataGeneral=self.data[groupName]

            if self.usedDfToNpInds:
                relIdx=self.mainGroupsIndexes[groupName].index(idx)
                idx=relIdx
        else:
            dataToSendTo_getBackForeCastDataGeneral=self.data
            if self.usedDfToNpInds:
                relIdx=self.indexes.index(idx)
                idx=relIdx

        return self.getBackForeCastDataGeneral(dataToSendTo_getBackForeCastDataGeneral,
                           idx=idx, mode=mode, colsOrIndexes=colsOrIndexes, shiftForward=shiftForward,
                           makeTensor=makeTensor,canBeOutStartIndex=canBeOutStartIndex,
                           canHaveShorterLength=canHaveShorterLength, rightPadIfShorter=rightPadIfShorter)

    def __len__(self):
        if self.indexes is None:
            return len(self.data)
        return len(self.indexes)

    def __getitem__(self, idx):
        self.assertIdxInIndexes(idx)
        if self.mainGroups:
            groupName, relIdx=self.findIdxInMainGroupsIndexes(idx)
            if isinstance(self.data[groupName], NpDict):
                return self.data[groupName][:][relIdx]
            return self.data[groupName].loc[idx]
        else:
            if isinstance(self.data, (pd.DataFrame, pd.Series)):
                return self.data.loc[idx]
            elif isinstance(self.data, NpDict):
                return self.data[:][idx]
            elif isinstance(self.data, (np.ndarray, torch.Tensor)):
                return self.data[idx]