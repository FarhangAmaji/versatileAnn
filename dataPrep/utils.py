# ---- imports
import os
from typing import Union

import numpy as np
import pandas as pd
import torch

from dataPrep.utils_innerFuncs import _exclude_mainGroupsWarn, _exclude_singleColWarn, \
    _split_splitNShuffle_startPointIndexes, _splitMakeWarning, _simpleSplit, \
    _makeSetDfWith_TailDataFrom_indexesNTailIndexes, _splitDataPrep, _extend_dfIndexes
from dataPrep.utils_innerFuncs import _splitApplyConditions
from dataPrep.utils_innerFuncs2 import _addNextNPrev_tailIndexes
from utils.globalVars import tsStartPointColName
from utils.typeCheck import argValidator
from utils.vAnnGeneralUtils import NpDict, npArrayBroadCast, regularizeBoolCol, varPasser, \
    pandasGroupbyAlternative


# ---- data split


@argValidator
def splitTsTrainValTest_DfNNpDict(df: Union[pd.DataFrame, NpDict],
                                  trainRatio, valRatio,
                                  seqLen=0, trainSeqLen=None, valSeqLen=None, testSeqLen=None,
                                  shuffle=True, shuffleSeed=None,
                                  conditions=None,
                                  tailIndexes_evenShorter=False,
                                  returnIndexes=False):
    # addTest1
    # goodToHave2 do it also for other datatypes other than df|NpDict
    # cccUsage
    #  - for seq lens pass (backcastLen+ forecastLen)
    #  - conditions:
    #       default (not passing with not having '__startPoint__' in df columns):
    #           there is no need to pass `conditions` and `__startPoint__` in df cols.
    #           '__startPoint__'s(point older in time|beginning and backer in sequence) are
    #           calculated, also split is done efficiently with other options
    #       default (not passing with not having '__startPoint__' in df columns):
    #           '__startPoint__' are acquired by True values in df columns.
    #       passing conditions:
    #           does queries on data and sets.
    #           it is suitable for `more complex splitting using df queries` or
    #           `preserving your starting points`.
    #           note '__startPoint__' is used in other functionalities of this project.
    #           preserve your starting points: as '__startPoint__' is manipulated, you may keep the
    #           `starting points` wanna have untouched in other columns and pass query conditions
    #           indicating `start points` needed for split
    #  - tailIndexes_evenShorter: allows to have shorter sequences
    #  - returnIndexes: by default dfs(or NpDicts of them) are returned but indexes may
    #       returned by this option
    # cccAlgo
    #  tails: points which are not startPoints are named as tails here. note tail for a sequence, if
    #  the starting point is not enough backer in sequence, may has to be shorter.
    #  set: is one of 'train', 'val' and 'test' sets
    conditions, df, dfCopy, npDictUsed, ratios, seqLens, setNames, shuffle = _splitDataPrep(
        conditions, df, seqLen, shuffle, shuffleSeed, testSeqLen, trainRatio, trainSeqLen, valRatio,
        valSeqLen)

    startPointsDf, isAnyConditionApplied = _splitApplyConditions(conditions, dfCopy)
    del dfCopy

    # returns start point indexes, for splitted sets
    setIndexes = _split_splitNShuffle_startPointIndexes(startPointsDf, isAnyConditionApplied,
                                                        ratios,
                                                        seqLens, shuffle, shuffleSeed, setNames)
    if returnIndexes:
        return setIndexes

    setDfs = {}
    for sn in setNames:
        _makeSetDfWith_TailDataFrom_indexesNTailIndexes(df, startPointsDf, seqLens, setDfs,
                                                        setIndexes,
                                                        sn, tailIndexes_evenShorter)
    if npDictUsed:
        setDfs = {sn: NpDict(setDfs[sn]) for sn in setNames}
    _splitMakeWarning(ratios, setDfs, setNames)
    return setDfs


def addNextNPrev_tailIndexes(indexes, seqLenWithSequents=0,
                             seqLenWithAntecedents=0):
    # cccDevStruct due not to get circular import the code is done this way
    return _addNextNPrev_tailIndexes(indexes, seqLenWithAntecedents, seqLenWithSequents)


def simpleSplit(data, ratios, setNames):
    # cccDevStruct due not to get circular import the code is done this way
    return _simpleSplit(data, ratios, setNames)


# ---- mainGroup
# bugPotentialCheck2
#  its doubtful what the difference between mainGroups and NSeries, sometimes they are the same,
#  rename the `_mainGroup` suffixes of funcs below
def splitTrainValTest_mainGroup(df, mainGroups, *, trainRatio, valRatio, seqLen=0,
                                trainSeqLen=None, valSeqLen=None, testSeqLen=None,
                                shuffle=False, shuffleSeed=None, conditions=None,
                                tailIndexes_evenShorter=False):
    if shuffleSeed:
        shuffle = True
    # cccAlgo
    #  ensures that tailIndexes are also from the same mainGroup,
    #  and different mainGroups data dont get mixed up
    # addTest1
    # mustHave2 shuffle should have seed
    grouped = pandasGroupbyAlternative(df, mainGroups)

    groupedDfs = {}
    groupNames = []

    for groupName, groupDf in grouped:
        groupNames += [groupName]
        kwargs = varPasser(localArgNames=['trainRatio', 'valRatio', 'seqLen', 'trainSeqLen',
                                          'valSeqLen', 'testSeqLen', 'shuffle', 'shuffleSeed',
                                          'conditions', 'tailIndexes_evenShorter'])
        groupedDfs[groupName] = splitTsTrainValTest_DfNNpDict(groupDf, returnIndexes=False,
                                                              **kwargs)
    del grouped

    setNames = ['train', 'val', 'test']
    setDfs = {sn: pd.DataFrame() for sn in setNames}
    for groupName in groupNames:
        for sn in ['train', 'val', 'test']:
            setDfs[sn] = pd.concat([setDfs[sn], groupedDfs[groupName][sn]])

    dropInd = lambda df: df.reset_index(drop=True)
    setDfs = {sn: dropInd(setDfs[sn]) for sn in setNames}
    return setDfs


def diffColValuesFromItsMin_mainGroups(df, mainGroups, col, resultColName=None):
    minValues = df.groupby(mainGroups)[col].transform('min')
    df[resultColName] = df[col] - minValues


def setExclusionFlag_seqEnd_mainGroups(df, mainGroups, excludeVal,
                                       col, resultColName):
    uniqueMainGroupMax, _ = _exclude_mainGroupsWarn(col, df, excludeVal, mainGroups,
                                                    resultColName, 'end')
    mask = df[col] <= uniqueMainGroupMax - excludeVal
    df[resultColName] = np.where(mask, True, False)


def setExclusionFlag_seqBeginning_mainGroups(df, mainGroups, excludeVal,
                                             col, resultColName):
    _, uniqueMainGroupMin = _exclude_mainGroupsWarn(col, df, excludeVal, mainGroups,
                                                    resultColName, 'beginning')
    mask = df[col] >= uniqueMainGroupMin + excludeVal
    df[resultColName] = np.where(mask, True, False)


# ---- multi series(NSeries) data
def splitToNSeries(df, pastCols, aggColName):
    if aggColName in df.columns:
        raise ValueError('splitToNSeries: aggColName must not be in df columns')
    processedData = pd.DataFrame({})
    otherCols = [col for col in df.columns if col not in pastCols]
    for i, pc in enumerate(pastCols):
        thisSeriesDf = df[otherCols + [pc]]
        thisSeriesDf = thisSeriesDf.rename(columns={pc: aggColName})
        thisSeriesDf[aggColName + 'Type'] = pc
        processedData = pd.concat([processedData, thisSeriesDf]).reset_index(
            drop=True)
    return processedData


def combineNSeries(df, aggColName, seriesTypes=None):
    # Find unique values in the 'aggColName' column to identify different series
    if seriesTypes is None:
        seriesTypes = df[aggColName + 'Type'].unique()

    combinedData = pd.DataFrame()

    for seriesType in seriesTypes:
        # Filter rows for the current series type
        seriesData = df[df[aggColName + 'Type'] == seriesType].copy()
        seriesData = seriesData.reset_index(drop=True)

        # Rename the columns to the original column name
        seriesData.rename(columns={aggColName: seriesType}, inplace=True)

        # Drop the type and aggColName column
        seriesData.drop(columns=[aggColName + 'Type'], inplace=True)

        colsNotPresentIn = [sc for sc in seriesData.columns if
                            sc not in combinedData.columns]
        # Merge the current series into the combined data
        combinedData = pd.concat([combinedData, seriesData[colsNotPresentIn]],
                                 axis=1)
    return combinedData


def addCorrespondentRow(df, correspondentRowsDf, targets, aggColName,
                        targetMapping=None):
    if targetMapping is None:
        targetMapping = {tr: idx for tr, idx in
                         zip(targets, correspondentRowsDf.index)}

    for target in targets:
        if target in targetMapping:
            target_index = targetMapping[target]
            condition = df[aggColName + 'Type'] == target
            df.loc[condition, correspondentRowsDf.columns] = correspondentRowsDf.iloc[
                target_index].values


# ---- padding
# ----      df & series
# mustHave2 refactor and comment these padding funcs
def rightPadSeriesIfShorter(series, maxLen, pad=0):
    if maxLen <= 0:
        return series
    currentLength = len(series)
    if currentLength > maxLen:
        raise ValueError(f"The series length is greater than {maxLen}: {currentLength}")
    if currentLength < maxLen:
        series = rightPadSeries(series, maxLen - currentLength, pad=pad)
    return series


def rightPadSeries(series, padLen, pad=0):
    if padLen <= 0:
        return series
    padding = pd.Series([pad] * padLen)
    series = pd.concat([series, padding], ignore_index=True)
    return series


@argValidator
def rightPadDfBaseFunc(func, dfOrSeries: Union[pd.DataFrame, pd.Series], padLen, pad=0):
    # cccDevAlgo dont refactor unless its fully tested
    # goodToHave2 do similar for left, and reduce all to another base func
    # goodToHave3 could have added colPad for each col, and if the specificColPad doesnt exist the 'pad'(which default would have used)
    'also works with series'
    if isinstance(dfOrSeries, pd.DataFrame):
        # acquire padded data
        tempDict = {}
        for i, col in enumerate(dfOrSeries.columns):
            if col == tsStartPointColName:
                tempDict[col] = func(dfOrSeries[col], padLen, pad=False)
            else:
                tempDict[col] = func(dfOrSeries[col], padLen, pad=pad)

        # assign padded data to df
        for i, col in enumerate(dfOrSeries.columns):
            if i == 0:
                dfOrSeries, newIndex = _extend_dfIndexes(col, dfOrSeries, tempDict)
            dfOrSeries[col] = pd.Series(tempDict[col].values, index=newIndex)
        return dfOrSeries
    elif isinstance(dfOrSeries, pd.Series):
        return func(dfOrSeries, padLen, pad=pad)


def rightPadIfShorter_df(dfOrSeries, maxLen, pad=0):
    return rightPadDfBaseFunc(rightPadSeriesIfShorter, dfOrSeries,
                              maxLen, pad=pad)


def rightPadDf(dfOrSeries, padLen, pad=0):
    return rightPadDfBaseFunc(rightPadSeries, dfOrSeries, padLen, pad=pad)


# ----      np array
def rightPadNpArrayBaseFunc(arr, padLen, pad=0):
    # cccDevAlgo dont refactor unless its fully tested
    # goodToHave2 do similar for left, and reduce all to another base func
    # goodToHave3 could have added colPad for each col, and if the specificColPad doesnt exist the 'pad'(which default would have used)
    if padLen <= 0:
        return arr
    currentLength = len(arr)
    if currentLength < padLen:
        padding = np.full(padLen - currentLength, pad)
        arrShape = list(arr.shape)
        arrShape[0] = len(padding)
        padding = npArrayBroadCast(padding, arrShape)
        arr = np.concatenate((arr, padding))
    return arr


def rightPadIfShorter_npArray(arr, maxLen, pad=0):
    if maxLen <= 0:
        return arr
    currentLength = len(arr)
    if currentLength > maxLen:
        raise ValueError(f"The array length is greater than {maxLen}: {currentLength}")
    if currentLength < maxLen:
        arr = rightPadNpArrayBaseFunc(arr, maxLen, pad=pad)
    return arr


def rightPadNpArray(arr, padLen, pad=0):
    return rightPadNpArrayBaseFunc(arr, padLen, pad=pad)


# ----      tensor
def rightPadTensorBaseFunc(tensor, padLen, pad=0):
    # goodToHave2 do similar for left, and reduce all to another base func
    # goodToHave3 could have added colPad for each col, and if the specificColPad doesnt exist the 'pad'(which default would have used)
    if padLen <= 0:
        return tensor
    currentLength = tensor.size(0)
    if currentLength < padLen:
        padding = torch.full((padLen - currentLength,) + tensor.shape[1:], pad)
        tensor = torch.cat((tensor, padding), dim=0)
    return tensor


def rightPadIfShorter_tensor(tensor, maxLen, pad=0):
    if maxLen <= 0:
        return tensor
    currentLength = tensor.size(0)
    if currentLength > maxLen:
        raise ValueError(f"The tensor length is greater than {maxLen}: {currentLength}")
    if currentLength < maxLen:
        tensor = rightPadTensorBaseFunc(tensor, maxLen, pad=pad)
    return tensor


def rightPadTensor(tensor, padLen, pad=0):
    return rightPadTensorBaseFunc(tensor, padLen, pad=pad)


# ---- misc
def diffColValuesFromItsMin_singleCol(df, col, resultColName):
    # this is non-mainGroups version of diffColValuesFromItsMin_mainGroups
    df[resultColName] = df[col] - df[col].min()


def setExclusionFlag_seqEnd_singleCol(df, excludeVal, col, resultColName):
    # this is non-mainGroups version of setExclusionFlag_seqEnd_mainGroups
    maxVal, _ = _exclude_singleColWarn(col, df, excludeVal, resultColName, 'end')
    mask = df[col] <= maxVal - excludeVal
    df[resultColName] = np.where(mask, True, False)


def setExclusionFlag_seqBeginning_singleCol(df, excludeVal, col, resultColName):
    # this is non-mainGroups version of setExclusionFlag_seqBeginning_mainGroups
    _, minVal = _exclude_singleColWarn(col, df, excludeVal, resultColName, 'beginning')
    mask = df[col] >= minVal + excludeVal
    df[resultColName] = np.where(mask, True, False)


def regularizeTsStartPoints(df):
    # addTest2 needs tests
    regularizeBoolCol(df, tsStartPointColName)
    nonTsStartPointsFalse(df)


def nonTsStartPointsFalse(df):
    # to make sure all non-True values are turned to False
    nonStartPointCondition = df[tsStartPointColName] != True
    df.loc[nonStartPointCondition, tsStartPointColName] = False

def _applyShuffleIfSeedExists(shuffle, shuffleSeed):
    if shuffleSeed:
        shuffle = True
    return shuffle