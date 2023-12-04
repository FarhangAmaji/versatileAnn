import numpy as np
import pandas as pd

from dataPrep.utils_innerFuncs2 import _getSetLens, _getIdxs_possibleSets, _getSetsDemand, \
    _getIdxsAssigned_ToGroupDontBelongTo, _assign_IdxsToRegroup_ToSetWithMaxDemand_orRandom, \
    _split_indexesNotInvolved, _assignAsMuchAs_IdxsWith1PossibleSet_loop, _sanityCheck, \
    _sortSetIndexes, _updateSetIndexes_WithMaxDemand_ofAllIdxs, _normalizeDictValues, \
    _splitLenAssignment, _splitNpDictUsed, _addNextNPrev_tailIndexes
from utils.globalVars import tsStartPointColName
from utils.typeCheck import argValidator
from utils.vAnnGeneralUtils import morePreciseFloat
from utils.warnings import Warn

# ----
splitDefaultCondition = f'{tsStartPointColName} == True'


def _convertDateTimeCols(df, dateTimeCols):
    for dc in dateTimeCols:
        df[dc] = pd.to_datetime(df[dc])


def _convertDatetimeNSortCols(df, dateTimeCols, sortCols):
    _convertDateTimeCols(df, dateTimeCols)
    df = df.sort_values(by=sortCols).reset_index(drop=True)


def _exclude_mainGroupsWarn(col, df, excludeVal, mainGroups, resultColName, excludeType):
    uniqueMainGroupMax = df.groupby(mainGroups)[col].transform('max')
    uniqueMainGroupMin = df.groupby(mainGroups)[col].transform('min')
    if (uniqueMainGroupMax - uniqueMainGroupMin).min() <= excludeVal:
        Warn.warn(
            f'by excluding values form the {excludeType}, for some groups there would be no {resultColName} equal to True')
    return uniqueMainGroupMax, uniqueMainGroupMin


def _exclude_singleColWarn(col, df, excludeVal, resultColName, excludeType):
    maxVal = df[col].max()
    minVal = df[col].min()
    if maxVal - minVal <= excludeVal:
        Warn.warn(
            f'by excluding values form the {excludeType}, no {resultColName} equal to True')
    return maxVal, minVal


def _ratiosCheck(ratios):
    mpf = morePreciseFloat
    # ratio > 0 check
    for sn in ['train', 'val']:
        if not (isinstance(ratios[sn], (int, float)) and ratios[sn] >= 0):
            raise ValueError(f'{sn}Ratio must be a float, greater equal than 0')

    trainNValRatios = mpf(ratios['train'] + ratios['val'])
    if trainNValRatios > 1:
        raise ValueError('trainRatio + valRatio must <= 1')

    ratios['test'] = mpf(1 - trainNValRatios)


# splitTsTrainValTest_DfNNpDict inner funcs
def _splitDataPrep(conditions, df, seqLen, shuffle, shuffleSeed, testSeqLen, trainRatio,
                   trainSeqLen, valRatio, valSeqLen):
    df = df.sort_index()
    conditions = conditions or [splitDefaultCondition]
    setNames = ['train', 'val', 'test']
    if shuffleSeed:
        shuffle = True
    ratios = {'train': trainRatio, 'val': valRatio, 'test': None}
    _ratiosCheck(ratios)
    seqLens = {'train': trainSeqLen, 'val': valSeqLen, 'test': testSeqLen}
    _splitLenAssignment(seqLen, seqLens, setNames)
    dfCopy = df.copy()
    dfCopy, npDictUsed = _splitNpDictUsed(dfCopy)
    return conditions, df, dfCopy, npDictUsed, ratios, seqLens, setNames, shuffle


def _splitApplyConditions(conditions, dfCopy):
    filteredDf = dfCopy
    isAnyConditionApplied = False
    doQuery = lambda df, con: (df.query(con), True)
    for condition in conditions:
        # cccAlgo
        #  the splitDefaultCondition is applied when no conditions passed.
        #  if `__startPoint__` is not in df cols, nothing is gonna be applied,
        #  otherwise splitDefaultCondition is applies `__startPoint__`
        if condition == splitDefaultCondition:
            try:
                filteredDf, isAnyConditionApplied = doQuery(filteredDf, condition)
            except:
                pass
        else:
            filteredDf, isAnyConditionApplied = doQuery(filteredDf, condition)
    return filteredDf, isAnyConditionApplied


def _split_splitNShuffle_startPointIndexes(df, isAnyConditionApplied, ratios, seqLens,
                                           shuffle, shuffleSeed, setNames):
    # cccDevAlgo
    #  this is a very complex problem take a look _split_splitNShuffleIndexes_Tests
    # cccAlgo
    #  split is done in a way that, to the most full tails(not shorter), also use all points for
    #  start points. therefore prevents that each set(train/val/test)+its seqLen exceeds from
    #  last Index of set indexes

    # bugPotentialCheck2
    #  multiple indexes are not considered for sorting
    indexes = list(df.sort_index().index)
    indexesCopyForLoggingPossibleErrors = indexes[:]
    setIdxExclusions = _split_indexesNotInvolved(indexes, isAnyConditionApplied,
                                                 seqLens, setNames)
    expectedSumOfLensOfSets = len(indexes)

    expectedSetLens, idxs_possibleSets, setsIndexes = _assignAsMuchAs_IdxsWith1PossibleSet(indexes,
                                                                                           ratios,
                                                                                           setIdxExclusions,
                                                                                           setNames)
    if not len(indexes):
        _sanityCheck(ratios, seqLens, setsIndexes, setIdxExclusions, expectedSumOfLensOfSets,
                     setNames,
                     indexesCopyForLoggingPossibleErrors, shuffle, shuffleSeed)
        _sortSetIndexes(setNames, setsIndexes)
        return setsIndexes
    if shuffle:
        # mustHave2 add compatibility to seed everything
        if shuffleSeed:
            np.random.seed(shuffleSeed)

        np.random.shuffle(indexes)
        _updateSetIndexes_WithMaxDemand_ofAllIdxs(expectedSetLens, idxs_possibleSets, indexes,
                                                  setNames,
                                                  setsIndexes, setIdxExclusions)
    else:
        # we involve ids with 2,3 possibleSets
        setsIndexes = _includeIdxsWith2_3possibleSets(expectedSetLens, indexes, setNames,
                                                      setsIndexes)

        # then try to fix idxs assigned to wrong sets (idxsToRegroup)
        idxsToRegroup_possibleSets, setsIndexes = _assignAsMuchAs_idxsToRegroup_With1PossibleSet(
            expectedSetLens, setIdxExclusions, setNames, setsIndexes)

        _assign_IdxsToRegroup_ToSetWithMaxDemand_orRandom(expectedSetLens,
                                                          idxsToRegroup_possibleSets,
                                                          setNames,
                                                          setsIndexes, setIdxExclusions)
        _sortSetIndexes(setNames, setsIndexes)
    _sanityCheck(ratios, seqLens, setsIndexes, setIdxExclusions, expectedSumOfLensOfSets, setNames,
                 indexesCopyForLoggingPossibleErrors, shuffle, shuffleSeed)
    return setsIndexes


def _assignAsMuchAs_idxsToRegroup_With1PossibleSet(expectedSetLens, setIdxExclusions, setNames,
                                                   setsIndexes):
    idxsToRegroup = _getIdxsAssigned_ToGroupDontBelongTo(setIdxExclusions, setNames,
                                                         setsIndexes)
    idxsToRegroup_possibleSets = _getIdxs_possibleSets(idxsToRegroup, setIdxExclusions,
                                                       setNames)
    # probably no idxs with 1 possible would exist, and this is just to be sure
    idxsToRegroup_possibleSets, setsIndexes = _assignAsMuchAs_IdxsWith1PossibleSet_loop(
        idxsToRegroup_possibleSets, idxsToRegroup,
        setsIndexes, setIdxExclusions,
        expectedSetLens, setNames)
    return idxsToRegroup_possibleSets, setsIndexes


def _includeIdxsWith2_3possibleSets(expectedSetLens, indexes, setNames, setsIndexes):
    setsDemandRatios = _normalizeDictValues(
        _getSetsDemand(expectedSetLens, setNames, setsIndexes))
    # cccAlgo
    #  setsIndexesWith2_3possibleSets (rest remained Idxs) but may(more likely) not gonna follow
    #  setIdxExclusions which needs to be fixed by having idxsToRegroup
    setsIndexesWith2_3possibleSets = _simpleSplit(indexes, setsDemandRatios, setNames)
    setsIndexes = {key: value + setsIndexesWith2_3possibleSets[key] for key, value in
                   setsIndexes.items()}
    return setsIndexes


def _assignAsMuchAs_IdxsWith1PossibleSet(indexes, ratios, setIdxExclusions, setNames):
    setsIndexes = _simpleSplit(indexes, ratios, setNames)  # this is only dummy
    expectedSetLens = _getSetLens(setsIndexes, setNames)
    idxs_possibleSets = _getIdxs_possibleSets(indexes, setIdxExclusions, setNames)
    setsIndexes = {sn: [] for sn in setNames}
    idxs_possibleSets, setsIndexes = _assignAsMuchAs_IdxsWith1PossibleSet_loop(idxs_possibleSets,
                                                                               indexes, setsIndexes,
                                                                               setIdxExclusions,
                                                                               expectedSetLens,
                                                                               setNames)
    return expectedSetLens, idxs_possibleSets, setsIndexes


def _makeSetDfWith_TailDataFrom_indexesNTailIndexes(df, filteredDf, seqLens, setDfs, setIndexes, sn,
                                                    tailIndexesAsPossible):
    setDfs[sn] = filteredDf.loc[setIndexes[sn]]
    setDfs[sn][tsStartPointColName] = True
    sequenceTailIndexes = _addNextNPrev_tailIndexes(setIndexes[sn],
                                                    seqLenWithSequents=seqLens[sn])
    sequenceTailIndexes = [item for item in sequenceTailIndexes if item not in setIndexes[sn]]

    try:
        sequenceTailData = df.loc[sequenceTailIndexes]
    except:
        if tailIndexesAsPossible:
            # cccAlgo
            #  having tailIndexes_evenShorter makes it possible for datasets to fetch sequences
            #  with inEqual length, some being shorter than their expected len
            dfIndexes = df.index
            sequenceTailIndexes = [sti for sti in sequenceTailIndexes if sti in dfIndexes]
            sequenceTailData = df.loc[sequenceTailIndexes]
        else:
            raise IndexError(
                "sequence tails(not start points; the rest of sequence)" +
                " df should have '__startPoint__'==False for last points (closer to end than seqLen)" +
                " or indicated with other query conditions")
    sequenceTailData[tsStartPointColName] = False
    setDfs[sn] = pd.concat([setDfs[sn], sequenceTailData]).sort_index().reset_index(drop=True)


@argValidator
def _simpleSplit(data, ratios: dict, setNames):
    if not all(sn in ratios.keys() for sn in setNames):
        raise ValueError(f"'train', 'val', 'test' must be in ratios")
    mpf = morePreciseFloat
    _ratiosCheck(ratios)
    trainEnd = int(mpf(mpf(ratios['train']) * len(data)))
    valEnd = int(mpf(mpf(ratios['train'] + ratios['val']) * len(data)))
    indexes = {'train': data[:trainEnd], 'val': data[trainEnd:valEnd], 'test': data[valEnd:]}
    return indexes


def _splitMakeWarning(ratios, setDfs, setNames):
    madeWarning = {sn: False for sn in setNames}
    for sn in setNames:
        if morePreciseFloat(ratios[sn]) != 0 and len(setDfs[sn]) == 0:
            madeWarning[sn] = True
    for mwk, mwv in madeWarning.items():
        if mwv:
            Warn.warn(f"{mwk} is empty. the {mwk}SeqLen seems to be high")
        # goodToHave2 make warnings type of vAnnWarning
        # goodToHave3 maybe print warnings in colored background


def _extend_dfIndexes(col, dfOrSeries, tempDict):
    if dfOrSeries.index.dtype in [np.int16, np.int32, np.int64]:
        dfStartInd = dfOrSeries.index.min()
        newIndex = [jj for jj in range(dfStartInd, dfStartInd + len(tempDict[col]))]
        dfOrSeries = dfOrSeries.reindex(newIndex)
    else:
        newIndex = tempDict[col].index
        dfOrSeries = dfOrSeries.reindex(newIndex)
    return dfOrSeries, newIndex
