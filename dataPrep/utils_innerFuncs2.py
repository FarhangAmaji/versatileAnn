import random

import numpy as np

from utils.vAnnGeneralUtils import NpDict


def _split_indexesNotInvolved(indexes, isAnyConditionApplied, seqLens, setNames):
    setIdxExclusions = {sn: [] for sn in setNames}

    # cccAlgo
    #  passing conditions means the user have considered not surpassing last indexes of indexes
    if not isAnyConditionApplied:

        for sn in setNames:
            # last 'seqLens[sn] - 1' should not be involved in this set
            if -(seqLens[sn] - 1) < 0:  # preventing bugs:like indexes[0:] or indexes[1:]or...
                setIdxExclusions[sn] = indexes[-(seqLens[sn] - 1):]

        elementsNotToInvolved_inAll3Sets = set(
            setIdxExclusions['train']).intersection(
            setIdxExclusions['val'], setIdxExclusions['test'])
        for eni in elementsNotToInvolved_inAll3Sets:
            indexes.remove(eni)
            # also remove from all setIdxExclusions
            for sn in setNames:
                if eni in setIdxExclusions[sn]:
                    setIdxExclusions[sn].remove(eni)
    return setIdxExclusions


def _assignAsMuchAs_IdxsWith1PossibleSet_loop(idxs_possibleSets, indexes, setsIndexes,
                                              setIdxExclusions, expectedSetLens, setNames):
    # cccAlgo
    #  repeatedly assignIdxsWith1PossibleSet then update idxs_possibleSets based on
    #  new doesSetHaveDemand until idxs_possibleSets is stabilized
    idxs_possibleSets_past = idxs_possibleSets.copy()

    notFirstPass = False
    while True:
        if notFirstPass:
            if _idxs_possibleSets_areEqual(idxs_possibleSets, idxs_possibleSets_past):
                return idxs_possibleSets, setsIndexes
            idxs_possibleSets_past = idxs_possibleSets.copy()
        notFirstPass = True

        idxs_possibleSets = _assignIdxsWith1PossibleSet_thenRemoveFrom_idxs_possibleSets(
            idxs_possibleSets, setsIndexes,
            indexes, removeFromIndexesAlso=True)

        # update idxs_possibleSets
        doesSetHaveDemand = _getWhichSetsHaveDemand(expectedSetLens, setNames, setsIndexes)
        idxs_possibleSets = _getIdxs_possibleSets(indexes, setIdxExclusions, setNames,
                                                  doesSetHaveDemand)


def _addSequentAndAntecedentIndexes(indexes, *, seqLenWithAntecedents=0, seqLenWithSequents=0):
    newIndexes = set()
    # Add sequent indexes
    if seqLenWithSequents > 0:
        for num in indexes:
            # adds sequents so the seqLen will be seqLenWithSequents
            newIndexes.update(range(num + 1, num + seqLenWithSequents))
    # Add antecedent indexes
    if seqLenWithAntecedents > 0:
        for num in indexes:
            # adds antecedents so the seqLen will be seqLenWithAntecedents
            newIndexes.update(range(num - seqLenWithAntecedents + 1, num))
    newIndexes.difference_update(indexes)  # Remove existing elements from the newIndexes set
    indexes = np.concatenate((indexes, np.array(list(newIndexes))))
    indexes.sort()
    return indexes


def _assignIdxsWith1PossibleSet_thenRemoveFrom_idxs_possibleSets(idxs_possibleSets, setsIndexes,
                                                                 indexes=None,
                                                                 removeFromIndexesAlso=False):
    idxsAssinged = []
    for idx in idxs_possibleSets.keys():
        # indexes which can only assigned to 1 group
        if len(idxs_possibleSets[idx]) == 1:
            snToBeAssigned = idxs_possibleSets[idx][0]
            setsIndexes[snToBeAssigned].append(idx)
            idxsAssinged.append(idx)
            if removeFromIndexesAlso:
                indexes.remove(idx)
    # remove idxs_possibleSets already assigned
    idxs_possibleSets = {idx: value for idx, value in idxs_possibleSets.items() if
                         idx not in idxsAssinged}
    return idxs_possibleSets


def _getIdxsAssigned_ToGroupDontBelongTo(setIdxExclusions, setNames, setsIndexes):
    idxsToRegroup = []
    for sn in setNames:
        for ini in setIdxExclusions[sn]:
            if ini in setsIndexes[sn]:
                idxsToRegroup.append(ini)
                setsIndexes[sn].remove(ini)
    idxsToRegroup.sort()
    return idxsToRegroup


def _assign_IdxsToRegroup_ToSetWithMaxDemand_orRandom(expectedSetLens, idxsToRegroup_possibleSets,
                                                      setNames, setsIndexes, setIdxExclusions):
    if not idxsToRegroup_possibleSets:
        return
    for intr_gch in idxsToRegroup_possibleSets.keys():
        setsDemand = _getSetsDemand(expectedSetLens, setNames, setsIndexes)

        possibleSets = idxsToRegroup_possibleSets[intr_gch]
        possibleSetsDemand = {sn: setsDemand[sn] for sn in setNames if sn in possibleSets}
        if possibleSetsDemand:
            possibleSetsMaxDemand = max(possibleSetsDemand.values())
            # multiple sets may have max
            setsPossibleWith_maxSetsDemand = [key for key, value in setsDemand.items() if
                                              value == possibleSetsMaxDemand]

            if len(setsPossibleWith_maxSetsDemand) == 1:
                setsIndexes[setsPossibleWith_maxSetsDemand[0]].append(intr_gch)
            else:
                # bugPotentialCheck2
                #  does it need to have seed assigned?!
                random.seed(65)
                randomSn = random.choice(setsPossibleWith_maxSetsDemand)
                setsIndexes[randomSn].append(intr_gch)
        else:
            # cccAlgo
            #  this is where the sets which intr_gch(idx) is not in their setIdxExclusions don't need
            #  more data as they have reached their expectedLen, but in order not let this data point
            #  wasted. we assign it to one of possible sets
            possibleSets = [sn for sn in setNames if intr_gch not in setIdxExclusions[sn]]
            # goodToHave3 could have assigned in a way a bit more fair, like add it to set which has less data
            random.seed(65)
            randomSn = random.choice(possibleSets)
            setsIndexes[randomSn].append(intr_gch)


def _getSetsDemand(expectedSetLens, setNames, setsIndexes, max0=True):
    currentSetLens = _getSetLens(setsIndexes, setNames)
    setsDemand = {sn: expectedSetLens[sn] - currentSetLens[sn] for sn in setNames}
    if max0:
        setsDemand = {sn: max(setsDemand[sn], 0) for sn in setNames}
    return setsDemand


def _getWhichSetsHaveDemand(expectedSetLens, setNames, setsIndexes):
    doesSetHaveDemand = _getSetsDemand(expectedSetLens, setNames, setsIndexes)
    doesSetHaveDemand = {sn: value > 0 for sn, value in doesSetHaveDemand.items()}
    return doesSetHaveDemand


def _idxs_possibleSets_areEqual(idxs_possibleSets1, idxs_possibleSets2):
    if set(idxs_possibleSets1.keys()) != set(idxs_possibleSets2.keys()):
        return False

    for key in idxs_possibleSets1.keys():
        if idxs_possibleSets1[key] != idxs_possibleSets2[key]:
            return False
    return True


def _getSetLens(setsIndexes, setNames):
    return {sn: len(setsIndexes[sn]) for sn in setNames}


def _getIdxs_possibleSets(indexes, setIdxExclusions, setNames, doesSetHaveDemand=None):
    if not doesSetHaveDemand:
        doesSetHaveDemand = {sn: True for sn in setNames}

    idxs_possibleSets = {idx: [] for idx in indexes}
    for idx in indexes:
        for sn in setNames:
            if idx not in setIdxExclusions[sn]:
                if doesSetHaveDemand[sn]:
                    idxs_possibleSets[idx].append(sn)
    return idxs_possibleSets


def _splitLenAssignment(seqLen, seqLens, setNames):
    for sn in setNames:
        seqLens[sn] = seqLens[sn] or seqLen


def _splitNpDictUsed(df):
    npDictUsed = False
    if isinstance(df, NpDict):
        df = df.df
        npDictUsed = True
    return df, npDictUsed


def _sanityCheck(ratios, seqLens, setsIndexes,
                 setIdxExclusions, expectedSumOfLensOfSets, setNames,
                 indexesCopyForLoggingPossibleErrors, shuffle, shuffleSeed):
    # raiseError = False
    raiseErrorTypes = []
    idxsToRegroup = _getIdxsAssigned_ToGroupDontBelongTo(setIdxExclusions, setNames, setsIndexes)
    if idxsToRegroup:
        raiseErrorTypes.append(f'{idxsToRegroup} were assigned to wrong set')

    currentSetLens = _getSetLens(setsIndexes, setNames)
    if sum(currentSetLens.values()) != expectedSumOfLensOfSets:
        raiseErrorTypes.append('all expected points are not distributed')

    for sn in setNames:
        if setsIndexes[sn]:
            if max(setsIndexes[sn]) > max(indexesCopyForLoggingPossibleErrors):
                raiseErrorTypes.append(f'{sn}Indexes exceeding from last index')

    if raiseErrorTypes:
        raiseErrorTypes = '\nalso '.join(raiseErrorTypes)
        # goodToHave3 log indexesCopyForLoggingPossibleErrors, ratios, seqLens, shuffle and shuffleSeed
        raise RuntimeError(f'internal logic error: {raiseErrorTypes}\n' +
                           f'{ratios=}, {seqLens=}, {shuffle=}, {shuffleSeed=}')


def _normalizeDictValues(inputDict):
    sum_ = sum(inputDict.values())
    if sum_ != 0:
        return {key: value / sum_ for key, value in inputDict.items()}
    return {key: 0 for key, value in inputDict.items()}


def _findMaxSetDemand_OfAllIdxs(data):
    maxValue = float('-inf')  # Initialize with negative infinity to find the maximum value
    maxIdx = None
    maxKey = None

    for idx, innerDict in data.items():
        for key, value in innerDict.items():
            if value > maxValue:
                maxValue = value
                maxIdx = idx
                maxKey = key

    return {'value': maxValue, 'idx': maxIdx, 'set': maxKey}


def _updateSetIndexes_WithMaxDemand_ofAllIdxs(expectedSetLens, idxs_possibleSets, indexes,
                                              setNames, setsIndexes, setIdxExclusions):
    while indexes:
        maxSetDemand_OfAllIdxs = _getMaxSetDemand_OfAllIdxs(expectedSetLens, idxs_possibleSets,
                                                            indexes, setNames, setsIndexes)
        setToAssign = maxSetDemand_OfAllIdxs['set']
        # cccAlgo
        #  when maxSetDemand_OfAllIdxs['value'] is 0, set which is not possible for idx may have returned
        #  note this is when idx possibleSets are already filled with their demand
        if maxSetDemand_OfAllIdxs['value'] == 0:
            setToAssign = [sn for sn in setNames if maxSetDemand_OfAllIdxs['idx']
                           not in setIdxExclusions[sn]]
            random.seed(65)
            setToAssign = random.choice(setToAssign)

        setsIndexes[setToAssign].append(maxSetDemand_OfAllIdxs['idx'])
        indexes.remove(maxSetDemand_OfAllIdxs['idx'])
        # del idxs_possibleSets[maxSetDemand_OfAllIdxs['idx']]


def _getMaxSetDemand_OfAllIdxs(expectedSetLens, idxs_possibleSets, indexes, setNames, setsIndexes):
    setsDemandRatios_ofIdx = {}
    setsDemand = _getSetsDemand(expectedSetLens, setNames, setsIndexes)
    for idx in indexes:
        whichSetsHaveDemand = _getWhichSetsHaveDemand(expectedSetLens, setNames, setsIndexes)
        # also checks if the set still has demand
        possibleSets = [s for s in idxs_possibleSets[idx] if whichSetsHaveDemand[s]]
        setsDemandRatios_ofIdx[idx] = {sn: setsDemand[sn] if sn in possibleSets else 0 \
                                       for sn in setNames}
        setsDemandRatios_ofIdx[idx] = _normalizeDictValues(setsDemandRatios_ofIdx[idx])
    maxSetDemand_OfAllIdxs = _findMaxSetDemand_OfAllIdxs(setsDemandRatios_ofIdx)
    return maxSetDemand_OfAllIdxs


def _sortSetIndexes(setNames, setsIndexes):
    for sn in setNames:
        setsIndexes[sn].sort()
