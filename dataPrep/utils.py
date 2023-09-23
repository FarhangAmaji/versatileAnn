#%% imports
import os
import torch
import pandas as pd
import numpy as np
from utils.globalVars import tsStartPointColName
from utils.vAnnGeneralUtils import NpDict, npArrayBroadCast, regularizeBoolCol
from utils.vAnnGeneralUtils import morePreciseFloat as mpf
import warnings
#%%
splitDefaultCondition=f'{tsStartPointColName} == True'
#%% datasets
datasetsRelativePath=r'..\data\datasets'
knownDatasetsDateTimeCols={"EPF_FR_BE.csv":{'dateTimeCols':["dateTime"],'sortCols':['dateTime']},
                           "stallion.csv":{'dateTimeCols':["date"],'sortCols':["agency", "sku"]},
                           "electricity.csv":{'dateTimeCols':["date"],'sortCols':['consumerId','hoursFromStart']}}

def sortDfByCols(df, sortCols):
    df=df.sort_values(by=sortCols).reset_index(drop=True)

def convertDateTimeCols(df, dateTimeCols):
    for dc in dateTimeCols:
        df[dc] = pd.to_datetime(df[dc])

def convertDatetimeNSortCols(df, dateTimeCols, sortCols):
    convertDateTimeCols(df, dateTimeCols)
    sortDfByCols(df, sortCols)

def getDatasetFiles(fileName: str, dateTimeCols=[],sortCols=[]):
    currentDir = os.path.dirname(os.path.abspath(__file__))
    filePath = os.path.normpath(os.path.join(currentDir, datasetsRelativePath, fileName))
    df=pd.read_csv(filePath)
    if fileName in knownDatasetsDateTimeCols.keys():
        dataset=knownDatasetsDateTimeCols[fileName]
        convertDatetimeNSortCols(df, dataset['dateTimeCols'],dataset['sortCols'])
    else:
        convertDatetimeNSortCols(df, dateTimeCols, sortCols)
    return df
#%% multi series(NSeries) data
def addCorrespondentRow(df, correspondentRowsDf, targets, summedColName, targetMapping={}):
    if targetMapping=={}:
        targetMapping = {tr:idx for tr,idx in zip(targets, correspondentRowsDf.index)}

    for target in targets:
        if target in targetMapping:
            target_index = targetMapping[target]
            condition = df[summedColName+'Type'] == target
            df.loc[condition, correspondentRowsDf.columns] = correspondentRowsDf.iloc[target_index].values

def splitToNSeries(df, pastCols, summedColName):
    assert summedColName not in df.columns,'splitToNSeries: summedColName must not be in df columns'
    processedData=pd.DataFrame({})
    otherCols= [col for col in df.columns if col not in pastCols]
    for i,pc in enumerate(pastCols):
        thisSeriesDf=df[otherCols+[pc]]
        thisSeriesDf=thisSeriesDf.rename(columns={pc:summedColName})
        thisSeriesDf[summedColName+'Type']=pc
        processedData = pd.concat([processedData,thisSeriesDf]).reset_index(drop=True)
    return processedData

def combineNSeries(df, summedColName, seriesTypes=None):
    # Find unique values in the 'summedColName' column to identify different series
    if seriesTypes is None:
        seriesTypes = df[summedColName + 'Type'].unique()
    
    combinedData = pd.DataFrame()
    
    for seriesType in seriesTypes:
        # Filter rows for the current series type
        seriesData = df[df[summedColName + 'Type'] == seriesType].copy()
        seriesData=seriesData.reset_index(drop=True)
        
        # Rename the columns to the original column name
        seriesData.rename(columns={summedColName: seriesType}, inplace=True)
        
        # Drop the type and summedColName column
        seriesData.drop(columns=[summedColName + 'Type'], inplace=True)

        colsNotPresentIn=[sc for sc in seriesData.columns if sc not in combinedData.columns]
        # Merge the current series into the combined data
        combinedData = pd.concat([combinedData, seriesData[colsNotPresentIn]], axis=1)
    return combinedData

def splitTrainValTest_NSeries(df, mainGroups, trainRatio, valRatio, seqLen=0,
                      trainSeqLen=None, valSeqLen=None, testSeqLen=None,
                      shuffle=True, conditions=[splitDefaultCondition], tailIndexesAsPossible=False):
    "#ccc this ensures that tailIndexes are also from this group of NSeries"
    grouped = df.groupby(mainGroups)

    groupedDfs = {}
    groupNames=[]
    
    for groupName, groupDf in grouped:
        groupNames+=[groupName]
        groupedDfs[groupName] = splitTsTrainValTest_DfNNpDict(groupDf, trainRatio=trainRatio, valRatio=valRatio,
                             seqLen=seqLen, trainSeqLen=trainSeqLen, valSeqLen=valSeqLen, testSeqLen=testSeqLen,
                              shuffle=shuffle, conditions=conditions, tailIndexesAsPossible=tailIndexesAsPossible,
                              giveStartPointsIndexes=False)
    del grouped

    trainDf, valDf, testDf=pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    #these loops could have been in 1 loop but seems this way the memory management is easier
    for groupName in groupNames:
        trainDf = pd.concat([trainDf, groupedDfs[groupName][0]])

    for groupName in groupNames:
        valDf = pd.concat([valDf, groupedDfs[groupName][1]])

    for groupName in groupNames:
        testDf = pd.concat([testDf, groupedDfs[groupName][2]])

    dropInd= lambda df: df.reset_index(drop=True)
    return dropInd(trainDf), dropInd(valDf), dropInd(testDf)

def calculateNSeriesMinDifference(df, mainGroups, valueCol, resultCol):
    minValues = df.groupby(mainGroups)[valueCol].transform('min')
    df[resultCol] = df[valueCol] - minValues

def excludeValuesFromEnd_NSeries(df, mainGroups, excludeVal, valueCol, resultCol):
    uniqueMainGroupMax = df.groupby(mainGroups)[valueCol].transform('max')
    uniqueMainGroupMin = df.groupby(mainGroups)[valueCol].transform('min')
    if (uniqueMainGroupMax-uniqueMainGroupMin).min()<=excludeVal:
        warnings.warn(f'by excluding values form the end, for some groups there would be no {resultCol} equal to True')
    df.loc[df[valueCol] <= uniqueMainGroupMax - excludeVal, resultCol] = True
    df.loc[df[resultCol] != True, resultCol] = False

def excludeValuesFromBeginning_NSeries(df, mainGroups, excludeVal, valueCol, resultCol):
    uniqueMainGroupMax = df.groupby(mainGroups)[valueCol].transform('max')
    uniqueMainGroupMin = df.groupby(mainGroups)[valueCol].transform('min')
    if (uniqueMainGroupMax-uniqueMainGroupMin).min()<=excludeVal:
        warnings.warn(f'by excluding values form the beginning, for some groups there would be no {resultCol} equal to True')
    df.loc[df[valueCol] >= uniqueMainGroupMin + excludeVal, resultCol] = True
    df.loc[df[resultCol] != True, resultCol] = False
#%% data split
def addSequentAndAntecedentIndexes(indexes, seqLenWithSequents=0, seqLenWithAntecedents=0):
    newIndexes = set()
    
    # Add sequent indexes
    if seqLenWithSequents>0:
        for num in indexes:
            newIndexes.update(range(num + 1, num + seqLenWithSequents))# adds sequents so the seqLen will be seqLenWithSequents
    
    # Add antecedent indexes
    if seqLenWithAntecedents>0:
        for num in indexes:
            newIndexes.update(range(num - seqLenWithAntecedents+ 1, num))# adds antecedents so the seqLen will be seqLenWithAntecedents
    
    newIndexes.difference_update(indexes)  # Remove existing elements from the newIndexes set
    indexes = np.concatenate((indexes, np.array(list(newIndexes))))
    indexes.sort()
    return indexes

def ratiosCheck(trainRatio, valRatio):
    trainRatio, valRatio=mpf(trainRatio), mpf(valRatio)
    testRatio=mpf(1-trainRatio-valRatio)
    assert mpf(sum([trainRatio, valRatio, testRatio]))==1, 'sum of train, val and test ratios must be 1'
    return trainRatio, valRatio, testRatio

def simpleSplit(data, trainRatio, valRatio):
    trainRatio, valRatio, testRatio=ratiosCheck(trainRatio, valRatio)
    trainEnd=int(mpf(mpf(trainRatio) * len(data)))
    valEnd=int(mpf(mpf(trainRatio + valRatio) * len(data)))
    train=data[:trainEnd]
    val=data[trainEnd:valEnd]
    test=data[valEnd:]
    return train, val, test

def regularizeTsStartPoints(df):
    #kkk needs tests
    regularizeBoolCol(df, tsStartPointColName)
    nonTsStartPointsFalse(df)

def nonTsStartPointsFalse(df):
    "to make sure all non-True values are turned to False"
    nonStartPointCondition =  df[tsStartPointColName]!=True
    df.loc[nonStartPointCondition, tsStartPointColName]=False

def subtractFromIndexes(indexes, trainRatio, valRatio, trainSeqLen, valSeqLen, testSeqLen, isAnyConditionApplied):
    "#ccc this is to prevent that each set(train/val/test)+its seqLen exceeds from last Index of indexes"
    #kkk this implementation may have some utilized points
    #ccc the problem of utilizing all points and keeping the order so train uses first items 
    #... and val and test have next items is impossible for some case due to respondant seqLens
    #... but without keeping the order, there may be some more complex algos to utilize all of the points and keep the set indexes together
    if not isAnyConditionApplied:
        maxLen=len(indexes)
        for ratio, sl in zip([trainRatio, trainRatio + valRatio, 1],
                              [trainSeqLen, valSeqLen           , testSeqLen]):
          maxLen = min((len(indexes)+1-sl)/ratio, maxLen)
        indexes = indexes[:int(mpf(maxLen))]

    return indexes

def splitTsTrainValTest_DfNNpDict(df, trainRatio, valRatio, seqLen=0,
                      trainSeqLen=None, valSeqLen=None, testSeqLen=None,
                      shuffle=True, conditions=[splitDefaultCondition], tailIndexesAsPossible=False, giveStartPointsIndexes=False):
    #kkk needs tests
    #kkk do it also for other datatypes
    """
    - for seq lens pass (backcastLen+ forecastLen)
    - note this func expects conditions which indicate the first point(older in time|backer in sequence), beginning of backcast point;
            this can be done with having '__startPoint__' with True values or other query conditions
    - note if u want to preserve u starting points use another column than '__startPoint__'(and use query conditions) 
            because it is gonna be manipulated in this func for each train,val or test set
    - note if your df has multipleSeries(NSeries) data, more likely you should gonna use df query conditions
    """
    trainRatio, valRatio, testRatio=ratiosCheck(trainRatio, valRatio)
    if trainSeqLen==None:
        trainSeqLen = seqLen
    if valSeqLen==None:
        valSeqLen = seqLen
    if testSeqLen==None:
        testSeqLen = seqLen

    npDictUsed=False
    if isinstance(df, NpDict):
        df = df.df
        npDictUsed=True
    filteredDf = df.copy()
    
    isAnyConditionApplied=False
    doQueryNTurnisAnyConditionApplied= lambda df,con,ica:(df.query(con),True)
    for condition in conditions:
        if condition==splitDefaultCondition:
            try:
                filteredDf, isAnyConditionApplied = doQueryNTurnisAnyConditionApplied(filteredDf, condition, isAnyConditionApplied)
            except:
                pass
        else:
            filteredDf, isAnyConditionApplied = doQueryNTurnisAnyConditionApplied(filteredDf, condition, isAnyConditionApplied)
    
    indexes=np.array(filteredDf.sort_index().index)
    indexes = subtractFromIndexes(indexes, trainRatio, valRatio, trainSeqLen,
                                  valSeqLen, testSeqLen, isAnyConditionApplied)

    if shuffle:
        #kkk add compatibility to seed everything
        np.random.shuffle(indexes)
    trainIndexes, valIndexes, testIndexes=simpleSplit(indexes, trainRatio, valRatio)
    if giveStartPointsIndexes:
        return trainIndexes, valIndexes, testIndexes

    sets=[]
    dfIndexes=df.index
    for idx,sl in zip([trainIndexes, valIndexes, testIndexes],
                      [trainSeqLen,valSeqLen,testSeqLen]):
        set_=filteredDf.loc[idx]
        set_[tsStartPointColName]=True
        idx2=addSequentAndAntecedentIndexes(idx, seqLenWithSequents=sl)
        sequenceTailIndexes=[item for item in idx2 if item not in idx]
        try:
            sequenceTailData=df.loc[sequenceTailIndexes]
        except:
            if tailIndexesAsPossible:
                "#ccc having tailIndexesAsPossible makes it possible for datasets to fetch data with inEqual length"
                sequenceTailIndexes=[sti for sti in sequenceTailIndexes if sti in dfIndexes]
                sequenceTailData=df.loc[sequenceTailIndexes]    
            else:
                raise IndexError("sequence tails(points which can't be the start point because of time series data)"\
                  +"df should have '__startPoint__', False or indicated with other query conditions")
        sequenceTailData[tsStartPointColName]=False
        set_=pd.concat([set_,sequenceTailData]).sort_index().reset_index(drop=True)
        sets+=[set_]
    trainDf, valDf, testDf=sets
    if npDictUsed:
        trainDf, valDf, testDf=NpDict(trainDf), NpDict(valDf), NpDict(testDf)
    warningMade=False
    for df_, ratio in zip([trainDf, valDf, testDf],[trainRatio, valRatio, testRatio]):
        if mpf(ratio)!=0 and len(df_)==0:
            warningMade=True
    if warningMade:
        warnings.warn("the backcastLen and forecastLen seem to be high. some of the sets(train|val|test) are empty")
        #kkk make warnings type of vAnnWarning
        #kkk maybe print warnings in colored background
    return trainDf, valDf, testDf
#%% padding
#%%      df & series
def rightPadSeriesIfShorter(series, maxLen, pad=0):
    if maxLen <= 0:
        return series
    currentLength = len(series)
    assert currentLength <= maxLen, f"The series length is greater than {maxLen}: {currentLength}"
    if currentLength < maxLen:
        series = rightPadSeries(series, maxLen - currentLength, pad=pad)
    return series

def rightPadSeries(series, padLen, pad=0):
    if padLen <= 0:
        return series
    padding = pd.Series([pad] * padLen)
    series = pd.concat([series, padding], ignore_index=True)
    return series

def rightPadDfBaseFunc(func, dfOrSeries, padLen, pad=0):
    #kkk do similar for left, and reduce all to another base func
    #kkk could have added colPad for each col, and if the specificColPad doesnt exist the 'pad'(which default would have used)
    'also works with series'
    if isinstance(dfOrSeries, pd.DataFrame):
        tempDict={}
        for i, col in enumerate(dfOrSeries.columns):
            if col==tsStartPointColName:
                tempDict[col]=func(dfOrSeries[col], padLen, pad=False)
            else:
                tempDict[col]=func(dfOrSeries[col], padLen, pad=pad)
        for i, col in enumerate(dfOrSeries.columns):
            if i==0:
                if dfOrSeries.index.dtype in [np.int16, np.int32, np.int64]:
                    dfStartInd=dfOrSeries.index.min()
                    newIndex=[jj for jj in range(dfStartInd,dfStartInd+len(tempDict[col]))]
                    dfOrSeries=dfOrSeries.reindex(newIndex)
                else:
                    newIndex=tempDict[col].index
                    dfOrSeries=dfOrSeries.reindex(newIndex)
            dfOrSeries[col]=pd.Series(tempDict[col].values,index=newIndex)
        return dfOrSeries
    elif isinstance(dfOrSeries, pd.Series):
        return func(dfOrSeries, padLen, pad=pad)
    else:
        raise ValueError("Input must be either a DataFrame or a Series")

def rightPadDfIfShorter(dfOrSeries, maxLen, pad=0):
    return rightPadDfBaseFunc(rightPadSeriesIfShorter, dfOrSeries, maxLen, pad=pad)

def rightPadDf(dfOrSeries, padLen, pad=0):
    return rightPadDfBaseFunc(rightPadSeries, dfOrSeries, padLen, pad=pad)
#%%      np array
def rightPadNpArrayBaseFunc(arr, padLen, pad=0):
    #kkk do similar for left, and reduce all to another base func
    #kkk could have added colPad for each col, and if the specificColPad doesnt exist the 'pad'(which default would have used)
    if padLen <= 0:
        return arr
    currentLength = len(arr)
    if currentLength < padLen:
        padding = np.full(padLen - currentLength, pad)
        arrShape=list(arr.shape)
        arrShape[0] = len(padding)
        padding= npArrayBroadCast(padding, arrShape)
        arr = np.concatenate((arr, padding))
    return arr

def rightPadNpArrayIfShorter(arr, maxLen, pad=0):
    if maxLen <= 0:
        return arr
    currentLength = len(arr)
    assert currentLength <= maxLen, f"The array length is greater than {maxLen}: {currentLength}"
    if currentLength < maxLen:
        arr = rightPadNpArrayBaseFunc(arr, maxLen, pad=pad)
    return arr

def rightPadNpArray(arr, padLen, pad=0):
    return rightPadNpArrayBaseFunc(arr, padLen, pad=pad)
#%%      tensor
def rightPadTensorBaseFunc(tensor, padLen, pad=0):
    #kkk do similar for left, and reduce all to another base func
    #kkk could have added colPad for each col, and if the specificColPad doesnt exist the 'pad'(which default would have used)
    if padLen <= 0:
        return tensor
    currentLength = tensor.size(0)
    if currentLength < padLen:
        padding = torch.full((padLen - currentLength,) + tensor.shape[1:], pad)
        tensor = torch.cat((tensor, padding), dim=0)
    return tensor

def rightPadTensorIfShorter(tensor, maxLen, pad=0):
    if maxLen <= 0:
        return tensor
    currentLength = tensor.size(0)
    assert currentLength <= maxLen, f"The tensor length is greater than {maxLen}: {currentLength}"
    if currentLength < maxLen:
        tensor = rightPadTensorBaseFunc(tensor, maxLen, pad=pad)
    return tensor

def rightPadTensor(tensor, padLen, pad=0):
    return rightPadTensorBaseFunc(tensor, padLen, pad=pad)
#%% misc
def calculateSingleColMinDifference(df, valueCol, resultCol):
    df[resultCol] = df[valueCol] - df[valueCol].min()

def excludeValuesFromBeginning_SingleCol(df, excludeVal, valueCol, resultCol):
    maxVal = df[valueCol].max()
    minVal = df[valueCol].min()
    if maxVal-minVal<=excludeVal:
        warnings.warn(f'by excluding values form the beginning, no {resultCol} equal to True')
    df.loc[df[valueCol] >= minVal + excludeVal, resultCol] = True
    df.loc[df[resultCol] != True, resultCol] = False

def excludeValuesFromEnd_SingleCol(df, excludeVal, valueCol, resultCol):
    maxVal = df[valueCol].max()
    minVal = df[valueCol].min()
    if maxVal-minVal<=excludeVal:
        warnings.warn(f'by excluding values form the end, no {resultCol} equal to True')
    df.loc[df[valueCol] <= maxVal - excludeVal, resultCol] = True
    df.loc[df[resultCol] != True, resultCol] = False