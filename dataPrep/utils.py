import pandas as pd
import numpy as np
import os
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.vAnnGeneralUtils import NpDict
#%% datasets
datasetsRelativePath=r'..\data\datasets'
knownDatasetsDateTimeCols={"EPF_FR_BE.csv":['dateTime']}

def convertdateTimeCols(df, dateTimeCols):
    for dc in dateTimeCols:
        df[dc] = pd.to_datetime(df[dc])

def getDatasetFiles(fileName: str, dateTimeCols=[]):
    currentDir = os.path.dirname(os.path.abspath(__file__))
    filePath = os.path.normpath(os.path.join(currentDir, datasetsRelativePath, fileName))
    df=pd.read_csv(filePath)
    if fileName in knownDatasetsDateTimeCols.keys():
        convertdateTimeCols(df, knownDatasetsDateTimeCols[fileName])
    else:
        convertdateTimeCols(df, dateTimeCols)
    return df
#%% multi series data
def addCorrespondentRow(df, correspondentRowsDf, targets, targetNewColNameType, targetMapping={}):
    if targetMapping=={}:
        targetMapping = {tr:idx for tr,idx in zip(targets, correspondentRowsDf.index)}

    for target in targets:
        if target in targetMapping:
            target_index = targetMapping[target]
            condition = df[targetNewColNameType] == target
            df.loc[condition, correspondentRowsDf.columns] = correspondentRowsDf.iloc[target_index].values

def splitToNSeries(df, pastCols, newColName):#kkk make a reverse func
    processedData=pd.DataFrame({})
    otherCols= [col for col in df.columns if col not in pastCols]
    for i,pc in enumerate(pastCols):
        thisSeriesDf=df[otherCols+[pc]]
        thisSeriesDf=thisSeriesDf.rename(columns={pc:newColName})
        thisSeriesDf[newColName+'Type']=pc
        processedData = pd.concat([processedData,thisSeriesDf]).reset_index(drop=True)
    return processedData
#%% data split
tsStartPointColName='__possibleStartPoint__'#kkk make this enviromental variable
splitDefaultCondition=f'{tsStartPointColName} == 1'
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
    trainRatio, valRatio=round(trainRatio,6), round(valRatio,6)
    testRatio=round(1-trainRatio-valRatio,6)
    assert round(sum([trainRatio, valRatio, testRatio]),6)==1, 'sum of train, val and test ratios must be 1'
    return trainRatio, valRatio, testRatio

def simpleSplit(obj, trainRatio, valRatio):
    trainRatio, valRatio, testRatio=ratiosCheck(trainRatio, valRatio)
    train=obj[:int(trainRatio*len(obj))]
    val=obj[int(trainRatio*len(obj)):int((trainRatio+valRatio)*len(obj))]
    test=obj[int((trainRatio+valRatio)*len(obj)):]
    return train, val, test

def splitTrainValTest(df, trainRatio, valRatio, seqLen=0,
                      trainSeqLen=None, valSeqLen=None, testSeqLen=None,
                      shuffle=True, conditions=[splitDefaultCondition], giveIndexes=False):
    """
    note this func expects conditions which indicate the first(older in time|backer in sequence);
    therefore for seq lens pass the (backcastLen+ forecastLen)
    note if you are passing multipleSeries data, more likely you should gonna use df query conditions
    """
    trainRatio, valRatio, testRatio=ratiosCheck(trainRatio, valRatio)
    if trainSeqLen==None:
        trainSeqLen = seqLen
    if valSeqLen==None:
        valSeqLen = seqLen
    if testSeqLen==None:
        testSeqLen = seqLen

    isCondtionsApplied=False
    filteredDf = df.copy()
    doQueryNTurnIsCondtionsApplied= lambda df,con,ica:(df.query(con),True)
    for condition in conditions:
        if condition==splitDefaultCondition:
            try:
                filteredDf, isCondtionsApplied = doQueryNTurnIsCondtionsApplied(filteredDf, condition, isCondtionsApplied)
            except:
                pass
        else:
            filteredDf, isCondtionsApplied = doQueryNTurnIsCondtionsApplied(filteredDf, condition, isCondtionsApplied)
    
    indexes=np.array(filteredDf.index)
    if isCondtionsApplied==False:
        lenToSubtract=0
        for endIdx,sl in zip([int(trainRatio*len(indexes)), int((trainRatio+valRatio)*len(indexes)), len(indexes)],[trainSeqLen, valSeqLen, testSeqLen]):
            lenToSubtract=max(endIdx-len(indexes)+sl-1,lenToSubtract)

        if lenToSubtract!=0:
            indexes=indexes[:-lenToSubtract]
    
    if shuffle:#kkk add compatibility to seed everything
        np.random.shuffle(indexes)
    trainIndexes, valIndexes, testIndexes=simpleSplit(indexes, trainRatio, valRatio)
    if giveIndexes:
        return trainIndexes, valIndexes, testIndexes

    filteredDf.loc[list(set([*trainIndexes,*valIndexes,*testIndexes])),tsStartPointColName]=True
    nonStartPointCondition=filteredDf[tsStartPointColName]!=True
    filteredDf.loc[nonStartPointCondition, tsStartPointColName]=False
    
    trainIndexes2=addSequentAndAntecedentIndexes(trainIndexes, seqLenWithSequents=trainSeqLen)
    valIndexes2=addSequentAndAntecedentIndexes(valIndexes, seqLenWithSequents=valSeqLen)
    testIndexes2=addSequentAndAntecedentIndexes(testIndexes, seqLenWithSequents=testSeqLen)
    
    trainData=filteredDf.loc[trainIndexes2]
    valData=filteredDf.loc[valIndexes2]
    testData=filteredDf.loc[testIndexes2]
    return trainData, valData, testData
#%% padding
def rightPadSeriesBatch(series, maxLen, pad=0):
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

def rightPadDfBaseFunc(func, dfOrSeries, padLen, pad=0):#kkk do similar for left, and reduce all to another base func
    'also works with series'
    if isinstance(dfOrSeries, pd.DataFrame):
        tempDict={}
        for i, col in enumerate(dfOrSeries.columns):
            tempDict[col]=rightPadSeries(dfOrSeries[col], padLen, pad=pad)
        for i, col in enumerate(dfOrSeries.columns):
            if i==0:
                dfOrSeries=dfOrSeries.reindex(tempDict[col].index)
            dfOrSeries[col]=tempDict[col]
        return dfOrSeries
    elif isinstance(dfOrSeries, pd.Series):
        return rightPadSeries(dfOrSeries, padLen, pad=pad)
    else:
        raise ValueError("Input must be either a DataFrame or a Series")

def rightPadDfBatch(dfOrSeries, maxLen, pad=0):
    return rightPadDfBaseFunc(rightPadSeriesBatch, dfOrSeries, maxLen, pad=pad)

def rightPadDf(dfOrSeries, padLen, pad=0):
    return rightPadDfBaseFunc(rightPadSeries, dfOrSeries, padLen, pad=pad)