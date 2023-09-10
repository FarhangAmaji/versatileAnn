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
#%% series
def splitToNSeries(df, pastCols, renameCol):
    processedData=pd.DataFrame({})
    otherCols= [col for col in df.columns if col not in pastCols]
    for i,pc in enumerate(pastCols):
        thisSeriesDf=df[otherCols+[pc]]
        thisSeriesDf=thisSeriesDf.rename(columns={pc:renameCol})
        thisSeriesDf[renameCol+'Type']=pc
        processedData = pd.concat([processedData,thisSeriesDf]).reset_index(drop=True)
    return processedData
#%% data split
splitDefaultCondition='__possibleStartPoint__ == 1'
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

def splitTrainValTest(df, trainRatio, valRatio,
                      trainSeqLen=0, valSeqLen=None, testSeqLen=None,
                      shuffle=True, conditions=[splitDefaultCondition]):
    """
    note this func expects conditions which indicate the first(older in time|backer in sequence);
    therefore for seq lens pass the (backcastLen+ forecastLen)
    """
    trainRatio, valRatio=round(trainRatio,6), round(valRatio,6)#kkk whole code
    testRatio=round(1-trainRatio-valRatio,6)
    assert sum([trainRatio, valRatio, testRatio])==1, 'sum of train, val and test ratios must be 1'

    if valSeqLen==None:
        valSeqLen = trainSeqLen
    if testSeqLen==None:
        testSeqLen = trainSeqLen

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

    if isCondtionsApplied:
        indexes=np.array(filteredDf.index)
    else:
        indexes=np.array(filteredDf.index.tolist()[:-max(trainSeqLen, valSeqLen, testSeqLen)])
    if shuffle:#kkk add compatibility to seed everything
        np.random.shuffle(indexes)

    trainIndexes=indexes[:int(trainRatio*len(indexes))]
    valIndexes=indexes[int(trainRatio*len(indexes)):int((trainRatio+valRatio)*len(indexes))]
    testIndexes=indexes[int((trainRatio+valRatio)*len(indexes)):]
    print(trainIndexes, valIndexes, testIndexes)

    trainIndexes=addSequentAndAntecedentIndexes(trainIndexes, seqLenWithSequents=trainSeqLen)
    valIndexes=addSequentAndAntecedentIndexes(valIndexes, seqLenWithSequents=valSeqLen)
    testIndexes=addSequentAndAntecedentIndexes(testIndexes, seqLenWithSequents=testSeqLen)

    train=filteredDf.loc[trainIndexes]
    val=filteredDf.loc[valIndexes]
    test=filteredDf.loc[testIndexes]
    return train, val, test
#%% utils misc
def equalDfs(df1, df2, floatPrecision=0.0001):
    # Check if both DataFrames have the same shape
    if df1.shape != df2.shape:
        return False

    # Iterate through columns and compare them individually
    for col in df1.columns:
        if pd.api.types.is_numeric_dtype(df1[col]) and pd.api.types.is_numeric_dtype(df2[col]):
            # Check if all elements in the numeric column are close
            if not np.allclose(df1[col], df2[col], rtol=floatPrecision):
                return False
        else:
            if any([pd.api.types.is_numeric_dtype(df1[col]), pd.api.types.is_numeric_dtype(df2[col])]):
                npd1=NpDict(df1).getDfDict(True)
                npd2=NpDict(df2).getDfDict(True)
                if any([pd.api.types.is_numeric_dtype(npd1[col]), pd.api.types.is_numeric_dtype(npd2[col])]):
                    return False
            # If the column is non-numeric, skip the check
            continue

    # If all numeric columns are close, return True
    return True

def checkAllItemsInList1ExistInList2(list1, list2):
    setList2 = set(list2)
    for item in list1:
        if item not in setList2:
            return False
    return True