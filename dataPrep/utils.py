import pandas as pd
import numpy as np
import os
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.vAnnGeneralUtils import NpDict
#%% datasets
datasetsRelativePath=r'..\data\datasets'
def getDatasetFiles(fileName: str):
    currentDir = os.path.dirname(os.path.abspath(__file__))
    datasetsDir = os.path.normpath(os.path.join(currentDir, datasetsRelativePath))
    os.makedirs(datasetsDir, exist_ok=True)
    filePath=os.path.join(datasetsDir, fileName)
    return pd.read_csv(filePath)
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