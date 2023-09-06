import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder
#%% datasets
datasetsRelativePath=r'..\..\data\datasets'

def getDatasetFiles(fileName: str):
    currentDir = os.path.dirname(os.path.abspath(__file__))
    datasetsDir = os.path.normpath(os.path.join(currentDir, datasetsRelativePath))
    os.makedirs(datasetsDir, exist_ok=True)
    filePath=os.path.join(datasetsDir, fileName)
    return pd.read_csv(filePath)
#%% normalizers
class StdScaler:
    def __init__(self, name=None):
        self.name = name
        self.scaler = StandardScaler()
        
    @property
    def isFitted(self):
        return hasattr(self.scaler, 'mean_') and self.scaler.mean_ is not None

    def fit(self, dataToFit, colShape=1):
        if not self.isFitted:
            self.scaler.fit(dataToFit.values.reshape(-1,colShape))
        else:
            print(f'StdScaler {self.name} is already fitted')

    def transform(self, dataToFit, colShape=1):
        dataToFit =dataToFit.values.reshape(-1,colShape)
        if self.isFitted:
            mean = dataToFit.mean()
            if -1 <= mean <= 1:
                print(f'StdScaler {self.name} skipping transform: Mean of dataToFit is between -1 and 1; so seems to be already fitted.')
                return dataToFit
            else:
                return self.scaler.transform(dataToFit)
        else:
            print(f'StdScaler {self.name} skipping transform: is not fitted; fit it first')
            return dataToFit

    def inverseTransform(self, dataToInverseTransformed, colShape=1):
        dataToInverseTransformed =dataToInverseTransformed.values.reshape(-1,colShape)
        if self.isFitted:
            mean = dataToInverseTransformed.mean()
            if -1 <= mean <= 1:
                return self.scaler.inverse_transform(dataToInverseTransformed)
            else:
                print(f'StdScaler {self.name} skipping inverse transform: Mean of dataToInverseTransformed is not between -1 and 1, since seems the dataToInverseTransformed not to be normalized')
                return dataToInverseTransformed
        else:
            print(f'StdScaler {self.name} is not fitted; cannot inverse transform.')
            return dataToInverseTransformed

def makeIntLabelsString(df, colName):
    #kkk needs asserts
    uniqueVals = df[colName].unique()
    intToLabelMapping = {intVal: f'{colName}{label}' for label, intVal in enumerate(uniqueVals)}
    df[colName] = df[colName].map(intToLabelMapping)

class LblEncoder:
    #kkk it cant handle None or np.nan or other common missing values
    def __init__(self, name=None):
        self.name = name
        self.encoder = LabelEncoder()

    @property
    def isFitted(self):
        return hasattr(self.encoder, 'classes_') and self.encoder.classes_ is not None

    def fit(self, dataToFit):
        if not self.isFitted:
            # Check if there are integer labels
            if any(isinstance(label, int) for label in dataToFit):
                raise ValueError("Integer labels detected. Use makeIntLabelsString to convert them to string labels.")
            self.encoder.fit(dataToFit.values.reshape(-1))
        else:
            print(f'LblEncoder {self.name} is already fitted')
    
    def doesDataSeemToBeAlreadyTransformed(self, data):
        return checkAllItemsInList1ExistInList2(np.unique(data), list(range(len(self.encoder.classes_))))
        
    def transform(self, dataToFit):
        dataToFit= dataToFit.values.reshape(-1)
        if self.isFitted:
            if self.doesDataSeemToBeAlreadyTransformed(dataToFit):
                print(f'LblEncoder {self.name} skipping transform: data already seems transformed.')
                return dataToFit
            else:
                return self.encoder.transform(dataToFit)
        else:
            print(f'LblEncoder {self.name} skipping transform: is not fitted; fit it first')
            return dataToFit

    def doesdataToInverseTransformedSeemToBeAlreadyDone(self, data):
        return checkAllItemsInList1ExistInList2(np.unique(data), list(self.encoder.classes_))

    def inverseTransform(self, dataToInverseTransformed):
        dataToInverseTransformed =dataToInverseTransformed.values.reshape(-1)
        if self.isFitted:
            if self.doesdataToInverseTransformedSeemToBeAlreadyDone(dataToInverseTransformed):
                print(f'LabelEncoder {self.name} skipping inverse transform: data already seems inverse transformed.')
                return dataToInverseTransformed
            else:
                return self.encoder.inverse_transform(dataToInverseTransformed)
        else:
            print(f'LblEncoder {self.name} is not fitted; cannot inverse transform.')
            return dataToInverseTransformed

class NormalizerStack:
    def __init__(self, *stdNormalizers):
        self._normalizers = {}
        for stdNormalizer in stdNormalizers:
            self.addNormalizer(stdNormalizer)

    def addNormalizer(self, newNormalizer):
        assert isinstance(newNormalizer, (BaseSingleColsNormalizer, BaseMultiColNormalizer))
        for col in newNormalizer.colNames:
            if col not in self._normalizers.keys():
                self._normalizers.update({col: newNormalizer})
            else:
                print(f'{col} is already in normalizers')
    
    @property
    def normalizers(self):
        return self._normalizers
    
    @property
    def uniqueNormalizers(self):
        uniqueNormalizers=[]
        [uniqueNormalizers.append(nrm) for nrm in self._normalizers.values() if nrm not in uniqueNormalizers]
        return uniqueNormalizers

    def fitNTransform(self, df):
        for nrm in self.uniqueNormalizers:
            nrm.fitNTransform(df)

    def inverseTransform(self, df):
        for col in self.normalizers.keys():
            df[col] = self.inverseTransformCol(df, col)

    def inverseTransformCol(self, df, col):
        assert col in self._normalizers.keys(),f'{col} is not in normalizers cols'
        return self._normalizers[col].inverseTransformCol(df[col], col)

class BaseSingleColsNormalizer:
    def __init__(self):
        self.makeIntLabelsStrings={}

    @property
    def colNames(self):
        return self.scalers.keys()
    
    def fitNTransform(self, df):
        for col in self.colNames:
            self.fitNTransformCol(df, col)

    def fitNTransformCol(self, df, col):
        assert col in df.columns, f'{col} is not in df columns'
        self.scalers[col].fit(df[col])
                
        df[col] = self.scalers[col].transform(df[[col]])

    def inverseTransformCol(self, dataToInverseTransformed, col):
        return self.scalers[col].inverseTransform(dataToInverseTransformed)

class SingleColsStdNormalizer(BaseSingleColsNormalizer):
    def __init__(self, colNames:list):
        self.scalers={col:StdScaler(f'std{col}') for col in colNames}

    def __repr__(self):
        return f"SingleColsStdNormalizer+{'_'.join(self.colNames)}"

class SingleColsLblEncoder(BaseSingleColsNormalizer):
    def __init__(self, colNames:list):
        self.encoders={col:LblEncoder(f'lbl{col}') for col in colNames}

    @property
    def scalers(self):
        return self.encoders

    def __repr__(self):
        return f"SingleColsLblEncoder+{'_'.join(self.colNames)}"

class BaseMultiColNormalizer:
    def __init__(self):
        pass

    def assertColNames(self, df):
        for col in self.colNames:
            assert col in df.columns, f'{col} is not in df columns'
    
    def fit(self, df):
        self.assertColNames(df)
        self.scaler.fit(df[self.colNames])
    
    def transform(self, df):
        self.assertColNames(df)
        df[self.colNames] = self.scaler.transform(df[self.colNames]).reshape(-1, len(self.colNames))
    
    def fitNTransform(self, df):
        self.fit(df)
        self.transform(df)
    
    #kkk could have add many fit, transform, assert and their other combinations for single col
    #kkk could have added inverseTransform which does inverse on self.colNames in df 
    
    def inverseTransformCol(self, dataToInverseTransformed, col=None):
        '#ccc col is not used and its just for compatibleness'#kkk is this acceptable in terms of software engineering
        return self.scaler.inverseTransform(dataToInverseTransformed)

class MultiColStdNormalizer(BaseMultiColNormalizer):
    def __init__(self, colNames):
        self.scaler = StdScaler('std'+'_'.join(colNames))
        self.colNames = colNames

    def __repr__(self):
        return f"MultiColStdNormalizer+{'_'.join(self.colNames)}"

class MultiColLblEncoder(BaseMultiColNormalizer):
    def __init__(self, colNames):
        self.encoder = LblEncoder('lbl'+'_'.join(colNames))
        self.colNames = colNames

    @property
    def scaler(self):
        return self.encoder
    
    def __repr__(self):
        return f"MultiColLblEncoder+{'_'.join(self.colNames)}"
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

#%% data conversion
def dfToNpDict(df):
    return {col:df[col].values for col in df.columns}

def npDictToDfForCol(df, dic, col):
    "works also with dfs with multiple indexes"
    assert col in dic.keys(),f'{col} is not in dictionary cols'
    assert col in df.columns,f'{col} is not in dataframe cols'
    assert len(dic[col])==len(df[col]),f'{col} lens in dataframe and dictionary are equal'
    df[col]=dic[col]
    
def npDictToDf(df, dic):
    for col in dic.keys():
        npDictToDfForCol(df, dic, col)
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
