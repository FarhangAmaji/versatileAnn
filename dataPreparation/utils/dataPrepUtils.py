import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
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
        self.name = name#kkk add names
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

class NormalizerStack:
    def __init__(self, *stdNormalizers):
        self._normalizers = {}
        for stdNormalizer in stdNormalizers:
            self.addNormalizer(stdNormalizer)

    def addNormalizer(self, newNormalizer):
        assert isinstance(newNormalizer, (SingleColsStdNormalizer, MultiColStdNormalizer))
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

class SingleColsStdNormalizer:
    def __init__(self, colNames:list):
        self.scalers={col:StdScaler(col) for col in colNames}

    @property
    def colNames(self):
        return self.scalers.keys()
    
    def fitNTransform(self, df):
        for col in self.colNames:
            self.fitNTransformCol(self, df, col)

    def fitNTransformCol(self, df, col):
        assert col in df.columns, f'{col} is not in df columns'
        self.scalers[col].fit(df[col])
        df[col] = self.scalers[col].transform(df[[col]])

    def inverseTransformCol(self, dataToInverseTransformed, col):
        return self.scalers[col].inverseTransform(dataToInverseTransformed)

    def repr(self):
        return f"SingleColsStdNormalizer+{'_'.join(self.colNames)}"

class MultiColStdNormalizer:
    def __init__(self, colNames):
        self.scaler = StdScaler('_'.join(colNames))
        self.colNames = colNames

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
    
    def repr(self):
        return f"MultiColStdNormalizer+{'_'.join(self.colNames)}"
