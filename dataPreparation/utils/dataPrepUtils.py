import pandas as pd
import os
from sklearn.preprocessing import StandardScaler

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

class SingleColsStdNormalizer:
    def init(self, colNames:list):
        self.scalers={col:StdScaler(col) for col in colNames}

    @property
    def colNames(self):
        return self.scalers.keys()

    def fitNTransform(self, df, col):
        assert col in df.columns, f'{col} is not in df columns'
        self.scalers[col].fit(df[col])
        df[col] = self.scalers[col].transform(df[[col]])

    def inverseTransform(self, dataToInverseTransformed):
        return self.scaler.inverseTransform(dataToInverseTransformed)

    def repr(self):
        return f"SingleColsStdNormalizer+{'_'.join(self.colNames)}"
