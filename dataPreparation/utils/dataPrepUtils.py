import pandas as pd
import numpy as np
import os
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import inspect
from sklearn.preprocessing import StandardScaler, LabelEncoder
from utils.vAnnGeneralUtils import NpDict
#%% general vars
datasetsRelativePath=r'..\..\data\datasets'
LblEncoderValueErrorMsg="Integer labels detected. Use makeIntLabelsString to convert them to string labels."
#%% datasets

def getDatasetFiles(fileName: str):
    currentDir = os.path.dirname(os.path.abspath(__file__))
    datasetsDir = os.path.normpath(os.path.join(currentDir, datasetsRelativePath))
    os.makedirs(datasetsDir, exist_ok=True)
    filePath=os.path.join(datasetsDir, fileName)
    return pd.read_csv(filePath)
#%% normalizers: base normalizers
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
            return self.scaler.transform(dataToFit)
        else:
            print(f'StdScaler {self.name} skipping transform: is not fitted; fit it first')#kkkMinor this is not in the tests
            return dataToFit

    def inverseTransform(self, dataToInverseTransformed, colShape=1):
        dataToInverseTransformed =dataToInverseTransformed.values.reshape(-1,colShape)
        if self.isFitted:
            return self.scaler.inverse_transform(dataToInverseTransformed)
        else:
            print(f'StdScaler {self.name} is not fitted; cannot inverse transform.')#kkkMinor this is not in the tests
            return dataToInverseTransformed

class LblEncoder:
    #kkk it cant handle None or np.nan or other common missing values
    def __init__(self, name=None):
        self.name = name
        self.encoder = LabelEncoder()

    @property
    def isFitted(self):
        return hasattr(self.encoder, 'classes_') and self.encoder.classes_ is not None

    def fit(self, dataToFit):
        assert isinstance(dataToFit, (pd.Series, pd.DataFrame)), "LblEncoder dataToFit is not a pandas Series or DataFrame"
        if not self.isFitted:
            # Check if there are integer labels
            if any(isinstance(label, int) for label in dataToFit):
                raise ValueError(LblEncoderValueErrorMsg)#kkk should in the tests
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
            print(f'LblEncoder {self.name} skipping transform: is not fitted; fit it first')#kkkMinor this is not in the tests
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
            print(f'LblEncoder {self.name} is not fitted; cannot inverse transform.')#kkkMinor this is not in the tests
            return dataToInverseTransformed

class makeIntLabelsString:
    def __init__(self, name):
        self.name = name
        self.isFitted = False

    def fitNTransform(self, inputData):
        if self.isFitted:
            print(f'skipping fit:{self.name} makeIntLabelsString is already fitted')
            return inputData
        array=inputData.values.reshape(-1)
        uniqueVals = np.unique(array)
        assert np.all(np.equal(uniqueVals, uniqueVals.astype(int))), "makeIntLabelsString {colName} All values should be integers."
        intToLabelMapping = {intVal: f'{self.name}:{label}' for label, intVal in enumerate(uniqueVals)}
        if isinstance(inputData, pd.Series):
            output = inputData.map(intToLabelMapping)
        elif isinstance(inputData, pd.DataFrame):
            output = inputData.applymap(lambda x: intToLabelMapping.get(x, x))
        self.intToLabelMapping={value: key for key, value in intToLabelMapping.items()}
        self.isFitted=True
        return output

    def inverseTransform(self, dataToInverseTransformed):
        return np.vectorize(self.intToLabelMapping.get)(dataToInverseTransformed)
#%% normalizers: NormalizerStack
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

    def ultimateInverseTransform(self, df):
        for col in self.normalizers.keys():
            df[col] = self._normalizers[col].ultimateInverseTransformCol(df, col)

    def ultimateInverseTransformCol(self, df, col):
        return self._normalizers[col].ultimateInverseTransformCol(df[col], col)
#%% normalizers: SingleColsNormalizers
class BaseSingleColsNormalizer:
    """for instances of SingleColsLblEncoder if they have/need makeIntLabelsStrings, we wont use 3 transforms.
    for i.e. in if we have 5->'colA0'->0, BaseSingleColsNormalizer transforms only the 'colA0'->0 and not 5->'colA0' or 5->0"""#kkk maybe comment needs a more detailed explanation
    def __init__(self):
        self.isFitted={col:False for col in self.colNames}

    @property
    def colNames(self):
        return self.scalers.keys()

    def fitNTransform(self, df):
        for col in self.colNames:
            self.fitNTransformCol(df, col)

    def fitNTransformCol(self, df, col):
        assert col in df.columns, f'{col} is not in df columns'
        if self.isFitted[col]:
            print(f'{self.__repr__()} {col} is already fitted')
            return
        try:
            self.scalers[col].fit(df[col])
        except ValueError as e:
            if str(e) == LblEncoderValueErrorMsg and isinstance(self, SingleColsLblEncoder):
                self.makeIntLabelsStrings[col]=makeIntLabelsString(col)
                df[col] = self.makeIntLabelsStrings[col].fitNTransform(df[col])
                self.scalers[col].fit(df[col])
        self.isFitted[col]=True
        df[col] = self.transformCol(df, col)

    def transform(self, df):
        for col in self.colNames:
            df[col]=self.transformCol(df, col)

    def transformCol(self, df, col):
        assert col in df.columns, f'{col} is not in df columns'
        if not self.isFitted[col]:
            print(f'{self.__repr__()} {col} is not fitted; fit it first')
            return df[col]
        return self.scalers[col].transform(df[col])

    def inverseTransformCol(self, dataToInverseTransformed, col):
        return self.scalers[col].inverseTransform(dataToInverseTransformed)

    def ultimateInverseTransformCol(self, dataToInverseTransformed, col):
        dataToInverseTransformed = self.inverseTransformCol(dataToInverseTransformed[col], col)
        if hasattr(self, 'makeIntLabelsStrings'):
            if col in self.makeIntLabelsStrings.keys():
                dataToInverseTransformed = self.makeIntLabelsStrings[col].inverseTransform(dataToInverseTransformed)
        return dataToInverseTransformed

class SingleColsStdNormalizer(BaseSingleColsNormalizer):
    def __init__(self, colNames:list):
        self.scalers={col:StdScaler(f'std{col}') for col in colNames}
        super().__init__()

    def __repr__(self):
        return f"SingleColsStdNormalizer+{'_'.join(self.colNames)}"

class SingleColsLblEncoder(BaseSingleColsNormalizer):
    def __init__(self, colNames:list):
        self.makeIntLabelsStrings={}
        self.encoders={col:LblEncoder(f'lbl{col}') for col in colNames}
        super().__init__()

    @property
    def scalers(self):
        return self.encoders

    def __repr__(self):
        return f"SingleColsLblEncoder+{'_'.join(self.colNames)}"
#%% normalizers: BaseMultiColNormalizers
class BaseMultiColNormalizer:
    def __init__(self):
        self.isFitted=False

    def assertColNames(self, df):
        for col in self.colNames:
            assert col in df.columns, f'{col} is not in df columns'

    def areTheseIntCols(self, df):
        return df[self.colNames].apply(lambda col: col.apply(lambda x: isinstance(x, int))).all().all()

    def fit(self, df):
        self.assertColNames(df)
        self.scaler.fit(df[self.colNames])
    
    def transform(self, df):
        self.assertColNames(df)
        df[self.colNames] = self.scaler.transform(df[self.colNames]).reshape(-1, len(self.colNames))

    def fitNTransform(self, df):
        if self.isFitted:
            print(f'{self.__repr__()} is already fitted')
            return
        if isinstance(self, MultiColLblEncoder) and self.areTheseIntCols(df):
            self.makeIntLabelsString=makeIntLabelsString(self.shortRep())
            df[self.colNames]=self.makeIntLabelsString.fitNTransform(df[self.colNames])
        self.fit(df)
        self.isFitted=True
        self.transform(df)
    
    #kkk could have add many fit, transform, assert and their other combinations for single col
    #kkk could have added inverseTransform which does inverse on self.colNames in df 
    
    def inverseTransformCol(self, dataToInverseTransformed, col=None):
        '#ccc col is not used and its just for compatibleness'#kkk is this acceptable in terms of software engineering
        return self.scaler.inverseTransform(dataToInverseTransformed)

    def ultimateInverseTransformCol(self, dataToInverseTransformed, col):
        assert col in dataToInverseTransformed.columns,'ultimateInverseTransformCol "{self}" "{col}" col is not in df columns'
        res = self.inverseTransformCol(dataToInverseTransformed[col])
        if self.makeIntLabelsString:
            res = self.makeIntLabelsString.inverseTransform(res)
        return res
        
class MultiColStdNormalizer(BaseMultiColNormalizer):
    def __init__(self, colNames):
        super().__init__()
        self.colNames = colNames
        self.scaler = StdScaler(self.shortRep())

    def shortRep(self):
        return 'std:'+'_'.join(self.colNames)

    def __repr__(self):
        return f"MultiColStdNormalizer+{'_'.join(self.colNames)}"

class MultiColLblEncoder(BaseMultiColNormalizer):
    def __init__(self, colNames):
        super().__init__()
        self.makeIntLabelsString = None
        self.colNames = colNames
        self.encoder = LblEncoder(self.shortRep())

    @property
    def scaler(self):
        return self.encoder

    def shortRep(self):
        return 'lbl:'+'_'.join(self.colNames)

    def __repr__(self):
        return f"MultiColLblEncoder+{'_'.join(self.colNames)}"
#%% normalizers: MainGroupNormalizers
class Combo:
    def __init__(self, defDict, mainGroupColNames):
        assert isinstance(defDict, dict) and all(key in defDict for key in mainGroupColNames), "defDict format is invalid."
        
        for key in defDict:
            if key not in mainGroupColNames:
                raise ValueError(f"'{key}' is not a valid column name in mainGroupColNames.")
        
        for col in mainGroupColNames:
            if col not in defDict:
                raise ValueError(f"'{col}' is missing in combo definition.")
        
        self.defDict=defDict

    def shortRepr_(self):
        return '_'.join(self.defDict.values())
    
    def __repr__(self):
        return str(self.defDict)
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
