import pandas as pd
import numpy as np
import os
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sklearn.preprocessing import StandardScaler, LabelEncoder
from utils.vAnnGeneralUtils import NpDict, checkAllItemsInList1ExistInList2
#%% general vars
LblEncoderValueErrorMsg="Integer labels detected. Use makeIntLabelsString to convert them to string labels."
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
        assert isinstance(newNormalizer, (BaseSingleColsNormalizer, BaseMultiColNormalizer, MainGroupBaseSingleColsStdNormalizer))
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
        return self._normalizers[col].inverseTransformCol(df, col)

    def ultimateInverseTransform(self, df):
        for col in self.normalizers.keys():
            df[col] = self._normalizers[col].ultimateInverseTransformCol(df, col)

    def ultimateInverseTransformCol(self, df, col):
        return self._normalizers[col].ultimateInverseTransformCol(df[col], col)

    def __repr__(self):
        return str(self.uniqueNormalizers)
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

    def inverseTransformCol(self, df, col):
        dataToInverseTransformed=df[col]
        return self.scalers[col].inverseTransform(dataToInverseTransformed)

    def ultimateInverseTransformCol(self, dataToInverseTransformed, col):
        dataToInverseTransformed = self.inverseTransformCol(dataToInverseTransformed, col)
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
    
    def inverseTransformCol(self, df, col):
        dataToInverseTransformed=df[col]
        return self.scaler.inverseTransform(dataToInverseTransformed)

    def ultimateInverseTransformCol(self, dataToInverseTransformed, col):
        assert col in dataToInverseTransformed.columns,'ultimateInverseTransformCol "{self}" "{col}" col is not in df columns'
        res = self.inverseTransformCol(dataToInverseTransformed, col)
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

class MainGroupBaseNormalizer:
    def __init__(self, df, mainGroupColNames):
        self.mainGroupColNames = mainGroupColNames
        self.uniqueCombos = self._getUniqueCombinations(df)

    def uniqueCombosShortReprs(self):
        return [combo.shortRepr_() for combo in self.uniqueCombos]

    def findMatchingShortReprCombo(self, combo):
        for uniqueCombo in self.uniqueCombos:
            if combo == uniqueCombo.shortRepr_():
                return uniqueCombo
        return None

    def uniqueCombosDictReprs(self):
        return [combo.defDict for combo in self.uniqueCombos]

    def findMatchingDictReprCombo(self, combo):
        for uniqueCombo in self.uniqueCombos:
            if combo == uniqueCombo.defDict:
                return uniqueCombo
        return None

    def comboInUniqueCombos(self, combo):
        if isinstance(combo, Combo):
            if combo in self.uniqueCombos:
                return combo
        elif isinstance(combo, str):
            if self.findMatchingShortReprCombo(combo):
                return self.findMatchingShortReprCombo(combo)
        elif isinstance(combo, dict):
            if self.findMatchingDictReprCombo(combo):
                return self.findMatchingDictReprCombo(combo)
        return

    def _getUniqueCombinations(self, df):
        uniqueCombos  = df.groupby(self.mainGroupColNames).size().reset_index().rename(columns={0: 'count'})
        uniqueCombos  = uniqueCombos.rename(columns={0: 'count'})
        comboObjs = []
        for index, row in uniqueCombos.iterrows():
            comboDict = {col: row[col] for col in self.mainGroupColNames}
            combo = Combo(comboDict, self.mainGroupColNames)
            comboObjs.append(combo)
        
        return comboObjs

    def getRowsByCombination(self, df, combo):
        comboObj=self.comboInUniqueCombos(combo)
        assert comboObj, "Combo is not in uniqueCombos"
        tempDf=df[(df[self.mainGroupColNames] == comboObj.defDict).all(axis=1)]
        
        # this is to correct dtypes
        npDict=NpDict(tempDf)
        tempDf=npDict.toDf(resetDtype=True)
        return tempDf
#%% MainGroupSingleColsStdNormalizer
class MainGroupBaseSingleColsStdNormalizer(MainGroupBaseNormalizer):
    def __init__(self, classType, df, mainGroupColNames, colNames:list):
        super().__init__(df, mainGroupColNames)
        self.colNames=colNames
        self.container={}
        for col in colNames:
            self.container[col]={}
            for combo in self.uniqueCombos:
                self.container[col][combo.shortRepr_()]=classType([col])

    def fitNTransform(self, df):
        for col in self.colNames:
            for combo in self.uniqueCombos:
                dfToFit=self.getRowsByCombination(df, combo)
                inds=dfToFit.index
                dfToFit=dfToFit.reset_index(drop=True)
                self.container[col][combo.shortRepr_()].fitNTransform(dfToFit)
                dfToFit.index=inds
                df.loc[inds,col]=dfToFit

    def inverseTransformColBase(self, df, col, funcName):
        for combo in self.uniqueCombos:
            dfToFit=self.getRowsByCombination(df, combo)
            inds=dfToFit.index
            dfToFit=dfToFit.reset_index(drop=True)
            func = getattr(self.container[col][combo.shortRepr_()], funcName)
            invRes=func(dfToFit, col)
            df.loc[inds,col]=invRes
        return df[col]

    def inverseTransformCol(self, df, col):
        return self.inverseTransformColBase(df, col, 'inverseTransformCol')

    def ultimateInverseTransformCol(self, df, col):
        return self.inverseTransformColBase(df, col, 'ultimateInverseTransformCol')

class MainGroupSingleColsStdNormalizer(MainGroupBaseSingleColsStdNormalizer):
    def __init__(self, df, mainGroupColNames, colNames:list):
        super().__init__(SingleColsStdNormalizer, df, mainGroupColNames, colNames)

class MainGroupSingleColsLblEncoder(MainGroupBaseSingleColsStdNormalizer):
    "this the lblEncoder version of MainGroupSingleColsStdNormalizer; its rarely useful, but in some case maybe used"
    def __init__(self, df, mainGroupColNames, colNames:list):
        super().__init__(SingleColsLblEncoder, df, mainGroupColNames, colNames)