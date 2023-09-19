import pandas as pd
import numpy as np
import os
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sklearn.preprocessing import StandardScaler, LabelEncoder
from utils.vAnnGeneralUtils import NpDict, checkAllItemsInList1ExistInList2
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
    LblEncoderValueErrorMsg="Integer labels detected. Use IntLabelsString to convert them to string labels."
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
                raise ValueError(LblEncoder.LblEncoderValueErrorMsg)#kkk should in the tests
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

class IntLabelsString:
    def __init__(self, name):
        self.name = name
        self.isFitted = False

    def fit(self, inputData):
        assert isinstance(inputData, (pd.Series, pd.DataFrame)),'IntLabelsString only accepts pd.Series, pd.DataFrame'#kkk add NpDict
        if self.isFitted:
            print(f'skipping fit:{self.name} intLabelsString is already fitted')
            return inputData
        array=inputData.values.reshape(-1)
        uniqueVals = np.unique(array)
        assert np.all(np.equal(uniqueVals, uniqueVals.astype(int))), "IntLabelsString {colName} All values should be integers."
        self.intToLabelMapping = {intVal: f'{self.name}:{label}' for label, intVal in enumerate(uniqueVals)}
        self.labelToIntMapping={value: key for key, value in self.intToLabelMapping.items()}
        self.isFitted=True

    def fitNTransform(self, inputData):
        self.fit(inputData)
        return self.transform(inputData)

    def transform(self, dataToTransformed):
        assert isinstance(dataToTransformed, (pd.Series, pd.DataFrame)),'IntLabelsString only accepts pd.Series, pd.DataFrame'
        if isinstance(dataToTransformed, pd.Series):
            output = dataToTransformed.map(self.intToLabelMapping)
        elif isinstance(dataToTransformed, pd.DataFrame):
            output = dataToTransformed.applymap(lambda x: self.intToLabelMapping.get(x, x))
        return output

    def inverseTransform(self, dataToInverseTransformed):
        return np.vectorize(self.labelToIntMapping.get)(dataToInverseTransformed)
#%% normalizers: NormalizerStack
class NormalizerStack:
    def __init__(self, *stdNormalizers):
        self._normalizers = {}
        for stdNormalizer in stdNormalizers:
            self.addNormalizer(stdNormalizer)

    def addNormalizer(self, newNormalizer):
        assert isinstance(newNormalizer, (BaseSingleColsNormalizer, BaseMultiColNormalizer, MainGroupSingleColsNormalizer))
        for col in newNormalizer.colNames:
            if col not in self._normalizers.keys():#kkk add ability to have a col which exists in 2 normalizers
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

    def transformCol(self, df, col):#kkk needs tests
        return self._normalizers[col].transformCol(df, col)

    def inverseMiddleTransform(self, df):
        for col in list(self.normalizers.keys())[::-1]:
            df[col] = self.inverseMiddleTransformCol(df, col)

    def inverseMiddleTransformCol(self, df, col):
        assert col in self._normalizers.keys(),f'{col} is not in normalizers cols'
        return self._normalizers[col].inverseMiddleTransformCol(df, col)

    def inverseTransform(self, df):
        for col in list(self.normalizers.keys())[::-1]:
            df[col] = self.inverseTransformCol(df, col)

    def inverseTransformCol(self, df, col):
        return self._normalizers[col].inverseTransformCol(df, col)

    def __repr__(self):
        return str(self.uniqueNormalizers)
#%% normalizers: baseNormalizerChecks
class BaseNormalizerChecks:
    def assertColNameInDf(self, df, col):
        assert col in df.columns, f'{col} is not in df columns'
#%% normalizers: SingleColsNormalizers
class BaseSingleColsNormalizer(BaseNormalizerChecks):
    """for instances of SingleColsLblEncoder if they have/need IntLabelsStrings, we wont use 3 transforms.
    for i.e. in if we have 5->'colA0'->0, BaseSingleColsNormalizer transforms only the 'colA0'->0 and not 5->'colA0' or 5->0"""#kkk maybe comment needs a more detailed explanation
    def __init__(self):
        self.isFitted={col:False for col in self.colNames}

    @property
    def colNames(self):
        return self.scalers.keys()

    def isColFitted(self, col, printFitted=False, printNotFitted=False):
        if self.isFitted[col]:
            if printFitted:
                print(f'{self.__repr__()} {col} is already fitted')
            return True
        if printNotFitted:
            print(f'{self.__repr__()} {col} is not fitted yet; fit it first')
        return False

    def fitCol(self, df, col):
        self.assertColNameInDf(df, col)
        if self.isColFitted(col, printFitted=True):
            return
        self.scalers[col].fit(df[col])
        self.isFitted[col]=True

    def fit(self, df):
        for col in self.colNames:
            self.fitCol(df, col)

    def fitNTransformCol(self, df, col):
        self.assertColNameInDf(df, col)
        if self.isColFitted(col, printFitted=True):
            return
        self.fitCol(df, col)
        df[col] = self.transformCol(df, col)

    def transform(self, df):
        for col in self.colNames:
            df[col]=self.transformCol(df, col)

    def fitNTransform(self, df):
        for col in self.colNames:
            self.fitNTransformCol(df, col)

    def inverseMiddleTransformCol(self, df, col):
        if not self.isColFitted(col, printNotFitted=True):
            return df[col]
        dataToInverseTransformed=df[col]
        return self.scalers[col].inverseTransform(dataToInverseTransformed)

    def inverseTransformCol(self, dataToInverseTransformed, col):
        if not self.isColFitted(col, printNotFitted=True):
            return dataToInverseTransformed
        dataToInverseTransformed = self.inverseMiddleTransformCol(dataToInverseTransformed, col)
        if hasattr(self, 'intLabelsStrings'):#kkk does this part have tests
            if col in self.intLabelsStrings.keys():
                dataToInverseTransformed = self.intLabelsStrings[col].inverseTransform(dataToInverseTransformed)
        return dataToInverseTransformed

class SingleColsStdNormalizer(BaseSingleColsNormalizer):
    def __init__(self, colNames:list):
        self.scalers={col:StdScaler(f'std{col}') for col in colNames}
        super().__init__()

    def transformCol(self, df, col):
        self.assertColNameInDf(df, col)
        if not self.isColFitted(col, printNotFitted=True):
            return df[col]
        return self.scalers[col].transform(df[col])

    def __repr__(self):
        return f"SingleColsStdNormalizer+{'_'.join(self.colNames)}"

class SingleColsLblEncoder(BaseSingleColsNormalizer):
    def __init__(self, colNames:list):
        self.intLabelsStrings={}
        self.encoders={col:LblEncoder(f'lbl{col}') for col in colNames}
        super().__init__()

    @property
    def scalers(self):
        return self.encoders

    def fitCol(self, df, col):
        try:
            super().fitCol(df, col)
        except ValueError as e:
            if str(e) == LblEncoder.LblEncoderValueErrorMsg:
                self.intLabelsStrings[col]=IntLabelsString(col)
                self.intLabelsStrings[col].fit(df[col])
                intLabelsStringsTransformed = self.intLabelsStrings[col].transform(df[col])
                self.scalers[col].fit(intLabelsStringsTransformed)
                self.isFitted[col]=True

    def transformCol(self, df, col):
        self.assertColNameInDf(df, col)
        if not self.isColFitted(col, printNotFitted=True):
            return df[col]
        if col in self.intLabelsStrings.keys():
            df[col] = self.intLabelsStrings[col].transform(df[col])
        return self.scalers[col].transform(df[col])

    def getClasses(self):
        return {col:enc.encoder.classes_ for col,enc in self.encoders.items()}

    def __repr__(self):
        return f"SingleColsLblEncoder+{'_'.join(self.colNames)}"
#%% normalizers: BaseMultiColNormalizers
class BaseMultiColNormalizer(BaseNormalizerChecks):
    def __init__(self):
        self.isFitted=False

    def assertColNames(self, df):
        for col in self.colNames:
            self.assertColNameInDf(df, col)

    def areTheseIntCols(self, df):
        return df[self.colNames].apply(lambda col: col.apply(lambda x: isinstance(x, int))).all().all()
    
    def isFittedFunc(self, printFitted=False, printNotFitted=False):
        if self.isFitted:
            if printFitted:
                print(f'{self.__repr__()} is already fitted')
            return True
        if printNotFitted:
            print(f'{self.__repr__()} is not fitted yet; fit it first')
        return False

    def transformCol(self, df, col):
        self.assertColNameInDf(df, col)
        if not self.isFittedFunc(printNotFitted=True):
            return df[col]
        res=df[col]
        if isinstance(self, MultiColLblEncoder) and self.intLabelsString:#kkk oop
            res=self.intLabelsString.transform(res)
        return self.scaler.transform(res)

    def transform(self, df):
        self.assertColNames(df)
        for col in self.colNames:
            df[col] = self.transformCol(df, col)

    def fit(self, df):
        self.assertColNames(df)
        if self.isFittedFunc(printFitted=True):
            return
        dfColsCopy=df[self.colNames].copy()#kkk is copying needed
        if isinstance(self, MultiColLblEncoder) and self.areTheseIntCols(df):#kkk oop
            self.intLabelsString=IntLabelsString(self.shortRep())
            self.intLabelsString.fit(dfColsCopy)
            dfColsCopy=self.intLabelsString.transform(dfColsCopy)
        self.scaler.fit(dfColsCopy)
        self.isFitted=True

    def fitNTransform(self, df):
        if self.isFittedFunc(printFitted=True):
            return
        self.fit(df)
        self.transform(df)

    #kkk could have add many fit, transform, assert and their other combinations for single col
    #kkk could have added inverseMiddleTransform and inverseTransform which does inverse on self.colNames in df

    def inverseMiddleTransformCol(self, df, col):
        if not self.isFittedFunc(printNotFitted=True):
            return df[col]
        dataToInverseTransformed=df[col]
        return self.scaler.inverseTransform(dataToInverseTransformed)

    def inverseTransformCol(self, dataToInverseTransformed, col):
        assert col in dataToInverseTransformed.columns,'inverseTransformCol "{self}" "{col}" col is not in df columns'
        if not self.isFittedFunc(printNotFitted=True):
            return dataToInverseTransformed[col]
        res = self.inverseMiddleTransformCol(dataToInverseTransformed, col)
        if isinstance(self, MultiColLblEncoder) and self.intLabelsString:
            res = self.intLabelsString.inverseTransform(res)#kkk does the singlecol has done intlabel invTrans after its main transform
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
        self.intLabelsString = None
        self.colNames = colNames
        self.encoder = LblEncoder(self.shortRep())

    @property
    def scaler(self):
        return self.encoder

    def getClasses(self):
        return self.encoder.encoder.classes_

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
        return '_'.join([str(item) for item in self.defDict.values()])
    
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
        comboObjs = []

        for groupName, groupDf in df.groupby(self.mainGroupColNames):
            comboDict = dict(zip(self.mainGroupColNames, groupName))
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
#%% MainGroupSingleColsNormalizer
class MainGroupSingleColsNormalizer(MainGroupBaseNormalizer, BaseNormalizerChecks):
    def __init__(self, classType, df, mainGroupColNames, colNames:list):
        super().__init__(df, mainGroupColNames)
        self.colNames=colNames
        self.container={}
        for col in colNames:
            self.container[col]={}
            for combo in self.uniqueCombos:
                self.container[col][combo.shortRepr_()]=classType([col])

    def fitCol(self, df, col):
        for combo in self.uniqueCombos:
            dfToFit=self.getRowsByCombination(df, combo)
            dfToFit=dfToFit.reset_index(drop=True)#kkk does it need reset_index
            self.container[col][combo.shortRepr_()].fit(dfToFit)

    def fit(self, df):
        for col in self.colNames:
            self.fitCol(df, col)

    def transformCol(self, df, col):
        dfCopy=df.copy()
        for combo in self.uniqueCombos:
            dfToFit=self.getRowsByCombination(df, combo)
            inds=dfToFit.index
            dfToFit=dfToFit.reset_index(drop=True)#kkk does it need reset_index
            self.container[col][combo.shortRepr_()].transform(dfToFit)
            dfToFit.index=inds
            dfCopy.loc[inds,col]=dfToFit
        return dfCopy[col]

    def transform(self, df):
        for col in self.colNames:
            df[col]=self.transformCol(df, col)

    def fitNTransform(self, df):
        self.fit(df)
        self.transform(df)

    def inverseTransformColBase(self, df, col, funcName):
        dfCopy=df.copy()
        for combo in self.uniqueCombos:
            dfToFit=self.getRowsByCombination(df, combo)
            inds=dfToFit.index
            dfToFit=dfToFit.reset_index(drop=True)
            func = getattr(self.container[col][combo.shortRepr_()], funcName)
            invRes=func(dfToFit, col)
            dfCopy.loc[inds,col]=invRes
        return dfCopy[col]

    def inverseMiddleTransformCol(self, df, col):
        return self.inverseTransformColBase(df, col, 'inverseMiddleTransformCol')

    def inverseTransformCol(self, df, col):
        return self.inverseTransformColBase(df, col, 'inverseTransformCol')

class MainGroupSingleColsStdNormalizer(MainGroupSingleColsNormalizer):
    def __init__(self, df, mainGroupColNames, colNames:list):
        super().__init__(SingleColsStdNormalizer, df, mainGroupColNames, colNames)

    def getMeanNStd(self, df):
        for col in self.colNames:
            for combo in self.uniqueCombos:
                dfToFit=self.getRowsByCombination(df, combo)
                inds=dfToFit.index
                dfToFit=dfToFit.reset_index(drop=True)
                self.container[col][combo.shortRepr_()].fitNTransform(dfToFit)
                dfToFit.index=inds
                df.loc[inds,col]=dfToFit

#kkk normalizer=NormalizerStack(SingleColsLblEncoder(['sku', 'month', 'agency', *specialDays]), MainGroupSingleColsStdNormalizer(df, mainGroups, target))
#... normalizer.fitNTransform(df)
#... this wont work because the unqiueCombos in MainGroupSingleColsStdNormalizer are determined first and after fitNTransform
#... of SingleColsLblEncoder, values of mainGroups are changed
#... kinda correct way right now: normalizer=NormalizerStack(MainGroupSingleColsStdNormalizer(df, mainGroups, target), SingleColsLblEncoder(['sku', 'agency', 'month', *specialDays]))
#kkk for this problem initing all normalizers in init of NormalizerStack doesnt seem to be a good solution
#kkk add test for this
    def __repr__(self):
        return f"MainGroupSingleColsStdNormalizer+{'_'.join(list(map(str, self.uniqueCombos)))}+{'_'.join(self.colNames)}"

class MainGroupSingleColsLblEncoder(MainGroupSingleColsNormalizer):
    "this the lblEncoder version of MainGroupSingleColsStdNormalizer; its rarely useful, but in some case maybe used"
    def __init__(self, df, mainGroupColNames, colNames:list):
        super().__init__(SingleColsLblEncoder, df, mainGroupColNames, colNames)
    #kkk maybe add getClasses()

    def __repr__(self):
        return f"MainGroupSingleColsLblEncoder+{'_'.join(list(map(str, self.uniqueCombos)))}+{'_'.join(self.colNames)}"