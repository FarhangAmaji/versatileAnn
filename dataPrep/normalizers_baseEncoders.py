from abc import ABC, abstractmethod
from typing import Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

from utils.typeCheck import argValidator
from utils.vAnnGeneralUtils import areItemsOfList1_InList2
from utils.warnings import Warn

forceMsgAdvice = " you may use 'force' option to force it."


# goodToHave2 split to NSeries could have been some baseNormalizer and combine its reverse
class BaseEncoder(ABC):
    @abstractmethod
    def __init__(self, name=None):
        ...

    def _isFitted(self):
        ...

    @abstractmethod
    def fit(self, dataToFit):
        ...

    @abstractmethod
    def transform(self, dataToFit):
        ...

    @abstractmethod
    def inverseTransform(self, dataToBeInverseTransformed):
        ...


class StdScaler(BaseEncoder):
    def __init__(self, name=None):
        self.name = name
        self.scaler = StandardScaler()

    @property
    def _isFitted(self):
        return hasattr(self.scaler, 'mean_') and self.scaler.mean_ is not None

    @argValidator
    def fit(self, dataToFit: Union[pd.DataFrame, pd.Series], colShape=1):
        if not self._isFitted:
            self.scaler.fit(dataToFit.values.reshape(-1, colShape))
        else:
            Warn.warn(f'StdScaler {self.name} is already fitted')

    @argValidator
    def transform(self, dataToFit: Union[pd.DataFrame, pd.Series], colShape=1):
        dataToFit_ = dataToFit.values.reshape(-1, colShape)
        if self._isFitted:
            return self.scaler.transform(dataToFit_)
        else:
            Warn.warn(f'StdScaler {self.name} skipping transform: is not fitted; fit it first')
            # addTest2 this is not in the tests; same for other already fitted cases
            return dataToFit_

    @argValidator
    def inverseTransform(self, dataToBeInverseTransformed: Union[
        pd.DataFrame, pd.Series], colShape=1):
        dataToBeInverseTransformed_ = dataToBeInverseTransformed.values.reshape(-1, colShape)
        if self._isFitted:
            return self.scaler.inverse_transform(dataToBeInverseTransformed_)
        else:
            Warn.warn(f'StdScaler {self.name} is not fitted; cannot inverse transform.')
            return dataToBeInverseTransformed_


class LblEncoder(BaseEncoder):
    # mustHave2 it cant handle None or np.nan or other common missing values
    intDetectedErrorMsg = "Integer labels detected. Use IntLabelsString to convert them to string labels."
    floatDetectedErrorMsg = "Float labels detected. for using float data as categories everything should be done manually by urself; also dont forget to do inverse transform."

    def __init__(self, name=None):
        self.name = name
        self.encoder = LabelEncoder()
        # cccDevAlgo
        #  note the LabelEncoder sorts its 'classes_' items(things fitted on);
        #  have this in mind in order to prevent potential bugs

    @property
    def _isFitted(self):
        return hasattr(self.encoder, 'classes_') and \
            self.encoder.classes_ is not None

    @argValidator
    def fit(self, dataToFit: Union[pd.DataFrame, pd.Series]):
        if not self._isFitted:
            # cccAlgo
            #  'sklearn.LabelEncoder' fits both int, float alongside with strings.
            #  in order not to accidentally retransfrom already transformed data,
            #  in this project if there are ints which are meant to be categories, there is a need to utilize IntLabelsString before.
            #  for float data as categories everything needs to be handled manually before and after operations.
            if any(isinstance(label, float) for label in dataToFit):
                raise ValueError(LblEncoder.floatDetectedErrorMsg)
            elif any(isinstance(label, int) for label in dataToFit):
                raise ValueError(LblEncoder.intDetectedErrorMsg)
            self.encoder.fit(dataToFit.values.reshape(-1))
        else:
            print(f'LblEncoder {self.name} is already fitted')

    def _doesDataSeemToBe_AlreadyTransformed(self, data):
        return areItemsOfList1_InList2(np.unique(data), list(range(len(self.encoder.classes_))))

    @argValidator
    def transform(self, dataToFit: Union[pd.DataFrame, pd.Series],
                  force=False):
        # cccUsage
        #  Warning: the transform/inverseTransform/inverseMiddleTransform in general can be applied multiple times; so if the data may differ as wanted.
        #  in LblEncoder has been tried to reduce this risk, but again there may be no guarantee
        dataToFit_ = dataToFit.values.reshape(-1)
        if self._isFitted:
            if (not force) and self._doesDataSeemToBe_AlreadyTransformed(dataToFit_):
                Warn.warn(
                    f"LblEncoder {self.name} skipping transform: data already seems transformed." + forceMsgAdvice)
                return dataToFit_
            else:
                print(f'LblEncoder applied on {self.name}')
                return self.encoder.transform(dataToFit_)
        else:
            Warn.warn(f'LblEncoder {self.name} skipping transform: is not fitted; fit it first')
            return dataToFit_

    def _doesdataToBeInverseTransformed_SeemToBeAlreadyDone(self, data):
        return areItemsOfList1_InList2(np.unique(data), list(self.encoder.classes_))

    @argValidator
    def inverseTransform(self, dataToBeInverseTransformed: Union[
        pd.DataFrame, pd.Series], force=False):
        dataToBeInverseTransformed_ = dataToBeInverseTransformed.values.reshape(-1)
        if self._isFitted:
            if (not force) and self._doesdataToBeInverseTransformed_SeemToBeAlreadyDone(
                    dataToBeInverseTransformed_):
                Warn.warn(
                    f'LblEncoder {self.name} skipping inverse transform: data already seems inverse transformed.' + forceMsgAdvice)
                return dataToBeInverseTransformed_
            else:
                return self.encoder.inverse_transform(
                    dataToBeInverseTransformed_)
        else:
            Warn.warn(f'LblEncoder {self.name} is not fitted; cannot inverse transform.')
            return dataToBeInverseTransformed_


class IntLabelsString(BaseEncoder):
    def __init__(self, name):
        self.name = name
        self.isFitted = False

    @argValidator
    def fit(self, inputData: Union[pd.DataFrame, pd.Series]):
        # goodToHave1 add NpDict
        if self.isFitted:
            Warn.warn(f'skipping fit:{self.name} intLabelsString is already fitted')
            return inputData
        array = inputData.values.reshape(-1)
        uniqueVals = np.unique(array)
        if not np.all(np.equal(uniqueVals, uniqueVals.astype(int))):
            raise ValueError(f"IntLabelsString {self.name} All values should be integers.")
        self.intToLabelMapping = {intVal: f'{self.name}:{label}' for
                                  label, intVal in enumerate(uniqueVals)}
        self.labelToIntMapping = {value: key for key, value in
                                  self.intToLabelMapping.items()}
        self.isFitted = True

    def fitNTransform(self, inputData):
        self.fit(inputData)
        return self.transform(inputData)

    @argValidator
    def transform(self, dataToTransformed: Union[pd.DataFrame, pd.Series]):
        if isinstance(dataToTransformed, pd.Series):
            output = dataToTransformed.map(self.intToLabelMapping)
        elif isinstance(dataToTransformed, pd.DataFrame):
            output = dataToTransformed.applymap(
                lambda x: self.intToLabelMapping.get(x, x))
        return output

    @argValidator
    def inverseTransform(self, dataToBeInverseTransformed: np.ndarray):
        return np.vectorize(self.labelToIntMapping.get)(
            dataToBeInverseTransformed)
