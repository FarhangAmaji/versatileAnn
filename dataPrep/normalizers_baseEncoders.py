from abc import ABC, abstractmethod
from typing import Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

from utils.typeCheck import argValidator
from utils.vAnnGeneralUtils import areItemsOfList1_InList2
from utils.warnings import Warn

forceMsgAdvice = " you may use 'force' option to force it."


# goodToHave2
#  split to NSeries could have been some _BaseEncoder and combine its reverse
#  note the normalizerstack should apply it first and inverse it last
class _BaseEncoder(ABC):
    @abstractmethod
    def __init__(self, name=None):
        ...

    def _isFitted(self):
        ...

    @abstractmethod
    def fit(self, data):
        ...

    @abstractmethod
    def transform(self, data):
        ...

    @abstractmethod
    def inverseTransform(self, data):
        ...


class _StdScaler(_BaseEncoder):
    def __init__(self, name=None):
        self.name = name
        self.scaler = StandardScaler()

    @property
    def _isFitted(self):
        # goodToHave3 name why these conditions are used
        return hasattr(self.scaler, 'mean_') and self.scaler.mean_ is not None

    @argValidator
    def fit(self, data: Union[pd.DataFrame, pd.Series]):
        # goodToHave2 add NpDict; to all these funcs in this file
        if not self._isFitted:
            self.scaler.fit(data.values.reshape(-1, 1))
        else:
            Warn.warn(f'StdScaler {self.name} is already fitted')
            # addTest2 this is not in the tests; same for other already fitted cases

    def _baseTransform(self, data: Union[pd.DataFrame, pd.Series], transformTypeName,
                       transformFunc, colShape=1):
        data_ = data.values.reshape(-1, colShape)
        if self._isFitted:
            return transformFunc(data_)
        else:
            Warn.warn(
                f'StdScaler {self.name} skipping {transformTypeName}: is not fitted; fit it first')
            return data_

    @argValidator
    def transform(self, data: Union[pd.DataFrame, pd.Series], colShape=1):
        return self._baseTransform(data, 'transform', self.scaler.transform, colShape)

    @argValidator
    def inverseTransform(self, data: Union[pd.DataFrame, pd.Series], colShape=1):
        return self._baseTransform(data, 'inverseTransform',
                                   self.scaler.inverse_transform, colShape)


class _LblEncoder(_BaseEncoder):
    # mustHave2 it cant handle None or np.nan or other common missing values
    intDetectedErrorMsg = "Integer labels detected. Use _IntLabelsString to convert them to string labels."
    floatDetectedErrorMsg = "Float labels detected. for using float data as categories everything should be done manually by urself; also dont forget to do inverse transform."

    def __init__(self, name=None):
        self.name = name
        self.encoder = LabelEncoder()
        # cccDevAlgo
        #  note the LabelEncoder sorts its 'classes_' items(things fitted on);
        #  have this in mind in order to prevent potential bugs

    @property
    def _isFitted(self):
        # goodToHave3 name why these conditions are used
        return hasattr(self.encoder, 'classes_') and \
            self.encoder.classes_ is not None

    @argValidator
    def fit(self, data: Union[pd.DataFrame, pd.Series]):
        if not self._isFitted:
            # cccAlgo
            #  'sklearn.LabelEncoder' fits both int, float alongside with strings.
            #  in order not to accidentally retransfrom already transformed data,
            #  in this project if there are `int`s which are meant to be `categories`, there is a need to utilize _IntLabelsString before.
            #  for `float` data as categories everything needs to be handled `manually` before and after operations.
            if any(isinstance(label, float) for label in data):
                raise ValueError(_LblEncoder.floatDetectedErrorMsg)
            elif any(isinstance(label, int) for label in data):
                raise ValueError(_LblEncoder.intDetectedErrorMsg)
            self.encoder.fit(data.values.reshape(-1))
        else:
            print(f'LblEncoder {self.name} is already fitted')

    def _doesDataSeemToBe_AlreadyTransformed(self, data):
        # bugPotentialCheck2 is it possible if the data is str and numeric together makes problem with 'list(range(len(self.encoder.classes_)))'
        return areItemsOfList1_InList2(np.unique(data), list(range(len(self.encoder.classes_))))

    def _doesdataToBeInverseTransformed_SeemToBeAlreadyDone(self, data):
        return areItemsOfList1_InList2(np.unique(data), list(self.encoder.classes_))

    def _baseTransform(self, data: Union[pd.DataFrame, pd.Series], transformTypeName,
                       transformFunc, TransformedAlreadyCheckFunc, force=False):
        # cccUsage
        #  Warning: the transform/inverseTransform in general can be applied multiple times; so if the data may differ as wanted.
        #  in _LblEncoder has been tried to reduce this risk, but again there may be no guarantee
        data_ = data.values.reshape(-1)
        if self._isFitted:
            if (not force) and TransformedAlreadyCheckFunc(data_):
                Warn.warn(
                    f"LblEncoder {self.name} skipping {transformTypeName}: data already seems transformed." + forceMsgAdvice)
                return data_
            else:
                print(f'LblEncoder applied {transformTypeName} on {self.name}')
                return transformFunc(data_)
        else:
            Warn.warn(
                f'LblEncoder {self.name} skipping {transformTypeName}: is not fitted; fit it first')
            return data_

    @argValidator
    def transform(self, data: Union[pd.DataFrame, pd.Series], force=False):
        return self._baseTransform(data, 'transform', self.encoder.transform,
                                   self._doesDataSeemToBe_AlreadyTransformed, force)

    @argValidator
    def inverseTransform(self, data: Union[pd.DataFrame, pd.Series], force=False):
        return self._baseTransform(data, 'inverseTransform', self.encoder.inverse_transform,
                                   self._doesdataToBeInverseTransformed_SeemToBeAlreadyDone, force)


class _IntLabelsString(_BaseEncoder):
    # cccAlgo same explanations of fit in _LblEncoder
    def __init__(self, name):
        self.name = name
        self.isFitted = False

    @argValidator
    def fit(self, data: Union[pd.DataFrame, pd.Series]):
        if self.isFitted:
            Warn.warn(f'skipping fit:{self.name} intLabelsString is already fitted')
            return
        array = data.values.reshape(-1)
        uniqueVals = np.unique(array)
        if not np.all(np.equal(uniqueVals, uniqueVals.astype(int))):
            raise ValueError(f"_IntLabelsString {self.name} All values should be integers.")
        self.intToLabelMapping = {intVal: f'{self.name}:{label}' for
                                  label, intVal in enumerate(uniqueVals)}
        self.labelToIntMapping = {value: key for key, value in
                                  self.intToLabelMapping.items()}
        self.isFitted = True

    def fitNTransform(self, data):
        self.fit(data)
        return self.transform(data)

    @argValidator
    def transform(self, data: Union[pd.DataFrame, pd.Series]):
        if isinstance(data, pd.Series):
            output = data.map(self.intToLabelMapping)
        elif isinstance(data, pd.DataFrame):
            output = data.applymap(
                lambda x: self.intToLabelMapping.get(x, x))
        else:
            output = None
            assert False, 'argValidator has not worked'
        return output

    @argValidator
    def inverseTransform(self, data: np.ndarray):
        return np.vectorize(self.labelToIntMapping.get)(data)
