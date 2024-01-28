from typing import Union

import numpy as np
import pandas as pd
import torch

from utils.typeCheck import argValidator
from utils.vAnnGeneralUtils import NpDict


# ---
@argValidator
def noNanOrNoneDf(data: Union[pd.DataFrame, pd.Series]):
    if data.isna().any().any():
        raise ValueError("The DataFrame contains NaN values.")


@argValidator
def noNanOrNoneNpArray(data: np.ndarray):
    if np.isnan(data).any():
        raise ValueError("The NumPy array contains NaN values.")


@argValidator
def noNanOrNoneNpDict(data: NpDict):
    if data.df.isna().any().any():
        raise ValueError("The NpDict contains NaN values.")


@argValidator
def noNanOrNoneTensor(data: torch.Tensor):
    if torch.isnan(data).any().item():
        raise ValueError("The PyTorch tensor contains NaN values.")

def noNanOrNoneData(data: Union[torch.Tensor, NpDict, np.ndarray, pd.DataFrame, pd.Series]):
    if isinstance(data, (pd.DataFrame, pd.Series)):
        noNanOrNoneDf(data)
    elif isinstance(data, np.ndarray):
        noNanOrNoneNpArray(data)
    elif isinstance(data, NpDict):
        noNanOrNoneNpDict(data)
    elif isinstance(data, torch.Tensor):
        noNanOrNoneTensor(data)
    # else:
    #     raise ValueError('noNanOrNoneData only gets torch.Tensor, NpDict, np.ndarray, pd.DataFrame or pd.Series')
