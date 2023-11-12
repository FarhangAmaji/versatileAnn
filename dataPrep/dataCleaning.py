import numpy as np
import pandas as pd
import torch

from utils.vAnnGeneralUtils import NpDict


# ---
def noNanOrNoneDf(data):
    assert isinstance(data, (pd.DataFrame, pd.Series))
    if data.isna().any().any():
        raise ValueError("The DataFrame contains NaN values.")


def noNanOrNoneNpArray(data):
    assert isinstance(data, np.ndarray)
    if np.isnan(data).any():
        raise ValueError("The NumPy array contains NaN values.")


def noNanOrNoneNpDict(data):
    assert isinstance(data, NpDict)
    if data.df.isna().any().any():
        raise ValueError("The NpDict contains NaN values.")


def noNanOrNoneTensor(data):
    assert isinstance(data, torch.Tensor)
    if torch.isnan(data).any().item():
        raise ValueError("The PyTorch tensor contains NaN values.")


def noNanOrNoneData(data):
    if isinstance(data, (pd.DataFrame, pd.Series)):
        noNanOrNoneDf(data)
    elif isinstance(data, np.ndarray):
        noNanOrNoneNpArray(data)
    elif isinstance(data, NpDict):
        noNanOrNoneNpDict(data)
    elif isinstance(data, torch.Tensor):
        noNanOrNoneTensor(data)
