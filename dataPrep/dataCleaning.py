from utils.vAnnGeneralUtils import NpDict
import torch
import pandas as pd
import numpy as np
#%%
def noNanOrNoneData(data):
    if isinstance(data, (pd.DataFrame, pd.Series)):
        if data.isna().any().any():
            raise ValueError("The DataFrame contains NaN values.")
    
    elif isinstance(data, np.ndarray):
        if np.isnan(data).any():
            raise ValueError("The NumPy array contains NaN values.")

    elif isinstance(data, NpDict):
        data=NpDict[:]
        if np.isnan(data).any():
            raise ValueError("The NumPy array contains NaN values.")

    elif isinstance(data, torch.Tensor):
        if torch.isnan(data).any().item():
            raise ValueError("The PyTorch tensor contains NaN values.")