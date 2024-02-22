import os
from inspect import getmro
from inspect import isclass
from typing import List, Optional

import pytorch_lightning as pl
import torch
import torch.optim.lr_scheduler as LrScheduler
from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.lr_monitor import LearningRateMonitor
from pytorch_lightning.loggers import Logger
from torch import nn
from torch.utils.data import DataLoader

from brazingTorchFolder.callbacks import StoreEpochData
from projectUtils.dataTypeUtils.tensor import getTorchDevice
from projectUtils.misc import inputTimeout
from projectUtils.typeCheck import argValidator
from projectUtils.warnings import Warn


def loadFromCheckpointPath(checkpointPath, ModelClassOrInstance):
    # bugPotentialCheck1
    #  note a normal checkpoint dictionary has these keys ['epoch', 'global_step',
    #  'pytorch-lightning_version', 'state_dict', 'loops', 'callbacks',
    #  'optimizer_states', 'lr_schedulers']
    #  - 'optimizer_states' and 'lr_schedulers' are used below and 'pytorch-lightning_version'
    #  is not gonna be used but I have doubts about the rest of the keys

    if not os.path.exists(checkpointPath):
        raise ValueError(f"{checkpointPath=} doesn't exist")

    # Load checkpoint
    checkpoint = torch.load(checkpointPath)

    # Load state_dict into model
    if isclass(ModelClassOrInstance):
        model = ModelClassOrInstance.load_from_checkpoint(checkpoint_path=checkpointPath)
    else:
        model = type(ModelClassOrInstance).load_from_checkpoint(
            checkpoint_path=checkpointPath)

    # Load optimizer and schedulers states
    if 'optimizer_states' in checkpoint:
        model.optimizer.load_state_dict(checkpoint['optimizer_states'])
    if 'lr_schedulers' in checkpoint:
        model.schedulers = checkpoint['lr_schedulers']

    return model


def isPytorchLightningScheduler(obj):
    """
    Robustly checks if an object can be used as a PyTorch Lightning scheduler.

    This function combines the accuracy and flexibility of both Response A and Response B,
    adding error handling and a more comprehensive check for compatibility.

    Args:
        obj: The object to be checked.

    Returns:
        True if the object is a PyTorch Lightning scheduler, False otherwise.
    """

    if isinstance(obj, (LrScheduler._LRScheduler, LrScheduler.LambdaLR, LrScheduler.OneCycleLR,
                        LearningRateMonitor)):
        # Direct instance of PyTorch Lightning schedulers (preferred)
        return True
    elif isinstance(obj, LightningModule):
        # Check if LightningModule has a compatible custom schedulers implementation
        schedulerMethods = [
            methodName for methodName in dir(obj)
            if methodName in {'onTrainEpochStart', 'onTrainBatchEnd', 'onValidationEpochStart'}
        ]
        customSchedulerMethodFound = len(schedulerMethods) > 0
        lightningModuleHasScheduler = hasattr(obj,
                                              'lrScheduler') or 'optimizer.lrScheduler' in obj.__dict__
        return customSchedulerMethodFound and lightningModuleHasScheduler
    else:
        # Check if obj is a subclass of a supported base class
        try:
            # Try using getmro() for more reliable inheritance path checks
            for baseClass in getmro(obj):
                if baseClass in (
                        LrScheduler._LRScheduler, LrScheduler.LambdaLR, LrScheduler.OneCycleLR,
                        LightningModule):
                    return True
        except TypeError:  # Handle cases where getmro() might not be supported
            pass
        return False
