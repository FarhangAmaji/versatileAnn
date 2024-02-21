from inspect import getmro

import torch
import torch.optim.lr_scheduler as LrScheduler
from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks.lr_monitor import LearningRateMonitor


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
