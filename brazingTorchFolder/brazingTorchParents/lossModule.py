from typing import List, Optional

import torch.nn as nn

from brazingTorchFolder.brazingTorchParents.innerClassesWithoutPublicMethods.loss_inner import \
    _BrazingTorch_loss_inner
from projectUtils.misc import _allowOnlyCreationOf_ChildrenInstances
from projectUtils.typeCheck import argValidator


class _BrazingTorch_loss(_BrazingTorch_loss_inner):
    def __init__(self, lossFuncs: Optional[List[nn.modules.loss._Loss]] = None, **kwargs):
        self.lossFuncs = lossFuncs or []
        # not allowing this class to have direct instance
        _allowOnlyCreationOf_ChildrenInstances(self, _BrazingTorch_loss)

    @property
    def lossFuncs(self):
        return self._lossFuncs

    @lossFuncs.setter
    @argValidator
    def lossFuncs(self, value: List[nn.modules.loss._Loss]):
        # bugPotn1
        #  when we pass [] doesn't give error
        self._lossFuncs = value
