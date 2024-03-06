import torch
from torch import nn

from projectUtils.customErrors import ImplementationError
from projectUtils.misc import _allowOnlyCreationOf_ChildrenInstances
from projectUtils.typeCheck import argValidator


class _BrazingTorch_specialModes_inner:
    VAEMode_implementationsMsg = "with VAEMode, the forward method " + \
                                 "must return normalForwardOutputs, mean, logvar"

    @argValidator
    def __init__(self):
        # not allowing this class to have direct instance
        _allowOnlyCreationOf_ChildrenInstances(self, _BrazingTorch_specialModes_inner)

    # ---- methods related to VAEs
    def _unpackForwardOutputs_autoEncoder(self, forwardOutputs):
        if self.VAEMode:
            if len(forwardOutputs) != 3:
                raise ImplementationError(self.VAEMode_implementationsMsg)
            forwardOutputs, mean, logvar = forwardOutputs
            return forwardOutputs, mean, logvar
        else:
            return forwardOutputs, None, None

    def _addKlDivergence_toLoss(self, loss, mean, logvar):
        if self.VAEMode:
            return loss + self.autoEncoder_KlDivergenceFunc(mean, logvar)
        else:
            return loss

    # ---- methods related to dropoutEnsemble

    def _activateDropouts_forEnsembleMode(self):
        # kkk addTest1
        # ccc1
        #  - self.modules() recursively includes all submodules, so if model has some
        #  layers(as attributes) which are other classes inherited from nn.module
        #  and have some nn.Dropout in them are accessed with self.modules()
        #  - `.train()` when gets applied, only changes the behaviour of
        #  `dropout` and `batch normalization` layers
        if self.dropoutEnsembleMode:
            for module in self.modules():
                if isinstance(module, nn.Dropout):
                    module.train()

    # ---- methods related to dropoutEnsemble and VAEMode
    def _forward_specialModes_inferencePhases(self, inputs, outputs):
        # ccc1
        #  this is used in inference phases(eval/test/predict)
        mean, logvar = None, None
        if self.dropoutEnsembleMode:
            if self.VAEMode:
                # addTest1 for when forwardOutputs is a tensor or a dict
                forwardOutputsList = [[] for _ in range(3)]
                for x in [inputs] * self.dropoutEnsembleNumSamples:
                    output = self.forward(x)
                    [forwardOutputsList[i].append(output[i]) for i in range(3)]
                forwardOutputs, mean, logvar = tuple(
                    torch.stack(forwardOutputsList[i]).squeeze().mean(dim=0).unsqueeze(1) for i in
                    range(3))
            else:
                # addTest1 for when forwardOutputs is a tensor or a dict
                forwardOutputs = torch.stack(
                    tuple(
                        map(lambda x: self.forward(x), [inputs] * self.dropoutEnsembleNumSamples)))
                forwardOutputs = forwardOutputs.squeeze().mean(dim=0).unsqueeze(1)
        else:
            # addTest1 for when forwardOutputs is a tensor or a dict
            forwardOutputs = self.forward(inputs, outputs)
            forwardOutputs, mean, logvar = self._unpackForwardOutputs_autoEncoder(forwardOutputs)

        return forwardOutputs, mean, logvar
