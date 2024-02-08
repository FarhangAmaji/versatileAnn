import torch
from torch import nn

from utils.customErrors import ImplementationError
from utils.typeCheck import argValidator
from utils.vAnnGeneralUtils import _allowOnlyCreationOf_ChildrenInstances
from utils.warnings import Warn


class _NewWrapper_specialModes:
    VAEMode_implementationsMsg = "with VAEMode, the forward method " + \
                                 "must return normalForwardOutputs, mean, logvar"
    # goodToHave2
    #  should implement detecting if the forward returns 3 outputs

    @argValidator
    def __init__(self, dropoutEnsembleMode: bool = False, VAEMode: bool = False,
                 dropoutEnsemble_samplesNum=100, **kwargs):
        # VAE stands for variationalAutoEncoder
        # not allowing this class to have direct instance
        _allowOnlyCreationOf_ChildrenInstances(self, _NewWrapper_specialModes)
        # kkk these mode values should be used in initArgs and model save and load

        self.dropoutEnsembleMode = dropoutEnsembleMode
        self.dropoutEnsemble_samplesNum = dropoutEnsemble_samplesNum

        self.VAEMode = VAEMode
        if VAEMode:
            Warn.info(self.VAEMode_implementationsMsg)
            self.autoEncoder_KlDivergenceFunc = self.klDivergence_normalDistributionLoss
            Warn.info('klDivergence_normalDistributionLoss is set as klDivergenceFunc.' +
                      '\nYou can change it by setting self.autoEncoder_KlDivergenceFunc to another function.')
        else:
            self.autoEncoder_KlDivergenceFunc = None

    def specialModesStep(self, batch, phase):
        # addTest1
        # cccAlgo
        #  this is a method for train/val/test/predict steps when special modes
        #  like VAEMode, dropoutEnsembleMode are applied
        # bugPotentialCheck1
        #  note this method should always be similar to commonStep
        #  so check commonStep and make similar changes here specially for the comments
        # cccUsage
        #  note we may need to if you are using variationalEncoderModes or dropoutEnsembleMode
        #  you may want to reimplement this method
        #  so take a look at this method to get an idea how to reimplement it yourself
        # reset tempVarStep
        self.resetTempVar_step(phase)

        inputs, targets = batch
        if phase == 'train':
            forwardOutputs = self.forward(inputs, targets)
            forwardOutputs, mean, logvar = self._unpackForwardOutputs_autoEncoder(forwardOutputs)
        else:  # inference phases like eval/test/predict
            self._activateDropouts_forEnsembleMode()
            # _forward_specialModes_inferencePhases also involves
            # VAEMode and dropoutEnsembleMode False
            forwardOutputs, mean, logvar = self._forward_specialModes_inferencePhases(inputs,
                                                                                      targets)

        # calculate loss
        # bugPotentialCheck1
        #  if the loss is not returned from _calculatedLosses because of
        #  not having self.lossFuncs would it make error
        loss, calculatedLosses = self._calculateLosses(forwardOutputs, targets)
        loss = self._addKlDivergence_toLoss(loss, mean, logvar)

        # Log losses
        self._logLosses(calculatedLosses, phase)
        # cccUsage
        #  Please ensure that your `training_step` method in PyTorch Lightning
        #  returns either the loss value directly or a dictionary containing
        #  the loss value under the key "loss". This is essential for the
        #  training process specially back propagation to function correctly.
        return loss

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

    @staticmethod
    def klDivergence_normalDistributionLoss(mean, logvar):
        klLoss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
        return klLoss

    @staticmethod
    def reparameterize(mean, logvar):
        # cccUsage
        #  this is not used in main commonStep but this is
        #  defined here so the user can use it in the forward method
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mean + eps * std
        return z

    # ---- methods related to dropoutEnsemble

    def _activateDropouts_forEnsembleMode(self):
        # kkk addTest1
        # cccDevStruct
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
        # cccAlgo
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
