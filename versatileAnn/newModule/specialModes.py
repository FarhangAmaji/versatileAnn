import torch
from torch import nn

from utils.customErrors import ImplementationError
from utils.typeCheck import argValidator
from utils.vAnnGeneralUtils import _allowOnlyCreationOf_ChildrenInstances
from utils.warnings import Warn


class _NewWrapper_specialModes:
    VAEMode_implementationsMsg = "with VAEMode, the forward method " + \
                                 "must return normalForwardOutputs, mean, logvar"

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
        #   you may want to reimplement this method
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
