# versatileAnn\layers\customLayers.py
from typing import Optional, Callable

import torch.nn as nn

from utils.typeCheck import argValidator
from versatileAnn.utils import LossRegularizator


class VAnnCustomLayer(nn.Module):
    @argValidator
    def __init__(self, innerSize: int, outterSize: int,
                 activation: Optional[Callable] = None,
                 dropoutRate: float = None, normalization: str = None,
                 regularization=None):
        # cccUsage
        #  note the order is 1. main layer 2. normalization 3. activation 4. dropout
        super(VAnnCustomLayer, self).__init__()
        self.innerSize = innerSize
        self.outterSize = outterSize
        self.dropoutRate = dropoutRate  # goodToHave2 if dropout changes also change dropout layer

        if regularization:
            self.regularization = LossRegularizator(regularization)
        else:
            # cccDevAlgo
            #  note 'LossRegularizator(None)' is not set, and 'None' is set.
            #  so this way is more efficient and in line #Llr1 we don't assume it has a
            #  regularization and we pass by it
            self.regularization = None

        layers = [nn.Linear(innerSize, outterSize)]

        if normalization is not None:
            if normalization not in ['batch', 'layer']:
                raise ValueError(f"Invalid normalization option: {normalization}")

            if normalization == 'batch':
                normLayer = nn.BatchNorm1d(outterSize)
            else:
                normLayer = nn.LayerNorm(outterSize)
            layers.append(normLayer)

        if activation is not None:
            layers.append(activation)

        if dropoutRate:
            if not 0 < dropoutRate < 1:
                raise ValueError(f"dropoutRate={dropoutRate} is not between 0 and 1")
            drLayer = nn.Dropout(p=dropoutRate)
            layers.append(drLayer)

        self.layer = nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class linLReluNormDropout(VAnnCustomLayer):
    def __init__(self, innerSize, outterSize, leakyReluNegSlope=0.05, dropoutRate=None,
                 normalization='layer', regularization=None):
        # goodToHave2
        #  think about removing this
        activation = nn.LeakyReLU(negative_slope=leakyReluNegSlope)
        super(linLReluNormDropout, self).__init__(innerSize, outterSize, activation, dropoutRate,
                                                  normalization, regularization=regularization)


class linLSigmoidNormDropout(VAnnCustomLayer):
    def __init__(self, innerSize, outterSize, dropoutRate=None, normalization='layer',
                 regularization=None):
        # goodToHave2
        #  think about removing this
        activation = nn.Sigmoid()
        super(linLSigmoidNormDropout, self).__init__(innerSize, outterSize, activation, dropoutRate,
                                                     normalization, regularization=regularization)
