import unittest

import pandas as pd
import torch
from torch import nn

from dataPrep.dataloader import VAnnTsDataloader
from dataPrep.dataset import VAnnTsDataset
from tests.baseTest import BaseTestClass
from versatileAnn.layers.customLayers import VAnnCustomLayer
from versatileAnn.newModule.brazingTorch import BrazingTorch


class RegularizationTests(BaseTestClass):
    def dataSetup(self, batch_size=5):
        # similar to DataloaderTests in dataloaderTests.py
        self.seed = 65

        class custom1Dataset(VAnnTsDataset):
            def __getitem__(self, idx):
                return torch.tensor(self.data['a'][idx]).unsqueeze(-1), \
                    torch.tensor(float(idx)).unsqueeze(-1)  # these are dummy values

        self.data = [float(i + 1000) for i in range(8, 73)]
        self.dataset = custom1Dataset(data=pd.DataFrame({'a': self.data}),
                                      backcastLen=0, forecastLen=0)
        self.dataloader = VAnnTsDataloader(self.dataset, phase='train', batch_size=batch_size,
                                           shuffle=True, randomSeed=self.seed)

    def testWithVAnnCustomLayer(self):
        # has a VAnnCustomLayer and a normal Linear layer with no regularization
        class NNDummy(BrazingTorch):
            def __init__(self):
                self.lay1 = VAnnCustomLayer(1, 7, regularization={'type': 'l1', 'value': 0.03})
                self.lay2 = nn.Linear(7, 5)

            def forward(self, inputs, targets):
                return self.lay2(self.lay1(inputs))

        self.dataSetup()
        model = NNDummy(lossFuncs=[nn.MSELoss(), nn.L1Loss()])
        kwargsApplied = {'max_epochs': 3, 'enable_checkpointing': False, }

        # checks that model runs also
        trainer = model.fit(self.dataloader, None, **kwargsApplied)

        model._setOperationalRegularizations()
        self.assertEqual(str(model._operationalRegularizations['lay1']),
                         "LossRegularizator{'type': 'l1', 'value': 0.03}")
        self.assertEqual(str(model._operationalRegularizations['lay2']),
                         "LossRegularizator{'type': 'l2', 'value': 0.001}")

    def test_addLayerRegularization_inModelDefinition(self):
        # uses addLayerRegularization in __init__ of model
        # also has a VAnnCustomLayer
        class NNDummy(BrazingTorch):
            def __init__(self):
                self.lay1 = VAnnCustomLayer(1, 7, regularization={'type': 'l1', 'value': 0.03})
                self.lay2 = nn.Linear(7, 5)
                # adding addLayerRegularization in model __init__
                self.addLayerRegularization({self.lay2: {'type': 'l1', 'value': .042}})

            def forward(self, inputs, targets):
                return self.lay2(self.lay1(inputs))

        self.dataSetup()
        model = NNDummy(lossFuncs=[nn.MSELoss(), nn.L1Loss()])
        kwargsApplied = {'max_epochs': 3, 'enable_checkpointing': False, }

        # checks that model runs also
        trainer = model.fit(self.dataloader, None, **kwargsApplied)

        model._setOperationalRegularizations()
        self.assertEqual(str(model._operationalRegularizations['lay1']),
                         "LossRegularizator{'type': 'l1', 'value': 0.03}")
        self.assertEqual(str(model._operationalRegularizations['lay2']),
                         "LossRegularizator{'type': 'l1', 'value': 0.042}")

    def test_addLayerRegularization_outOfModelDefinition(self):
        # uses addLayerRegularization after __init__ of model
        # also has a VAnnCustomLayer
        class NNDummy(BrazingTorch):
            def __init__(self):
                self.lay1 = VAnnCustomLayer(1, 7, regularization={'type': 'l1', 'value': 0.03})
                self.lay2 = nn.Linear(7, 5)

            def forward(self, inputs, targets):
                return self.lay2(self.lay1(inputs))

        self.dataSetup()
        model = NNDummy(lossFuncs=[nn.MSELoss(), nn.L1Loss()])
        # adding addLayerRegularization after model initiation
        model.addLayerRegularization({model.lay2: {'type': 'l1', 'value': .042}})
        kwargsApplied = {'max_epochs': 3, 'enable_checkpointing': False, }

        # checks that model runs also
        trainer = model.fit(self.dataloader, None, **kwargsApplied)

        model._setOperationalRegularizations()
        self.assertEqual(str(model._operationalRegularizations['lay1']),
                         "LossRegularizator{'type': 'l1', 'value': 0.03}")
        self.assertEqual(str(model._operationalRegularizations['lay2']),
                         "LossRegularizator{'type': 'l1', 'value': 0.042}")


# ---- run test
if __name__ == '__main__':
    unittest.main()
