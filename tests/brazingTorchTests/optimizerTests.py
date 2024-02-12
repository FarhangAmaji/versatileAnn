import unittest

import torch
from torch import nn

from tests.baseTest import BaseTestClass
from brazingTorchFolder.brazingTorch import BrazingTorch


class brazingTorchTests_optimizer(BaseTestClass):
    def setUp(self):
        class ChildClass(BrazingTorch):
            def __init__(self, **kwargs):
                self.l1 = nn.Linear(1, 1)

            def forward(self, inputs, targets):
                pass

        return ChildClass

    def setupSgdOpt(self):
        ChildClass = self.setUp()
        model = ChildClass()
        model.optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9,
                                          dampening=0, weight_decay=0, nesterov=False)
        return model

    def testDefaultOptimizerSetup(self):
        def innerFunc():
            ChildClass = self.setUp()
            model = ChildClass(testPrints=True)
            self.assertTrue(isinstance(model.optimizer, torch.optim.Adam))

        expectedPrint = """BrazingTorch __new__ method initiated for "ChildClass" class
_BrazingTorch_postInit func
ChildClass
"""
        self.assertPrint(innerFunc, expectedPrint)

    def test_OptimizerInitArgs1(self):
        ChildClass = self.setUp()
        model = ChildClass()
        model.optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999),
                                           eps=1e-08)

        self.assertEqual(model._optimizerInitArgs['type'], torch.optim.Adam)
        initArgs_args = {'lr': 0.0001, 'betas': (0.9, 0.999), 'eps': 1e-08, 'weight_decay': 0,
                         'amsgrad': False, 'maximize': False, 'foreach': None, 'capturable': False,
                         'differentiable': False, 'fused': None}
        self.assertEqual(model._optimizerInitArgs['args'], initArgs_args)

    def test_OptimizerInitArgs2(self):
        model = self.setupSgdOpt()

        self.assertEqual(model._optimizerInitArgs['type'], torch.optim.SGD)
        initArgs_args = {'lr': 0.0001, 'momentum': 0.9, 'dampening': 0, 'weight_decay': 0,
                         'nesterov': False, 'maximize': False, 'foreach': None,
                         'differentiable': False}
        self.assertEqual(model._optimizerInitArgs['args'], initArgs_args)

    def test_changingSelfLr_changesOptimizerLr(self):
        model = self.setupSgdOpt()

        model.lr = .007
        self.assertEqual(model.optimizer.param_groups[0]['lr'], model.lr)

    def test_OptimizerReset_keepLr(self):
        model = self.setupSgdOpt()

        model.lr = .007
        model.resetOptimizer()
        self.assertEqual(model.optimizer.param_groups[0]['lr'], model.lr)

    def test_OptimizerReset_noKeepLr(self):
        model = self.setupSgdOpt()

        model.lr = .007
        model.resetOptimizer(keepLr=False)
        self.assertEqual(model.optimizer.param_groups[0]['lr'], 0.0001)


# ---- run test
if __name__ == '__main__':
    unittest.main()
