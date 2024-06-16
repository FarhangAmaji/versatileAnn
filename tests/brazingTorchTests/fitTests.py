import os
import unittest
from unittest.mock import Mock

from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import Logger
from torch import nn

from brazingTorchFolder.brazingTorch import BrazingTorch
from commonDatasets.commonDatasetsPrep.epfFrBe import getEpfFrBeDataloaders, \
    dataInfo as epfFrBeDataInfo
from projectUtils.misc import getProjectDirectory
from tests.baseTest import BaseTestClass
from tests.utils import simulateInput


# ---- dummy classes to be used in tests
class NNDummy1(BrazingTorch):
    def __init__(self, **kwargs):
        # this just in order to see does it run of not so 1 neuron is enough
        self.l1 = nn.Linear(4, 1)
        self.l2 = nn.Linear(1, 4)

    def forward(self, inputs, targets):
        output = {}
        output['output'] = self.l2(self.l1(targets['output']))
        return output


class NNDummy2(NNDummy1):
    def __init__(self, **kwargs):
        self.l3 = nn.Linear(5, 1)


# ---- actual tests
class FitTestsSetup(BaseTestClass):
    readySetupsBySeed = {}

    def readySetupsChecker(self, seed):
        if seed in self.readySetupsBySeed:
            self._readySetupAssign(seed)
            return True
        return False

    def readySetupsAssigner(self, seed):
        if seed not in self.readySetupsBySeed:
            self._readySetupAssign(seed)

    def _readySetupAssign(self, seed):
        attributes = ['trainDataloader', 'valDataloader', 'testDataloader', 'normalizer',
                      'model']
        for attribute in attributes:
            setattr(self, attribute, self.readySetupsBySeed[seed][attribute])

    def setup(self, seed):
        self.seed = seed
        if self.readySetupsChecker(seed):
            return

        backcastLen = 7
        forecastLen = 4
        batchSize = 4
        dataInfo = epfFrBeDataInfo
        shuffle = False
        devTestMode = True
        self.trainDataloader, self.valDataloader, self.testDataloader, self.normalizer = getEpfFrBeDataloaders(
            backcastLen=backcastLen, forecastLen=forecastLen,
            batchSize=batchSize, shuffle=shuffle, dataInfo=dataInfo, devTestMode=devTestMode)
        # bugPotn1
        #  gives 'train is empty. the trainSeqLen seems to be high'; check is it giving warn
        #  with correct reason or not

        # ccc1
        #  note with define this modelName so the models get saved in modelName path and
        #  not 'NNDummy1'(className) which is default
        #  note also files for DetermineFitRunStateTests are put in
        #  getProjectDirectory(), 'tests', 'DetermineFitRunStateTests_mockSavedModels'
        self.model = NNDummy1(modelName='DetermineFitRunStateTests_mockSavedModels',
                              testPrints=True, seed=self.seed,
                              lossFuncs=[nn.MSELoss(), nn.L1Loss()])


class BaseFit_getBaseFit_appliedKwargsTests(FitTestsSetup):
    def test_basicCombination(self):
        self.setup(seed=71)
        kwargs = {'a': 1, 'b': 2}
        listOfKwargs = [{'c': 3}, {'d': 4}]

        def innerFunc(kwargs, listOfKwargs):
            return self.model._getBaseFit_appliedKwargs(kwargs, listOfKwargs)

        expectedPrint = """you have included "c" but it doesn't match with args can be passed to pl.Trainer, pl.Trainer.fit or pl.LightningModule.log; even their camelCase names
you have included "d" but it doesn't match with args can be passed to pl.Trainer, pl.Trainer.fit or pl.LightningModule.log; even their camelCase names
you have included "a" but it doesn't match with args can be passed to pl.Trainer, pl.Trainer.fit or pl.LightningModule.log; even their camelCase names
you have included "b" but it doesn't match with args can be passed to pl.Trainer, pl.Trainer.fit or pl.LightningModule.log; even their camelCase names
"""
        result, printed = self.assertPrint(innerFunc, expectedPrint, returnPrinted=True,
                                           **{'kwargs': kwargs, 'listOfKwargs': listOfKwargs})
        expected = {'trainer': {}, 'trainerFit': {}, 'log': {}}
        self.assertEqual(result, expected)

    def test_emptyInput(self):
        self.setup(seed=71)
        kwargs = {}
        listOfKwargs = []
        result = self.model._getBaseFit_appliedKwargs(kwargs, listOfKwargs)
        expected = {'trainer': {}, 'trainerFit': {}, 'log': {}}
        self.assertEqual(result, expected)

    def test_outputsBoolsCorrectly(self):
        self.setup(seed=71)
        kwargsApplied = {'max_epochs': 3, 'enable_checkpointing': False, }
        res = self.model._getBaseFit_appliedKwargs(kwargsApplied, [])
        expectedRes = {'trainer': {'enable_checkpointing': False, 'max_epochs': 3},
                       'trainerFit': {}, 'log': {}}
        self.assertEqual(res, expectedRes)


class BaseFit_plKwargUpdaterTests(FitTestsSetup):

    def test_emptyKwargs(self):
        self.setup(seed=71)

        appliedKwargs = {}
        kwarg = {}

        expected = {'trainer': {}, 'trainerFit': {}, 'log': {}}

        result = self.model._plKwargUpdater(appliedKwargs, kwarg)
        self.assertEqual(result, expected)

    def test_noOverlapInKwargs(self):
        self.setup(seed=71)

        appliedKwargs = {'trainer': {'max_epochs': 10}}
        kwarg = {'log': {'on_step': {'else': True}, 'prog_bar': True}}

        expected = {'trainer': {'max_epochs': 10}, 'trainerFit': {},
                    'log': {'on_step': {'else': True}, 'prog_bar': {'else': True}}}

        result = self.model._plKwargUpdater(appliedKwargs, kwarg)
        self.assertEqual(result, expected)

    def test_inputsUntouched(self):
        self.setup(seed=71)

        appliedKwargs = {'trainer': {'max_epochs': 10}}
        kwarg = {'log': {'on_step': {'else': True}, 'prog_bar': True}}
        appliedKwargsCopy = appliedKwargs.copy()
        kwargCopy = kwarg.copy()

        self.model._plKwargUpdater(appliedKwargs, kwarg)
        self.assertEqual(appliedKwargs, appliedKwargsCopy)
        self.assertEqual(kwarg, kwargCopy)

    def test_conflictingKwargs_nonListable(self):
        self.setup(seed=71)

        appliedKwargs = {'trainer': {'max_epochs': 10}, 'trainerFit': {'ckpt_path': 'path1'}}
        kwarg = {'trainer': {'max_epochs': 20}, 'trainerFit': {'ckpt_path': 'path2'}}

        expected = {'trainer': {'max_epochs': 20}, 'trainerFit': {'ckpt_path': 'path2'}, 'log': {}}

        result = self.model._plKwargUpdater(appliedKwargs, kwarg)
        self.assertEqual(result, expected)

    def test_conflictingListable_bothHaveLoggers(self):
        self.setup(seed=71)

        appliedKwargs = {'trainer': {'logger': [Mock(spec=Logger)]}}
        kwarg = {'trainer': {'logger': Mock(spec=Logger)}}

        expected = {'trainer': {
            'logger': appliedKwargs['trainer']['logger'] + [kwarg['trainer']['logger']]},
            'trainerFit': {}, 'log': {}}

        result = self.model._plKwargUpdater(appliedKwargs, kwarg)
        self.assertEqual(result, expected)

    def test_conflictingListable_bothHaveCallbacks(self):
        self.setup(seed=71)

        callback1 = Mock(spec=Callback)
        callback2 = Mock(spec=Callback)

        appliedKwargs = {'trainer': {'callbacks': [callback1]}}
        kwarg = {'trainer': {'callbacks': callback2}}

        expected = {'trainer': {'callbacks': [callback1, callback2]}, 'trainerFit': {}, 'log': {}}

        result = self.model._plKwargUpdater(appliedKwargs, kwarg)
        self.assertEqual(result, expected)

    def test_logMethod_nonLogger(self):
        self.setup(seed=71)

        appliedKwargs = {
            'log': {'on_step': {'train': False, 'val': True, 'predict': [], 'else': True},
                    'prog_bar': True}}
        kwarg = {'log': {'on_step': {'train': True, 'else': False}, 'prog_bar': False}}

        expected = {'trainer': {}, 'trainerFit': {},
                    'log': {'on_step': {'train': True, 'val': True, 'else': False},
                            'prog_bar': {'else': False}}}

        result = self.model._plKwargUpdater(appliedKwargs, kwarg)
        self.assertEqual(result, expected)

    def test_trainerNTrainerFit_combined(self):
        self.setup(seed=71)

        logger1 = Mock(spec=Logger)
        logger2 = Mock(spec=Logger)
        callback1 = Mock(spec=Callback)
        callback2 = Mock(spec=Callback)

        appliedKwargs = {'trainer': {'max_epochs': 10, 'logger': logger1, 'callbacks': [callback1]}}
        kwarg = {'trainer': {'logger': logger2, 'callbacks': callback2}}

        expected = {'trainer': {'max_epochs': 10, 'logger': [logger1, logger2],
                                'callbacks': [callback1, callback2]}, 'trainerFit': {}, 'log': {}}

        result = self.model._plKwargUpdater(appliedKwargs, kwarg)
        self.assertEqual(result, expected)

    def test_trainerCallbacks_emptyInApplied(self):
        self.setup(seed=71)

        callback2 = Mock(spec=Callback)

        appliedKwargs = {'trainer': {'callbacks': []}}
        kwarg = {'trainer': {'callbacks': callback2}}

        expected = {'trainer': {'callbacks': [callback2]}, 'trainerFit': {}, 'log': {}}

        result = self.model._plKwargUpdater(appliedKwargs, kwarg)
        self.assertEqual(result, expected)

    def test_trainerCallbacks_emptyInKwarg(self):
        self.setup(seed=71)

        callback1 = Mock(spec=Callback)

        appliedKwargs = {'trainer': {'callbacks': [callback1]}}
        kwarg = {'trainer': {'callbacks': []}}

        expected = {'trainer': {'callbacks': [callback1]}, 'trainerFit': {}, 'log': {}}

        result = self.model._plKwargUpdater(appliedKwargs, kwarg)
        self.assertEqual(result, expected)


class BaseFit_getArgsRelated_toEachMethodSeparately_Tests(FitTestsSetup):
    # addTest1
    #  camel and snake keys test; also check for _warnNotUsedKwargs_baseFit
    def test_followsByMethodFormat_allMethodsInvolved_valid(self):
        # Test follows format with valid arguments for all methods
        self.setup(seed=71)

        appliedKwargs = {'trainer': {'max_epochs': 10, 'log_every_n_steps': 50},
                         'trainerFit': {'ckpt_path': 'path/to/checkpoint'},
                         'log': {'on_step': {'else': True}, 'prog_bar': True}}

        expected = {'trainer': {'max_epochs': 10, 'log_every_n_steps': 50},
                    'trainerFit': {'ckpt_path': 'path/to/checkpoint'},
                    'log': {'on_step': {'else': True}, 'prog_bar': True}}

        result = self.model._getArgsRelated_toEachMethodSeparately(appliedKwargs)
        self.assertEqual(result, expected)

    def test_followsByMethodFormat_someMethodsInvolved_valid(self):
        # Test follows format with valid arguments for some methods
        self.setup(seed=71)

        appliedKwargs = {'trainer': {'max_epochs': 15},
                         'log': {'name': 'test_metric', 'value': 0.7}}

        expected = {'trainer': {'max_epochs': 15},
                    'log': {'name': 'test_metric', 'value': 0.7}, }

        result = self.model._getArgsRelated_toEachMethodSeparately(appliedKwargs)
        self.assertEqual(result, expected)

    def test_followsByMethodFormat_invalid(self):
        # Test follows format with invalid arguments
        self.setup(seed=71)

        appliedKwargs = {'trainer': {'max_epochs': 10}, 'trainerFit': {'train_dataloaders': None},
                         'log': {'invalid_arg': 123}}

        with self.assertRaises(ValueError) as context:
            self.model._getArgsRelated_toEachMethodSeparately(appliedKwargs)
        expectedErrorText = 'apparently you have sent your kwargs in `{"trainer":{...},"trainerFit":{...},"log":{...}}` format;[\'invalid_arg\'] are not related to log'
        self.assertIn(expectedErrorText, str(context.exception))

    def test_followsByMethodFormat_butHasAdditionalKeys(self):
        # Test follows format with additional unrelated keys
        self.setup(seed=71)

        appliedKwargs = {'trainer': {'max_epochs': 10},
                         'trainerFit': {'ckpt_path': 'path/to/checkpoint'},
                         'log': {'name': 'test_metric', 'value': 0.5},
                         'extra': {'some_extra_key': 'extra_value'}}

        expected = {'log': {}, 'trainer': {}, 'trainerFit': {}}
        expectedPrint = """you have included "trainerFit" but it doesn't match with args can be passed to pl.Trainer, pl.Trainer.fit or pl.LightningModule.log; even their camelCase names
you have included "trainer" but it doesn't match with args can be passed to pl.Trainer, pl.Trainer.fit or pl.LightningModule.log; even their camelCase names
you have included "log" but it doesn't match with args can be passed to pl.Trainer, pl.Trainer.fit or pl.LightningModule.log; even their camelCase names
you have included "extra" but it doesn't match with args can be passed to pl.Trainer, pl.Trainer.fit or pl.LightningModule.log; even their camelCase names
"""

        def innerFunc(appliedKwargs):
            return self.model._getArgsRelated_toEachMethodSeparately(appliedKwargs)

        result = self.assertPrint(innerFunc, expectedPrint=expectedPrint,
                                  appliedKwargs=appliedKwargs)
        self.assertEqual(result, expected)

    def test_doesNotFollowByMethodFormat_valid(self):
        # Test does not follow format with valid arguments
        self.setup(seed=71)

        appliedKwargs = {'max_epochs': 20, 'ckpt_path': 'path/to/checkpoint',
                         'name': 'test_metric', 'value': 0.8}

        expected = {'trainer': {'max_epochs': 20},
                    'trainerFit': {'ckpt_path': 'path/to/checkpoint'},
                    'log': {'name': 'test_metric', 'value': 0.8}}

        result = self.model._getArgsRelated_toEachMethodSeparately(appliedKwargs)
        self.assertEqual(result, expected)

    def test_doesNotFollowByMethodFormat_invalid(self):
        # Test does not follow format with invalid arguments and catch it with self.assertPrint
        self.setup(seed=71)

        appliedKwargs = {'max_epochs': 25, 'invalid_arg': 999,
                         'name': 'test_metric', 'value': 0.9}

        expectedPrint = 'you have included "invalid_arg" but it doesn' + "'t match with args can be passed to pl.Trainer, pl.Trainer.fit or pl.LightningModule.log; even their camelCase names"

        def innerFunc(appliedKwargs):
            return self.model._getArgsRelated_toEachMethodSeparately(appliedKwargs)

        result = self.assertPrint(innerFunc, expectedPrint=expectedPrint,
                                  appliedKwargs=appliedKwargs)

        expectedRes = {'trainer': {'max_epochs': 25}, 'trainerFit': {},
                       'log': {'name': 'test_metric', 'value': 0.9}}
        self.assertEqual(result, expectedRes)

    def test_inputsUntouced(self):
        # for this test; doesNotFollowByMethodFormat should be tested
        self.setup(seed=71)

        appliedKwargs = {'max_epochs': 25, 'invalid_arg': 999,
                         'name': 'test_metric', 'value': 0.9}
        appliedKwargsCopy = appliedKwargs.copy()
        self.model._getArgsRelated_toEachMethodSeparately(appliedKwargs)
        self.assertEqual(appliedKwargs, appliedKwargsCopy)

    def test_emptyAppliedKwargs(self):
        self.setup(seed=71)

        appliedKwargs = {}
        expected = {}

        result = self.model._getArgsRelated_toEachMethodSeparately(appliedKwargs)
        self.assertEqual(result, expected)


class BaseFit_putTogetherPlLoggersTests(FitTestsSetup):
    def test_bothNone(self):
        self.setup(seed=71)
        result = self.model._putTogether_plLoggers_normal(None, None)
        self.assertIsNone(result)

    def test_firstNone(self):
        self.setup(seed=71)
        logger2 = Mock(spec=Logger)
        result = self.model._putTogether_plLoggers_normal(None, logger2)
        self.assertEqual(result, logger2)

    def test_secondNone(self):
        self.setup(seed=71)
        logger1 = Mock(spec=Logger)
        result = self.model._putTogether_plLoggers_normal(logger1, None)
        self.assertEqual(result, logger1)

    def test_bothBool(self):
        self.setup(seed=71)
        result = self.model._putTogether_plLoggers_normal(True, False)
        self.assertFalse(result)

    def test_firstBool(self):
        self.setup(seed=71)
        logger2 = Mock(spec=Logger)
        result = self.model._putTogether_plLoggers_normal(True, logger2)
        self.assertEqual(result, logger2)

    def test_secondBool(self):
        self.setup(seed=71)
        logger1 = Mock(spec=Logger)
        result = self.model._putTogether_plLoggers_normal(logger1, True)
        self.assertEqual(result, logger1)

    def test_bothSingleLogger(self):
        self.setup(seed=71)
        logger1 = Mock(spec=Logger)
        logger2 = Mock(spec=Logger)
        result = self.model._putTogether_plLoggers_normal(logger1, logger2)
        self.assertEqual(result, [logger1, logger2])

    def test_firstListLogger(self):
        self.setup(seed=71)
        logger1 = [Mock(spec=Logger), Mock(spec=Logger)]
        logger2 = Mock(spec=Logger)
        result = self.model._putTogether_plLoggers_normal(logger1, logger2)
        self.assertEqual(result, logger1 + [logger2])

    def test_secondListLogger(self):
        self.setup(seed=71)
        logger1 = Mock(spec=Logger)
        logger2 = [Mock(spec=Logger), Mock(spec=Logger)]
        result = self.model._putTogether_plLoggers_normal(logger1, logger2)
        self.assertEqual(result, [logger1] + logger2)

    def test_bothListLogger(self):
        self.setup(seed=71)
        logger1 = [Mock(spec=Logger), Mock(spec=Logger)]
        logger2 = [Mock(spec=Logger), Mock(spec=Logger)]
        result = self.model._putTogether_plLoggers_normal(logger1, logger2)
        self.assertEqual(result, logger1 + logger2)


class BaseFit_putTogetherPlLoggersWithPhasedBasedLoggingTests(FitTestsSetup):
    phases1 = ['train', 'predict']
    phases2 = ['train', 'test', 'else']

    def _assertEqual(self, expected, result):
        for phase in self.model._phaseBasedLoggingTypes:
            print(phase, result[phase], expected[phase])
            self.assertEqual(result[phase], expected[phase])

    def test_bothEmptyDict(self):
        self.setup(seed=71)
        result = self.model._putTogether_plLoggers_withPhasedBasedLogging({}, {})
        expected = {phase: [] for phase in self.model._phaseBasedLoggingTypes}
        self._assertEqual(expected, result)

    def test_firstEmptyDict(self):
        self.setup(seed=71)
        logger2 = {phase: [Mock(spec=Logger)] for phase in self.phases1}
        result = self.model._putTogether_plLoggers_withPhasedBasedLogging({}, logger2)
        expected = {phase: logger2.get(phase, []) for phase in
                    self.model._phaseBasedLoggingTypes}
        self._assertEqual(expected, result)

    def test_secondEmptyDict(self):
        self.setup(seed=71)
        logger1 = {phase: [Mock(spec=Logger)] for phase in self.phases2}
        result = self.model._putTogether_plLoggers_withPhasedBasedLogging(logger1, {})
        expected = {phase: logger1.get(phase, []) for phase in
                    self.model._phaseBasedLoggingTypes}
        self._assertEqual(expected, result)

    def test_bothListLogger(self):
        self.setup(seed=71)
        logger1 = [Mock(spec=Logger), Mock(spec=Logger)]
        logger2 = [Mock(spec=Logger), Mock(spec=Logger)]
        result = self.model._putTogether_plLoggers_withPhasedBasedLogging(logger1, logger2)
        expected = {phase: [] for phase in self.model._phaseBasedLoggingTypes}
        expected['else'] = logger1 + logger2
        self._assertEqual(expected, result)

    def test_firstListLogger_2ndPhaseBased(self):
        self.setup(seed=71)
        logger1 = [Mock(spec=Logger), Mock(spec=Logger)]
        logger2 = {phase: [Mock(spec=Logger)] for phase in self.phases1}
        result = self.model._putTogether_plLoggers_withPhasedBasedLogging(logger1, logger2)
        expected = {phase: [] for phase in self.model._phaseBasedLoggingTypes}
        expected.update(logger2)
        expected['else'] = expected['else'] + logger1
        self._assertEqual(expected, result)

    def test_secondListLogger_1stPhaseBased(self):
        self.setup(seed=71)
        logger1 = {phase: [Mock(spec=Logger)] for phase in self.phases2}
        logger2 = [Mock(spec=Logger), Mock(spec=Logger)]
        result = self.model._putTogether_plLoggers_withPhasedBasedLogging(logger1, logger2)
        expected = {phase: [] for phase in self.model._phaseBasedLoggingTypes}
        expected.update(logger1)
        expected['else'] = expected['else'] + logger2
        self._assertEqual(expected, result)

    def test_bothPhaseBased(self):
        self.setup(seed=71)
        logger1 = {phase: [Mock(spec=Logger)] for phase in self.phases1}
        logger2 = {phase: [Mock(spec=Logger)] for phase in self.phases2}
        result = self.model._putTogether_plLoggers_withPhasedBasedLogging(logger1, logger2)
        expected = {phase: logger1.get(phase, []) + logger2.get(phase, []) for phase in
                    self.model._phaseBasedLoggingTypes}
        self._assertEqual(expected, result)


class BaseFit_putTogetherPlCallbacksTests(FitTestsSetup):
    def test_bothNone(self):
        self.setup(seed=71)
        result = self.model._putTogether_plCallbacks(None, None)
        self.assertIsNone(result)

    def test_firstNone(self):
        self.setup(seed=71)
        callback2 = Mock(spec=Callback)
        result = self.model._putTogether_plCallbacks(None, callback2)
        self.assertEqual(result, callback2)

    def test_secondNone(self):
        self.setup(seed=71)
        callback1 = Mock(spec=Callback)
        result = self.model._putTogether_plCallbacks(callback1, None)
        self.assertEqual(result, callback1)

    def test_bothSingleCallback(self):
        self.setup(seed=71)
        callback1 = Mock(spec=Callback)
        callback2 = Mock(spec=Callback)
        result = self.model._putTogether_plCallbacks(callback1, callback2)
        self.assertEqual(result, [callback1, callback2])

    def test_firstListCallback(self):
        self.setup(seed=71)
        callback1 = [Mock(spec=Callback), Mock(spec=Callback)]
        callback2 = Mock(spec=Callback)
        result = self.model._putTogether_plCallbacks(callback1, callback2)
        self.assertEqual(result, callback1 + [callback2])

    def test_secondListCallback(self):
        self.setup(seed=71)
        callback1 = Mock(spec=Callback)
        callback2 = [Mock(spec=Callback), Mock(spec=Callback)]
        result = self.model._putTogether_plCallbacks(callback1, callback2)
        self.assertEqual(result, [callback1] + callback2)

    def test_bothListCallback(self):
        self.setup(seed=71)
        callback1 = [Mock(spec=Callback), Mock(spec=Callback)]
        callback2 = [Mock(spec=Callback), Mock(spec=Callback)]
        result = self.model._putTogether_plCallbacks(callback1, callback2)
        self.assertEqual(result, callback1 + callback2)

    def test_bothListCallbacks(self):
        self.setup(seed=71)
        callback1 = [Mock(spec=Callback), Mock(spec=Callback)]
        callback2 = [Mock(spec=Callback), Mock(spec=Callback)]
        result = self.model._putTogether_plCallbacks(callback1, callback2)
        self.assertEqual(result, callback1 + callback2)


class FitTests(FitTestsSetup):
    pass

    # def testFit(self):  # kkk do it later
    #     # ccc it's just to see does it run or not
    #     self.setup(71)
    #
    #     self.model.fit(trainDataloader=self.trainDataloader, valDataloader=self.valDataloader,
    #                    lossFuncs=[nn.MSELoss(), nn.L1Loss()], max_epochs=3)


class DetermineFitRunStateTests(FitTests):
    # ccc1
    #  note there are some saved files essential for this test(DetermineFitRunStateTests_mockSavedModels)
    #  python import path is different when this file is run by runAllTests.py and when it's run
    #  from this file itself; so assertEqual_pathCompatibile func tries to make the tests pass for both
    #  cases

    expectedLoggerPathSeed71 = os.path.join(getProjectDirectory(), 'tests',
                                            'DetermineFitRunStateTests_mockSavedModels',
                                            'arch1',
                                            'mainRun_seed71')
    expectedLoggerPathSeed81 = os.path.join(getProjectDirectory(), 'tests',
                                            'DetermineFitRunStateTests_mockSavedModels',
                                            'arch1',
                                            'mainRun_seed81')

    # mustHave1
    #  for some of these cases also add matchedSeedDict (should )
    # ccc1
    #  note test func which have _matchedSeedDict have seed=71 as there is a model saved with
    #  this seed

    def assertEqual_pathCompatibile(self, loggerPath, path2):
        pathArgs = path2.split(os.sep)
        if 'tests' in pathArgs:
            if loggerPath != path2:
                testsIndex = pathArgs.index('tests')
                pathArgsWithBrazingTorchTestsFolder = list(pathArgs[:])
                pathArgsWithBrazingTorchTestsFolder.insert(testsIndex + 1,
                                                           'brazingTorchTests')
                if ':' in pathArgsWithBrazingTorchTestsFolder[0] and not '\\' in \
                                                                         pathArgsWithBrazingTorchTestsFolder[
                                                                             0]:
                    pathArgsWithBrazingTorchTestsFolder[0] += '\\'
                self.assertEqual(loggerPath,
                                 os.path.join(*pathArgsWithBrazingTorchTestsFolder))
        else:
            self.assertEqual(loggerPath, os.path.join(*pathArgs))

    def test_resume_seedSensitive_matchedSeedDict(self):
        self.setup(seed=71)
        architectureName, loggerPath, fitRunState, checkpointPath, isModelChanged = self.model._determineFitRunState(
            seed=self.seed, resume=True, seedSensitive=True)
        self.assertEqual(fitRunState, 'resume')
        self.assertFalse(isModelChanged)
        self.assertEqual_pathCompatibile(checkpointPath,
                                         os.path.join(self.expectedLoggerPathSeed71,
                                                      'BrazingTorch.ckpt'))

    def test_resume_seedSensitive_noMatchedSeedDict(self):
        self.setup(seed=81)
        architectureName, loggerPath, fitRunState, checkpointPath, isModelChanged = self.model._determineFitRunState(
            seed=self.seed, resume=True, seedSensitive=True)
        self.assertEqual(fitRunState, "beginning")
        self.assertEqual_pathCompatibile(loggerPath,
                                         self.expectedLoggerPathSeed81)  # note its 81

    def test_noResume_seedSensitive_matchedSeedDict_dontReplace(self):
        self.setup(seed=71)
        with simulateInput("this is some random input"):
            architectureName, loggerPath, fitRunState, checkpointPath, isModelChanged = self.model._determineFitRunState(
                seed=self.seed, resume=False, seedSensitive=True)

        self.assertEqual(fitRunState, "don't run")

    def test_noResume_seedSensitive_matchedSeedDict_replace(self):
        self.setup(seed=71)
        with simulateInput("yes"):
            architectureName, loggerPath, fitRunState, checkpointPath, isModelChanged = self.model._determineFitRunState(
                seed=self.seed, resume=False, seedSensitive=True)

        self.assertEqual(fitRunState, "beginning")
        self.assertEqual_pathCompatibile(loggerPath, self.expectedLoggerPathSeed71)

    def test_noResume_seedSensitive_noMatchedSeedDict(self):
        # note this one also exactly lands in the condition which
        # test_resume_seedSensitive_noMatchedSeedDict is landed and has exact same results
        self.setup(seed=81)
        architectureName, loggerPath, fitRunState, checkpointPath, isModelChanged = self.model._determineFitRunState(
            seed=self.seed, resume=False, seedSensitive=True)
        self.assertEqual(fitRunState, "beginning")
        self.assertEqual_pathCompatibile(loggerPath,
                                         self.expectedLoggerPathSeed81)  # note its 81

    def test_resume_noSeedSensitive_matchedSeedDict(self):
        self.setup(seed=71)
        architectureName, loggerPath, fitRunState, checkpointPath, isModelChanged = self.model._determineFitRunState(
            seed=self.seed, resume=True, seedSensitive=False)
        self.assertEqual(fitRunState, "resume")
        self.assertFalse(isModelChanged)
        self.assertEqual_pathCompatibile(loggerPath, self.expectedLoggerPathSeed71)
        self.assertEqual_pathCompatibile(checkpointPath,
                                         os.path.join(self.expectedLoggerPathSeed71,
                                                      'BrazingTorch.ckpt'))

    def test_resume_noSeedSensitive_noMatchedSeedDict(self):
        self.setup(seed=81)
        architectureName, loggerPath, fitRunState, checkpointPath, isModelChanged = self.model._determineFitRunState(
            seed=self.seed, resume=True, seedSensitive=False)
        self.assertEqual(fitRunState, "resume")
        self.assertTrue(isModelChanged)
        self.assertEqual_pathCompatibile(loggerPath, self.expectedLoggerPathSeed71)
        # note it's not 81 and it's 71 as the model has been changed

        self.assertEqual_pathCompatibile(checkpointPath,
                                         os.path.join(self.expectedLoggerPathSeed71,
                                                      'BrazingTorch.ckpt'))

    def test_noResume_noSeedSensitive_matchedSeedDict_dontReplace(self):
        self.setup(seed=71)
        with simulateInput("this is some random input"):
            architectureName, loggerPath, fitRunState, checkpointPath, isModelChanged = self.model._determineFitRunState(
                seed=self.seed, resume=False, seedSensitive=False)

        self.assertEqual(fitRunState, "don't run")

    def test_noResume_noSeedSensitive_matchedSeedDict_replace(self):
        self.setup(seed=71)
        with simulateInput("yes"):
            architectureName, loggerPath, fitRunState, checkpointPath, isModelChanged = self.model._determineFitRunState(
                seed=self.seed, resume=False, seedSensitive=False)

        self.assertEqual(fitRunState, "beginning")
        self.assertEqual_pathCompatibile(loggerPath, self.expectedLoggerPathSeed71)

    def test_noResume_noSeedSensitive_noMatchedSeedDict(self):
        self.setup(seed=81)
        architectureName, loggerPath, fitRunState, checkpointPath, isModelChanged = self.model._determineFitRunState(
            seed=self.seed, resume=False, seedSensitive=False)
        self.assertEqual(fitRunState, "beginning")
        self.assertEqual_pathCompatibile(loggerPath,
                                         self.expectedLoggerPathSeed81)  # note its 81

    def test_modelWithSameName_anotherArchitecture(self):
        model = NNDummy2(modelName='DetermineFitRunStateTests_mockSavedModels',
                         testPrints=True, seed=71,
                         lossFuncs=[nn.MSELoss(), nn.L1Loss()])
        architectureName, loggerPath, fitRunState, checkpointPath, isModelChanged = model._determineFitRunState(
            seed=71, resume=False, seedSensitive=False)
        self.assertEqual(fitRunState, "beginning")
        self.assertEqual(architectureName, 'arch2')

    def test_modelWithAnotherName(self):
        model = NNDummy1(modelName='NNDummy1',
                         testPrints=True, seed=71,
                         lossFuncs=[nn.MSELoss(), nn.L1Loss()])
        architectureName, loggerPath, fitRunState, checkpointPath, isModelChanged = model._determineFitRunState(
            seed=71)
        self.assertEqual(fitRunState, "beginning")

        self.assertEqual_pathCompatibile(loggerPath,
                                         os.path.join(getProjectDirectory(), 'tests',
                                                      'NNDummy1', 'arch1',
                                                      'mainRun_seed71'))
        self.assertEqual(architectureName, 'arch1')


# ---- run test
if __name__ == '__main__':
    unittest.main()
