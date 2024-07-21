import os
from typing import Iterable, Optional, List, Union

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import Logger

from projectUtils.dataTypeUtils.str import snakeToCamel
from projectUtils.misc import _allowOnlyCreationOf_ChildrenInstances, inputTimeout, nFoldersBack, \
    giveOnlyKwargsRelated_toMethod
from projectUtils.typeCheck import argValidator
from projectUtils.warnings import Warn


# goodToHave1
#  the phasedBasedFormat and appliedKwargs_byMethod should have classes
#  and not a format of dictionary
# kkk
#  does pl set trainer to model after training once? if so then in continuation
#  (for i.e. after loading model) we may not use .fit of this class

class _BrazingTorch_modelFitter_inner:
    def __init__(self):
        # not allowing this class to have direct instance
        _allowOnlyCreationOf_ChildrenInstances(self, _BrazingTorch_modelFitter_inner)
        # bugPotn1
        #  defining vars here(in inner parents) are not possible?!?!

    @property
    def _logOptions(self):
        # kkk does it need to be a public method?
        return self.__logOptions

    @_logOptions.setter
    @argValidator
    def _logOptions(self, value: dict):
        for key, val in value.items():
            self._assertPhaseBased_logOptions(val)
        self.__logOptions = value

    # ---- _getBaseFit_appliedKwargs and inners
    def _getBaseFit_appliedKwargs(self, kwargs, listOfKwargs):
        listOfKwargs_ = listOfKwargs or []
        listOfKwargs_ = listOfKwargs_[:]
        listOfKwargs_.append(kwargs)

        appliedKwargs = {method: {} for method in self._methodsFitCanHandle_names}
        for kw in listOfKwargs_:
            appliedKwargs = self._plKwargUpdater(appliedKwargs, kw)
        return appliedKwargs

    def _plKwargUpdater(self, appliedKwargs, kwarg):
        # ccc4
        #  - note _plKwargUpdater has been used in _getBaseFit_appliedKwargs
        #  also _mergeKwargsWith_runKwargs and preRunTests
        kwarg_ = kwarg.copy()
        appliedKwargs_ = self._getArgsRelated_toEachMethodSeparately(appliedKwargs)
        kwarg_ = self._getArgsRelated_toEachMethodSeparately(kwarg_)

        def nonListableKwargs_AddFunc(first, second):
            if isinstance(second, list) and len(second) == 0:  # if 2nd one is []
                return first
            return second

        for method in self._methodsFitCanHandle_names:
            appliedKwargs_[method] = appliedKwargs_.get(method, {})
            kwarg_[method] = kwarg_.get(method, {})

            for kw in set(list(appliedKwargs_[method].keys()) +
                          list(kwarg_[method].keys())):

                if kw not in appliedKwargs_[method]:
                    appliedKwargs_[method][kw] = []
                kwarg_kwValue = kwarg_[method].get(kw, [])

                # ccc2
                #  PyTorch Lightning for some args may get different types, and this
                #  part ensures those options are correctly applied (put together).
                #  note informally those kwargs are called listable.
                #  For example, a logger can be a Logger object or a list of Logger objects. See
                #  _putTogether_plLoggers_normal and _putTogether_plCallbacks.
                #  Note: only the kwargs 'logger', 'callbacks', 'plugins', and 'devices' can
                #  have lists and get their own combinator,
                #  while for other kwargs the second one replaces the earlier one.
                if method == 'log':
                    appliedKwargs_[method][kw] = \
                        self._putTogether_items_inPhasedBasedLoggingFormat(
                            appliedKwargs_[method][kw], kwarg_kwValue, nonListableKwargs_AddFunc)
                    # remove keys which have empty lists([])
                    appliedKwargs_[method][kw] = {k: v for k, v in
                                                  appliedKwargs_[method][kw].items() if
                                                  not (isinstance(v, list) and len(v) == 0)}
                elif method == 'trainer':
                    if kw == 'logger':
                        appliedKwargs_[method][kw] = self._putTogether_plLoggers_normal(
                            appliedKwargs_[method][kw], kwarg_kwValue)
                    elif kw == 'callbacks':
                        appliedKwargs_[method][kw] = self._putTogether_plCallbacks(
                            appliedKwargs_['trainer']['callbacks'], kwarg_kwValue)
                        # bugPotn3
                        #  'plugins' and 'devices' are also like 'logger' and 'callbacks' can also
                        #  take list alongside other types. but I don't take care of them, as these
                        #  options are rare
                    else:
                        appliedKwargs_[method][kw] = nonListableKwargs_AddFunc(
                            appliedKwargs_[method][kw], kwarg_kwValue)

                elif method == 'trainerFit':
                    appliedKwargs_[method][kw] = nonListableKwargs_AddFunc(
                        appliedKwargs_[method][kw], kwarg_kwValue)

        return appliedKwargs_

    def _getArgsRelated_toEachMethodSeparately(self, appliedKwargs):
        appliedKwargs_byMethod = {}
        keysNotRelated = []

        if self._doesDictFollow_byMethod(appliedKwargs):
            for meth, methName in zip([pl.Trainer, pl.Trainer.fit, pl.LightningModule.log],
                                      self._methodsFitCanHandle_names):
                # so here the dict seems to have method format.
                # but we check to give errors
                if methName not in appliedKwargs:
                    # should work with half filled method type
                    # (for i.e. missing trainerFit)
                    continue
                _, keysNotRelated_ = giveOnlyKwargsRelated_toMethod(meth,
                                                                    updater=appliedKwargs[methName])
                if keysNotRelated_:
                    raise ValueError('apparently you have sent your kwargs in `{"trainer":{...}' +
                                     ',"trainerFit":{...},"log":{...}}` format;' +
                                     f"{keysNotRelated_} are not related to {methName}")
                appliedKwargs_byMethod = appliedKwargs.copy()
        else:
            for meth, methName in zip([pl.Trainer, pl.Trainer.fit, pl.LightningModule.log],
                                      self._methodsFitCanHandle_names):
                # ccc3
                #  - note giveOnlyKwargsRelated_toMethod has camelCase compatibility
                #  for i.e. if the method takes `my_arg` but updater has
                #  `myArg`, includes `my_arg` as 'myArg'
                appliedKwargs_byMethod[methName] = {}
                appliedKwargs_byMethod[methName], keysNotRelated_ = giveOnlyKwargsRelated_toMethod(
                    meth, updater=appliedKwargs, updatee=appliedKwargs_byMethod[methName])
                keysNotRelated.extend(keysNotRelated_)

        self._warnNotUsedKwargs_baseFit(appliedKwargs_byMethod, keysNotRelated)
        return appliedKwargs_byMethod

    def _warnNotUsedKwargs_baseFit(self, appliedKwargs_byMethod, keysNotRelated):
        # ccc3
        #  note there might be some args passed by user (in listOfKwargs or kwargs) but the
        #  appliedKwargs_byMethod only keeps the kwargs related to the pytorch lightning trainer,
        #  trainer.fit, and self.log; therefore to prevent mistakes we warn the user

        # get keysRelatedToMethods
        keysRelatedToMethods = []
        for methName in self._methodsFitCanHandle_names:
            if methName not in appliedKwargs_byMethod:
                # should work with half filled method type
                # (for i.e. missing trainerFit)
                continue
            for key in appliedKwargs_byMethod[methName].keys():
                keysRelatedToMethods.append(key)

        # bugPotn3
        #  now we take the _byMethod format dict; so there is a possibility of the
        #  real snakeCase of pytorchlightning is not got corrected and it's been missed
        #  to get corrected somewhere

        # warning part
        keysNotRelated = list(set(keysNotRelated) - set(keysRelatedToMethods))
        for key in keysNotRelated:
            warnMsg = f'you have included "{key}" but it ' + "doesn't match with args " + \
                      "can be passed to pl.Trainer, pl.Trainer.fit or " + \
                      "pl.LightningModule.log; even their camelCase names"
            Warn.warn(warnMsg)
            self._printTestPrints(warnMsg)

    # ---- methods for putting together listable Pytorch Lightning kwargs
    @argValidator
    def _putTogether_plLoggers_normal(self,
                                      var1: Optional[Union[Logger, Iterable[Logger], bool]],
                                      var2: Optional[Union[Logger, Iterable[Logger], bool]]) \
            -> Union[Logger, List[Logger], None, bool]:
        # ccc2
        #  - each pytorch lightning arg may get a Logger object or a list of Logger
        #       objects or None Or bool
        #  - note Logger and Iterable[Logger] have higher importance in setting than None or
        #       bool (would replace them)
        #  - also if the one side is Iterable[Logger] so the other one is either appended or
        #       extended to the one with Iterable[Logger]
        #  - also note var1 is the earlier one and var2 is the later

        # convert list_iterator back to list; note the hints make the list convert to list_iterator
        if isinstance(var1, type(iter([]))):
            var1 = list(var1)
        if isinstance(var2, type(iter([]))):
            var2 = list(var2)
        # Check if either var1 or var2 is None or bool
        if var1 is None:
            return var2
        elif isinstance(var1, bool):
            if var2 is not None:
                return var2
            else:
                return var1

        if var2 is None:
            return var1
        elif isinstance(var2, bool):
            # we know var1 is not None now (because of previous if block)
            if isinstance(var1, bool):
                # when both are bool var2 has higher importance
                return var2
            else:
                # when var1 is not None or bool means it's Logger or Iterable[Logger] so
                # it's returned
                return var1

        result = []
        if isinstance(var1, Iterable):
            # bugPotn1
            #  I checked and found that list(here result) can extend tuples and sets as well
            #  but I don't know what happens for other iterables
            result.extend(var1)
        else:
            result.append(var1)

        if isinstance(var2, Iterable):
            result.extend(var2)
        else:
            result.append(var2)
        return result

    @argValidator
    def _putTogether_plCallbacks(self,
                                 var1: Optional[Union[List[Callback], Callback]],
                                 var2: Optional[Union[List[Callback], Callback]]) \
            -> Union[Callback, List[Callback], None]:
        # ccc3
        #  - each pytorch lightning arg may get a Callback object or a list of Callback or None
        #  - note have higher importance in setting Callback, Iterable[Callback] than None

        # convert list_iterator back to list; note the hints make the list convert to list_iterator
        if isinstance(var1, type(iter([]))):
            var1 = list(var1)
        if isinstance(var2, type(iter([]))):
            var2 = list(var2)
        # Check if either var1 or var2 is None
        if var1 is None:
            return var2
        if var2 is None:
            return var1

        result = []
        if isinstance(var1, Iterable):
            # goodToHave3 like above
            if not all(isinstance(callback, Callback) for callback in var1):
                raise ValueError('var1 has some elements which are not Callback objects')
            # bugPotn1 like above
            result.extend(var1)
        else:
            result.append(var1)

        if isinstance(var2, Iterable):
            # goodToHave3 like above
            if not all(isinstance(callback, Callback) for callback in var2):
                raise ValueError('var2 has some elements which are not Callback objects')
            # bugPotn1 like above
            result.extend(var2)
        else:
            result.append(var2)
        return result

    # ---- methods for phaseBased
    @argValidator
    def _keysDontFollow_phaseBasedFormat(self, dict_: dict):
        # ccc3
        #  having keysDontFollow empty mean it does follow phaseBasedFormat
        keysDontFollow = []
        for key, value in dict_.items():
            if isinstance(value, dict):
                if key not in ['train', 'val', 'test', 'predict', 'else']:
                    keysDontFollow.append(key)
        return keysDontFollow

    @argValidator
    def _assertPhaseBased_logOptions(self, logOptions: dict):
        # assert phaseBased _logOptions to fit in the format it should have
        keysDontFollow = self._keysDontFollow_phaseBasedFormat(logOptions)
        if keysDontFollow:
            raise ValueError("it seems you are trying to use phaseBased logOptions" + \
                             f"\nthe key '{keysDontFollow[0]}' must be one of " + \
                             "'train', 'val', 'test', 'predict', 'else'")

    def _get_phaseBasedFormat(self, var):
        if isinstance(var, dict) and not self._keysDontFollow_phaseBasedFormat(var):
            return var
        res = {phase: [] for phase in self._phaseBasedLoggingTypes}
        res['else'] = var
        return res

    @argValidator
    def _putTogether_items_inPhasedBasedLoggingFormat(self, var1, var2, addFunc=None):
        # ccc1
        #  there is a phasedBasedLogging feature which allows to send kwargs related to logging in
        #  a dictionary consisting keys 'train', 'val', 'test', 'predict' or 'else'
        #  this code unifies the loggers to have that format
        result = {phase: [] for phase in self._phaseBasedLoggingTypes}

        var1PhaseBased = self._get_phaseBasedFormat(var1)
        var2PhaseBased = self._get_phaseBasedFormat(var2)

        for phase in self._phaseBasedLoggingTypes:
            result[phase] = addFunc(var1PhaseBased.get(phase, []),
                                    var2PhaseBased.get(phase, []))

        return result

    # ---- check to have byMethod format
    def _doesDictFollow_byMethod(self, dict_: dict):
        return all(method in self._methodsFitCanHandle_names for method in dict_.keys())

    # ---- other baseFit methods
    def _removeNotAllowedArgs(self, appliedKwargs_byMethod, notAllowedArgs):
        # addTest3
        for naa in notAllowedArgs:
            for method in self._methodsFitCanHandle_names:
                for key in appliedKwargs_byMethod[method].keys():
                    if naa == key or naa == snakeToCamel(key):
                        warnMsg = f'you have include {naa} in kwargs you passed, but' + \
                                  'this is not allowed and omitted'
                        Warn.warn(warnMsg)
                        self._printTestPrints(warnMsg)
                        del appliedKwargs_byMethod[method][key]

    # ---- determineFitRunState
    def _determineFitRunState(self, seed, resume=True, seedSensitive=False):
        # ccc2
        #  this is similar to _determineShouldRun_preRunTests

        # ccc3(same as _determineShouldRun_preRunTests)
        #  there was a idea about ""architectureName should be figured out in postInit"" but it may
        #  cause problems with loggerPath in preRunTests and .fit method

        # by default these values are assumed and will be
        # changed depending on the case
        fitRunState = 'beginning'  # fitRunState can have these: "beginning", "resume", "don't run"
        architectureName = 'arch1'
        checkpointPath = ''
        runName = f'mainRun_seed{seed}'
        isModelChanged = False  # kkk the name is meaningLess

        dummyLogger = pl.loggers.TensorBoardLogger(self.modelName)
        loggerPath = os.path.abspath(dummyLogger.log_dir)
        # loggerPath is fullPath including 'modelName/someName/version_0'

        if os.path.exists(nFoldersBack(loggerPath, n=2)):
            # there is a model run before with the name of this model

            architectureDicts = self._collectArchDicts(loggerPath)
            # update loggerPath with now found archDicts
            if architectureDicts:
                loggerPath = self._updateLoggerPath_withExistingArchName(architectureDicts, runName)
            architectureDicts_withMatchedAllDefinitions = self._getArchitectureDicts_withMatchedAllDefinitions(
                architectureDicts)
            # matchedAllDefinitions means the exact same model structure as all
            # layers and their definitions are exactly the same
            # note architectureDicts matches _saveArchitectureDict

            # remove "architecture.pkl" which is in the "preRunTests" folder
            architectureDicts_withMatchedAllDefinitions = self._remove_architectureDicts_fromPreRunTestsFolder(
                architectureDicts_withMatchedAllDefinitions)

            if architectureDicts_withMatchedAllDefinitions:
                # note being here means some models with the exact same
                # structure as this model has run before

                # Check if there is a model with the same seed
                # note this is gonna be used in various cases below
                matchedSeedDict, matchedSeedDict_filePath = self._findSeedMatch_inArchitectureDicts(
                    architectureDicts_withMatchedAllDefinitions, seed, returnCheckPointPath=True)

                checkpointPath, fitRunState, runName, isModelChanged = self._fitRunState_conditions(
                    checkpointPath, matchedSeedDict, matchedSeedDict_filePath, resume,
                    runName, seedSensitive, isModelChanged)
            else:
                # there are models with the name of this model but with different structures

                # finds new architectureName
                architectureName = self._findAvailableArchName(nFoldersBack(loggerPath, n=2))

        else:
            # no model with this name in directory has never run
            pass  # so default fitRunState and architectureName are applied

        dummyLogger = pl.loggers.TensorBoardLogger(self.modelName,
                                                   name=architectureName,
                                                   version=runName)
        loggerPath = os.path.abspath(dummyLogger.log_dir)

        return architectureName, loggerPath, fitRunState, checkpointPath, isModelChanged

    def _fitRunState_conditions(self, checkpointPath,
                                matchedSeedDict, matchedSeedDict_filePath,
                                resume, runName, seedSensitive, isModelChanged):
        # ccc1
        #  - note being here means some models with the exact same
        #  structure as this model has run before
        #  - note there is no option to have duplicate models with the
        #  same seed as the result would be the same anyway, so having
        #  duplicate models with the same seed is useless
        #  - 'no resume' means tend to run from beginning
        if seedSensitive:
            if matchedSeedDict:
                if resume:
                    # Resume the model with the same seed
                    fitRunState = 'resume'
                    checkpointPath = matchedSeedDict_filePath
                else:
                    # 'no resume' means tend to run from beginning

                    # Ask the user whether to run the model with the same
                    # seed from beginning and replace the old save
                    if self._askUserToReplaceModel():
                        fitRunState = 'beginning'
                    else:
                        fitRunState = "don't run"
            else:
                # ccc1
                #  there is no matched seed and it's seedSensitive(meaning that
                #  doesn't prefer to resume or start from the beginning models
                #  saved with other seeds)
                #  - thus only option is to run from the beginning
                fitRunState = 'beginning'
        else:
            if resume:  # not seedSensitive and resume
                # ccc1
                #  even it's no seedSensitive if there is a model with the same
                #  seed, so resumes that because it is probably more desired
                if matchedSeedDict:
                    # Resume the model which has the same seed
                    fitRunState = 'resume'
                    checkpointPath = matchedSeedDict_filePath
                else:
                    # Resumes any other seed
                    # ccc1
                    #  note as the seeds are not the same, this option only wants to resume
                    #  sth, later in .fit when the model(with different seed) is loaded
                    #  there would be a warning for 'seed is changing'
                    fitRunState = 'resume'
                    # ccc1
                    #  note as you have guessed matchedSeedDict_filePath is checkPointPath
                    #  for a model with different seed
                    checkpointPath = matchedSeedDict_filePath
                    runName = matchedSeedDict_filePath.split(os.sep)[-2]
                    isModelChanged = True
                    Warn.info("loading a model with another seed")
            else:  # no resume and no seedSensitive
                # ccc1
                #  even it's no seedSensitive if there is a model with the same seed,
                #  thats more prefered; also note duplicate models with same seeds
                #  are not allowed
                if matchedSeedDict:
                    # Ask the user whether to run the model with the same
                    # seed from beginning and replace the old save
                    if self._askUserToReplaceModel():
                        fitRunState = 'beginning'
                    else:
                        fitRunState = "don't run"
                else:  # not seedSensitive and not resume and no matchedSeedDict
                    # here the seed differs and 'no resume' means tend to run from
                    # beginning with its own seed
                    fitRunState = 'beginning'
        return checkpointPath, fitRunState, runName, isModelChanged

    def _askUserToReplaceModel(self):
        # Use inputTimeout to ask the user whether to replace the model
        answer = inputTimeout("Do you want to replace the model? (yes/no)", timeout=30)
        if answer:
            return answer.lower() == 'yes'
        return False

    def _remove_architectureDicts_fromPreRunTestsFolder(self, listOfDicts):
        """
        removes architectureDict which is "architectureDict" folder
        """
        newListOfDicts = []

        for currentDict in listOfDicts:
            for key, value in currentDict.items():
                pathComponents = key.split(os.sep)
                # Check if "architecture.pkl" is not in "preRunTests" folder
                if pathComponents[-2] != "preRunTests":
                    newListOfDicts.append(currentDict)
        return newListOfDicts

    @staticmethod
    def _warnIf_modelIsChanged():
        Warn.error("WARNING: note .fit method returns instance(or self) and also the trainer." + \
                   "\nspecially in this case, which the instance is replaced, make sure" + \
                   "that you are catching new instance like 'instance, trainer=instance.fit(...)'")
        answer = inputTimeout("do you want to stop the code to follow the pattern? (yes/no)",
                              timeout=30)
        return answer

    def getArchitectureNameNVersion_fromLoggerPath(self, loggerPath):
        # Split the logger path into components
        pathParts = loggerPath.split(os.sep)

        # Initialize the variables
        architectureName = None
        version = None

        # Find the index of modelName in pathParts
        if self.modelName in pathParts:
            model_index = pathParts.index(self.modelName)

            # Check if architectureName exists
            if model_index + 1 < len(pathParts):
                architectureName = pathParts[model_index + 1]

            # Check if version exists
            if model_index + 2 < len(pathParts):
                version = pathParts[model_index + 2]

        # Return the extracted details in a dictionary
        return {'modelName': self.modelName, 'architectureName': architectureName,
                'version': version}

    # ---- not used: obsolete; delete later
    @argValidator
    def _putTogether_plLoggers_withPhasedBasedLogging(self,
                                                      var1: Optional[Union[
                                                          dict, Logger, Iterable[Logger], bool]],
                                                      var2: Optional[Union[
                                                          dict, Logger, Iterable[Logger], bool]]) \
            -> dict:
        # goodToHave3
        #  this is not used; remove this later
        # ccc3
        #  this is not used in the code directly anymore and not it's applied in _plKwargUpdater but
        #  because there are some tests for it and the logic in _plKwargUpdater is exactly the same
        #  so we keep it
        return self._putTogether_items_inPhasedBasedLoggingFormat(var1, var2,
                                                                  self._putTogether_plLoggers_normal)

    def _flatten_byMethodDict(self, dict_: dict):
        # goodToHave3
        #  this is not used; remove this later
        flatDict = {}
        if self._doesDictFollow_byMethod(dict_):
            for method in self._methodsFitCanHandle_names:
                flatDict.update(dict_[method])
            return flatDict
        return dict_
