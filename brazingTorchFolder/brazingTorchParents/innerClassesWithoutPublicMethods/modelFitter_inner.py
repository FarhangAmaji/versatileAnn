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
        self._assertPhaseBased_logOptions(value)
        self.__logOptions = value

    # ---- other baseFit methods
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
                # bugPotn1
                #  here if the 'logger' exist in appliedKwargs must go in 'trainer' as
                #  the logger for 'log' can only be passed with method format
                #  this has not been implemented yet
                # ccc1
                #  - note I have checked these 3 methods and they don't have args with mutual names
                #           except pl.Trainer and pl.LightningModule.log which take 'logger' arg
                #           which is ok, even though logger in self.log is just a bool or None but in
                #           pl.Trainer is a Logger object or a list of Logger objects or bool or None
                #  to be mutual
                #  - note giveOnlyKwargsRelated_toMethod has camelCase compatibility
                #  for i.e. if the method takes `my_arg` but updater has
                #  `myArg`, includes `my_arg` as 'myArg'
                appliedKwargs_byMethod[methName] = {}
                appliedKwargs_byMethod[methName], keysNotRelated_ = giveOnlyKwargsRelated_toMethod(
                    meth, updater=appliedKwargs, updatee=appliedKwargs_byMethod[methName])
                keysNotRelated.extend(keysNotRelated_)

        self._warnNotUsedKwargs_baseFit(appliedKwargs_byMethod, keysNotRelated)
        return appliedKwargs_byMethod

    def _removeNotAllowedArgs(self, appliedKwargs, appliedKwargs_byMethod, notAllowedArgs):
        # addTest3
        for naa in notAllowedArgs:
            for ak in appliedKwargs_byMethod.keys():
                for ak2 in appliedKwargs_byMethod[ak].keys():
                    if naa == ak2 or naa == snakeToCamel(ak2):
                        Warn.warn(f'you have include {naa} in kwargs you passed, but' + \
                                  'this is not allowed and omitted')
                        del appliedKwargs_byMethod[ak][ak2]
                        del appliedKwargs[ak2]  # in order to give warning again in leftOvers part

    def _warnForNotUsedArgs(self, appliedKwargs, appliedKwargs_byMethod):
        # addTest3
        # warn for left over kwargs, the kwargs which user has passed but don't
        # fit in pl.Trainer, pl.Trainer.fit or pl.LightningModule.log
        for auk in appliedKwargs:
            isInVolved = False
            for ak in appliedKwargs_byMethod.keys():
                if isInVolved:
                    break
                for ak2 in appliedKwargs_byMethod[ak].keys():
                    if auk == ak2 or auk == snakeToCamel(ak2):
                        isInVolved = True
                        break
            if not isInVolved:
                Warn.warn(f"you have included {auk} but it doesn't match with args " + \
                          "can be passed to pl.Trainer, pl.Trainer.fit or " + \
                          "pl.LightningModule.log; even their camelCase names")

    @argValidator
    def _assertPhaseBased_logOptions(self, logOptions: dict):
        # assert phaseBased _logOptions to fit in the format it should have
        for akl, aklV in logOptions.items():
            if isinstance(aklV, dict):
                if akl not in ['train', 'val', 'test', 'predict', 'else']:
                    raise ValueError("it seems you are trying to use phaseBased logOptions" + \
                                     f"\nthe key '{akl}' must be one of " + \
                                     "'train', 'val', 'test', 'predict', 'else'")

    # ---- _getBaseFit_appliedKwargs and inners
    def _getBaseFit_appliedKwargs(self, kwargs, listOfKwargs):
        listOfKwargs = listOfKwargs or []
        listOfKwargs.append(kwargs)
        appliedKwargs = {}
        for kw in listOfKwargs:
            self._plKwargUpdater(appliedKwargs, kw)
        return appliedKwargs

    def _plKwargUpdater(self, appliedKwargs, kw):
        # ccc2
        #  pytorch lightning for some args may get different type
        #  this methods makes sure those options are correctly applied (put together)
        #  for i.e. logger can be a Logger object or a list of Logger objects
        kw_ = kw.copy()
        correctArgs = {}
        if 'logger' in appliedKwargs and 'logger' in kw:
            correctArgs['logger'] = self._putTogether_plLoggers_withPhasedBasedLogging(
                appliedKwargs['logger'],
                kw['logger'])
            del kw_['logger']
        # 'plugins' is also like callbacks. but I don't take care of as this option is rare
        if 'callbacks' in appliedKwargs and 'callbacks' in kw:
            correctArgs['callbacks'] = self._putTogether_plCallbacks(appliedKwargs['callbacks'],
                                                                     kw['callbacks'])
            del kw_['callbacks']
        appliedKwargs.update(kw_)
        appliedKwargs.update(correctArgs)
        return appliedKwargs

    @argValidator
    def _putTogether_plLoggers_withPhasedBasedLogging(self,
                                                      var1: Optional[Union[
                                                          dict, Logger, Iterable[Logger], bool]],
                                                      var2: Optional[Union[
                                                          dict, Logger, Iterable[Logger], bool]]) \
            -> dict:
        # ccc1
        #  there is a phasedBasedLogging feature which allows to send kwargs related to logging in
        #  a dictionary consisting keys 'train', 'val', 'test', 'predict' or 'else'
        #  this code unifies the loggers to have that format
        result = {phase: [] for phase in self.phaseBasedLoggingTypes}

        def get_phaseBasedOptions(var):
            if isinstance(var, dict):
                self._assertPhaseBased_logOptions(var)
                return var
            res = {phase: [] for phase in self.phaseBasedLoggingTypes}
            res['else'] = var
            return res

        var1PhaseBased = get_phaseBasedOptions(var1)
        var2PhaseBased = get_phaseBasedOptions(var2)

        for phase in self.phaseBasedLoggingTypes:
            result[phase] = self._putTogether_plLoggers_normal(var1PhaseBased.get(phase, []),
                                                               var2PhaseBased.get(phase, []))

        return result

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
            # goodToHave3
            #  check does argValidator check for Iterable[Logger] or not
            # Check if all elements of var1 are Logger
            if not all(isinstance(logger, Logger) for logger in var1):
                raise ValueError('var1 has some elements which are not Logger objects')
            # bugPotn1
            #  I checked and found that list(here result) can extend tuples and sets as well
            #  but I don't know what happens for other iterables
            result.extend(var1)
        else:
            result.append(var1)

        if isinstance(var2, Iterable):
            # Check if all elements of var2 are Logger
            if not all(isinstance(logger, Logger) for logger in var2):
                raise ValueError('var2 has some elements which are not Logger objects')
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

    # ----
    # ---- check to have byMethod format
    def _doesDictFollow_byMethod(self, dict_: dict):
        return all(method in self._methodsFitCanHandle_names for method in dict_.keys())
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
