# current
.fit should have:
            1. ModelCheckpoint to save best model on loss
            2. StepLR Scheduler, ReduceLROnPlateau, warmUp
            3. EarlyStopping
            4. log_every_n_epoch=1 (on trainer I think)
create:
    1. new todos tagnames
    2. new commit prefixes
clean this file

1. determine where should architectureDict and model checkpoint to be saved(have 'seed' and preRunTests in mind)
2. check preRunTests follows 1 correctly
3. try save and load with .fit(without _determineFitRunState)
4. apply checkpoints save path to .fit
5. (***here)complete load or from begging to .fit
6. loadModel should replace with self:
    7. it's not possible so with a trick .fit should be able to load model and replace it with self
   8. not it should have codeClarifier
   9. make sure optimizer and schedulers are also loaded
   10. also on_load_checkpoint should be applied
6. make sure comments of determineFitRunState are correct and complete
7. there should be some files saved for dummy1
7. - .fit should have:
            1. ModelCheckpoint to save best model on loss
            2. StepLR Scheduler, ReduceLROnPlateau, warmUp
            3. EarlyStopping
            4. log_every_n_epoch=1 (on trainer I think)
on getBackForeCastData_general if the device is mps it should check for changing int/float 64 to 32: no no no getBackForeCastData_general doesn't move to device so it's not needed
- problems:
- on_save_checkpoint is not called
- magic number error
- macos errors:
    1. requirements.txt
    8. dtype difference
       9. dataloaderTests
          10. 352
          11. 363
       12. tsRowFetcherTests
           13. 243
           14. 236
           15. 274
           17. 281
           16. 289
           18. 320
    3. equalDf
        2. splitTsTrainValTest_DfNNpDict
           3. splitTests
               4. 95
              5. 125
        1. environment difference error
    1. mpsDeviceName

- todos:

    
    - 
    - 
    - correct architecture to have seed for version and in subdir it should take preRunTests or runName
    - _determineShouldRun_preRunTests saves architectureDict in architectureFolder and not runFolder
    - .fit should have:
            1. ModelCheckpoint to save best model on loss
            2. StepLR Scheduler, ReduceLROnPlateau
            3. EarlyStopping
            4. log_every_n_epoch=1
    - how to model 
    - right now default features of .fit are not customizable
            for example, the callbacks are preset but I should have self.fit_preset_variables and from that user may modify it
    - check soleymanis errors?!?!
    - baseFit and fit
        - think about resuming model

    - caching data for tests
    - revise todos
        - add `bugPotential_hardcode`
        - add `bugPotential_fixedLocation` to conventions and colorcoding todos of pycharm and vscode
          - also add it to code
    - revise commit prefixes
    - add linter for vscode
    - add to cleancode .md
        - check 'a=a or []'; must be '[] if a is None else a'



    - make test for addTests of lossNRegularization
    - check when both of forwardOutputs and targets are tensor in _warnIf_forwardOutputsNTargets_haveNoSamePattern
    - self.lossFuncs should be always True or not:
    - masked loss
    - in tests have a model which has accuracy 
    - improve modelDifferentiator to get classObj instead of warn(didnt understand this when reading again)
            if its not necessary I should addTest for working example with no self.lossFuncs
    - final resetModel args with should be used from init args so current state of model
    - make test for resetModel
    - make preRunTests logs as I want
    - modelFitter:
        - think about adding _printFirstNLast_valLossChanges to modelFitter
    - add finding best shuffleSeed in random for preRunTests
    - make sure specialModesStep and commonStep are similar
    - add restructureNeeded to todos

# after current dev spring
# major
- universal seed in dataprep and BrazingTorch; it should not mess the funcs which take seed as arg; ofc it seems easy to solve it with `if` seed as arg is None then use universal seed; maybe load should set universal seed too
- brazingTorch should be separated from the rest of the project
- related to modelDifferentiator and architectures: we may save performance of each arch of models with the same name in .csv file(ofc we have tensorboard but this one also may be useful)
- change todo conventions: remove cccDevStruct and cccDevAlgo and add ccY(=ccWhy) and ccWhat maybe with 1,2,3; also change the
# minor
- is it possible to cache some data for tests? for i.e. creating some dataloaders takes so much time, and theyre used in many tests
- (not needed because we want to provide a package like pandas which should work on any machine)add .dockerfile(but how docker is going to be useful, we are not running some commands and we are providing a frame work and libraries)
- choose and add linter to project
- make formatter for vscode(currently is not working)
- export formatter options of pycharm(put it in devDocs)
- export comments style of pycharm and put it in devDocs
