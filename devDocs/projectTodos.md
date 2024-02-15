# current

todos:

    
    - 
    - _determineShouldRun_preRunTests saves architectureDict in architectureFolder and not runFolder
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
