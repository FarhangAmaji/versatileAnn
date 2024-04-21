# current
- readme for brazing torch

2. readMe
   3. explain folders in root folder
       4. where to look at code, for dive in it more
   4. explain private and public func/class names `_`
   4. reading comments due to importance/scale of scope affecting(if u add color codding devDocs/conventions/convention_todoTypes.py)
   5. readme for each folder
3. readMe for
    4. brazingTorch
   5. dataPrep
       6. short for commonDatasets
   6. 
6. brazingTorch should be separated from the rest of the project
   2. readMe
6. merge to development and master
6. refactor:
   3. add docstring and moduleStrings
   4. change old todos
   5. add addtests comment
   6. add tests for the ones needed the most
       - caching data for tests
       - check regularization check optimizer when it's been set on optimizer or reversed
11. solve macOs errors
        1. splitTsTrainValTest_DfNNpDict
            1. splitTests
               4. 95
               5. 125
        2. mpsDeviceName
16. machine learning models and optimizer
17. adapt models to new wrapper
18. user examples


# later todos:
## limited scope features:
   - right now default features of .fit are not customizable
           for example, the callbacks are preset but I should have self.fit_preset_variables and from that user may modify it
   - check when both of forwardOutputs and targets are tensor in _warnIf_forwardOutputsNTargets_haveNoSamePattern
   - self.lossFuncs should be always True or not:
   - masked loss
   - in tests have a model which has accuracy 
   - final resetModel args with should be used from init args so current state of model
   - make test for resetModel
   - make preRunTests logs as I want
    
   - add finding best shuffleSeed in random for preRunTests
   - related to modelDifferentiator and architectures: we may save performance of each arch of models with the same name in .csv file(ofc we have tensorboard but this one also may be useful)
## wider scope features:
   - universal seed in dataprep and BrazingTorch; it should not mess the funcs which take seed as arg; ofc it seems easy to solve it with `if` seed as arg is None then use universal seed; maybe load should set universal seed too
## whole code:
   - code error prevention:
       - check 'a=a or []'; must be '[] if a is None else a'
       - add to cleancode .md
   
## general:
   - progamming helpers:
       - add linter for vscode
   - add .dockerfile I don't find it useful but maxim is going to add this probably
   - add explanation for:
      - `mustHave`s, `goodToHave`s, `bug`s and `bugPotn`s
   - make formatter for vscode(currently is not working)
   - revise pycharm formatter
## past comments didnt understand this when reading again:
   - improve modelDifferentiator to get classObj instead of warn
       if its not necessary I should addTest for working example with no self.lossFuncs
   - modelFitter:
       - think about adding _printFirstNLast_valLossChanges to modelFitter
  

# big steps
mustHave:
adapt models to new wrapper
user examples

goodTohave:
1. machine learning models and optimizer
2. conformal prediction
2. models:
       4. deep reinforcement learning
       7. lamaish
      8. diffusion
      9. gan
hyperparam opt with optuna
data downloader(easier)
trading pipeline(maybe utilize ready made packages)
