# current

todos:
    - modelTrainer:
        - check it
        - replace preRunTests with modelTrainer
        - correct callbacks and logger:
            - I think they can get single or list
            - what are other options which may get single or list
            - correct it in preRuntTests and modelFitter
        - add phase based logOptions to:
            - modelTrainer
            - _logLosses
        - make preRunTests logs as I want
        - think about adding _printFirstNLast_valLossChanges
    - logging options:
            - may need to have my implementation of trainer with:
                    - checking methods pl.Trainer, pl.Trainer.fit and self.log and
                         self._logLosses options in a more unified manner
                    - maybe with this I easily can implement variables for each run
                    - maybe after this contextManager are not necessary
            - re_design logging for preRunTests
    - variable for each run:
            - which enables to flag sths to do 'once'
            - after this move _warnIf_forwardOutputsNTargets_haveNoSamePattern
    - make test for addTests of lossNRegularization
    - in preRunTests don't allow some args to be set by user like overfitBatches in overfitBatches
    - check when both of forwardOutputs and targets are tensor in _warnIf_forwardOutputsNTargets_haveNoSamePattern
    - restore corrected version of preRunTests_Tests with check for (stallion and epf datasets)
    - add regularization features
    - self.lossFuncs should be always True or not:
            if its not necessary I should addTest for working example with no self.lossFuncs
    - final resetModel args with should be used from init args so current state of model
    - make test for resetModel

# after current dev spring
# major

# minor
- add `bugPotential_hardcode`
- add `bugPotential_fixedLocation` to conventions and colorcoding todos of pycharm and vscode
  - also add it to code
- 
