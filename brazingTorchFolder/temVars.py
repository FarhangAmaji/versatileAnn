from utils.dataTypeUtils.str import joinListWithComma
from utils.generalUtils import _allowOnlyCreationOf_ChildrenInstances


class _BrazingTorch_tempVars:
    # addTest1 these need tests but are checked with debugging through
    def __init__(self, **kwargs):
        # cccDevAlgo
        #  can keep temp variables in step/epoch/run for each phase
        self.tempVarStep = {"train": {}, "val": {}, "test": {}, "predict": {}}
        self.tempVarEpoch = {"train": {}, "val": {}, "test": {}, "predict": {}}
        self.tempVarRun = {"train": {}, "val": {}, "test": {}, "predict": {}}
        self.tempVarRun_allPhases = {}
        self._tempVarRun_allPhases_hidden = {}  # for internal use
        # note self.tempVarRun and tempVarRun_allPhases reset on fit start

        # not allowing this class to have direct instance
        _allowOnlyCreationOf_ChildrenInstances(self, _BrazingTorch_tempVars)

    def resetTempVar_step(self, phase):
        self._phaseValidator(phase)
        self.tempVarStep[phase] = {}

    def resetTempVar_epoch(self, phase):
        self._phaseValidator(phase)
        self.tempVarEpoch[phase] = {}

    def resetTempVar_run(self, phase):
        self._phaseValidator(phase)
        self.tempVarRun[phase] = {}

    def resetTempVarRun_allPhases(self):
        self.tempVarRun_allPhases = {}

    def _phaseValidator(self, phase):
        phases = list(self.phases.keys())
        if phase not in phases:
            phasesStr = joinListWithComma(phases)
            raise ValueError(f'{phase} for reset TempVar must be one of {phasesStr}')
