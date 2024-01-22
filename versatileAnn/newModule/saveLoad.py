from utils.vAnnGeneralUtils import _allowOnlyCreationOf_ChildrenInstances


class _NewWrapper_saveLoad:  # kkk1 do it later
    def __init__(self, **kwargs):
        # not allowing this class to have direct instance
        _allowOnlyCreationOf_ChildrenInstances(self, _NewWrapper_saveLoad)
