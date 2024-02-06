from utils.vAnnGeneralUtils import _allowOnlyCreationOf_ChildrenInstances


class _NewWrapper_saveLoad:  # kkk1 do it later
    def __init__(self, **kwargs):
        # not allowing this class to have direct instance
        _allowOnlyCreationOf_ChildrenInstances(self, _NewWrapper_saveLoad)


    def on_save_checkpoint(self, checkpoint: dict):
        # reimplement this method to save additional information to the checkpoint

        # Add additional information to the checkpoint
        checkpoint['newWrapper'] = {
            '_initArgs': self._initArgs,
            'allDefinitions': self.allDefinitions,
            'warnsFrom_getAllNeededDefinitions': self.warnsFrom_getAllNeededDefinitions,
        }

    def on_load_checkpoint(self, checkpoint: dict):
        # Load additional information from the checkpoint
        additionalInfo = checkpoint['newWrapper']

        _initArgs = additionalInfo['_initArgs']
        pl.seed_everything(_initArgs['__plSeed__'])
        # I guess setting seed here doesn't really make difference
        # on the most models but some models which may use some random
        # variables in their implementation, may benefit from this
