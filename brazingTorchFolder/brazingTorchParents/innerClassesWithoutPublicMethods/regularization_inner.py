from brazingTorchFolder.layers.customLayers import VAnnCustomLayer
from brazingTorchFolder.lossRegulator import LossRegulator
from projectUtils.globalVars import regularizationTypes
from projectUtils.misc import _allowOnlyCreationOf_ChildrenInstances
from projectUtils.typeCheck import argValidator


class _BrazingTorch_regularization_inner:
    _regularizationTypes = regularizationTypes
    nullRegulator = LossRegulator(LossRegulator.nullDictValue)

    @argValidator
    def __init__(self, **kwargs):

        # not allowing this class to have direct instance
        _allowOnlyCreationOf_ChildrenInstances(self, _BrazingTorch_regularization_inner)

    # ---- specific layer regularizations
    def _register_VAnnCustomLayers_regularizations(self):
        # goodToHave1
        #  now only detects if VAnnCustomLayer is the main layer in model but if
        #  VAnnCustomLayer is in a class and that class is the main layer in model
        #  regularization won't get detected; think about self.modules() and not self._modules
        # ccc1
        #  VAnnCustomLayers can have regularization on their layer
        # ccc1
        #  vars(self)['_modules'](ofc not used here but may be used elsewhere) is similar to
        #  self._modules. these contain only the direct children modules of the model
        for layerName, layer in self._modules.items():
            if isinstance(layer, VAnnCustomLayer):
                if layer.regularization:  # Llr1
                    if layerName not in self._specificLayerRegularization.keys():
                        self._specificLayerRegularization[layerName] = layer.regularization

    # ----
    def _setOperationalRegularizations(self):
        # note _operationalRegularizations is set on each run not to slow down
        # also renewed if there are changes on different runs
        self._operationalRegularizations = {}  # reset

        self._register_VAnnCustomLayers_regularizations()
        hasGeneralRegularization = False if self.generalRegularization.type == 'None' else True

        for name, param in self.named_parameters():
            layerName = name.split('.')[0]
            # note name looks like 'layer1Name.layer.0.weight' which is self.layer1Name.layer[0].weight
            # note layer1Name is the name of attribute of the model class so layerName here gets
            # name of attribute of the model:layer1Name

            # if the layer has a regularization of it's own then that one is applied
            # otherwise the general regularization if available is applied
            if layerName in self._specificLayerRegularization.keys():
                self._operationalRegularizations[layerName] = \
                    self._specificLayerRegularization[layerName]
            else:
                if hasGeneralRegularization:
                    self._operationalRegularizations[layerName] = self.generalRegularization
