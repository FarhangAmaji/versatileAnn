#%% imports
# trainAnn.py
import os
baseFolder = os.path.dirname(os.path.abspath(__file__))
os.chdir(baseFolder)
from annModule import ann
import inspect
import torch
import torch.optim as optim
#%% define model
class myAnn(ann):
    def __init__(self, inputSize, outputSize):
        super(myAnn, self).__init__(inspect.currentframe())
        self.layer1 = self.linLReluDropout(inputSize, inputSize*4, dropoutRate=0.5)
        self.layer2 = self.linLReluDropout(inputSize*4, inputSize*4, dropoutRate=0.8)
        self.layer3 = self.linLReluDropout(inputSize*4, outputSize)
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x
#%% make model instance
z1=myAnn(40,1)
#%%
'#ccc how to set optimizer manually'
# z1.lr=0.001
# z1.learningRate=0.001
# z1.changeLearningRate(0.001)
# z1.optimizer=optim.Adam(z1.parameters(), lr=0.4)
# z1.batchSize=32
# z1.evalBatchSize=1024
# z1.device=torch.device(type='cpu')

# z1.patience=10
# z1.saveOnDiskPeriod=1
#%% regression test
# Set random seed for reproducibility
torch.manual_seed(42)
import time
t0=time.time()
trainInputs = torch.randn(100, 40)  # Assuming 100 training samples with 40 features each
trainOutputs = torch.randn(100, 1)  # Assuming 100 training output values

testInputs = torch.randn(50, 40)  # Assuming 50 testing samples with 40 features each
testOutputs = torch.randn(50, 1)  # Assuming 50 testing output values

# Define the criterion (loss function)
criterion = torch.nn.MSELoss()  # Example: Mean Squared Error (MSE) loss

# Train the model
z1.trainModel(trainInputs, trainOutputs, testInputs, testOutputs, criterion, numEpochs=10, savePath=r'data\bestModels\a1')

# Evaluate the model
evalLoss = z1.evaluateModel(testInputs, testOutputs, criterion)
print("Evaluation Loss:", evalLoss)
print('time:',time.time()-t0)
#%% 
runcell('imports', 'F:/projects/public github projects/private repos/versatileAnnModule/trainAnn.py')
runcell('define model', 'F:/projects/public github projects/private repos/versatileAnnModule/trainAnn.py')
runcell('make model instance', 'F:/projects/public github projects/private repos/versatileAnnModule/trainAnn.py')
runcell('regression test', 'F:/projects/public github projects/private repos/versatileAnnModule/trainAnn.py')
#%%
#%%
#%%
#%%
#%%
#%%
import inspect

class ann():
    def __init__(self, arg1):
        super(ann, self).__init__()
        self.getInitInpArgs()
    def getInitInpArgs(self):
        args, _, _, values = inspect.getargvalues(inspect.currentframe().f_back)
        self.inputArgs = {arg: values[arg] for arg in args if arg != 'self'}
class myAnn(ann):
    def __init__(self, inputSize, outputSize):
        super(myAnn, self).__init__(4)
        self.layer1 = 8
        self.layer2 = 16
z1=myAnn(40,1)
#%% gpt 
import inspect

class ann():
    def __init__(self, arg1):
        super(ann, self).__init__()
        self.getInitInpArgs()
    
    def getInitInpArgs(self):
        args, _, _, values = inspect.getargvalues(inspect.currentframe().f_back)
        self.inputArgs = {arg: values[arg] for arg in args if arg != 'self'}
        
class myAnn(ann):
    def __init__(self, inputSize, outputSize):
        super(myAnn, self).__init__(4)
        self.layer1 = 8
        self.layer2 = 16

z1 = myAnn(40, 1)
z1.inputArgs
#%% bit good
import inspect

class ann():
    def __init__(self, arg1):
        super(ann, self).__init__()
        self.getInitInpArgs()
    
    def getInitInpArgs(self):
        argspec = inspect.getfullargspec(self.__init__)
        args = argspec.args[1:]
        defaults = argspec.defaults or ()
        default_args = dict(zip(args[-len(defaults):], defaults))
        self.inputArgs = {arg: default_args.get(arg) for arg in args}

class myAnn(ann):
    def __init__(self, inputSize, outputSize):
        super(myAnn, self).__init__(4)

z1 = myAnn(40, 1)
z1.inputArgs
#%%
import inspect

class ann():
    def __init__(self, arg1):
        super(ann, self).__init__()
        self.getInitInpArgs()
    
    def getInitInpArgs(self):
        frame = inspect.currentframe().f_back
        argspec = inspect.getfullargspec(frame.f_code)
        args = argspec.args[1:]
        self.inputArgs = {arg: frame.f_locals.get(arg) for arg in args}

class myAnn(ann):
    def __init__(self, inputSize, outputSize):
        super(myAnn, self).__init__(4)
        self.layer1 = 8
        self.layer2 = 16

z1 = myAnn(40, 1)
z1.inputArgs
#%%
import inspect

class ann():
    def __init__(self, arg1):
        super(ann, self).__init__()
        self.getInitInpArgs()
    
    def getInitInpArgs(self):
        frame = inspect.currentframe().f_back
        argspec = inspect.signature(frame.f_code)
        args = list(argspec.parameters)[1:]
        self.inputArgs = {arg: frame.f_locals.get(arg) for arg in args}

class myAnn(ann):
    def __init__(self, inputSize, outputSize):
        super(myAnn, self).__init__(4)
        self.layer1 = 8
        self.layer2 = 16

z1 = myAnn(40, 1)
z1.inputArgs
#%%
import inspect

class ann():
    def __init__(self, arg1):
        super(ann, self).__init__()
        self.getInitInpArgs()
    
    def getInitInpArgs(self):
        frame = inspect.currentframe().f_back
        args = inspect.getargs(frame.f_code.co_consts[0])
        args = args[1:]  # Exclude 'self' argument
        self.inputArgs = {arg: frame.f_locals.get(arg) for arg in args}

class myAnn(ann):
    def __init__(self, inputSize, outputSize):
        super(myAnn, self).__init__(4)
        self.layer1 = 8
        self.layer2 = 16

z1 = myAnn(40, 1)
z1.inputArgs
#%% 
class ann():
    def __init__(self, arg1, **kwargs):
        super(ann, self).__init__()
        self.inputArgs = kwargs

class myAnn(ann):
    def __init__(self, inputSize, outputSize):
        super(myAnn, self).__init__(4, inputSize=inputSize, outputSize=outputSize)
        self.layer1 = 8
        self.layer2 = 16

z1 = myAnn(40, 1)
z1.inputArgs
#%%
class ann():
    def __new__(cls, *args, **kwargs):
        instance = super().__new__(cls)
        instance.getInitInpArgs(*args, **kwargs)
        return instance

    def getInitInpArgs(self, *args, **kwargs):
        self.inputArgs = {**kwargs}

class myAnn(ann):
    def __init__(self, inputSize, outputSize):
        super().__init__()
        self.layer1 = 8
        self.layer2 = 16

z1 = myAnn(40, 1)
z1.inputArgs
#%%
class ann():
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.getInitInpArgs(*args, **kwargs)

    def getInitInpArgs(self, *args, **kwargs):
        self.inputArgs = {**kwargs}

class myAnn(ann):
    def __init__(self, inputSize, outputSize):
        super().__init__(inputSize=inputSize, outputSize=outputSize)
        self.layer1 = 8
        self.layer2 = 16

z1 = myAnn(40, 1)
z1.inputArgs
#%%
import inspect

class ann():
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.getInitInpArgs()

    def getInitInpArgs(self):
        frame = inspect.currentframe().f_back
        argspec = inspect.getfullargspec(frame.f_code)
        args = argspec.args[1:]
        self.inputArgs = {arg: frame.f_locals.get(arg) for arg in args}

class myAnn(ann):
    def __init__(self, inputSize, outputSize):
        super().__init__()
        self.inputSize = inputSize
        self.outputSize = outputSize
        self.layer1 = 8
        self.layer2 = 16

z1 = myAnn(40, 1)
z1.inputArgs
#%%
import inspect

class ann():
    def __init__(self, arg1):
        super(ann, self).__init__()
        self.getInitInpArgs()
    
    def getInitInpArgs(self):
        argspec = inspect.getfullargspec(self.__init__)
        args = argspec.args[1:]
        defaults = argspec.defaults or ()
        num_args = len(args)
        num_defaults = len(defaults)
        
        default_args = {}
        if num_args > num_defaults:
            default_args = dict(zip(args[num_args - num_defaults:], defaults))
        
        self.inputArgs = {arg: default_args.get(arg) for arg in args}

class myAnn(ann):
    def __init__(self, inputSize, outputSize):
        super(myAnn, self).__init__(4)

z1 = myAnn(40, 1)
z1.inputArgs
#%%
import inspect

class ann():
    def __init__(self, arg1):
        super().__init__()
        self.getInitInpArgs()
    
    def getInitInpArgs(self):
        frame = inspect.currentframe().f_back
        argspec = inspect.getfullargspec(frame.f_code)
        args = argspec.args[1:]
        values = frame.f_locals
        self.inputArgs = {arg: values.get(arg) for arg in args}

class myAnn(ann):
    def __init__(self, inputSize, outputSize):
        super().__init__(4)

z1 = myAnn(40, 1)
z1.inputArgs

#%%
import inspect

class ann():
    def __init__(self, arg1):
        super().__init__()
        self.getInitInpArgs()
    
    def getInitInpArgs(self):
        sig = inspect.signature(self.__init__)
        params = sig.parameters
        self.inputArgs = {name: param.default for name, param in params.items() if param.default != inspect.Parameter.empty}

class myAnn(ann):
    def __init__(self, inputSize, outputSize):
        super().__init__(4)

z1 = myAnn(40, 1)
z1.inputArgs
#%%
import inspect

class ann():
    def __init__(self, arg1):
        super().__init__()
        self.getInitInpArgs()
    
    def getInitInpArgs(self):
        cls = type(self)
        sig = inspect.signature(cls)
        params = sig.parameters
        self.inputArgs = {name: param.default for name, param in params.items() if param.default != inspect.Parameter.empty}

class myAnn(ann):
    def __init__(self, inputSize, outputSize):
        super().__init__(4)

z1 = myAnn(40, 1)
z1.inputArgs
#%%
class ann():
    def __init__(self, arg1):
        super().__init__()
        self.getInitInpArgs()
    
    def getInitInpArgs(self):
        cls_annotations = getattr(self.__class__, '__annotations__', {})
        self.inputArgs = {arg: getattr(self, arg, None) for arg in cls_annotations.keys()}

class myAnn(ann):
    def __init__(self, inputSize, outputSize):
        super().__init__(4)
        self.inputSize = inputSize
        self.outputSize = outputSize

z1 = myAnn(40, 1)
z1.inputArgs
#%%
import inspect

class ann():
    def __init__(self, arg1):
        super().__init__()
        self.getInitInpArgs()
    
    def getInitInpArgs(self):
        frame = inspect.currentframe().f_back
        argspec = inspect.getargvalues(frame)
        self.inputArgs = {arg: value for arg, value in argspec.locals.items() if arg != 'self'}

class myAnn(ann):
    def __init__(self, inputSize, outputSize):
        super().__init__(4)

z1 = myAnn(40, 1)
z1.inputArgs
#%%
import inspect

class ann():
    def __init__(self, arg1):
        super().__init__()
    
    def getInitInpArgs(self):
        argspec = inspect.getfullargspec(self.__init__)
        args = argspec.args[1:]
        defaults = argspec.defaults or ()
        default_args = dict(zip(args[-len(defaults):], defaults))
        self.inputArgs = {arg: getattr(self, arg, default_args.get(arg)) for arg in args}

class myAnn(ann):
    def __init__(self, inputSize, outputSize):
        super().__init__(4)

z1 = myAnn(40, 1)
z1.getInitInpArgs()
z1.inputArgs
#%%
import inspect

class InitArgsMeta(type):
    def __call__(cls, *args, **kwargs):
        obj = super().__call__(*args, **kwargs)
        obj.getInitInpArgs()
        return obj

class ann():
    def __init__(self, arg1):
        super().__init__()
    
    def getInitInpArgs(self):
        frame = inspect.currentframe().f_back
        argspec = inspect.getfullargspec(frame.f_code)
        args = argspec.args[1:]
        defaults = argspec.defaults or ()
        default_args = dict(zip(args[-len(defaults):], defaults))
        self.inputArgs = {arg: getattr(self, arg, default_args.get(arg)) for arg in args}

class myAnn(ann, metaclass=InitArgsMeta):
    def __init__(self, inputSize, outputSize):
        super().__init__(4)

z1 = myAnn(40, 1)
z1.inputArgs
#%%
import inspect

class InitArgsMeta(type):
    def __call__(cls, *args, **kwargs):
        obj = super().__call__(*args, **kwargs)
        obj.getInitInpArgs()
        return obj

class ann():
    def __init__(self, arg1):
        super().__init__()

    def getInitInpArgs(self):
        frame = inspect.currentframe().f_back
        sig = inspect.signature(frame.f_locals['self'].__class__.__init__)
        self.inputArgs = {param.name: getattr(self, param.name, param.default) for param in sig.parameters.values() if param.name != 'self'}

class myAnn(ann, metaclass=InitArgsMeta):
    def __init__(self, inputSize, outputSize):
        super().__init__(4)

z1 = myAnn(40, 1)
z1.inputArgs
#%%
import inspect

class InitArgsMeta(type):
    def __call__(cls, *args, **kwargs):
        obj = super().__call__(*args, **kwargs)
        obj.getInitInpArgs()
        return obj

class ann():
    def __init__(self, arg1):
        super().__init__()

    def getInitInpArgs(self):
        frame = inspect.currentframe().f_back.f_back
        sig = inspect.signature(frame.f_locals['self'].__class__.__init__)
        self.inputArgs = {param.name: getattr(self, param.name, param.default) for param in sig.parameters.values() if param.name != 'self'}

class myAnn(ann, metaclass=InitArgsMeta):
    def __init__(self, inputSize, outputSize):
        super().__init__(4)

z1 = myAnn(40, 1)
z1.inputArgs
#%%
import inspect

class InitArgsMeta(type):
    def __call__(cls, *args, **kwargs):
        obj = super().__call__(*args, **kwargs)
        obj.getInitInpArgs()
        return obj

class ann():
    def __init__(self, arg1):
        super().__init__()

    def getInitInpArgs(self):
        frame = inspect.currentframe().f_back
        self_locals = frame.f_locals
        self_class = self_locals.get('self')
        if self_class:
            sig = inspect.signature(self_class.__class__.__init__)
            self.inputArgs = {param.name: getattr(self_class, param.name, param.default) for param in sig.parameters.values() if param.name != 'self'}
        else:
            self.inputArgs = {}

class myAnn(ann, metaclass=InitArgsMeta):
    def __init__(self, inputSize, outputSize):
        super().__init__(4)

z1 = myAnn(40, 1)
z1.inputArgs
#%%
import inspect

class InitArgsMeta(type):
    def __call__(cls, *args, **kwargs):
        obj = super().__call__(*args, **kwargs)
        obj.getInitInpArgs()
        return obj

class ann():
    def __init__(self, arg1):
        super().__init__()

    def getInitInpArgs(self):
        frame = inspect.currentframe().f_back
        self_locals = frame.f_locals
        self_class = self_locals.get('self')
        if self_class:
            sig = inspect.signature(self_class.__class__.__init__)
            self.inputArgs = {param.name: getattr(self_class, param.name, param.default) for param in sig.parameters.values() if param.name != 'self'}
        else:
            self.inputArgs = {}

class myAnn(ann, metaclass=InitArgsMeta):
    def __init__(self, inputSize, outputSize):
        super().__init__(4)

z1 = myAnn(40, 1)
z1.inputArgs
#%%
import inspect

class ann():
    def __init__(self, arg1):
        super().__init__()
        self.getInitInpArgs()
    
    def getInitInpArgs(self):
        frame = inspect.currentframe().f_back
        self_class = frame.f_locals.get('self')
        if self_class:
            self.inputArgs = self._get_input_args(self_class)
        else:
            self.inputArgs = {}
    
    def _get_input_args(self, instance):
        init_signature = inspect.signature(instance.__init__)
        init_params = init_signature.parameters
        args = {param_name: instance.__dict__[param_name] for param_name in init_params if param_name != 'self'}
        return args

class myAnn(ann):
    def __init__(self, inputSize, outputSize):
        super().__init__(4)
        self.inputSize = inputSize
        self.outputSize = outputSize

z1 = myAnn(40, 1)
z1.inputArgs
#%%
import inspect

class InitArgsMeta(type):
    def __call__(cls, *args, **kwargs):
        obj = super().__call__(*args, **kwargs)
        obj.getInitInpArgs()
        return obj

class ann():
    def __init__(self, arg1):
        super().__init__()

    def getInitInpArgs(self):
        frame = inspect.currentframe().f_back
        self_locals = frame.f_locals
        self_class = self_locals.get('self')
        if self_class:
            frame_info = inspect.getframeinfo(frame)
            code = frame_info.code_context[0].strip()
            code = code[code.index('(') + 1 : code.rindex(')')]
            arg_values = frame.f_locals.copy()
            arg_values.pop('self', None)
            self.inputArgs = {arg: arg_values[arg] for arg in code.split(',') if arg.strip() in arg_values}
        else:
            self.inputArgs = {}

class myAnn(ann, metaclass=InitArgsMeta):
    def __init__(self, inputSize, outputSize):
        super().__init__(4)

z1 = myAnn(40, 1)
z1.inputArgs
#%%
import inspect

class ann():
    def __init__(self, arg1):
        self.getInitInpArgs()
    
    def getInitInpArgs(self):
        frame = inspect.currentframe().f_back
        argspec = inspect.getfullargspec(frame.f_locals['__init__'])
        args = argspec.args[1:]
        defaults = argspec.defaults or ()
        default_args = dict(zip(args[-len(defaults):], defaults))
        self.inputArgs = {arg: frame.f_locals.get(arg, default_args.get(arg)) for arg in args}

class myAnn(ann):
    def __init__(self, inputSize, outputSize):
        super().__init__(4)

z1 = myAnn(40, 1)
z1.inputArgs
#%%
import inspect

class ann():
    def __init__(self, arg1):
        self.getInitInpArgs()
    
    def getInitInpArgs(self):
        frame = inspect.currentframe().f_back
        _, _, _, values = inspect.getargvalues(frame)
        self.inputArgs = {arg: values[arg] for arg in values if arg != 'self'}

class myAnn(ann):
    def __init__(self, inputSize, outputSize):
        super().__init__(4)

z1 = myAnn(40, 1)
z1.inputArgs
#%%
import inspect

class ann():
    def __init__(self, arg1):
        self.getInitInpArgs()
    
    def getInitInpArgs(self):
        frame = inspect.currentframe().f_back
        _, _, _, values = inspect.getargvalues(frame)
        self.inputArgs = {arg: values[arg] for arg in values if arg != 'self'}
    
    def __init_subclass__(cls, **kwargs):
        frame = inspect.currentframe().f_back
        argspec = inspect.getfullargspec(frame.f_code)
        args = argspec.args[1:]
        defaults = argspec.defaults or ()
        default_args = dict(zip(args[-len(defaults):], defaults))
        cls.inputArgs = {arg: default_args.get(arg) for arg in args}

class myAnn(ann):
    def __init__(self, inputSize, outputSize):
        super().__init__(4)

z1 = myAnn(40, 1)
z1.inputArgs
#%%
import inspect

class ann():
    def __init__(self, arg1):
        self.getInitInpArgs()
    
    def getInitInpArgs(self):
        frame = inspect.currentframe().f_back
        _, _, _, values = inspect.getargvalues(frame)
        self.inputArgs = {arg: values[arg] for arg in values if arg != 'self'}
    
    def __init_subclass__(cls, **kwargs):
        frame = inspect.currentframe().f_back
        signature = inspect.signature(frame.f_locals['__init__'])
        parameters = signature.parameters
        default_args = {name: parameter.default for name, parameter in parameters.items() if parameter.default is not inspect.Parameter.empty}
        cls.inputArgs = default_args

class myAnn(ann):
    def __init__(self, inputSize, outputSize):
        super().__init__(4)

z1 = myAnn(40, 1)
z1.inputArgs
#%%
class PostInitCaller(type):
    def __call__(cls, *args, **kwargs):
        obj = type.__call__(cls, *args, **kwargs)
        obj.__post_init__()
        return obj


class BaseClass(metaclass=PostInitCaller):  

    def __init__(self,frame_):
        args, _, _, values = inspect.getargvalues(frame_)
        print(args)
        print('base __init__')
        self.common1()

    def common1(self):
        print('common 1')
        
    def getInitInpArgs(self):
        args, _, _, values = inspect.getargvalues(inspect.currentframe().f_back.f_back.f_back.f_back.f_back.f_back.f_back.f_back.f_back.f_back.f_back.f_back.f_back.f_back.f_back)
        print(args)
        self.inputArgs = {arg: values[arg] for arg in args if arg != 'self'}

    def finalizeInitialization(self):
        print('finalizeInitialization [common2]')
        self.getInitInpArgs()

    def __post_init__(self): # this is called at the end of __init__
        self.finalizeInitialization()

class Subclass1(BaseClass):
    def __init__(self,jj,zz):
        super().__init__(inspect.currentframe())
        self.specific()

    def specific(self):
        print('specific')


s = Subclass1(11,22) 



