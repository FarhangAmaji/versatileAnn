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
'#ccc access to tensorboard with "tensorboard --logdir=data" from terminal'
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
#%% bad
from functools import wraps
class Base:

    @staticmethod
    def get_initial_args_wrapper(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            # This code will run automatically before each __init__ method
            # in each subclass: 
            
            # if you want to annotate arguments passed in order,
            # as just "args", then you will indeed have to resort
            # to the "inspect" module - 
            # but, `inspect.signature`, followed by `signature.bind` calls 
            # instead of dealing with frames.
    
            # for named args, this will just work to annotate all of them:
            input_args = getattr(self, "input_args", {})
            
            input_args.update(kwargs)
            self.input_args = input_args
    
            # after the arguments are recorded as instance attributes, 
            # proceed to the original __init__ call:
            return func(self, *args, **kwargs)
        return wrapper
        
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if "__init__" in cls.__dict__:
            setattr(cls, "__init__", cls.get_initial_args_wrapper(cls.__init__))
class Ann(Base):
    def __init__(self, arg1):
        super().__init__()
    
class MyAnn(Ann):
    def __init__(self, input_size, output_size):
        super().__init__(4)
z1=MyAnn(40, 1)
print(z1.input_args)
# outputs:

# {'input_size': 40, 'output_size': 1, 'arg1': 4}
#%%

#%%

#%%

#%%

#%%

#%%


