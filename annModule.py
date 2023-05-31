# annModule.py
import torch
import torch.nn as nn
import torch.optim as optim
import inspect
import types
import os
from torch.utils.tensorboard import SummaryWriter
import concurrent.futures

def randomIdFunc(stringLength=4):
    import random
    import string
    characters = string.ascii_letters + string.digits
    
    return ''.join(random.choices(characters, k=stringLength))

class PostInitCaller(type):
    def __call__(cls, *args, **kwargs):
        obj = type.__call__(cls, *args, **kwargs)
        obj.__post_init__()
        return obj

class ann(nn.Module, metaclass=PostInitCaller):#kkk do hyperparam search maybe with mopso(note to utilize the tensorboard)
    def __init__(self):
        super(ann, self).__init__()
        self.getInitInpArgs()
        self.device= torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.patience = 10
        self.optimizer = None
        self.tensorboardWriter = None
        self.batchSize = 64
        self.evalBatchSize = 1024
        self.saveOnDiskPeriod = 5
        self.regularization = ['l2', 1e-3]
        
        # define model here
        # self.layer1 = self.linLReluDropout(40, 160, dropoutRate=0.5)
        # self.layer2 = self.linLSigmoidDropout(200, 300, dropoutRate=0.2)
        #self.layer3 = nn.Linear(inputSize, 4*inputSize)
        # Add more layers as needed
    def __post_init__(self):
        '# ccc this is ran after child class constructor'
        if isinstance(self, ann) and not type(self) == ann:
            self.to(self.device)
            # self.initOptimizer()

    def forward(self, x):
        # define forward step here
        # x = self.layer1(x)
        # x = self.layer2(x)
        # x = self.layer3(x)
        return x
    
    @property
    def device(self):
        return self._device
    
    @device.setter
    def device(self, value):
        assert value in [torch.device(type='cuda'),torch.device(type='cpu')],"device={value} must be 'cuda' or 'cpu'"
        self._device = value
        self.to(value)
    
    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, value):
        assert isinstance(value, (optim.Optimizer, type(None))),f'optimizerType={type(value)} is not correct'
        self._optimizer = value
    
    def getInitInpArgs(self):
        numOfFbacksNeeded=0
        foundAnn=False
        classHierarchy = self.__class__.mro()
        for i in range(len(classHierarchy)):
            if classHierarchy[i]==ann:
                numOfFbacksNeeded = i +1
                foundAnn=True
                break
        assert foundAnn,'it seems the object is not from ann class'
        frame_ = inspect.currentframe()
        for i in range(numOfFbacksNeeded):
            frame_=frame_.f_back
        args, _, _, values = inspect.getargvalues(frame_)
        "#ccc patience and optimizer's change should not affect that doesnt let the model rerun"
        excludeInputArgs = ['patience', 'optimizer', 'device']
        for eia in excludeInputArgs:
            if eia in args:
                args.remove(eia)
        self.inputArgs = {arg: values[arg] for arg in args if arg != 'self'}
        
    def linLReluDropout(self, innerSize, outterSize, leakyReluNegSlope=0.05, dropoutRate=False, normalization='layer'):
        activation = nn.LeakyReLU(negative_slope=leakyReluNegSlope)
        return self.linActivationDropout(innerSize, outterSize, activation, dropoutRate, normalization=normalization)
    
    def linLSigmoidDropout(self, innerSize, outterSize, dropoutRate=False, normalization='layer'):
        activation = nn.Sigmoid()
        return self.linActivationDropout(innerSize, outterSize, activation, dropoutRate, normalization=normalization)
    
    def linActivationDropout(self, innerSize, outterSize, activation, dropoutRate=None, normalization=None):
        layer = [nn.Linear(innerSize, outterSize)]
    
        if normalization is not None:
            assert normalization in ['batch', 'layer'], f"Invalid normalization option: {normalization}"
            if normalization == 'batch':
                normLayer = nn.BatchNorm1d(outterSize)
            else:
                normLayer = nn.LayerNorm(outterSize)
            layer.append(normLayer)
        layer.append(activation)
    
        if dropoutRate:
            assert isinstance(dropoutRate, (int, float)), f"dropoutRateType={type(dropoutRate)} is not int or float"
            assert 0 <= dropoutRate <= 1, f"dropoutRate={dropoutRate} is not between 0 and 1"
            drLayer = nn.Dropout(p=dropoutRate)
            layer.append(drLayer)
    
        return nn.Sequential(*layer)

    @property
    def tensorboardWriter(self):
        return self._tensorboardWriter
    
    @tensorboardWriter.setter
    def tensorboardWriter(self, tensorboardPath):
        if tensorboardPath:
            os.makedirs(os.path.dirname(tensorboardPath+'_Tensorboard'), exist_ok=True)
            self._tensorboardWriter = SummaryWriter(tensorboardPath+'_Tensorboard')
    
    def initOptimizer(self):
        if list(self.parameters()):
            self.optimizer = optim.Adam(self.parameters(), lr=0.00001)
        else:
            self.optimizer = None
    def tryToSetDefaultOptimizerIfItsNotSet(self):
        if not isinstance(self.optimizer, optim.Optimizer):
            self.initOptimizer()
    
    def havingOptimizerCheck(self):
        self.tryToSetDefaultOptimizerIfItsNotSet()
        assert isinstance(self.optimizer, optim.Optimizer), "model's optimizer is not defined"
    @property
    def lr(self):
        return self.optimizer.param_groups[0]['lr']
    
    @lr.setter
    def lr(self, value):
        self.changeLearningRate(value)
        
    @property
    def learningRate(self):
        return self.optimizer.param_groups[0]['lr']
    
    @learningRate.setter
    def learningRate(self, value):
        self.changeLearningRate(value)
        
    def changeLearningRate(self, newLearningRate):
        self.tryToSetDefaultOptimizerIfItsNotSet()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = newLearningRate
    
    def divideLearningRate(self,factor):
        self.changeLearningRate(self.optimizer.param_groups[0]['lr']/factor)
    
    def saveModel(self, bestModel):
        torch.save({'className':self.__class__.__name__,'classDefinition':inspect.getsource(self.__class__),'inputArgs':self.inputArgs,
                    'model':bestModel}, self.savePath)
        #kkk model save subclass definition for i.e. if there is some property of this object has a encoder class I should save their definitions also
    
    @classmethod
    def loadModel(cls, savePath):
        bestModelDic = torch.load(savePath)
        exec(bestModelDic['classDefinition'], globals())
        classDefinition = globals()[bestModelDic['className']]
        emptyModel = classDefinition(**bestModelDic['inputArgs'])
        emptyModel.load_state_dict(bestModelDic['model'])
        return emptyModel
    
    @property
    def regularization(self):
        return [self._regularizationType, self._regularizationValue]
    
    @regularization.setter
    def regularization(self, value):
        assert value[0] in [None,'l1','l2'],'regularization type should be either None , "l1", "l2"'
        self._regularizationType = value[0]
        self._regularizationValue = value[1]
    
    @property
    def l1Reg(self):
        return self._l1Reg
    
    @l1Reg.setter
    def l1Reg(self, value):
        self._l1Reg = value
        self.regularization=['l1',value]
    
    @property
    def l2Reg(self):
        return self._l2Reg
    
    @l2Reg.setter
    def l2Reg(self, value):
        self._l2Reg = value
        self.regularization=['l2',value]
    
    @property
    def noReg(self):
        return self._noReg
    
    @noReg.setter
    def noReg(self, value):
        self._noReg = value
        self.regularization=[None,None]
    
    def addL1Regularization(self,loss):
        regType,regVal = z1.regularization[0]
        assert regType=='l1','to add l1 regularization the type should be "l1"'
        l1Reg = torch.tensor(0., requires_grad=True)
        for name, param in self.named_parameters():
            if 'weight' in name:
                l1Reg = l1Reg + torch.linalg.norm(param, 1)
        
        loss += regVal * l1Reg
        return loss
    
    def addL2Regularization(self,loss):
        regType,regVal = z1.regularization[0]
        assert regType=='l2','to add l2 regularization the type should be "l2"'
        l2Reg = torch.tensor(0., requires_grad=True)
        for param in self.parameters():
            l2Reg += torch.norm(param)
        
        loss += regVal * l2Reg
        return loss
    
    def addRegularizationToLoss(self, loss):
        regType,regVal=self.regularization
        assert regType in [None,'l1','l2'],'regularization type should be either None , "l1", "l2"'
        if regType==None:
            return loss
        elif regType=='l1':
            return self.addL1Regularization(loss)
        elif regType=='l2':
            return self.addL2Regularization(loss)
            
        
    def batchDatapreparation(self,indexesIndex, indexes, inputs, outputs, batchSize, identifier=None):
        batchIndexes = indexes[indexesIndex*batchSize:indexesIndex*batchSize + batchSize]
        appliedBatchSize = len(batchIndexes)
        
        batchInputs = inputs[batchIndexes].to(self.device)
        batchOutputs = outputs[batchIndexes].to(self.device)
        return batchInputs, batchOutputs, appliedBatchSize, identifier
    
    def getPreTrainStats(self, trainInputs):
        bestValScore = float('inf')
        patienceCounter = 0
        bestModel = None
        bestModelCounter = 1
        
        # Create random indexes for sampling
        indexes = torch.randperm(trainInputs.shape[0])
        batchIterLen = len(trainInputs)//self.batchSize if len(trainInputs) % self.batchSize == 0 else len(trainInputs)//self.batchSize + 1
        return indexes, batchIterLen, bestValScore, patienceCounter, bestModel, bestModelCounter
    
    def checkPatience(self, valScore, bestValScore, patienceCounter, epoch, bestModel, bestModelCounter):#jjj does it need to be on the class
        if valScore < bestValScore:
            bestValScore = valScore
            patienceCounter = 0
            bestModel = self.state_dict()
            bestModelCounter += 1
        else:
            patienceCounter += 1
            if patienceCounter >= self.patience:
                print(f"Early stopping! in {epoch+1} epoch")
                raise StopIteration
        
        if patienceCounter == 0 and (bestModelCounter -1 ) % self.saveOnDiskPeriod == 0:
            # Save the best model to the hard disk
            self.saveModel(bestModel)
        
        return bestValScore, patienceCounter, bestModel, bestModelCounter
    
    def singleProcessTrainModel(self,numEpochs,trainInputs, trainOutputs,valInputs, valOutputs,criterion):
        indexes, batchIterLen, bestValScore, patienceCounter, bestModel, bestModelCounter = self.getPreTrainStats(trainInputs)
        for epoch in range(numEpochs):
            trainLoss = 0.0
            
            for i in range(0, batchIterLen):
                self.optimizer.zero_grad()
                
                with torch.no_grad():
                    batchTrainInputs, batchTrainOutputs, appliedBatchSize, _ = self.batchDatapreparation(i, indexes, trainInputs, trainOutputs, self.batchSize)
                
                batchTrainOutputsPred = self.forward(batchTrainInputs)
                loss = criterion(batchTrainOutputsPred, batchTrainOutputs)
                
                loss.backward()
                self.optimizer.step()
                
                trainLoss += addRegularizationToLoss(loss.item())
            
            trainLoss = trainLoss / len(trainInputs)
            self.tensorboardWriter.add_scalar('train loss', trainLoss, epoch + 1)
            
            valScore = self.evaluateModel(valInputs, valOutputs, criterion, epoch + 1, 'eval')
            print(f"Epoch [{epoch+1}/{numEpochs}], aveItemLoss: {trainLoss:.6f}, evalScore:{valScore}")
            
            bestValScore, patienceCounter, bestModel, bestModelCounter = self.checkPatience(valScore, bestValScore, patienceCounter, epoch, bestModel, bestModelCounter)
        return bestModel
    
    def workerNumRegularization(self,workerNum):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            maxCpus = executor._max_workers
        workerNum=max(min(maxCpus-4,workerNum),1)
        return workerNum
    
    def multiProcessTrainModel(self,numEpochs,trainInputs, trainOutputs,valInputs, valOutputs,criterion, workerNum):
        indexes, batchIterLen, bestValScore, patienceCounter, bestModel, bestModelCounter = self.getPreTrainStats(trainInputs)
        workerNum=self.workerNumRegularization(workerNum)
        totEpochBatchItersNum=numEpochs*batchIterLen
        with concurrent.futures.ThreadPoolExecutor(max_workers=workerNum) as executor:
            readyArgs = []
            identifiers=[]
            futures = []
            for epoch in range(numEpochs):
                trainLoss = 0.0
                
                for i in range(0, batchIterLen):
                    with torch.no_grad():
                        parallelInputArgs = []
                        idIdx = epoch * batchIterLen +i
                        if len(readyArgs) < 10:
                            for jj in range(20 - len(readyArgs)):
                                idIdx2=idIdx+jj
                                if idIdx2>=totEpochBatchItersNum:
                                    continue
                                if idIdx2 in identifiers:
                                    continue
                                if len(parallelInputArgs)<workerNum:
                                    parallelInputArgs.append([idIdx2%batchIterLen, indexes, trainInputs, trainOutputs, self.batchSize, idIdx2])
                                    identifiers.append(idIdx2)
                                else:
                                    break
                            for args in parallelInputArgs:
                                future = executor.submit(self.batchDatapreparation, *args)
                                futures.append(future)
    
                    # Wait until at least one future is completed
                    while not readyArgs:
                        while futures:
                            result = futures[0].result()
                            readyArgs.append(result)
                            futures.pop(0)#
                        continue
                    self.optimizer.zero_grad()
                    
                    batchTrainInputs, batchTrainOutputs, appliedBatchSize, identifier = readyArgs[0]
                    identifiers.remove(identifier)
                    readyArgs.pop(0)
                    
                    batchTrainOutputsPred = self.forward(batchTrainInputs)
                    loss = criterion(batchTrainOutputsPred, batchTrainOutputs)
                    
                    loss.backward()
                    self.optimizer.step()
                    
                    trainLoss += addRegularizationToLoss(loss.item())
                
                trainLoss = trainLoss / len(trainInputs)
                self.tensorboardWriter.add_scalar('train loss', trainLoss, epoch + 1)
                
                valScore = self.evaluateModel(valInputs, valOutputs, criterion, epoch + 1, 'eval', workerNum)
                print(f"Epoch [{epoch+1}/{numEpochs}], aveItemLoss: {trainLoss:.6f}, evalScore:{valScore}")
                
                bestValScore, patienceCounter, bestModel, bestModelCounter = self.checkPatience(valScore, bestValScore, patienceCounter, epoch, bestModel, bestModelCounter)
        return bestModel
    
    def trainModel(self, trainInputs, trainOutputs, valInputs, valOutputs, criterion, numEpochs, savePath, tensorboardPath='', workerNum=0):
        self.havingOptimizerCheck()
        randomId=randomIdFunc()#kkk keepId or not
        self.savePath=savePath+'_'+randomId
        print(f'model will be saved in {self.savePath}')
        os.makedirs(os.path.dirname(self.savePath), exist_ok=True)
        if tensorboardPath:
            tensorboardPath+=randomId
        else:
            tensorboardPath = self.savePath
        self.tensorboardWriter = tensorboardPath#kkk may add print 'access to tensorboard with "tensorboard --logdir=data" from terminal' (I need to take first part of path from tensorboardPath)
        
        self.train()
        if workerNum:
            bestModel = self.multiProcessTrainModel(numEpochs, trainInputs, trainOutputs, valInputs, valOutputs, criterion, workerNum)
        else:
            bestModel = self.singleProcessTrainModel(numEpochs, trainInputs, trainOutputs, valInputs, valOutputs, criterion)
        
        print("Training finished.")
        
        # Save the best model to the hard disk
        self.saveModel(bestModel)
        
        # Load the best model into the current instance
        self.load_state_dict(bestModel)
        
        # Return the best model
        return self
    
    def evaluateModel(self, inputs, outputs, criterion, stepNum=0, evalOrTest='Test', workerNum=0):
        self.eval()
        with torch.no_grad():
            if workerNum:
                evalLoss = self.multiProcessEvaluateModel(inputs, outputs, criterion, stepNum, evalOrTest, workerNum)
            else:
                evalLoss = self.singleProcessEvaluateModel(inputs, outputs, criterion, stepNum, evalOrTest)
            if hasattr(self, 'tensorboardWriter'):
                self.tensorboardWriter.add_scalar(f'{evalOrTest} loss', evalLoss, stepNum)
            return evalLoss
    
    def getPreEvalStats(self, inputs):
        indexes = torch.arange(len(inputs))
        batchIterLen = len(inputs)//self.evalBatchSize if len(inputs) % self.evalBatchSize == 0 else len(inputs)//self.evalBatchSize + 1
        return indexes, batchIterLen
    
    def singleProcessEvaluateModel(self, inputs, outputs, criterion, stepNum, evalOrTest):
        evalLoss = 0.0
        indexes, batchIterLen = self.getPreEvalStats(inputs)
        for i in range(0, batchIterLen):
            batchEvalInputs, batchEvalOutputs, appliedBatchSize,_ = self.batchDatapreparation(i, indexes, inputs, outputs, batchSize=self.evalBatchSize)
            
            batchEvalOutputsPred = self.forward(batchEvalInputs)
            loss = criterion(batchEvalOutputsPred, batchEvalOutputs)
            
            evalLoss += loss.item()
        
        evalLoss /= inputs.shape[0]
        return evalLoss

    def multiProcessEvaluateModel(self, inputs, outputs, criterion, stepNum, evalOrTest, workerNum):
        evalLoss = 0.0
        workerNum=self.workerNumRegularization(workerNum)
        indexes, batchIterLen = self.getPreEvalStats(inputs)
        with concurrent.futures.ThreadPoolExecutor(max_workers=workerNum) as executor:
            readyArgs = []
            identifiers=[]
            futures = []
            for i in range(0, batchIterLen):
                parallelInputArgs = []
                if len(readyArgs) < 10:
                    for jj in range(20 - len(readyArgs)):
                        idIdx2=i+jj
                        if idIdx2>=batchIterLen:
                            continue
                        if idIdx2 in identifiers:
                            continue
                        if len(parallelInputArgs)<workerNum:
                            parallelInputArgs.append([idIdx2%batchIterLen, indexes, inputs, outputs, self.evalBatchSize, idIdx2])
                            identifiers.append(idIdx2)
                        else:
                            break
                    for args in parallelInputArgs:
                        future = executor.submit(self.batchDatapreparation, *args)
                        futures.append(future)
                # Wait until at least one future is completed
                while not readyArgs:
                    while futures:
                        result = futures[0].result()
                        readyArgs.append(result)
                        futures.pop(0)#
                    continue
                batchEvalInputs, batchEvalOutputs, appliedBatchSize, identifier = readyArgs[0]
                identifiers.remove(identifier)
                readyArgs.pop(0)
                
                batchEvalOutputsPred = self.forward(batchEvalInputs)
                loss = criterion(batchEvalOutputsPred, batchEvalOutputs)
                
                evalLoss += loss.item()
            
            evalLoss /= inputs.shape[0]
            return evalLoss
    
