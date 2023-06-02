# versatileAnn\annModule.py
import torch
import torch.nn as nn
import torch.optim as optim
import inspect
import types
import os
from torch.utils.tensorboard import SummaryWriter
import concurrent.futures
from .utils import randomIdFunc

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
        self.neededDefinitions=None
        
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
        torch.save({'className':self.__class__.__name__,'classDefinition':self.neededDefinitions,'inputArgs':self.inputArgs,
                    'model':bestModel}, self.savePath)
    
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
        regType,regVal = self.regularization
        assert regType=='l1','to add l1 regularization the type should be "l1"'
        l1Reg = torch.tensor(0., requires_grad=True)
        for name, param in self.named_parameters():
            if 'weight' in name:
                l1Reg = l1Reg + torch.linalg.norm(param, 1)
        
        loss = loss + regVal * l1Reg
        return loss
    
    def addL2Regularization(self,loss):
        regType,regVal = self.regularization
        assert regType=='l2','to add l2 regularization the type should be "l2"'
        l2Reg = torch.tensor(0., requires_grad=True)
        for param in self.parameters():
            l2Reg = l2Reg + torch.norm(param)
        
        loss = loss + regVal * l2Reg
        return loss
    
    def addRegularizationToLoss(self, loss):#kkk add layer specific regularization
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
    
    # Get the definitions of all instance variables of an object
    def getAllNeededDefinitions(self,obj):#kkk add metaclass, function and classmethod and methods of class 
        
        def findClassNamesAndDependencies(classDefinitions):
            classDefinitionsWithInfo = []

            for i in range(len(classDefinitions)):
                classDef=classDefinitions[i]
                # Extract the class name
                classNameWithdependencies = classDef.split("class ")[1].split(":")[0].strip()

                # Find the superclass names (if any)
                if '(' in classNameWithdependencies:
                    class_name = classNameWithdependencies.split("(")[0].strip()
                    superclasses = [s.strip() for s in classNameWithdependencies.split("(")[1].split(")")[0].split(",")]
                    superclasses= [s.split('=')[1].strip() if '=' in s else s for s in superclasses]

                    classDefinitionsWithInfo.append({'class_name':class_name,'dep': superclasses, 'def': classDef})
                else:
                    class_name=classNameWithdependencies
                    classDefinitionsWithInfo.append({'class_name':class_name,'dep': [], 'def': classDef})
            return classDefinitionsWithInfo
        def getOrderedClassDefinitions(classDefinitionsWithInfo):
            changes=1
            while changes!=0:
                changes=0
                for i in range(len(classDefinitionsWithInfo)):
                    dependencies=classDefinitionsWithInfo[i]['dep']
                    for k in range(len(dependencies)):
                        for j in range(len(classDefinitionsWithInfo)):
                            if dependencies[k]==classDefinitionsWithInfo[j]['class_name']:
                                if j>i:
                                    classDefinitionsWithInfo[j], classDefinitionsWithInfo[i] = classDefinitionsWithInfo[i], classDefinitionsWithInfo[j]
                                    changes+=1
            classDefinitions=[]
            for i in range(len(classDefinitionsWithInfo)):
                classDefinitions.append(classDefinitionsWithInfo[i]['def'])
            return classDefinitions
        unorderedClasses=self.getAllNeededClassDefinitions(obj)
        unorderedClasses=findClassNamesAndDependencies(unorderedClasses)
        orderedClasses=getOrderedClassDefinitions(unorderedClasses)
        return '\n'.join(orderedClasses)
    def getAllNeededClassDefinitions(self,obj, visitedClasses=set()):
        import builtins
        import pkg_resources

        def isCustomClass(cls_):
            if cls_ is None or cls_ is types.NoneType:
                return False
            moduleName = getattr(cls_, '__module__', '')
            return not (
                cls_ in builtins.__dict__.values()
                or any(moduleName.startswith(package.key) for package in pkg_resources.working_set)
                or moduleName.startswith('collections')
            )

        # Get the definition of a class if it's a custom class
        def getCustomClassDefinition(cls_):
            if cls_==ann:
                return None
            return inspect.getsource(cls_) if isCustomClass(cls_) else None

        def getClassDefinitionsIfNotDoneBefore(cls_, visitedClasses, classDefinitions):
            if cls_ not in visitedClasses:
                visitedClasses.add(cls_)
                classDefinition = getCustomClassDefinition(cls_)
                if classDefinition:
                    classDefinitions.append(classDefinition)

        def getClassAndItsParentsDefinitions(cls_, visitedClasses, classDefinitions):
            getClassDefinitionsIfNotDoneBefore(cls_, visitedClasses, classDefinitions)
            classDefinitions.extend(getParentClassDefinitions(cls_, visitedClasses))

        # Get the definitions of the parent classes recursively
        def getParentClassDefinitions(cls_, visited=set()):
            parentClassDefinitions = []
            for parentClass in cls_.__bases__:
                getClassDefinitionsIfNotDoneBefore(parentClass, visited, parentClassDefinitions)
                if parentClass not in visited:
                    parentClassDefinitions.extend(getParentClassDefinitions(parentClass, visited))
            return parentClassDefinitions
        
        classDefinitions = []
        getClassAndItsParentsDefinitions(obj.__class__, visitedClasses, classDefinitions)

        objVars = vars(obj)
        for varName, varValue in objVars.items():
            # if inspect.isfunction(varValue):
            #     if inspect.ismethod(varValue):
                    
            if inspect.isclass(varValue):
                getClassAndItsParentsDefinitions(varValue, visitedClasses, classDefinitions)
            if varName == '_modules':
                for varNameModules, varValueModules in varValue.items():
                    classDefinitions.extend(self.getAllNeededClassDefinitions(varValueModules, visitedClasses))
            getClassAndItsParentsDefinitions(type(varValue), visitedClasses, classDefinitions)

        return classDefinitions
    
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
                
                trainLoss += self.addRegularizationToLoss(loss.item())
            
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
                    
                    trainLoss += self.addRegularizationToLoss(loss.item())
                
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
        
        if not self.neededDefinitions:
            self.neededDefinitions=self.getAllNeededDefinitions(self)
        
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
                        futures.pop(0)
                    continue
                batchEvalInputs, batchEvalOutputs, appliedBatchSize, identifier = readyArgs[0]
                identifiers.remove(identifier)
                readyArgs.pop(0)
                
                batchEvalOutputsPred = self.forward(batchEvalInputs)
                loss = criterion(batchEvalOutputsPred, batchEvalOutputs)
                
                evalLoss += loss.item()
            
            evalLoss /= inputs.shape[0]
            return evalLoss