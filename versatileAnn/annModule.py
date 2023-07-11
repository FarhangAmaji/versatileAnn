# versatileAnn\annModule.py
import torch
import torch.nn as nn
import torch.optim as optim
import inspect
import os
from torch.utils.tensorboard import SummaryWriter
import concurrent.futures
from .utils import randomIdFunc
from .layers.customLayers import CustomLayer

#kkk make a module to copy trained models weights to raw model with same architecture
#kkk add weight inits orthogonal he_uniform he_normal glorot_uniform glorot_normal lecun_normal
class PostInitCaller(type):
    def __call__(cls, *args, **kwargs):
        obj = type.__call__(cls, *args, **kwargs)
        obj.__post_init__()
        return obj

class ann(nn.Module, metaclass=PostInitCaller):#kkk do hyperparam search maybe with mopso(note to utilize the tensorboard)
    modeNames = ['evalMode', 'variationalAutoEncoderMode', 'dropoutEnsembleMode','timeSeriesMode','transformerMode']
    def __init__(self):#kkk add comments 
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
        self.layersRegularization = {}
        self.layersRegularizationOperational = {}
        self.modelNameDifferentiator = True
        self.evalMode='loss'
        self.variationalAutoEncoderMode=False
        self.dropoutEnsembleMode=False
        self.dropoutEnsembleNumSamples = 100
        self.timeSeriesMode=False
        self.transformerMode=False
        
    def __post_init__(self):
        '# ccc this is ran after child class constructor'
        if isinstance(self, ann) and not type(self) == ann:
            self.to(self.device)
            self.addCustomLayersRegularizations()
            self.makeLayersRegularizationOperational()
            # self.initOptimizer()

    def forward(self, x):
        return x
    
    def autoEncoderOutputAssign(self, predicts):
        if self.variationalAutoEncoderMode:
            assert len(predicts) == 3, 'In variationalAutoEncoderMode, you should pass predicts, mean, logvar'
            predicts, mean, logvar = predicts
            return predicts, mean, logvar
        else:
            return predicts, None, None
    
    def forwardModelForEvalDueToMode(self, inputs, outputs, appliedBatchSize):
        'this is used in eval phases'
        mean, logvar = None, None
        if self.dropoutEnsembleMode:
            predicts = torch.zeros((self.dropoutEnsembleNumSamples, appliedBatchSize)).to(self.device)
            if self.variationalAutoEncoderMode:
                predictsList = [[] for _ in range(3)]
                for x in [inputs] * self.dropoutEnsembleNumSamples:
                    output = self.forward(x)
                    [predictsList[i].append(output[i]) for i in range(3)]
                predicts, mean, logvar = tuple(torch.stack(predictsList[i]).squeeze().mean(dim=0).unsqueeze(1) for i in range(3))
            else:
                predicts = torch.stack(tuple(map(lambda x: self.forward(x), [inputs] * self.dropoutEnsembleNumSamples)))
                predicts = predicts.squeeze().mean(dim=0).unsqueeze(1)
        else:
            predicts = self.transformerModeForward(inputs, outputs)
            predicts, mean, logvar = self.autoEncoderOutputAssign(predicts)
    
        return predicts, mean, logvar
    
    def autoEncoderAddKlDivergence(self,loss, mean, logvar):
        if self.variationalAutoEncoderMode:
            return loss + self.autoEncoderKlDivergence(mean, logvar)
        else:
            return loss
    
    @property
    def timeSeriesMode(self):
        return self._timeSeriesMode
    
    @timeSeriesMode.setter
    def timeSeriesMode(self, value):
        assert isinstance(value, bool), 'timeSeriesMode should be bool'
        if value:
            assert getattr(self, 'tsInputWindow') and getattr(self, 'tsOutputWindow'),'with timeSeriesMode u should first introduce tsInputWindow and tsOutputWindow to model'
            assert not (self.dropoutEnsembleMode or self.variationalAutoEncoderMode),'with timeSeriesMode the dropoutEnsembleMode and variationalAutoEncoderMode should be off'
        self._timeSeriesMode = value
    
    @property
    def transformerMode(self):
        return self._transformerMode
    
    @transformerMode.setter
    def transformerMode(self, value):
        assert isinstance(value, bool), 'transformerMode should be bool'
        if value:
            assert not (self.dropoutEnsembleMode or self.variationalAutoEncoderMode),'with transformerMode the dropoutEnsembleMode and variationalAutoEncoderMode should be off'
        self._transformerMode = value
    
    @property
    def dropoutEnsembleMode(self):
        return self._dropoutEnsembleMode
    
    @dropoutEnsembleMode.setter
    def dropoutEnsembleMode(self, value):
        assert isinstance(value, bool), 'dropoutEnsembleMode should be bool'
        self._dropoutEnsembleMode = value
        
    
    @property
    def variationalAutoEncoderMode(self):
        return self._variationalAutoEncoderMode
    
    @variationalAutoEncoderMode.setter
    def variationalAutoEncoderMode(self, value):
        assert isinstance(value, bool), 'variationalAutoEncoderMode should be bool'
        self._variationalAutoEncoderMode = value
        if value:
            self.autoEncoderKlDivergence = ann.klDivergenceNormalDistributionLoss
            print('tip: in variationalAutoEncoderMode u should pass predicts, mean, logvar')
    
    @property
    def evalMode(self):
        return self._evalMode
    
    @evalMode.setter
    def evalMode(self, value):
        assert value in ['loss','accuracy','noEval'],f"{self.evalMode} must be either 'loss' or 'accuracy' or 'noEval'"
        self._evalMode=value
        if value=='loss':
            self.evalCompareFunc=lambda valScore, bestValScore: valScore< bestValScore
        elif value=='accuracy':
            self.evalCompareFunc=lambda valScore, bestValScore: valScore> bestValScore
        elif value=='noEval':
            self.evalCompareFunc=lambda valScore, bestValScore: True
    
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
    
    def saveModel(self, bestModel, bestValScore):
        dicToSave={'className':self.__class__.__name__,'classDefinition':self.neededDefinitions,'inputArgs':self.inputArgs,
                    'bestValScore': bestValScore,'model':bestModel}
        dicToSave['modes'] = {mode:getattr(self, mode) for mode in self.modeNames}
        torch.save(dicToSave, self.savePath)
    
    @classmethod
    def loadModel(cls, savePath):
        bestModelDic = torch.load(savePath)
        exec(bestModelDic['classDefinition'], globals())
        classDefinition = globals()[bestModelDic['className']]
        emptyModel = classDefinition(**bestModelDic['inputArgs'])
        emptyModel.load_state_dict(bestModelDic['model'])
        [setattr(emptyModel, mode, bestModelDic['modes'][mode]) for mode in cls.modeNames]
        print('bestValScore:',bestModelDic['bestValScore'])
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
    
    def addLayerRegularization(self,regList):
        for rg in regList:
            assert rg[1] in [None,'l1','l2'],f'regularization type for {rg[0]} should be either None , "l1", "l2"'
            foundLayer=False
            for layerName, layer in vars(self)['_modules'].items():
                if layer==rg[0]:
                    self.layersRegularization[layerName]={'layer':layer, 'regularization':rg[1:3]}
                    foundLayer=True
            assert foundLayer,f'{rg[1]} is not in proper layer'
    
    def makeLayersRegularizationOperational(self):
        defaultRegType,defaultRegVal=self.regularization
        assert defaultRegType in [None,'l1','l2'],'regularization type should be either None , "l1", "l2"'
        defaultRegAddFunc=self.getRegAddFunc(defaultRegType)
        
        layersRegularizationNames=self.layersRegularization.keys()
        
        for name, param in self.named_parameters():
            layerName = name.split('.')[0]
            if layerName in layersRegularizationNames:
                layerRegType, layerRegVal = self.layersRegularization[layerName]['regularization']
                layerRegAddFunc=self.getRegAddFunc(layerRegType)
                self.layersRegularizationOperational[name]=[layerRegAddFunc, layerRegVal]
            else:
                self.layersRegularizationOperational[name]=[defaultRegAddFunc, defaultRegVal]
    
    def addCustomLayersRegularizations(self):
        for layerName, layer in vars(self)['_modules'].items():
            if isinstance(layer, CustomLayer):
                if layer.regularization:
                    if layerName not in self.layersRegularization.keys():
                        self.layersRegularization[layerName]={'layer':layer, 'regularization':layer.regularization}
    
    def addNoRegularization(self,param, regVal):
        return torch.tensor(0)
    
    def addL1Regularization(self,param, regVal):
        return torch.linalg.norm(param, 1) * regVal
    
    def addL2Regularization(self,param, regVal):
        return torch.norm(param)*regVal

    def getRegAddFunc(self, regType):
        if regType==None:
            return self.addNoRegularization
        elif regType=='l1':
            return self.addL1Regularization
        elif regType=='l2':
            return self.addL2Regularization
    
    def addRegularizationToLoss(self, loss):
        lReg = torch.tensor(0., requires_grad=True)
        for name, param in self.named_parameters():
            layerRegAddFunc, layerRegVal=self.layersRegularizationOperational[name]
            lReg = lReg + layerRegAddFunc(param, layerRegVal)
        return loss + lReg
    
    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mean + eps * std
        return z
    
    def klDivergenceNormalDistributionLoss(mean, logvar):
        klLoss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
        return klLoss
    
    def batchDatapreparation(self,indexesIndex, indexes, inputs, outputs, batchSize, identifier=None):
        if self.timeSeriesMode:
            raise NotImplementedError("with timeSeriesMode 'batchDatapreparation' needs to be reimplemented.")
        batchIndexes = indexes[indexesIndex*batchSize:indexesIndex*batchSize + batchSize]
        appliedBatchSize = len(batchIndexes)
        
        batchInputs = inputs[batchIndexes].to(self.device)#kkk add memory or speed priority to have the whole data in gpu or add it here to gpu
        batchOutputs = outputs[batchIndexes].to(self.device)
        return batchInputs, batchOutputs, appliedBatchSize, identifier
    
    # Get all definitions needed to create an instance
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
        def isCustomClass(cls_):
            import builtins
            import pkg_resources
            import types
            if cls_ is None or cls_ is types.NoneType:
                return False
            moduleName = getattr(cls_, '__module__', '')
            return (
                isinstance(cls_, type) and
                not (
                    cls_ in builtins.__dict__.values()
                    or any(moduleName.startswith(package.key) for package in pkg_resources.working_set)
                    or moduleName.startswith('collections')
                )
            ) and not issubclass(cls_, types.FunctionType)

        # Get the definition of a class if it's a custom class
        def getCustomClassDefinition(cls_):
            if cls_ is ann:
                return None
            if isCustomClass(cls_):
                return inspect.getsource(cls_)
            return None

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
        if self.evalMode == 'accuracy':
            bestValScore = 0
        elif self.evalMode == 'loss':
            bestValScore = float('inf')
        patienceCounter = 0
        bestModel = None
        bestModelCounter = 1
        
        # Create random indexes for sampling
        lenOfIndexes=trainInputs.shape[0]
        if self.timeSeriesMode:
            lenOfIndexes += -(self.tsInputWindow+self.tsOutputWindow) + 1
        indexes = torch.randperm(lenOfIndexes)
        batchIterLen = len(trainInputs)//self.batchSize if len(trainInputs) % self.batchSize == 0 else len(trainInputs)//self.batchSize + 1
        return indexes, batchIterLen, bestValScore, patienceCounter, bestModel, bestModelCounter
    
    def checkPatience(self, valScore, bestValScore, patienceCounter, epoch, bestModel, bestModelCounter):
        if self.evalCompareFunc(valScore, bestValScore):
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
            self.saveModel(bestModel, bestValScore)
        
        return bestValScore, patienceCounter, bestModel, bestModelCounter
    
    def singleProcessTrainModel(self,numEpochs,trainInputs, trainOutputs, valInputs, valOutputs, criterion):
        indexes, batchIterLen, bestValScore, patienceCounter, bestModel, bestModelCounter = self.getPreTrainStats(trainInputs)
        for epoch in range(numEpochs):
            trainLoss = 0.0
            
            for i in range(0, batchIterLen):
                self.optimizer.zero_grad()
                
                with torch.no_grad():
                    batchTrainInputs, batchTrainOutputs, appliedBatchSize, _ = self.batchDatapreparation(i, indexes, trainInputs, trainOutputs, self.batchSize)
                
                batchTrainOutputsPred = self.transformerModeForward(batchTrainInputs, batchTrainOutputs)
                batchTrainOutputsPred, mean, logvar = self.autoEncoderOutputAssign(batchTrainOutputsPred)
                
                loss = criterion(batchTrainOutputsPred, batchTrainOutputs)
                loss = self.autoEncoderAddKlDivergence(loss, mean, logvar)
                loss = self.addRegularizationToLoss(loss)
                
                loss.backward()
                self.optimizer.step()
                
                trainLoss += loss.item()
            
            trainLoss = trainLoss / len(trainInputs)
            self.tensorboardWriter.add_scalar('train loss', trainLoss, epoch + 1)
            
            iterPrint=f"Epoch [{epoch+1}/{numEpochs}], aveItemLoss: {trainLoss:.6f}"
            if self.evalMode!= 'noEval':
                valScore = self.evaluateModel(valInputs, valOutputs, criterion, epoch + 1, 'eval')
                iterPrint += f', evalScore:{valScore}'
            print(iterPrint)
            
            bestValScore, patienceCounter, bestModel, bestModelCounter = self.checkPatience(valScore, bestValScore, patienceCounter, epoch, bestModel, bestModelCounter)
        return bestModel, bestValScore
    
    def workerNumRegularization(self,workerNum):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            maxCpus = executor._max_workers
        workerNum=max(min(maxCpus-4,workerNum),1)
        return workerNum
    
    def multiProcessTrainModel(self,numEpochs,trainInputs, trainOutputs,valInputs, valOutputs, criterion, workerNum):
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
                        if len(readyArgs) < 16:
                            for raI in range(24 - len(readyArgs)):
                                idIdx2=idIdx+raI
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
                    
                    batchTrainOutputsPred = self.transformerModeForward(batchTrainInputs, batchTrainOutputs)
                        
                    batchTrainOutputsPred, mean, logvar = self.autoEncoderOutputAssign(batchTrainOutputsPred)
                    
                    loss = criterion(batchTrainOutputsPred, batchTrainOutputs)
                    loss =self.autoEncoderAddKlDivergence(loss, mean, logvar)
                    loss = self.addRegularizationToLoss(loss)
                    
                    loss.backward()
                    self.optimizer.step()
                    
                    trainLoss += loss.item()
                
                trainLoss = trainLoss / len(trainInputs)
                self.tensorboardWriter.add_scalar('train loss', trainLoss, epoch + 1)
                
                iterPrint=f"Epoch [{epoch+1}/{numEpochs}], aveItemLoss: {trainLoss:.6f}"
                if self.evalMode!= 'noEval':
                    valScore = self.evaluateModel(valInputs, valOutputs, criterion, epoch + 1, 'eval', workerNum)
                    iterPrint += f', evalScore:{valScore}'
                print(iterPrint)
                
                bestValScore, patienceCounter, bestModel, bestModelCounter = self.checkPatience(valScore, bestValScore, patienceCounter, epoch, bestModel, bestModelCounter)
        return bestModel, bestValScore
    
    def trainModel(self, trainInputs, trainOutputs, valInputs, valOutputs, criterion, numEpochs, savePath, tensorboardPath='', workerNum=0):
        self.havingOptimizerCheck()
        randomId=randomIdFunc()
        nameDifferentiator=''
        if self.modelNameDifferentiator:
            nameDifferentiator='_'+randomId
        self.savePath=savePath+nameDifferentiator
        print(f'model will be saved in {self.savePath}')
        os.makedirs(os.path.dirname(self.savePath), exist_ok=True)
        if tensorboardPath:
            tensorboardPath+=nameDifferentiator
        else:
            tensorboardPath = self.savePath
        self.tensorboardWriter = tensorboardPath#kkk may add print 'access to tensorboard with "tensorboard --logdir=data" from terminal' (I need to take first part of path from tensorboardPath)
        
        if not self.neededDefinitions:
            self.neededDefinitions=self.getAllNeededDefinitions(self)
        
        if valInputs is None or valOutputs is None:
            self.evalMode='noEval'
        
        self.train()
        if workerNum:
            bestModel, bestValScore = self.multiProcessTrainModel(numEpochs, trainInputs, trainOutputs, valInputs, valOutputs, criterion, workerNum)
        else:
            bestModel, bestValScore = self.singleProcessTrainModel(numEpochs, trainInputs, trainOutputs, valInputs, valOutputs, criterion)
        
        print("Training finished.")
        
        # Save the best model to the hard disk
        self.saveModel(bestModel, bestValScore)
        
        # Load the best model into the current instance
        self.load_state_dict(bestModel)
        
        # Return the best model
        return self
    
    def transformerModeForward(self, inputs, outputs):
        if self.transformerMode:
            return self.forward(inputs, outputs)
        return self.forward(inputs)
    
    def activateDropouts(self):
        if self.dropoutEnsembleMode:
            for module in self.modules():
                if isinstance(module, nn.Dropout):
                    module.train()
    
    def evaluateModel(self, inputs, outputs, criterion, stepNum=0, evalOrTest='Test', workerNum=0):
        self.eval()
        with torch.no_grad():
            self.activateDropouts()
            if workerNum:
                evalScore = self.multiProcessEvaluateModel(inputs, outputs, criterion, stepNum, evalOrTest, workerNum)
            else:
                evalScore = self.singleProcessEvaluateModel(inputs, outputs, criterion, stepNum, evalOrTest)
            if hasattr(self, 'tensorboardWriter'):
                self.tensorboardWriter.add_scalar(f'{evalOrTest} {self.evalMode}', evalScore, stepNum)
            return evalScore
    
    def getPreEvalStats(self, inputs):
        lenOfIndexes=len(inputs)
        if self.timeSeriesMode:
            lenOfIndexes+=-(self.tsInputWindow+self.tsOutputWindow)+1
        indexes = torch.arange(lenOfIndexes)
        batchIterLen = len(inputs)//self.evalBatchSize if len(inputs) % self.evalBatchSize == 0 else len(inputs)//self.evalBatchSize + 1
        return indexes, batchIterLen
    
    def updateEvalScoreBasedOnMode(self, evalScore, batchEvalOutputs, batchEvalOutputsPred, criterion):
        if self.evalMode == 'accuracy':#kkk add dropoutEnsemble certainty based accuracy
            predicted = torch.argmax(batchEvalOutputsPred, dim=1)
            evalScore += (predicted == batchEvalOutputs).sum().item()
        elif self.evalMode == 'loss':
            loss = criterion(batchEvalOutputsPred, batchEvalOutputs)
            evalScore += loss.item()
        return evalScore
    
    def singleProcessEvaluateModel(self, inputs, outputs, criterion, stepNum, evalOrTest):
        evalScore = 0.0
        indexes, batchIterLen = self.getPreEvalStats(inputs)
        for i in range(0, batchIterLen):
            batchEvalInputs, batchEvalOutputs, appliedBatchSize,_ = self.batchDatapreparation(i, indexes, inputs, outputs, batchSize=self.evalBatchSize)
            
            batchEvalOutputsPred, mean, logvar = self.forwardModelForEvalDueToMode(batchEvalInputs, batchEvalOutputs, appliedBatchSize)
            evalScore=self.updateEvalScoreBasedOnMode(evalScore, batchEvalOutputs, batchEvalOutputsPred, criterion)#kkk check for onehot vectors
        
        evalScore /= inputs.shape[0]
        return evalScore

    def multiProcessEvaluateModel(self, inputs, outputs, criterion, stepNum, evalOrTest, workerNum):
        evalScore = 0.0
        workerNum=self.workerNumRegularization(workerNum)
        indexes, batchIterLen = self.getPreEvalStats(inputs)
        with concurrent.futures.ThreadPoolExecutor(max_workers=workerNum) as executor:
            readyArgs = []
            identifiers=[]
            futures = []
            for i in range(0, batchIterLen):
                parallelInputArgs = []
                if len(readyArgs) < 16:
                    for raI in range(24 - len(readyArgs)):
                        idIdx2=i+raI
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
                
                batchEvalOutputsPred, mean, logvar = self.forwardModelForEvalDueToMode(batchEvalInputs, batchEvalOutputs, appliedBatchSize)
                evalScore=self.updateEvalScoreBasedOnMode(evalScore, batchEvalOutputs, batchEvalOutputsPred, criterion)
            
            evalScore /= inputs.shape[0]
            return evalScore
