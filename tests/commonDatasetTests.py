#%% imports
import os
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tests.baseTest import BaseTestClass
import unittest
from dataPrep.commonDatasets.epfFrBe import getEpfFrBeProcessed_loadData, getEpfFrBeProcessed, getEpfFrBeDataloaders
from dataPrep.utils import combineNSeries
from utils.vAnnGeneralUtils import equalDfs
import pandas as pd
import torch
#%% epfFrBeTests
class epfFrBeTests(BaseTestClass):
    def setup(self):
        self.devTestMode=True
        self.backcastLen=7
        self.forecastLen=4
        self.aggColName='price'

    def processedSetup(self):
        self.setup()
        self.mainDf, staticDf=getEpfFrBeProcessed_loadData(devTestMode=self.devTestMode, backcastLen=self.backcastLen, forecastLen=self.forecastLen)
        self.trainDf, self.valDf, self.testDf, self.normalizer = getEpfFrBeProcessed(backcastLen=self.backcastLen, forecastLen=self.forecastLen,
                       trainRatio=.6, valRatio=.2, rightPadTrain=True, aggColName=self.aggColName, devTestMode=True)

    def testGetEpfFrBeProcessed(self):
        #kkk could have had test for innerSteps also
        #ccc hard written values is chosen because other ways had to just follow the same procedure done int the getEpfFrBeProcessed
        self.processedSetup()
        trainDfCheck=pd.DataFrame({'genForecast': [0.3186052798375584, 0.844584345427674, 1.0116957943796043, 1.4090496841097493, 1.0809035661677773, 0.1633097919226333, -0.24315926775034438, -0.4396418198514017, -0.06760784662912478, 1.3229619679829974, 1.712888682204168, 1.00865740439866, 0.09106363015351601, -0.23944567999585706, 0.3705955084003811, -0.1763146881695723, -0.8545508516936688, -1.4872111655036027]+3*[0.0]+[0.3186052798375584, 0.844584345427674, 1.0116957943796043, 1.4090496841097493, 1.0809035661677773, 0.1633097919226333, -0.24315926775034438, -0.4396418198514017, -0.06760784662912478, 1.3229619679829974, 1.712888682204168, 1.00865740439866, 0.09106363015351601, -0.23944567999585706, 0.3705955084003811, -0.1763146881695723, -0.8545508516936688, -1.4872111655036027]+3*[0.0], 
                                    'systemLoad': [-0.2683097548411284, 0.4512716844203051, 0.6859123151938649, 1.0382521213608356, 1.4835389158213155, 0.3421600025105335, -0.34812292957142427, -0.5208830925952294, -0.25896454134422664, 1.2377850581865752, 1.775008547589548, 1.0372418280098192, 0.27194461461489344, -0.3569629963928178, 0.8018434772229972, 0.5939756202513721, -0.4365235977853596, -0.5107801590850652]+3*[0.0]+[-0.2683097548411284, 0.4512716844203051, 0.6859123151938649, 1.0382521213608356, 1.4835389158213155, 0.3421600025105335, -0.34812292957142427, -0.5208830925952294, -0.25896454134422664, 1.2377850581865752, 1.775008547589548, 1.0372418280098192, 0.27194461461489344, -0.3569629963928178, 0.8018434772229972, 0.5939756202513721, -0.4365235977853596, -0.5107801590850652]+3*[0.0], 
                                    'weekDay': 15*[-0.3872983346207408]+3*[2.5819888974716125]+3*[0.0]+15*[-0.3872983346207408]+3*[2.5819888974716125]+3*[0.0], 
                                    'mask': 18*[True]+3*[False]+18*[True]+3*[False], 
                                    '__startPoint__': 11*[True]+10*[False]+11*[True]+10*[False], 
                                    'price': [-0.797110549573975, -0.7089587073042353, -0.7040613827336943, -0.6044824497993591, -0.7889483419564064, -0.9521924943077759, -1.3496920052833608, -1.1203339712296865, -0.753850849200862, -0.27962658662013373, -0.010273735240374053, -0.27064815824080846, -0.7448724208215367, -0.16862056302120254, -0.016803501334428695, -0.0755713961809219, -0.29839966414054125, -0.06496052627808296]+3*[0.0]+[0.9602127504885175, 1.5617674519033138, 1.8915206396530801, 2.016402416201878, 1.6923627737844096, 0.6435190949268605, 0.6769841461588916, 0.9846993733412227, 1.5642161141885844, 2.617140896854918, 1.1528408502631335, 0.615767589027128, 0.5814863170333403, 0.578221433986313, 0.41089617782615895, -0.0755713961809219, -0.29839966414054125, -0.06496052627808296]+3*[0.0], 
                                    'priceType': 21*['priceFr']+21*['priceBe'], 
                                    'market0': 21*[1.0]+21*[0.0], 
                                    'market1': 21*[0.0]+21*[1.0],})

        valDfCheck=pd.DataFrame({'genForecast': [-1.686732107585604, -0.9024898936152327, -0.03080956797102298, 0.3186052798375584, 0.844584345427674, 1.0116957943796043, 1.4090496841097493, 1.0809035661677773, 0.1633097919226333, -0.24315926775034438, -0.4396418198514017, -0.06760784662912478, 1.3229619679829974, -1.686732107585604, -0.9024898936152327, -0.03080956797102298, 0.3186052798375584, 0.844584345427674, 1.0116957943796043, 1.4090496841097493, 1.0809035661677773, 0.1633097919226333, -0.24315926775034438, -0.4396418198514017, -0.06760784662912478, 1.3229619679829974], 
                                'systemLoad': [-1.5145066033198622, -1.1386774767417602, -0.6221650010346238, -0.2683097548411284, 0.4512716844203051, 0.6859123151938649, 1.0382521213608356, 1.4835389158213155, 0.3421600025105335, -0.34812292957142427, -0.5208830925952294, -0.25896454134422664, 1.2377850581865752, -1.5145066033198622, -1.1386774767417602, -0.6221650010346238, -0.2683097548411284, 0.4512716844203051, 0.6859123151938649, 1.0382521213608356, 1.4835389158213155, 0.3421600025105335, -0.34812292957142427, -0.5208830925952294, -0.25896454134422664, 1.2377850581865752], 
                                'weekDay': 26*[-0.3872983346207408], 
                                'mask': 26*[True], 
                                '__startPoint__': 3*[True]+10*[False]+3*[True]+10*[False], 
                                'price': [-1.3154107332895728, -0.9823926624927793, -0.8934245994612829, -0.797110549573975, -0.7089587073042353, -0.7040613827336943, -0.6044824497993591, -0.7889483419564064, -0.9521924943077759, -1.3496920052833608, -1.1203339712296865, -0.753850849200862, -0.27962658662013373, -1.3154107332895728, 0.12358646968774896, 0.36273915288250524, 0.9602127504885175, 1.5617674519033138, 1.8915206396530801, 2.016402416201878, 1.6923627737844096, 0.6435190949268605, 0.6769841461588916, 0.9846993733412227, 1.5642161141885844, 2.617140896854918], 
                                'priceType': 13*['priceFr']+13*['priceBe'], 
                                'market0': 13*[1.0]+13*[0.0], 
                                'market1': 13*[0.0]+13*[1.0],})

        testDfCheck=pd.DataFrame({'genForecast': [-1.4521008812793588, -1.7542518849399193, -1.686732107585604, -0.9024898936152327, -0.03080956797102298, 0.3186052798375584, 0.844584345427674, 1.0116957943796043, 1.4090496841097493, 1.0809035661677773, 0.1633097919226333, -0.24315926775034438, -1.4521008812793588, -1.7542518849399193, -1.686732107585604, -0.9024898936152327, -0.03080956797102298, 0.3186052798375584, 0.844584345427674, 1.0116957943796043, 1.4090496841097493, 1.0809035661677773, 0.1633097919226333, -0.24315926775034438], 
                                'systemLoad': [-1.8059762350880948, -1.937061797382473, -1.5145066033198622, -1.1386774767417602, -0.6221650010346238, -0.2683097548411284, 0.4512716844203051, 0.6859123151938649, 1.0382521213608356, 1.4835389158213155, 0.3421600025105335, -0.34812292957142427, -1.8059762350880948, -1.937061797382473, -1.5145066033198622, -1.1386774767417602, -0.6221650010346238, -0.2683097548411284, 0.4512716844203051, 0.6859123151938649, 1.0382521213608356, 1.4835389158213155, 0.3421600025105335, -0.34812292957142427], 
                                'weekDay': 24*[-0.3872983346207408], 
                                'mask': 24*[True], 
                                '__startPoint__': 2*[True]+10*[False]+2*[True]+10*[False], 
                                'price': [-1.534157897440408, -1.5178334822052715, -1.3154107332895728, -0.9823926624927793, -0.8934245994612829, -0.797110549573975, -0.7089587073042353, -0.7040613827336943, -0.6044824497993591, -0.7889483419564064, -0.9521924943077759, -1.3496920052833608, -0.2151451464413429, -0.5122495037208351, -1.3154107332895728, 0.12358646968774896, 0.36273915288250524, 0.9602127504885175, 1.5617674519033138, 1.8915206396530801, 2.016402416201878, 1.6923627737844096, 0.6435190949268605, 0.6769841461588916], 
                                'priceType': 12*['priceFr']+12*['priceBe'], 
                                'market0': 12*[1.0]+12*[0.0], 
                                'market1': 12*[0.0]+12*[1.0],})

        trainDf=self.trainDf.drop(columns='dateTime')
        valDf=self.valDf.drop(columns='dateTime')
        testDf=self.testDf.drop(columns='dateTime')

        self.assertTrue(equalDfs(trainDf,trainDfCheck))
        self.assertTrue(equalDfs(valDf,valDfCheck))
        self.assertTrue(equalDfs(testDf,testDfCheck))

    def testInvNormalizer(self):
        self.processedSetup()
        for set1 in [self.trainDf, self.valDf, self.testDf]:
            set_=combineNSeries(set1, self.aggColName)
            set_=set_.drop(columns=['__startPoint__', 'market0', 'market1', 'mask'])

            #ccc in real use case the zero pad rows are not inverse transformed so for purpose of checking if invTransfrom work we can delete them
            setIndsWith0dateCond= set_[set_['dateTime']==0].index
            set_ = set_.drop(setIndsWith0dateCond)
            self.normalizer.inverseTransform(set_)

            mainDfIndexes_whichAreInSetDf=self.mainDf[self.mainDf['dateTime'].isin(set_['dateTime'].values)].index
            setCheck=self.mainDf.loc[mainDfIndexes_whichAreInSetDf]

            self.equalDfs(set_, setCheck, checkIndex=False, floatApprox=True)

    def testDataloader_data(self):
        device = torch.device('cuda')
        self.processedSetup()
        epfFrBe_TrainDataloader, epfFrBe_ValDataloader, epfFrBe_TestDataloader, normalizer=getEpfFrBeDataloaders(
                                        backcastLen=self.backcastLen, forecastLen=self.forecastLen,
                                        batchSize=64, trainRatio=.7, valRatio=.2,
                                        rightPadTrain=True, aggColName=self.aggColName, devTestMode=self.devTestMode)


        # train
        trainDataloader_inputs, trainDataloader_outputs = next(iter(epfFrBe_TrainDataloader))# ccc these 2 are just here in order if they make error, get detected
        valDataloader_inputs, valDataloader_outputs =next(iter(epfFrBe_ValDataloader))


        testDataloader_inputs, testDataloader_outputs =next(iter(epfFrBe_TestDataloader))
        trainDf, valDf, testDf, normalizer = getEpfFrBeProcessed(backcastLen=self.backcastLen, forecastLen=self.forecastLen,
                       trainRatio=.7, valRatio=.2, rightPadTrain=True, aggColName=self.aggColName, devTestMode=True)
        
        
        expectedInputs={}
        expectedInputs['target']=[[-1.5341578722000122, -1.5178334712982178, -1.3154107332229614, -0.9823926687240601, -0.8934245705604553, -0.7971105575561523, -0.7089586853981018], [-0.21514514088630676, -0.5122495293617249, -1.3154107332229614, 0.12358646839857101, 0.3627391457557678, 0.960212767124176, 1.5617674589157104]]
        expectedInputs['mask']=[[True, True, True, True, True, True, True], [True, True, True, True, True, True, True]]
        expectedInputs['historyExogenous']=[[-1.8059762716293335, -1.9370617866516113, -1.5145065784454346, -1.1386774778366089, -0.6221650242805481, -0.26830974221229553, 0.4512716829776764], [-1.8059762716293335, -1.9370617866516113, -1.5145065784454346, -1.1386774778366089, -0.6221650242805481, -0.26830974221229553, 0.4512716829776764]]
        expectedInputs['staticExogenous']=[[[1.0, 0.0]], [[0.0, 1.0]]]
        expectedInputs['futureExogenous']=[[[-1.4521008729934692, -0.3872983455657959], [-1.7542518377304077, -0.3872983455657959], [-1.6867320537567139, -0.3872983455657959], [-0.9024899005889893, -0.3872983455657959], [-0.030809568241238594, -0.3872983455657959], [0.3186052739620209, -0.3872983455657959], [0.8445843458175659, -0.3872983455657959], [1.0116957426071167, -0.3872983455657959], [1.4090496301651, -0.3872983455657959], [1.0809035301208496, -0.3872983455657959], [0.16330979764461517, -0.3872983455657959]], [[-1.4521008729934692, -0.3872983455657959], [-1.7542518377304077, -0.3872983455657959], [-1.6867320537567139, -0.3872983455657959], [-0.9024899005889893, -0.3872983455657959], [-0.030809568241238594, -0.3872983455657959], [0.3186052739620209, -0.3872983455657959], [0.8445843458175659, -0.3872983455657959], [1.0116957426071167, -0.3872983455657959], [1.4090496301651, -0.3872983455657959], [1.0809035301208496, -0.3872983455657959], [0.16330979764461517, -0.3872983455657959]]]
        
        for key, value in testDataloader_inputs.items():
            self.assertTrue(torch.equal(value, torch.tensor(expectedInputs[key]).to(device)))

        expectedOutputs={}
        expectedOutputs['output']=[[-0.7040613889694214, -0.6044824719429016, -0.7889483571052551, -0.952192485332489], [1.891520619392395, 2.01640248298645, 1.6923627853393555, 0.6435191035270691]]
        expectedOutputs['outputMask']=[[True, True, True, True],[True, True, True, True]]
        for key, value in testDataloader_outputs.items():
            self.assertTrue(torch.equal(value, torch.tensor(expectedOutputs[key]).to(device)))
#%% run test
if __name__ == '__main__':
    unittest.main()