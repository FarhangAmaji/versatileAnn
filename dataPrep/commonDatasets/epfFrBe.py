# ---- imports
"""
this is data preparation steps of hourly electricity price forecasts (EPF) for France and Belgium markets
the data exists in data\datasets EPF_FR_BE.csv, EPF_FR_BE_futr.csv and EPF_FR_BE_static.csv files
"""
from dataPrep.dataloader import VAnnTsDataloader
from dataPrep.dataset import VAnnTsDataset
from dataPrep.normalizers_multiColNormalizer import MultiColStdNormalizer
from dataPrep.normalizers_singleColsNormalizer import SingleColsStdNormalizer
from dataPrep.normalizers_normalizerStack import NormalizerStack
from dataPrep.utils import getDatasetFiles, splitTsTrainValTest_DfNNpDict, addCorrespondentRow, \
    rightPadDf, splitToNSeries, regularizeTsStartPoints
from utils.globalVars import tsStartPointColName
from utils.vAnnGeneralUtils import regularizeBoolCol

# ----
futureExogenousCols = ['genForecast', 'weekDay']
historyExogenousCols = ['systemLoad']
staticExogenousCols = ['market0', 'market1']
targets = ['priceFr', 'priceBe']
datasetKwargs = {'futureExogenousCols': futureExogenousCols,
                 'historyExogenousCols': historyExogenousCols,
                 'staticExogenousCols': staticExogenousCols, 'targets': ['price']}


# ---- getEpfFrBeProcessed_innerSteps
def getEpfFrBeProcessed_loadData(devTestMode=False, backcastLen=110, forecastLen=22):
    mainDf, staticDf = getDatasetFiles('EPF_FR_BE.csv'), getDatasetFiles('EPF_FR_BE_static.csv')
    if devTestMode:
        mainDf = mainDf.loc[20 * (backcastLen + forecastLen):22 * (backcastLen + forecastLen)]
    return mainDf, staticDf


def getEpfFrBeProcessed_normalizerFit_split(mainDf, backcastLen, forecastLen, trainRatio=.7,
                                            valRatio=.2):
    normalizer = NormalizerStack(
        SingleColsStdNormalizer([*futureExogenousCols, *historyExogenousCols]),
        MultiColStdNormalizer(targets))
    "#ccc MultiColStdNormalizer normalize data of multiple cols based on all of those cols"
    "#ccc here we use MultiColStdNormalizer for targets('priceFr', 'priceBe'), which are from 1unit(Euro€)"
    "#ccc SingleColsStdNormalizer normalize data of each col separately"
    mainDf['mask'] = True
    normalizer.fitNTransform(mainDf)
    trainDf, valDf, testDf = splitTsTrainValTest_DfNNpDict(mainDf, trainRatio=trainRatio,
                                                           valRatio=valRatio,
                                                           seqLen=backcastLen + forecastLen,
                                                           shuffle=False)
    return trainDf, valDf, testDf, normalizer


def getEpfFrBeProcessed_rightPadTrain(rightPadTrain, trainDf, backcastLen, forecastLen):
    if rightPadTrain:
        "#ccc tsStartPointColName(=='__startPoint__) should change, because they are determined on the the "
        trainDf = rightPadDf(trainDf, forecastLen - 1)
        regularizeBoolCol(trainDf, 'mask')
        trainDf[tsStartPointColName] = True
        trainDf.loc[len(trainDf) - (backcastLen + forecastLen - 1):, tsStartPointColName] = False
    return trainDf


def getEpfFrBeProcessed_splitNSeries(trainDf, valDf, testDf, staticDf, aggColName):
    newSets = []
    for set_ in [trainDf, valDf, testDf]:
        set_ = splitToNSeries(set_, targets, aggColName)
        addCorrespondentRow(set_, staticDf, targets, aggColName, targetMapping={})
        regularizeTsStartPoints(set_)  # this is not necessary but to make sure
        newSets += [set_]
    trainDf, valDf, testDf = newSets
    return trainDf, valDf, testDf


# ---- getEpfFrBeProcessed
def getEpfFrBeProcessed(backcastLen=110, forecastLen=22, trainRatio=.7, valRatio=.2,
                        rightPadTrain=True, aggColName='price', devTestMode=False):
    mainDf, staticDf = getEpfFrBeProcessed_loadData(devTestMode=devTestMode,
                                                    backcastLen=backcastLen,
                                                    forecastLen=forecastLen)
    if rightPadTrain:
        "#ccc note if we have rightPadTrain==True, we get this order, 'testDf, valDf, trainDf', and pass trainRatio=1-(trainRatio+valRatio)"
        "... so the last indexes of mainDf are dedicated to trainDf and the paddings are after them"
        testDf, valDf, trainDf, normalizer = getEpfFrBeProcessed_normalizerFit_split(mainDf=mainDf,
                                                                                     backcastLen=backcastLen,
                                                                                     forecastLen=forecastLen,
                                                                                     trainRatio=1 - (
                                                                                                 trainRatio + valRatio),
                                                                                     valRatio=valRatio)
    else:
        trainDf, valDf, testDf, normalizer = getEpfFrBeProcessed_normalizerFit_split(mainDf=mainDf,
                                                                                     backcastLen=backcastLen,
                                                                                     forecastLen=forecastLen,
                                                                                     trainRatio=trainRatio,
                                                                                     valRatio=valRatio)

    trainDf = getEpfFrBeProcessed_rightPadTrain(rightPadTrain=rightPadTrain, trainDf=trainDf,
                                                backcastLen=backcastLen, forecastLen=forecastLen)

    trainDf, valDf, testDf = getEpfFrBeProcessed_splitNSeries(trainDf=trainDf, valDf=valDf,
                                                              testDf=testDf,
                                                              staticDf=staticDf,
                                                              aggColName=aggColName)
    return trainDf, valDf, testDf, normalizer


# ----
class EpfFrBeDataset(VAnnTsDataset):
    def __getitem__(self, idx):
        inputs = {}
        inputs['target'] = self.getBackForeCastData(idx, mode=self.castModes.backcast,
                                                    colsOrIndexes=self.targets)
        inputs['mask'] = self.getBackForeCastData(idx, mode=self.castModes.backcast,
                                                  colsOrIndexes=['mask'])
        inputs['historyExogenous'] = self.getBackForeCastData(idx, mode=self.castModes.backcast,
                                                              colsOrIndexes=self.historyExogenousCols)
        inputs['staticExogenous'] = self.getBackForeCastData(idx, mode=self.castModes.singlePoint,
                                                             colsOrIndexes=self.staticExogenousCols)
        inputs['futureExogenous'] = self.getBackForeCastData(idx, mode=self.castModes.fullcast,
                                                             colsOrIndexes=self.futureExogenousCols)

        outputs = {}
        outputs['output'] = self.getBackForeCastData(idx, mode=self.castModes.forecast,
                                                     colsOrIndexes=self.targets)
        outputs['outputMask'] = self.getBackForeCastData(idx, mode=self.castModes.forecast,
                                                         colsOrIndexes=['mask'])
        return inputs, outputs


# ---- dataloader
def getEpfFrBeDataloaders(backcastLen=110, forecastLen=22, batchSize=64,
                          trainRatio=.7, valRatio=.2,
                          rightPadTrain=True, aggColName='price', devTestMode=False):
    trainDf, valDf, testDf, normalizer = getEpfFrBeProcessed(backcastLen=backcastLen,
                                                             forecastLen=forecastLen,
                                                             trainRatio=trainRatio,
                                                             valRatio=valRatio,
                                                             rightPadTrain=rightPadTrain,
                                                             aggColName=aggColName,
                                                             devTestMode=devTestMode)

    epfFrBeTrainDataset = EpfFrBeDataset(trainDf, backcastLen=backcastLen, forecastLen=forecastLen,
                                         mainGroups=[aggColName + 'Type'], indexes=None,
                                         **datasetKwargs)
    epfFrBeValDataset = EpfFrBeDataset(valDf, backcastLen=backcastLen, forecastLen=forecastLen,
                                       mainGroups=[aggColName + 'Type'], indexes=None,
                                       **datasetKwargs)
    epfFrBeTestDataset = EpfFrBeDataset(testDf, backcastLen=backcastLen, forecastLen=forecastLen,
                                        mainGroups=[aggColName + 'Type'], indexes=None,
                                        **datasetKwargs)
    del trainDf, valDf, testDf

    epfFrBe_TrainDataloader = VAnnTsDataloader(epfFrBeTrainDataset, batch_size=batchSize)
    epfFrBe_ValDataloader = VAnnTsDataloader(epfFrBeValDataset, batch_size=batchSize)
    epfFrBe_TestDataloader = VAnnTsDataloader(epfFrBeTestDataset, batch_size=batchSize)
    return epfFrBe_TrainDataloader, epfFrBe_ValDataloader, epfFrBe_TestDataloader, normalizer
