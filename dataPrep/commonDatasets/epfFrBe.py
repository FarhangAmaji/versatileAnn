"""
this is data preparation steps of hourly electricity price forecasts (EPF) for France and Belgium markets
the data exists in data\datasets EPF_FR_BE.csv, EPF_FR_BE_futr.csv and EPF_FR_BE_static.csv files
"""
# ---- imports
from dataPrep.dataloader import VAnnTsDataloader
from dataPrep.dataset import VAnnTsDataset
from dataPrep.normalizers_multiColNormalizer import MultiColStdNormalizer
from dataPrep.normalizers_normalizerStack import NormalizerStack
from dataPrep.normalizers_singleColsNormalizer import SingleColsStdNormalizer
from dataPrep.utils import getDatasetFiles, splitTsTrainValTest_DfNNpDict, addCorrespondentRow, \
    rightPadDf, splitToNSeries, regularizeTsStartPoints
from utils.globalVars import tsStartPointColName
from utils.vAnnGeneralUtils import regularizeBoolCol, varPasser

# ----
futureExogenousCols = ['genForecast', 'weekDay']
historyExogenousCols = ['systemLoad']
staticExogenousCols = ['market0', 'market1']
targets = ['priceFr', 'priceBe']
datasetInfos = {'futureExogenousCols': futureExogenousCols,
                 'historyExogenousCols': historyExogenousCols,
                 'staticExogenousCols': staticExogenousCols, 'targets': ['price']}


# ---- getEpfFrBeProcessed_innerSteps
def getEpfFrBeProcessed_loadData(devTestMode=False, backcastLen=110, forecastLen=22):
    mainDf, staticDf = getDatasetFiles('EPF_FR_BE.csv'), getDatasetFiles('EPF_FR_BE_static.csv')
    if devTestMode: # goodToHave3 poor implementation
        mainDf = mainDf.loc[20 * (backcastLen + forecastLen):22 * (backcastLen + forecastLen)]
    return mainDf, staticDf


def _getEpfFrBeProcessed_normalizerFit_split(mainDf, backcastLen, forecastLen, trainRatio=.7,
                                             valRatio=.2):
    normalizer = NormalizerStack(
        SingleColsStdNormalizer([*futureExogenousCols, *historyExogenousCols]),
        MultiColStdNormalizer(targets))
    # cccUsage SingleColsStdNormalizer normalize data of each col separately
    # cccUsage MultiColStdNormalizer normalize data of multiple cols based on all of those cols
    # cccUsage here we use MultiColStdNormalizer for targets('priceFr', 'priceBe'), which have same unit(Euro€)
    mainDf['mask'] = True
    normalizer.fitNTransform(mainDf)
    trainDf, valDf, testDf = splitTsTrainValTest_DfNNpDict(mainDf, trainRatio=trainRatio, valRatio=valRatio,
                                                           seqLen=backcastLen + forecastLen,
                                                           shuffle=False)
    return trainDf, valDf, testDf, normalizer


def _getEpfFrBeProcessed_rightPadTrain(rightPadTrain, trainDf, backcastLen, forecastLen):
    if rightPadTrain:
        trainDf = rightPadDf(trainDf, forecastLen - 1)
        # because rightPadDf adds pad=0, but the mask and tsStartPointColName must have bool data
        regularizeBoolCol(trainDf, 'mask')
        trainDf[tsStartPointColName] = True
        trainDf.loc[len(trainDf) - (backcastLen + forecastLen - 1):, tsStartPointColName] = False
    return trainDf


def _getEpfFrBeProcessed_splitNSeries(trainDf, valDf, testDf, staticDf, aggColName):
    newSets = []
    for set_ in [trainDf, valDf, testDf]:
        set_ = splitToNSeries(set_, targets, aggColName)
        addCorrespondentRow(set_, staticDf, targets, aggColName)
        regularizeTsStartPoints(set_)  # this is just for safety
        newSets += [set_]
    trainDf, valDf, testDf = newSets
    return trainDf, valDf, testDf


# ---- getEpfFrBeProcessed
def getEpfFrBeProcessed(backcastLen=110, forecastLen=22,
                        trainRatio=.7, valRatio=.2,
                        rightPadTrain=True, aggColName='price', devTestMode=False):
    mainDf, staticDf = getEpfFrBeProcessed_loadData(devTestMode=devTestMode,
                                                    backcastLen=backcastLen,
                                                    forecastLen=forecastLen)
    kwargs = varPasser(localArgNames=['mainDf', 'backcastLen', 'forecastLen', 'valRatio'])
    if rightPadTrain:
        # cccAlgo
        #  note if we have rightPadTrain==True, we get this order, 'testDf, valDf, trainDf',
        #  and pass trainRatio=1-(trainRatio+valRatio). so the last indexes of mainDf are dedicated
        #  to trainDf and the paddings are after them
        testDf, valDf, trainDf, normalizer = _getEpfFrBeProcessed_normalizerFit_split(
                                            trainRatio=1 - (trainRatio + valRatio), **kwargs)
    else:
        trainDf, valDf, testDf, normalizer = _getEpfFrBeProcessed_normalizerFit_split(trainRatio=trainRatio,
                                                                                      **kwargs)

    kwargs = varPasser(localArgNames=['rightPadTrain', 'trainDf', 'backcastLen', 'forecastLen'])
    trainDf = _getEpfFrBeProcessed_rightPadTrain(**kwargs)

    kwargs = varPasser(localArgNames=['trainDf', 'valDf', 'testDf', 'staticDf', 'aggColName'])
    trainDf, valDf, testDf = _getEpfFrBeProcessed_splitNSeries(**kwargs)
    return trainDf, valDf, testDf, normalizer


# ----
class EpfFrBeDataset(VAnnTsDataset):
    def __getitem__(self, idx):
        inputs = {}
        inputs['target'] = self.getBackForeCastData(idx, mode=self.castModes.backcast,
                                                    colsOrIndexes=self.additionalInfo['targets'])
        inputs['mask'] = self.getBackForeCastData(idx, mode=self.castModes.backcast,
                                                  colsOrIndexes=['mask'])
        inputs['historyExogenous'] = self.getBackForeCastData(idx, mode=self.castModes.backcast,
                                                              colsOrIndexes=self.additionalInfo['historyExogenousCols'])
        inputs['staticExogenous'] = self.getBackForeCastData(idx, mode=self.castModes.singlePoint,
                                                             colsOrIndexes=self.additionalInfo['staticExogenousCols'])
        inputs['futureExogenous'] = self.getBackForeCastData(idx, mode=self.castModes.fullcast,
                                                             colsOrIndexes=self.additionalInfo['futureExogenousCols'])

        outputs = {}
        outputs['output'] = self.getBackForeCastData(idx, mode=self.castModes.forecast,
                                                     colsOrIndexes=self.additionalInfo['targets'])
        outputs['outputMask'] = self.getBackForeCastData(idx, mode=self.castModes.forecast,
                                                         colsOrIndexes=['mask'])
        return inputs, outputs


# ---- dataloader
def getEpfFrBeDataloaders(backcastLen=110, forecastLen=22, batchSize=64,
                          trainRatio=.7, valRatio=.2,
                          rightPadTrain=True, aggColName='price', devTestMode=False):
    kwargs = varPasser(localArgNames=['backcastLen', 'forecastLen', 'trainRatio', 'valRatio',
                                    'rightPadTrain', 'aggColName', 'devTestMode'])
    trainDf, valDf, testDf, normalizer = getEpfFrBeProcessed(**kwargs)

    epfFrBeTrainDataset = EpfFrBeDataset(trainDf, backcastLen=backcastLen, forecastLen=forecastLen,
                                         mainGroups=[aggColName + 'Type'], indexes=None,
                                         additionalInfo=datasetInfos)
    epfFrBeValDataset = EpfFrBeDataset(valDf, backcastLen=backcastLen, forecastLen=forecastLen,
                                       mainGroups=[aggColName + 'Type'], indexes=None,
                                       additionalInfo=datasetInfos)
    epfFrBeTestDataset = EpfFrBeDataset(testDf, backcastLen=backcastLen, forecastLen=forecastLen,
                                        mainGroups=[aggColName + 'Type'], indexes=None,
                                        additionalInfo=datasetInfos)
    del trainDf, valDf, testDf

    epfFrBe_TrainDataloader = VAnnTsDataloader(epfFrBeTrainDataset, batch_size=batchSize)
    epfFrBe_ValDataloader = VAnnTsDataloader(epfFrBeValDataset, batch_size=batchSize)
    epfFrBe_TestDataloader = VAnnTsDataloader(epfFrBeTestDataset, batch_size=batchSize)
    return epfFrBe_TrainDataloader, epfFrBe_ValDataloader, epfFrBe_TestDataloader, normalizer
