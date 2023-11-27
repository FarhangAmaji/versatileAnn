"""
https://www.kaggle.com/datasets/utathya/future-volume-prediction
this dataset is usually treated as multiple series(NSeries) with mainGroups of `agency` and `sku`(stock keeping units (SKU))
it has also these features:
    date:'date'(months)
    production volumes:'sku'(volume that agency has stocked),'volume', 'industryVolume', 'sodaVolume'
    weather temperature:'avgMaxTemp'
    discounted price:'priceRegular', 'priceActual', 'discount', 'discountInPercent'
    population info:'avgPopulation2017', 'avgYearlyHouseholdIncome2017'
    specialDays:'easterDay',  'goodFriday', 'newYear', 'christmas', 'laborDay', 'independenceDay',
                'revolutionDayMemorial', 'regionalGames', 'fifaU17WorldCup', 'footballGoldCup',
                'beerCapital', 'musicFest'
"""
# ---- imports
import numpy as np
import pandas as pd

from dataPrep.dataloader import VAnnTsDataloader
from dataPrep.dataset import VAnnTsDataset
from dataPrep.normalizers_mainGroupNormalizers import NormalizerStack, SingleColsLblEncoder, \
    MainGroupSingleColsStdNormalizer, SingleColsStdNormalizer
from dataPrep.utils import getDatasetFiles, splitTrainValTest_NSeries
# kkk replace this from a particular model, to general embedding files
from dataPrep.utils import rightPadIfShorter_df, rightPadIfShorter_npArray
from models.temporalFusionTransformers_components import getEmbeddingSize
from utils.globalVars import tsStartPointColName

# ----
# kkk explain timeVarying|static;real|categorical;known|unknown
timeIdx = 'timeIdx'
mainGroups = ['agency', 'sku']
target = ['volume']
specialDays = ['easterDay',
               'goodFriday', 'newYear', 'christmas', 'laborDay', 'independenceDay',
               'revolutionDayMemorial', 'regionalGames', 'fifaU17WorldCup',
               'footballGoldCup', 'beerCapital', 'musicFest']
categoricalVariableGroups = {"specialDays": specialDays}
categoricalSingularVariables = ["agency", "sku", "month"]

staticCategoricals = ["agency", "sku"]
staticReals = ["avgPopulation2017", "avgYearlyHouseholdIncome2017"]
timeVaryingknownCategoricals = ["specialDays", "month"]
timeVaryingknownReals = ["timeIdx", "priceRegular", "discountInPercent"]
timeVaryingUnknownCategoricals = []
timeVaryingUnknownReals = ["volume", "logVolume", "industryVolume", "sodaVolume", "avgMaxTemp",
                           "avgVolumeByAgency", "avgVolumeBySku"]

dataInfo = {'mainGroups': mainGroups, 'target': target,
            'categoricalVariableGroups': categoricalVariableGroups,
            'categoricalSingularVariables': categoricalSingularVariables,
            'staticCategoricals': staticCategoricals,
            'staticReals': staticReals,
            'timeVaryingknownCategoricals': timeVaryingknownCategoricals,
            'timeVaryingknownReals': timeVaryingknownReals,
            'timeVaryingUnknownCategoricals': timeVaryingUnknownCategoricals,
            'timeVaryingUnknownReals': timeVaryingUnknownReals}


def getStallionTftProcessed(maxEncoderLength=24, maxPredictionLength=6, minEncoderLength=12,
                            minPredictionLength=1, **dataInfo):
    dataInfo = dataInfo['dataInfo']
    mainGroups = dataInfo['mainGroups']
    target = dataInfo['target']
    categoricalVariableGroups = dataInfo['categoricalVariableGroups']
    categoricalSingularVariables = dataInfo['categoricalSingularVariables']
    staticCategoricals = dataInfo['staticCategoricals']
    staticReals = dataInfo['staticReals']
    timeVaryingknownCategoricals = dataInfo['timeVaryingknownCategoricals']
    timeVaryingknownReals = dataInfo['timeVaryingknownReals']
    timeVaryingUnknownCategoricals = dataInfo['timeVaryingUnknownCategoricals']
    timeVaryingUnknownReals = dataInfo['timeVaryingUnknownReals']

    df = getDatasetFiles('stallion.csv')
    df["timeIdx"] = df["date"].dt.year * 12 + df["date"].dt.month
    df["timeIdx"] -= df["timeIdx"].min()

    # adding some other features
    df["month"] = df.date.dt.month.astype(str).astype("category")  # categories have be strings
    df["logVolume"] = np.log(df.volume + 1e-8)
    df["avgVolumeBySku"] = df.groupby(["timeIdx", "sku"], observed=True).volume.transform("mean")
    df["avgVolumeByAgency"] = df.groupby(["timeIdx", "agency"], observed=True).volume.transform(
        "mean")

    df['relativeTimeIdx'] = 0
    df['encoderLength'] = 0
    timeVaryingknownReals += ['relativeTimeIdx']
    staticReals += ['encoderLength']
    df = df.sort_values(mainGroups + [timeIdx]).reset_index(drop=True)

    normalizer = NormalizerStack(MainGroupSingleColsStdNormalizer(df, mainGroups, target),
                                 SingleColsLblEncoder(['sku', 'agency', 'month', *specialDays]))
    """#ccc pay attention if the MainGroupSingleColsStdNormalizer was passed after SingleColsLblEncoder,
     because it sets up uniquecombos first and after SingleColsLblEncoder's fitNTransform those values would have changed,
     we have to pass it before the SingleColsLblEncoder"""
    normalizer.fitNTransform(df)
    for col in mainGroups:
        df[col] = normalizer.inverseTransformCol(df, col)

    normalizer.uniqueNormalizers[0].getMeanNStd(df)
    staticReals.extend([f'{target[0]}Mean', f'{target[0]}Std'])
    for col in mainGroups:
        df[col] = normalizer.transformCol(df, col)

    categoricalClasses = normalizer.uniqueNormalizers[1].getClasses()
    embeddingSizes = {}
    for col in ['sku', 'month', 'agency']:
        embeddingSizes[col] = [len(categoricalClasses[col]),
                               getEmbeddingSize(len(categoricalClasses[col]))]
    classesLen = 0
    for col in specialDays:
        classesLen += len(categoricalClasses[col])
    embeddingSizes['specialDays'] = [classesLen, getEmbeddingSize(classesLen)]

    "time Varying Encoder= time Varying known + time Varying unkown"
    "time Varying Decoder= time Varying known"
    timeVaryingCategoricalsEncoder = list(
        set(timeVaryingknownCategoricals + timeVaryingUnknownCategoricals))
    timeVaryingRealsEncoder = list(set(timeVaryingknownReals + timeVaryingUnknownReals))
    timeVaryingCategoricalsDecoder = timeVaryingknownCategoricals[:]
    timeVaryingRealsDecoder = timeVaryingknownReals[:]

    allReals = list(set(staticReals + timeVaryingknownReals + timeVaryingUnknownReals))
    allReals = list(set(allReals) - set(target))
    normalizer.addNormalizer(SingleColsStdNormalizer(allReals))
    normalizer.uniqueNormalizers[2].fitNTransform(df)

    df[timeIdx] = normalizer.inverseTransformCol(df, timeIdx)

    uniqueMainGroupMax = df.groupby(mainGroups)[timeIdx].transform('max')
    for col, maxLength in zip(['encoderLength', 'decoderLength', 'sequenceLength'],
                              [maxEncoderLength, maxPredictionLength,
                               maxEncoderLength + maxPredictionLength]):
        df[col] = uniqueMainGroupMax - df[timeIdx] + 1
        df[col] = df[col].apply(lambda x: min(x, maxLength))

    fullLenConditions = (df['sequenceLength'] == maxEncoderLength + maxPredictionLength)
    df.loc[fullLenConditions, 'fullLenConditions'] = True
    df.loc[~fullLenConditions, 'fullLenConditions'] = False

    notFullLenButMoreThanMinEncoderNPredictLenConditions = (df['fullLenConditions'] == False) & (
            df['encoderLength'] >= minEncoderLength) & (df[
                                                            'decoderLength'] >= minPredictionLength)
    df.loc[
        notFullLenButMoreThanMinEncoderNPredictLenConditions, 'notFullLenButMoreThanMinEncoderNPredictLenConditions'] = True
    df.loc[
        ~notFullLenButMoreThanMinEncoderNPredictLenConditions, 'notFullLenButMoreThanMinEncoderNPredictLenConditions'] = False

    df[timeIdx] = normalizer.transformCol(df, timeIdx)

    trainDf1, valDf1, testDf1 = splitTrainValTest_NSeries(df, mainGroups, trainRatio=.7,
                                                          valRatio=.2,
                                                          seqLen=maxEncoderLength + maxPredictionLength,
                                                          shuffle=True,
                                                          conditions=['fullLenConditions==True'])

    trainDf2, valDf2, testDf2 = splitTrainValTest_NSeries(df, mainGroups, trainRatio=.7,
                                                          valRatio=.2,
                                                          seqLen=maxEncoderLength + maxPredictionLength,
                                                          shuffle=True,
                                                          conditions=[
                                                              'notFullLenButMoreThanMinEncoderNPredictLenConditions==True'],
                                                          tailIndexesAsPossible=True)

    sets = []
    for df1, df2 in zip([trainDf1, valDf1, testDf1], [trainDf2, valDf2, testDf2]):
        concatDf = pd.concat([df1, df2])
        concatDf[tsStartPointColName] = concatDf.groupby(concatDf.index)[
            tsStartPointColName].transform('any')
        concatDf = concatDf[~concatDf.index.duplicated(keep='first')]
        sets += [concatDf]
    trainDf, valDf, testDf = sets

    datasetKwargs = {'target': target, 'categoricalVariableGroups': categoricalVariableGroups,
                     'categoricalSingularVariables': categoricalSingularVariables,
                     'staticCategoricals': staticCategoricals,
                     'staticReals': staticReals,
                     'timeVaryingknownCategoricals': timeVaryingknownCategoricals,
                     'timeVaryingknownReals': timeVaryingknownReals,
                     'timeVaryingUnknownCategoricals': timeVaryingUnknownCategoricals,
                     'timeVaryingUnknownReals': timeVaryingUnknownReals,
                     'minPredictionLength': minPredictionLength,
                     'maxPredictionLength': maxPredictionLength,
                     'maxEncoderLength': maxEncoderLength,
                     'minEncoderLength': minEncoderLength,
                     'timeVaryingCategoricalsEncoder': timeVaryingCategoricalsEncoder,
                     'timeVaryingRealsEncoder': timeVaryingRealsEncoder,
                     'timeVaryingCategoricalsDecoder': timeVaryingCategoricalsDecoder,
                     'timeVaryingRealsDecoder': timeVaryingRealsDecoder, 'allReals': allReals}
    return trainDf, valDf, testDf, normalizer, datasetKwargs


# ----
class StallionTftDataset(VAnnTsDataset):
    def __getitem__(self, idx):
        encoderLength = self.data.loc[idx, 'encoderLength']
        decoderLength = self.data.loc[idx, 'decoderLength']

        inputs = {}
        inputs['encoderLengths'] = self.getBackForeCastData(idx, mode=self.castModes.singlePoint,
                                                            colsOrIndexes='encoderLength',
                                                            rightPadIfShorter=True)
        inputs['decoderLengths'] = self.getBackForeCastData(idx, mode=self.castModes.singlePoint,
                                                            colsOrIndexes='decoderLength',
                                                            rightPadIfShorter=True)

        inputs['allReals'] = {}
        for ar in self.allReals:
            inputs['allReals'][ar] = self.getBackForeCastData(idx, mode=self.castModes.fullcast,
                                                              colsOrIndexes=ar,
                                                              rightPadIfShorter=True)

        fullcastLen = self.backcastLen + self.forecastLen
        inputs['allReals']['relativeTimeIdx'] = pd.Series(
            [i for i in range(-encoderLength, decoderLength)])
        inputs['allReals']['relativeTimeIdx'] /= self.maxEncoderLength
        inputs['allReals']['relativeTimeIdx'] = rightPadIfShorter_df(
            inputs['allReals']['relativeTimeIdx'], fullcastLen)

        inputs['allReals']['encoderLength'] = pd.Series(
            [(encoderLength - .5 * self.maxEncoderLength)
             for i in range(encoderLength + decoderLength)])
        inputs['allReals']['encoderLength'] /= self.maxEncoderLength * 2
        inputs['allReals']['encoderLength'] = rightPadIfShorter_df(
            inputs['allReals']['encoderLength'], fullcastLen)

        inputs['categorical'] = {}
        inputs['categorical']['singular'] = {}
        inputs['categorical']['groups'] = {}
        for sc in self.categoricalSingularVariables:
            inputs['categorical']['singular'][sc] = self.getBackForeCastData(idx,
                                                                             mode=self.castModes.fullcast,
                                                                             colsOrIndexes=sc,
                                                                             rightPadIfShorter=True)

        for gc in self.categoricalVariableGroups:
            inputs['categorical']['groups'][gc] = {}
            for gc1 in gc.keys():
                inputs['categorical']['groups'][gc][gc1] = self.getBackForeCastData(idx,
                                                                                    mode=self.castModes.fullcast,
                                                                                    colsOrIndexes=gc1,
                                                                                    rightPadIfShorter=True)

        outputs = {}
        groupName, relIdx = self.findIdxInMainGroupsIndexes(idx)
        outputs['volume'] = self.data[groupName]['volume'][
                            relIdx + encoderLength:relIdx + encoderLength + decoderLength]
        outputs['volume'] = rightPadIfShorter_npArray(outputs['volume'], fullcastLen)

        return inputs, outputs


# ---- dataloader
def getStallionTftDataloaders(maxEncoderLength=24, maxPredictionLength=6, minEncoderLength=12,
                              minPredictionLength=1, mainGroups=['agency', 'sku'], batchSize=64,
                              dataInfo=dataInfo):
    trainDf, valDf, testDf, normalizer, datasetKwargs = getStallionTftProcessed(
        maxEncoderLength=maxEncoderLength,
        maxPredictionLength=maxPredictionLength, minEncoderLength=minEncoderLength,
        minPredictionLength=minPredictionLength, dataInfo=dataInfo)

    stallionTftTrainDataset = StallionTftDataset(trainDf, backcastLen=maxEncoderLength,
                                                 forecastLen=maxPredictionLength,
                                                 mainGroups=mainGroups, indexes=None,
                                                 **datasetKwargs)
    stallionTftValDataset = StallionTftDataset(valDf, backcastLen=maxEncoderLength,
                                               forecastLen=maxPredictionLength,
                                               mainGroups=mainGroups, indexes=None, **datasetKwargs)
    stallionTftTestDataset = StallionTftDataset(testDf, backcastLen=maxEncoderLength,
                                                forecastLen=maxPredictionLength,
                                                mainGroups=mainGroups, indexes=None,
                                                **datasetKwargs)
    del trainDf, valDf, testDf

    stallionTftTrainDataloader = VAnnTsDataloader(stallionTftTrainDataset, batch_size=batchSize)
    stallionTftValDataloader = VAnnTsDataloader(stallionTftValDataset, batch_size=batchSize)
    stallionTftTestDataloader = VAnnTsDataloader(stallionTftTestDataset, batch_size=batchSize)
    return stallionTftTrainDataloader, stallionTftValDataloader, stallionTftTestDataloader, normalizer
