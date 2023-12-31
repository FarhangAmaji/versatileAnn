import asyncio
import os

import aiohttp
import pandas as pd

from utils.vAnnGeneralUtils import downloadFileAsync

datasetsRelativePath = r'.\data\datasets'
knownDatasets_dateTimeCols = {
    "EPF_FR_BE.csv": {'dateTimeCols': ["dateTime"], 'sortCols': ['dateTime']},
    "stallion.csv": {'dateTimeCols': ["date"], 'sortCols': ["agency", "sku"]},
    "electricity.csv": {'dateTimeCols': ["date"],
                        'sortCols': ['consumerId', 'hoursFromStart']}
}
knownDatasets = {"EPF_FR_BE.csv": "demand/EPF_FR_BE.csv",
                 "EPF_FR_BE_futr.csv": "demand/EPF_FR_BE_futr.csv",
                 "EPF_FR_BE_static.csv": "demand/EPF_FR_BE_static.csv",
                 "electricity.csv": "demand/electricity.csv",
                 "stallion.csv": "demand/stallion.csv",
                 }


def getDatasetFiles(fileName: str, dateTimeCols=None, sortCols=None):
    dateTimeCols = dateTimeCols or []
    sortCols = sortCols or []

    filePath = _getFilePathInDataStoreLocation(fileName)

    if not os.path.exists(filePath):
        if fileName in knownDatasets.keys():
            _downloadMissingKnownDataset(fileName)
        else:
            raise Exception(f"File {fileName} does not exist in the data/datasets folder")

    df = pd.read_csv(filePath)
    _convertDatetimeNSortCols(dateTimeCols, df, fileName, sortCols)
    return df


def _getFilePathInDataStoreLocation(fileName):
    currentDir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    filePath = os.path.normpath(os.path.join(currentDir, datasetsRelativePath, fileName))

    os.makedirs(os.path.dirname(filePath), exist_ok=True)
    return filePath


def _downloadMissingKnownDataset(fileName):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(_downloadMissingKnownDataset_Async(fileName))


async def _downloadMissingKnownDataset_Async(fileName):
    event = asyncio.Event()

    async def inner_download():
        nonlocal event
        url = f"https://raw.githubusercontent.com/FarhangAmaji/Datasets/main/{knownDatasets[fileName]}"
        filePath = _getFilePathInDataStoreLocation(fileName)

        await downloadFileAsync(url, filePath, event)

    await inner_download()
    await event.wait()


def _convertDatetimeNSortCols(dateTimeCols, df, fileName, sortCols):
    if fileName in knownDatasets_dateTimeCols.keys():
        dataset = knownDatasets_dateTimeCols[fileName]
        _convertDf_DatetimeNSortCols(df, dataset['dateTimeCols'], dataset['sortCols'])
    else:
        _convertDf_DatetimeNSortCols(df, dateTimeCols, sortCols)


def _convertDfDateTimeCols(df, dateTimeCols):
    for dc in dateTimeCols:
        df[dc] = pd.to_datetime(df[dc])


def _convertDf_DatetimeNSortCols(df, dateTimeCols, sortCols):
    _convertDfDateTimeCols(df, dateTimeCols)
    df = df.sort_values(by=sortCols).reset_index(drop=True)


