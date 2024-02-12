import asyncio
import os
import subprocess

import pandas as pd

from utils.generalUtils import downloadFileAsync
from utils.warnings import Warn

datasetsRelativePath = os.path.join('data', 'datasets')
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
                 "downloadDummy.csv": "downloadDummy.csv",
                 }


def getDatasetFiles(fileName: str, dateTimeCols=None, sortCols=None, try_=0):
    dateTimeCols = dateTimeCols or []
    sortCols = sortCols or []

    filePath = _getFilePathInDataStoreLocation(fileName)

    if not os.path.exists(filePath):
        if fileName in knownDatasets.keys():
            _downloadMissingKnownDataset(fileName)
        else:
            raise Exception(f"File {fileName} does not exist in the data/datasets folder")

    if isFileIn_gitlfsFormat(filePath):
        # cccDevStruct
        #  the files bigger than 100Mb are uploaded in github repos with gitlens
        #  but this url pattern `https://raw.githubusercontent.com/FarhangAmaji/Datasets/main/...`
        #  don't have csv data, instead they have sth like `version https://git-lfs.github.com/...`
        #  so it's detected is this pattern exists or not and if yes uses _downloadFileWithCurl
        #  which downloads from another url with curl
        os.remove(filePath)
        _downloadFileWithCurl(fileName, filePath)

    try:
        try_ += 1
        df = pd.read_csv(filePath)
    except:
        # this is for the case in the past file was download half way. and it's not working correctly
        if try_ < 4:  # 3 tries in total
            os.remove(filePath)
            return getDatasetFiles(fileName, dateTimeCols, sortCols, try_=try_)
    _convertDatetimeNSortCols(dateTimeCols, df, fileName, sortCols)
    return df


def _getFilePathInDataStoreLocation(fileName):
    currentDir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    filePath = os.path.normpath(os.path.join(currentDir, datasetsRelativePath, fileName))

    os.makedirs(os.path.dirname(filePath), exist_ok=True)
    return filePath


def _downloadMissingKnownDataset(fileName):
    # cccDevStruct
    #  this how we call an async function from a sync function
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


# --- funcs to download files which are uploaded with gitlens
def isFileIn_gitlfsFormat(filePath):
    with open(filePath, 'r') as file:
        first_line = file.readline().strip()
        return first_line.startswith('version https://git-lfs.github.com/')


async def _downloadFileWithCurl_async(url, filePath, event=None):
    # goodToHave3
    #  add printing online (changing with changes of cmd output)
    Warn.info(f"starting to download and save: {filePath} from {url}")
    Warn.info(f"this is probably a big file so be patient")

    workingDirectory = os.path.dirname(filePath)
    command = ['curl', '-LJO', url]
    process = await asyncio.create_subprocess_exec(*command, stdout=subprocess.PIPE,
                                                   stderr=subprocess.PIPE, cwd=workingDirectory)

    stdout, stderr = await process.communicate()

    if process.returncode == 0:
        Warn.info(f"Downloaded and saved: {filePath}")

    else:
        errMsg = f"Failed to download {url}. Error: {stderr.decode('utf-8')}"
        Warn.error(errMsg)
        raise RuntimeError(errMsg)


def _downloadFileWithCurl(fileName, filePath):
    url = f'https://github.com/FarhangAmaji/Datasets/raw/main/demand/{fileName}'
    # goodToHave2 # bugPotentialCheck1
    #  this loop gets closed and gives error below at end of program
    #  it's not a really important error but keeps the loop open till the program is running
    #  error:
    #  """
    #  File "C:\Program Files\Python310\lib\asyncio\proactor_events.py", line 116, in __del__
    #     self.close()
    #   File "C:\Program Files\Python310\lib\asyncio\proactor_events.py", line 108, in close
    #     self._loop.call_soon(self._call_connection_lost, None)
    #   File "C:\Program Files\Python310\lib\asyncio\base_events.py", line 745, in call_soon
    #     self._check_closed()
    #   File "C:\Program Files\Python310\lib\asyncio\base_events.py", line 510, in _check_closed
    #     raise RuntimeError('Event loop is closed')
    #  """
    # Explicitly create and manage the event loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(_downloadFileWithCurl_async(url, filePath))
