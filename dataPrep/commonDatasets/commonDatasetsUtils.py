from utils.vAnnGeneralUtils import DotDict


def _dataInfoAssert(dataInfo, necessaryKeys):
    if isinstance(dataInfo, dict):
        dataInfo = DotDict(dataInfo)
    if not all([key in dataInfo.keys() for key in necessaryKeys]):
        raise ValueError(f"dataInfo should provided with {necessaryKeys}")
    dataInfo = dataInfo.copy()
    return dataInfo