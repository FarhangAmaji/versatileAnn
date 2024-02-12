import numpy as np


def npArrayBroadCast(arr, shape):
    shape = tuple(shape)
    arrShape = arr.shape
    arrShapeLen = len(arrShape)
    if arrShape[:arrShapeLen] != shape[:arrShapeLen]:
        raise ValueError('np array and the given shape, doesnt have same first dims')
    repeatCount = np.prod(shape[arrShapeLen:])
    res = np.repeat(arr, repeatCount).reshape(shape)
    return res


def equalArrays(array1, array2, checkType=True, floatApprox=False, floatPrecision=1e-4):
    dtypeEqual = True
    if checkType:
        # Check if the data types are equal
        dtypeEqual = array1.dtype == array2.dtype

    equalVals = True
    if floatApprox:
        # Check if the arrays are equal with precision
        equalVals = np.allclose(array1, array2, atol=floatPrecision)
    else:
        equalVals = np.array_equal(array1, array2)

    # Return True if both data type and precision are equal
    return dtypeEqual and equalVals
