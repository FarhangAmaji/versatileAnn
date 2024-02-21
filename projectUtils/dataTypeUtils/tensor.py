import torch

from projectUtils.warnings import Warn


def tensor_floatDtypeChangeIfNeeded(tensor):
    # mustHave3
    #  the whole code which uses this func should be revised to apply pytorch lightning
    #  model precision to tensors and not just changing to float32 by default
    if tensor.dtype == torch.float16 or tensor.dtype == torch.float64:
        tensor = tensor.to(torch.float32)
        # kkk make it compatible to global precision
    return tensor


def equalTensors(tensor1, tensor2, checkType=True, floatApprox=False, floatPrecision=1e-6,
                 checkDevice=True):
    tensor1_ = tensor1.clone()
    tensor2_ = tensor2.clone()

    dtypeEqual = True
    if checkType:
        dtypeEqual = tensor1_.dtype == tensor2_.dtype
    else:
        tensor1_ = tensor1_.to(torch.float32)
        tensor2_ = tensor2_.to(torch.float32)
    if not dtypeEqual:
        return False

    deviceEqual = tensor1_.device == tensor2_.device
    if not checkDevice:
        if not deviceEqual:
            #  even though device check is not need but make both tensors to cpu
            #  in order not to get different device error in equal line below
            tensor1_ = tensor1_.to(torch.device('cpu'))
            tensor2_ = tensor2_.to(torch.device('cpu'))
        deviceEqual = True
    if not deviceEqual:
        return False

    equalVals = True
    if floatApprox:
        # Check if the tensors are equal with precision
        equalVals = torch.allclose(tensor1_, tensor2_, atol=floatPrecision)
    else:
        equalVals = torch.equal(tensor1_, tensor2_)
    return equalVals


def toDevice(tensor, device):
    # cccDevAlgo
    #  'mps' device doesn't support float64 and int64
    # check if the device.type is 'mps' and it's float64 or int64; first change
    # dtype to float32 or int32, after that change device
    if device.type == 'mps':
        if tensor.dtype == torch.float64:
            tensor = tensor.to(torch.float32)
            Warn.info('float64 tensor is changed to float32 to be compatible with mps')
        elif tensor.dtype == torch.int64:
            tensor = tensor.to(torch.int32)
            Warn.info('int64 tensor is changed to int32 to be compatible with mps')
    return tensor.to(device)


# ---- torch utils
def getTorchDevice():
    # bugPotentialCheck1
    #  this func may still not work with macbooks; ofc in general they don't work with 'cuda' but
    #  may also not work with Mps
    # Check if CUDA is available
    if torch.cuda.is_available():
        device = torch.device('cuda')
    # Check if MPS is available (only for MacOS with Metal support)
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    # Check if XLA is available (only if `torch_xla` package is installed)
    elif 'xla' in torch.__dict__:
        device = torch.device('xla')
    else:
        device = torch.device('cpu')
    return device


def getDefaultTorchDevice_name():
    device_ = torch.tensor([1, 2]).to(getTorchDevice()).device
    deviceName = ''

    if device_.type == 'cuda':
        if hasattr(device_, 'index'):
            deviceName = f"{device_.type}:{device_.index}"
        else:
            deviceName = f"{device_.type}"

    elif device_.type == 'mps':
        deviceName = f'{device_.type}'
    elif device_.type == 'cpu':
        deviceName = 'cpu'
    # bugPotentialCheck2
    #  not sure about xla devices

    return deviceName


def getDefaultTorchDevice_printName():
    # gives printing device name of getTorchDevice()
    deviceName = getDefaultTorchDevice_name()
    devicePrintName = ''
    if 'cuda' in deviceName or 'mps' in deviceName:
        devicePrintName = f", device='{deviceName}'"
    elif 'cpu' in deviceName:
        devicePrintName = ''
    # bugPotentialCheck2
    #  not sure about xla devices

    return devicePrintName
