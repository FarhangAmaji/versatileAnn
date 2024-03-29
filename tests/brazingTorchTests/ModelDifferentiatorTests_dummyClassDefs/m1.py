from torch import nn


class NNDummyModule1ClassForStaticAndInstanceMethod:
    def __init__(self):
        self.ke = 78

    @staticmethod
    def static_Method1():
        print('staticmethod for NNDummyModule1')

    def instanceMeth1(self):
        print('instancemethod for NNDummyModule1')


class NNDummyModule1(nn.Module):
    def __init__(self):
        super(NNDummyModule1, self).__init__()
        self.lay11 = nn.Linear(1, 3)
        self.lay12 = nn.Linear(3, 1)
        self.statMeth = NNDummyModule1ClassForStaticAndInstanceMethod.static_Method1
        self.instanceMeth = NNDummyModule1ClassForStaticAndInstanceMethod.instanceMeth1

    def forward(self, inputs, targets):
        return self.lay12(self.lay11(inputs))
