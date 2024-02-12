from torch import nn


class NNDummyModule2ClassForStaticMethod:
    def __init__(self):
        self.ke = 43

    @staticmethod
    def static_Method2():
        print('staticmethod for NNDummyModule2')


class NNDummyModule2(nn.Module):
    def __init__(self):
        super(NNDummyModule2, self).__init__()
        self.lay21 = nn.Linear(1, 4)
        self.lay22 = nn.Linear(4, 1)
        self.statMeth2 = NNDummyModule2ClassForStaticMethod.static_Method2

    def forward(self, inputs, targets):
        return self.lay22(self.lay21(inputs))


class NNDummyModule3ClassForInstanceMethod:
    def __init__(self):
        self.ke = 47

    def instanceMethod3(self):
        print('instance method for NNDummyModule3')


class NNDummyModule3(nn.Module):
    def __init__(self):
        super(NNDummyModule3, self).__init__()
        self.lay21 = nn.Linear(1, 4)
        self.lay22 = nn.Linear(4, 1)
        self.insMeth = NNDummyModule3ClassForInstanceMethod.instanceMethod3

    def forward(self, inputs, targets):
        return self.lay22(self.lay21(inputs))


class NNDummyModule4:
    def __init__(self):
        self.a2 = 24

    def md(self):
        return ''
