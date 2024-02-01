from torch import nn


class NNDummyModule2(nn.Module):
    def __init__(self):
        super(NNDummyModule2, self).__init__()
        self.lay21 = nn.Linear(1, 4)
        self.lay22 = nn.Linear(4, 1)

    def forward(self, inputs, targets):
        return self.lay22(self.lay21(inputs))


class NNDummyModule3Parent:
    def __init__(self):
        self.m3p = 7


class NNDummyModule3(nn.Module):
    def __init__(self):
        super(NNDummyModule3, self).__init__()
        self.lay21 = nn.Linear(1, 4)
        self.lay22 = nn.Linear(4, 1)

    def forward(self, inputs, targets):
        return self.lay22(self.lay21(inputs))


class NNDummyModule4:
    def __init__(self):
        self.a2 = 24

    def md(self):
        return ''
