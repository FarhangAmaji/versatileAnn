from torch import nn


class NNDummyModule1(nn.Module):
    def __init__(self):
        super(NNDummyModule1, self).__init__()
        self.lay11 = nn.Linear(1, 3)
        self.lay12 = nn.Linear(3, 1)

    def forward(self, inputs, targets):
        return self.lay12(self.lay11(inputs))