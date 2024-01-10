from abc import ABC
class _NewWrapper_optimizer(ABC):  # kkk1 do it later
    def __init__(self, **kwargs):
        self.untouchedOptimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
