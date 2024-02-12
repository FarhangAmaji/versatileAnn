import pandas as pd
from torch.utils.data import Dataset, DataLoader

from utils.generalUtils import gpuMemoryUsed


class CDS(Dataset):
    def __init__(self) -> None:
        super().__init__()
        self.data = pd.DataFrame({'a': [[j for j in range(i, i + 20)] for i in range(8, 170000)]})

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data['a'][idx]


ds = CDS()
dl = DataLoader(ds, batch_size=10000)
fb = next(iter(dl))
gpuMemoryUsed()
