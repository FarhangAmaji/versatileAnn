# cccDevAlgo
#  note this file and VAnnDatasetAndLoader, show that the gpuMemoryEfficiency of VAnnDataloader lies
#  only in the fact that VAnnDataloader helps us not to put cuda tensors in dataset, which is super
#  useful. but the otherwise the default dataloader doesnt move any tensors to the cuda, which
#  obviously occupies no gpu memory. so the gpu memory efficiency alongside a code to help us to do
#  less labor code for putting the tensors in gpu is achieved

import pandas as pd

from dataPrep.dataloader import VAnnTsDataloader
from dataPrep.dataset import VAnnTsDataset
from projectUtils.misc import gpuMemoryUsed


class custDataset(VAnnTsDataset):
    def __getitem__(self, idx):
        return self.data['a'][idx]


ds = custDataset(
    data=pd.DataFrame({'a': [[j for j in range(i, i + 20)] for i in range(8, 170000)]}),
    backcastLen=0, forecastLen=0)

dl = VAnnTsDataloader(ds, batch_size=10000)
fb = next(iter(dl))
gpuMemoryUsed()
