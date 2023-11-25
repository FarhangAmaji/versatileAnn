'''
this file is just to see, what how things work and
have a quick dummy example of doing things with regular dataset and dataloader
'''
from torch.utils.data import DataLoader, Dataset


class custDataset(Dataset):

    def __init__(self) -> None:
        super().__init__()
        self.ii = -1
        self.i1 = [{'a': [1, 2, 3], 'b': [88, 97, 103]}]
        self.i2 = [{'a': [6, 7, 8], 'b': [89, 98, 104]}]

    def __len__(self):
        return 5

    def __getitem__(self, index):
        self.ii += 1
        if self.ii % 2 == 0:
            return self.i1
        return self.i2


ds = custDataset()
dl = DataLoader(ds, batch_size=3)
dlRes = next(iter(dl))
'''
with self.i1=[{'a': [1, 2, 3], 'b': [88, 97, 103]}]
 and self.i2=[{'a': [6, 7, 8], 'b': [89, 98, 104]}]
-> dlRes=[{'a': [tensor([1, 6, 1]),    tensor([2, 7, 2]),    tensor([3, 8, 3])],
        'b': [tensor([88, 89, 88]), tensor([97, 98, 97]), tensor([103, 104, 103])]}]
each output is 1list-> 2keyDict-> 1list3items
but dlRes is 1list-> 2keyDict-> 3tensor(3,)
I expected to be [{'a':tensor([[1, 2, 3],[6, 7, 8],[1, 2, 3]]),
                   'b':tensor()}]
'''
from torch.utils.data.dataloader import default_collate

colRes = default_collate(dlRes)
'''colRes={'a': [tensor([[1, 6, 1]]), tensor([[2, 7, 2]]), tensor([[3, 8, 3]])],
           'b': [tensor([[88, 89, 88]]), tensor([[97, 98, 97]]), tensor([[103, 104, 103]])]}
and colRes is 2keyDict->1list-> 3tensor(1,3)'''
