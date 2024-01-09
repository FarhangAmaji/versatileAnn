# ----
import unittest

import numpy as np
import pandas as pd
import torch

# ----
from dataPrep.dataloader import _NestedDictStruct, appendValue_ToNestedDictPath, VAnnTsDataloader
from dataPrep.dataloader import _ObjectToBeTensored as bstObjInit
from dataPrep.dataset import VAnnTsDataset
from tests.baseTest import BaseTestClass
from utils.vAnnGeneralUtils import DotDict, NpDict, shuffleData


# ---- dataloader tests
# ----     batch data tests
class nestedDictStructTests(BaseTestClass):
    def setUp(self):
        self.item1 = {'a': {'a1': [2],
                            'a2': {'b1': [2], 'b2': {}, 'b3': 4, 'b4': True},
                            'a3': 4, 'a4': True}}
        self.item1Res = {'a': {'a1': bstObjInit([2]),
                               'a2': {'b1': bstObjInit([2]), 'b2': bstObjInit({}),
                                      'b3': bstObjInit(4), 'b4': bstObjInit(True)},
                               'a3': bstObjInit(4), 'a4': bstObjInit(True)}}

    def test(self):
        self.setUp()
        self.assertEqual(str(_NestedDictStruct(self.item1)), str(self.item1Res))
        """#ccc seems to be a not secure way
        but because we have only nestedDictStruct and _ObjectToBeTensored types with defined __repr__s
        so seems to be ok"""

    def testToList(self):
        res = _NestedDictStruct({'a': {'b': {'c': [1, 2, 3]}}, 'z': [23, 5]},
                                giveFilledStruct=True).toList()
        self.assertEqual(res, [[1, 2, 3], [23, 5]])


# ----     appendValue_ToNestedDictPathTests
class appendValue_ToNestedDictPathForSimpleDictionaryTests(BaseTestClass):
    def setUp(self):
        self.item1 = {'a': {'a1': {'b1': []},
                            'a2': {'b1': [], 'b2': [], 'b3': [], 'b4': []},
                            'a3': [],
                            'a4': []}}
        self.avtl = appendValue_ToNestedDictPath

    def test1(self):
        self.setUp()
        self.avtl(self.item1, ['a', 'a1', 'b1'], 5)
        self.assertEqual(self.item1, {
            'a': {'a1': {'b1': [5]}, 'a2': {'b1': [], 'b2': [], 'b3': [], 'b4': []}, 'a3': [],
                  'a4': []}})

    def test2(self):
        self.setUp()
        self.avtl(self.item1, ['a', 'a1', 'b1'], 6)
        self.avtl(self.item1, ['a', 'a1', 'b1'], 5)
        self.assertEqual(self.item1, {
            'a': {'a1': {'b1': [6, 5]}, 'a2': {'b1': [], 'b2': [], 'b3': [], 'b4': []}, 'a3': [],
                  'a4': []}})

    def testDoesntLeadToAList(self):
        self.setUp()
        with self.assertRaises(AssertionError) as context:
            self.avtl(self.item1, ['a', 'a1'], 4)
        self.assertEqual(str(context.exception), "['a', 'a1'] doesn't lead to a list")

    def testNoKeyWithSomeName(self):
        self.setUp()
        with self.assertRaises(AssertionError) as context:
            self.avtl(self.item1, ['a', 'a1', 'b2'], 4)
        self.assertEqual(str(context.exception), "b2 is not in ['a', 'a1', 'b2']")

    def testNotADict(self):
        self.setUp()
        with self.assertRaises(AssertionError) as context:
            self.avtl(self.item1, ['a', 'a4', 'b2', 'xs'], 4)
        self.assertEqual(str(context.exception), "['a', 'a4', 'b2'] is not a dict or DotDict")

    def testKeyNotInDict(self):
        self.setUp()
        with self.assertRaises(AssertionError) as context:
            self.avtl(self.item1, ['a', 'a1', 'b2', 'b3'], 4)
        self.assertEqual(str(context.exception), "b2 is not in ['a', 'a1']")


# ----     fillBatchStructWithDataTests
class fillBatchStructWithDataTests(BaseTestClass):
    def setUp(self):
        self.item1 = {'a': {'a1': [2, 2.4],
                            'a2': {'b1': 3.4, 'b2': {}, 'b3': 4, 'b4': True},
                            'a3': np.array([[3.1, 4.1], [5.1, 6.1]]),
                            'a4': {'z1': True},
                            'a5': {'m2': bytearray(i * 15 for i in range(10)),
                                   'm3': bytes(i * 12 for i in range(10)),
                                   'm4': (4, 3.2, 6.7),
                                   'm5': torch.tensor(np.array([[3.1, 4.1], [5.1, 6.1]])),
                                   'm6': set([12, 43]),
                                   'm7': 'dsdnk',
                                   'm8': None,
                                   },
                            'df': {'df': pd.DataFrame({'y1': [1, 2, 3], 'y2': [4, 5, 6]}),
                                   'series': pd.Series(np.array([3.1, 4.1]))},
                            }}
        self.item2 = {'a': {'a1': [6.1, 7.3],
                            'a2': {'b1': 4.11, 'b2': {}, 'b3': 53, 'b4': False},
                            'a3': np.array([[9.2, 14.3], [6.1, 1.1]]),
                            'a4': {'z1': False},
                            'a5': {'m2': bytearray(i * 9 for i in range(10)),
                                   'm3': bytes(i * 13 for i in range(10)),
                                   'm4': (4, 1.3, 7.8),
                                   'm5': torch.tensor(np.array([[77.6, 8.5], [7.2, 7.3]])),
                                   'm6': set([13, 24]),
                                   'm7': 'fvb r',
                                   'm8': None,
                                   },
                            'df': {'df': pd.DataFrame({'y1': [8, 7, 16], 'y2': [14, 35, 61]}),
                                   'series': pd.Series(np.array([6.5, 7.2]))},
                            }}
        "#ccc dictToFillRes is concat of item1 and item2 items"
        self.dictToFillRes = {'a': {'a1': [[6.1, 7.3], [2, 2.4]],
                                    'a2': {'b1': [4.11, 3.4], 'b2': [{}, {}], 'b3': [53, 4],
                                           'b4': [False, True]},
                                    'a3': [np.array([[9.2, 14.3], [6.1, 1.1]]),
                                           np.array([[3.1, 4.1], [5.1, 6.1]])],
                                    'a4': {'z1': [False, True]},
                                    'a5': {'m2': [bytearray(i * 9 for i in range(10)),
                                                  bytearray(i * 15 for i in range(10))],
                                           'm3': [bytes(i * 13 for i in range(10)),
                                                  bytes(i * 12 for i in range(10))],
                                           'm4': [(4, 1.3, 7.8), (4, 3.2, 6.7)],
                                           'm5': [torch.tensor(np.array([[77.6, 8.5], [7.2, 7.3]])),
                                                  torch.tensor(np.array([[3.1, 4.1], [5.1, 6.1]]))],
                                           'm6': [set([13, 24]), set([12, 43])],
                                           'm7': ['fvb r', 'dsdnk'],
                                           'm8': [None, None],
                                           },
                                    'df': {
                                        'df': [pd.DataFrame({'y1': [8, 7, 16], 'y2': [14, 35, 61]}),
                                               pd.DataFrame({'y1': [1, 2, 3], 'y2': [4, 5, 6]})],
                                        'series': [pd.Series(np.array([6.5, 7.2])),
                                                   pd.Series(np.array([3.1, 4.1]))]},
                                    }}
        self.item3 = [4, 6, 8]
        self.item4 = [7, 9, 6]
        self.nonDictionaryRes = [[4, 6, 8], [7, 9, 6]]

    def toTensorFunc(self, obj):
        obj = torch.tensor(obj)
        if obj.dtype == torch.float16 or obj.dtype == torch.float64:
            obj = obj.to(torch.float32)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return obj.to(device)

    def tensorSetUp(self):
        toTensorFunc = self.toTensorFunc

        self.dictToFillTensorRes = {'a': {'a1': toTensorFunc([[6.1, 7.3], [2, 2.4]]),
                                          'a2': {'b1': toTensorFunc([4.11, 3.4]), 'b2': [{}, {}],
                                                 'b3': toTensorFunc([53, 4]),
                                                 'b4': toTensorFunc([False, True])},
                                          'a3': toTensorFunc([np.array([[9.2, 14.3], [6.1, 1.1]]),
                                                              np.array([[3.1, 4.1], [5.1, 6.1]])]),
                                          'a4': {'z1': toTensorFunc([False, True])},
                                          'a5': {'m2': toTensorFunc(
                                              [bytearray(i * 9 for i in range(10)),
                                               bytearray(i * 15 for i in range(10))]),
                                              'm3': [bytes(i * 13 for i in range(10)),
                                                     bytes(i * 12 for i in range(10))],
                                              'm4': toTensorFunc([(4, 1.3, 7.8), (4, 3.2, 6.7)]),
                                              'm5': toTensorFunc(torch.stack([torch.tensor(
                                                  np.array([[77.6, 8.5], [7.2, 7.3]])),
                                                  torch.tensor(
                                                      np.array([[3.1,
                                                                 4.1],
                                                                [5.1,
                                                                 6.1]]))])),
                                              'm6': [set([13, 24]), set([12, 43])],
                                              'm7': ['fvb r', 'dsdnk'],
                                              'm8': [None, None],
                                          },
                                          'df': {'df': torch.stack([toTensorFunc(pd.DataFrame(
                                              {'y1': [8, 7, 16], 'y2': [14, 35, 61]}).values),
                                                                    # first converted to nparray then tensor
                                                                    toTensorFunc(pd.DataFrame(
                                                                        {'y1': [1, 2, 3],
                                                                         'y2': [4, 5,
                                                                                6]}).values)]),
                                                 'series': torch.stack(
                                                     [toTensorFunc(pd.Series(np.array([6.5, 7.2]))),
                                                      toTensorFunc(
                                                          pd.Series(np.array([3.1, 4.1])))])},
                                          }}

    def setUpAllTypesCheck(self):
        toTensorFunc = self.toTensorFunc

        self.inputs = {
            'df': pd.DataFrame(
                {'y1': [i for i in range(10, 20)], 'y2': [i for i in range(20, 30)]}),
            'npDict': NpDict(
                pd.DataFrame({'y1': [i for i in range(10, 20)], 'y2': [i for i in range(20, 30)]})),
            'series': pd.Series(np.array([i for i in range(10, 20)])),
            'npArray': np.array([i for i in range(10, 20)]),
            'tensor': torch.tensor(np.array([i for i in range(10, 20)])),
            'npArray2d': np.array([[i, i + 3] for i in range(10, 20)]),
            'tensor2d': torch.tensor(np.array([[i, i + 3] for i in range(10, 20)])),
            'byteArray': bytearray(i for i in range(10, 20)),
            'bytes': bytes(i for i in range(10, 20)),
            'list': [i + .1 for i in range(10, 20)],
            'list2Elements': [[i + .1, i + 3.1] for i in range(10, 20)],
            'tuple': [i + .1 for i in range(10, 20)],
            'set': set(i + .1 for i in range(10, 20)),
            'listStr': ['fvb r' for i in range(10, 20)],
            'listBool': [bool(i % 5) for i in range(10, 20)],
            'listNone': [None for i in range(10, 20)],
            'dict': {'s': 1},
            'dotDict': DotDict({'s': 1}),
        }
        self.inputsRes = {
            'df': toTensorFunc(pd.DataFrame(
                {'y1': [i for i in range(10, 20)], 'y2': [i for i in range(20, 30)]}).values),
            'npDict': toTensorFunc(NpDict(pd.DataFrame(
                {'y1': [i for i in range(10, 20)], 'y2': [i for i in range(20, 30)]})).df.values),
            'series': toTensorFunc(pd.Series(np.array([i for i in range(10, 20)]))),
            'npArray': toTensorFunc(np.array([i for i in range(10, 20)])),
            'tensor': toTensorFunc(torch.tensor(np.array([i for i in range(10, 20)]))),
            'npArray2d': toTensorFunc(np.array([[i, i + 3] for i in range(10, 20)])),
            'tensor2d': toTensorFunc(torch.tensor(np.array([[i, i + 3] for i in range(10, 20)]))),
            'byteArray': toTensorFunc(bytearray(i for i in range(10, 20))),
            'bytes': [bytes(i for i in range(10, 20))],
            'list': toTensorFunc([i + .1 for i in range(10, 20)]),
            'list2Elements': toTensorFunc([[i + .1, i + 3.1] for i in range(10, 20)]),
            'tuple': toTensorFunc([i + .1 for i in range(10, 20)]),
            'set': [{i + .1 for i in range(10, 20)}],
            # kkk working correct for set because of append to BatchStruct .values
            'listStr': [['fvb r' for i in range(10, 20)]],
            # kkk these extra [ ] around is because of appendValue_ToNestedDictPath
            'listBool': toTensorFunc([bool(i % 5) for i in range(10, 20)]),
            'listNone': [[None for i in range(10, 20)]],
            'dict': {'s': toTensorFunc([1])},  # [1] is is because of appendValue_ToNestedDictPath
            'dotDict': [DotDict({'s': 1})],  # kkk didnt work for dotDict
        }

    def testNestedDictStruct_forADictItem(self):
        self.setUp()
        dictToFill = _NestedDictStruct(self.item1)
        dictToFill.fillWithData(self.item2)
        dictToFill.fillWithData(self.item1)
        self.assertEqual(str(dictToFill.getData_single()), str(self.dictToFillRes))
        """#ccc seems to be a not secure way, but because we have verified types
        which are either default python types or torch, np, pd types so seems to be ok"""

    def testNestedDictStruct_forA_NonDictItem(self):
        self.setUp()
        dictToFill = _NestedDictStruct(self.item3)
        dictToFill.fillWithData(self.item3)
        dictToFill.fillWithData(self.item4)
        self.assertEqual(dictToFill.getData_single(), self.nonDictionaryRes)

    def testGetDictStructTensors(self):
        self.setUp()
        self.tensorSetUp()
        dictToFill = _NestedDictStruct(self.item1)
        dictToFill.fillWithData(self.item2)
        dictToFill.fillWithData(self.item1)
        self.assertEqual(str(dictToFill.getDataAsGpuTensors_single()),
                         str(self.dictToFillTensorRes))

    def testAllTypesCheck(self):
        self.setUpAllTypesCheck()
        dictToFill = _NestedDictStruct(self.inputs)
        dictToFill.fillWithData(self.inputs)
        self.assertEqual(str(dictToFill.getDataAsGpuTensors_single()), str(self.inputsRes))


class DataloaderTests(BaseTestClass):
    def setup1(self, batch_size=5):
        self.seed = 65

        class customDataset(VAnnTsDataset):
            def __getitem__(self, idx):
                return self.data['a'][idx]

        self.dataset = customDataset(data=pd.DataFrame({'a': [i + 1000 for i in range(8, 170)]}),
                                     backcastLen=0, forecastLen=0)
        self.dataloader = VAnnTsDataloader(self.dataset, batch_size=batch_size, shuffle=True,
                                           randomSeed=self.seed)

    def testShuffledWithSeed(self):
        self.setup1()
        firstBatch = next(iter(self.dataloader))
        expectedFirstBatch = torch.tensor([1114, 1081, 1168, 1139, 1064], dtype=torch.int64)
        self.equalTensors(firstBatch, expectedFirstBatch, checkDevice=False)

    def testShuffleAllResults_1stBatch(self):
        self.setup1(batch_size=700)
        firstBatch = next(iter(self.dataloader))
        expectedFirstBatch = shuffleData([i + 1000 for i in range(8, 170)], self.seed)
        expectedFirstBatch = torch.tensor(expectedFirstBatch, dtype=torch.int64)
        self.equalTensors(firstBatch, expectedFirstBatch, checkDevice=False)

    def testShuffleAllResults_2ndBatch(self):
        self.setup1(batch_size=700)
        firstBatch = next(iter(self.dataloader))
        secondBatch = next(iter(self.dataloader))
        expectedFirstBatch = shuffleData([i + 1000 for i in range(8, 170)], self.seed)
        expected2ndBatch = shuffleData(expectedFirstBatch, self.seed)
        expected2ndBatch = torch.tensor(expected2ndBatch, dtype=torch.int64)
        self.equalTensors(secondBatch, expected2ndBatch, checkDevice=False)

    def testChangeShuffleBeforeGettingResultWithShuffleState(self):
        self.setup1()  # self.dataloader is shuffle=True
        # cccAlgo(same as _iterShuffleLogic)
        #  note indexes by getting shuffled result get changed inplace
        #  and there is way back even by making shuffle False
        self.dataloader.shuffle = False
        firstBatch = next(iter(self.dataloader))
        expectedFirstBatch = torch.tensor([1008, 1009, 1010, 1011, 1012], dtype=torch.int64)
        self.equalTensors(firstBatch, expectedFirstBatch, checkDevice=False)


# ---- run test
if __name__ == '__main__':
    unittest.main()
