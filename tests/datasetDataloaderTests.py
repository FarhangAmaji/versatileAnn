import os
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tests.baseTest import BaseTestClass
import unittest
#%%
"#ccc some of the tests related to dataset and TsRowFetcher are TsRowFetcherTests"
from dataPrep.dataset import VAnnTsDataset
from dataPrep.dataloader import BatchStructTemplate, appendValue_ToNestedDictPath
from dataPrep.dataloader import BatchStructTemplate_Non_BatchStructTemplate_Objects as bstObjInit
from utils.vAnnGeneralUtils import DotDict, NpDict
from utils.globalVars import tsStartPointColName
import torch
import pandas as pd
import numpy as np
#%% dataset tests
#%%         VAnnTsDataset_NoIndexesAssertionTests
#kkk this test also does some of check related to VAnnTsDataset_indexesSettingTests
class VAnnTsDataset_NoIndexesAssertionTests(BaseTestClass):
    def setUp(self):
        self.df = pd.DataFrame({'A': [1, 2, 3, 4], 'B': [5, 6, 7, 8], '__startPoint__': [False, True, True, False]}, index=[8,9,10,11])

    def testDf_NoIndexesPassed_NoStartPoint_No0lens(self):
        self.setUp()
        self.df = self.df.drop('__startPoint__', axis=1)
        with self.assertRaises(AssertionError) as context:
            VAnnTsDataset(self.df, backcastLen=1, forecastLen=1, useNpDictForDfs=False)
        self.assertEqual(str(context.exception), VAnnTsDataset.noIndexesAssertionMsg)

    def testDf_NoIndexesPassed_NoStartPoint_0lens(self):
        self.setUp()
        self.df=self.df.drop('__startPoint__', axis=1)
        dataset = VAnnTsDataset(self.df, backcastLen=0, forecastLen=0, useNpDictForDfs=False)
        self.assertEqual(dataset.indexes, [0, 1, 2, 3])
        self.assertEqual(len(dataset), 4)  # All rows should be included

    def testDf_NoIndexesPassed_WithStartPoint_No0lens(self):
        self.setUp()
        dataset = VAnnTsDataset(self.df, backcastLen=1, forecastLen=1, useNpDictForDfs=False)
        self.assertTrue(list(dataset.indexes) == [9, 10])#ccc note indexes has kept their values
        self.assertEqual(len(dataset), 2)  # Only rows with '__startPoint__' as True should be included

    def testDf_NoIndexesPassed_WithStartPoint_0lens(self):
        self.setUp()
        dataset = VAnnTsDataset(self.df, backcastLen=0, forecastLen=0, useNpDictForDfs=False)
        self.assertTrue(list(dataset.indexes) == [9, 10])
        self.assertEqual(len(dataset), 2)  # Only rows with '__startPoint__' as True should be included

    def testNpArray_NoIndexesPassed_0lens(self):
        self.setUp()
        npArray = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        dataset = VAnnTsDataset(npArray, backcastLen=0, forecastLen=0)
        self.assertEqual(dataset.indexes, [0, 1, 2])
        self.assertEqual(len(dataset), 3)

    def testNpArray_NoIndexesPassed_No0lens(self):
        self.setUp()
        npArray = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        with self.assertRaises(AssertionError) as context:
            VAnnTsDataset(npArray, backcastLen=1, forecastLen=1)
        self.assertEqual(str(context.exception), VAnnTsDataset.noIndexesAssertionMsg)

    def testNpDict_NoIndexesPassed_WithStartPoint_No0lens(self):
        self.setUp()
        npDict = NpDict(self.df)
        dataset = VAnnTsDataset(npDict, backcastLen=1, forecastLen=1)
        self.assertTrue(list(dataset.indexes) == [1, 2])
        self.assertEqual(len(dataset), 2)  # Only rows with '__startPoint__' as True should be included

    def testNpDict_NoIndexesPassed_NoStartPoint_No0lens(self):
        self.setUp()
        npDict = NpDict(pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9]}))
        with self.assertRaises(AssertionError) as context:
            VAnnTsDataset(npDict, backcastLen=1, forecastLen=1)
        self.assertEqual(str(context.exception), VAnnTsDataset.noIndexesAssertionMsg)

    def testNpArrayWithIndexes_No0lens(self):
        self.setUp()
        npArray = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        indexes = [0, 2]  # Include only rows at index 0 and 2
        dataset = VAnnTsDataset(npArray, backcastLen=1, forecastLen=1, indexes=indexes)
        self.assertEqual(len(dataset), 2)  # Only specified indexes should be included
#%%         VAnnTsDataset_indexesSettingTests
class VAnnTsDataset_indexesSettingTests(BaseTestClass):
    '#ccc test assumes, noIndexes are passed and there is some value for backcast and forecast lens'
    def setUp(self):
        self.df = pd.DataFrame({'A': list(range(10)),
                                '__startPoint__': [True, False, True, True, False, True, False, True, False, True]},
                               index=list(range(10, 20)))

    def testDf_StartPointsInCols_useNpDictForDfs(self):
        dataset = VAnnTsDataset(self.df, backcastLen=3, forecastLen=2, useNpDictForDfs=True)
        self.assertEqual(dataset.indexes, list(self.df[self.df['__startPoint__']==True].index))
        self.assertEqual(dataset.usedDfToNpInds, True)

    def testDf_StartPointsInCols_noUseNpDictForDfs(self):
        dataset = VAnnTsDataset(self.df, backcastLen=3, forecastLen=2, useNpDictForDfs=False)
        self.assertEqual(dataset.indexes, list(self.df[self.df['__startPoint__']==True].index))
        self.assertEqual(dataset.usedDfToNpInds, False)

    def testNpDict_StartPointsInCols(self):
        dataset = VAnnTsDataset(NpDict(self.df), backcastLen=3, forecastLen=2)
        self.assertEqual(dataset.indexes, [0, 2, 3, 5, 7, 9])
        self.assertEqual(dataset.usedDfToNpInds, False)
#%%         VAnnTsDataset_NoNanOrNoneDataAssertionTests
class VAnnTsDataset_NoNanOrNoneDataAssertionTests(BaseTestClass):
    def setUp(self):
        self.df = pd.DataFrame({'A': [1, 2, 3, 4], 'B': [5, 6, np.nan, 8], '__startPoint__': [False, True, True, False]}, index=[8,9,10,11])

    def testDf_WithStartPoint(self):
        with self.assertRaises(ValueError) as context:
            VAnnTsDataset(self.df, backcastLen=1, forecastLen=1, useNpDictForDfs=False)
        self.assertEqual(str(context.exception), "The DataFrame contains NaN values.")

    def testNpDict_WithStartPoint(self):
        with self.assertRaises(ValueError) as context:
            VAnnTsDataset(NpDict(self.df), backcastLen=1, forecastLen=1)
        self.assertEqual(str(context.exception), "The NpDict contains NaN values.")

    def testNpArray_WithStartPoint(self):
        npArray = np.array([[1, 2, 3], [4, 5, 6], [7, 8, np.nan]])
        with self.assertRaises(ValueError) as context:
            VAnnTsDataset(npArray, backcastLen=0, forecastLen=0)
        self.assertEqual(str(context.exception), "The NumPy array contains NaN values.")
#%%         VAnnTsDataset_NSeries_assignData
class VAnnTsDataset_NSeries_assignData(BaseTestClass):
    def setUp(self):
        self.df = pd.DataFrame({
            'A': 26*['A1'],
            'B': 7*['B1']+19*['B2'],
            tsStartPointColName: 3*[True]+4*[False]+15*[True]+4*[False],
            'y1': list(range(30, 56)),
            'y2': list(range(130, 156))},index=range(100, 126))
        self.dataset=VAnnTsDataset(self.df,backcastLen=2, forecastLen=3, mainGroups=['A','B'], useNpDictForDfs=False)
        self.expectedGroup1 = pd.DataFrame({'A': ['A1', 'A1', 'A1', 'A1', 'A1', 'A1', 'A1'],
                                            'B': ['B1', 'B1', 'B1', 'B1', 'B1', 'B1', 'B1'],
                                            '__startPoint__': [ True,  True,  True, False, False, False, False],
                                            'y1': [30, 31, 32, 33, 34, 35, 36], 'y2': [130, 131, 132, 133, 134, 135, 136]}, index=list(range(100,107)))
        self.expectedGroup2 = pd.DataFrame({'A': 19*['A1'], 'B': 19*['B2'],
                                            '__startPoint__':15*[True]+4*[False], 'y1': list(range(37,56)),
                                            'y2': list(range(137,156))}, index=list(range(107,126)))

    def test(self):
        self.assertTrue(list(self.dataset.data.keys())==[('A1', 'B1'), ('A1', 'B2')])
        self.assertTrue(self.expectedGroup1.equals(self.dataset.data[('A1', 'B1')]))
        self.assertTrue(self.expectedGroup2.equals(self.dataset.data[('A1', 'B2')]))
#%%         VAnnTsDataset_NSeries_GetItemTests
#kkk implement later
#%%         VAnnTsDataset_NoNSeries_GetItemTests
class VAnnTsDataset_NoNSeries_GetItemTests(BaseTestClass):
    def setUp(self):
        self.df = pd.DataFrame({'A': [1, 2, 3, 4], 'B': [5, 6, 7, 8], '__startPoint__': [False, True, True, False]}, index=[8,9,10,11])
        self.npDict = NpDict(self.df)
        self.npArray = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

    def testGetItemDfInStartPoints(self):
        dataset = VAnnTsDataset(self.df, backcastLen=0, forecastLen=0, useNpDictForDfs=False)
        idx = 9  # Index is in indexes
        expected = self.df.loc[idx]
        result = dataset[idx]
        self.assertTrue(result.equals(expected))

    def testGetItemDfNotInStartPoints(self):
        dataset = VAnnTsDataset(self.df, backcastLen=0, forecastLen=0, useNpDictForDfs=False)
        idx = 8  # Index is not in indexes
        with self.assertRaises(AssertionError) as context:
            dataset[idx]
        self.assertEqual(str(context.exception), f"{idx} is not in indexes")

    def testGetItemNpDictInStartPoints(self):
        dataset = VAnnTsDataset(self.npDict, backcastLen=0, forecastLen=0)
        idx = 2  # Index is in indexes
        expected = np.array([3, 7, 1])
        result = dataset[idx]
        np.testing.assert_array_equal(result, expected)

    def testGetItemNpDictNotInStartPoints(self):
        dataset = VAnnTsDataset(self.npDict, backcastLen=0, forecastLen=0)
        idx = 0  # Index is not in indexes
        with self.assertRaises(AssertionError) as context:
            dataset[idx]
        self.assertEqual(str(context.exception), f"{idx} is not in indexes")

    def testGetItemNpArrayInStartPoints(self):
        dataset = VAnnTsDataset(self.npArray, backcastLen=0, forecastLen=0)
        idx = 1
        expected = np.array([4, 5, 6])
        result = dataset[idx]
        np.testing.assert_array_equal(result, expected)

    def testGetItemNpArrayNotInStartPoints(self):
        dataset = VAnnTsDataset(self.npArray, backcastLen=0, forecastLen=0, indexes=[0,2])
        idx = 1
        with self.assertRaises(AssertionError) as context:
            dataset[idx]
        self.assertEqual(str(context.exception), f"{idx} is not in indexes")
#%% dataloader tests
#%%     batch data tests
class batchStructTemplateTests(BaseTestClass):
    def setUp(self):
        self.item1={'a':{'a1':[2],
                         'a2':{'b1':[2],'b2':{},'b3':4,'b4':True},
                         'a3':4,'a4':True}}
        self.item1Res={'a':{'a1':bstObjInit([2]),
                         'a2':{'b1':bstObjInit([2]),'b2':bstObjInit({}),'b3':bstObjInit(4),'b4':bstObjInit(True)},
                         'a3':bstObjInit(4),'a4':bstObjInit(True)}}

    def test(self):
        self.setUp()
        self.assertEqual(str(BatchStructTemplate(self.item1)),str(self.item1Res))
        """#ccc seems to be a not secure way
        but because we have only batchStructTemplate and BatchStructTemplate_Non_BatchStructTemplate_Objects types with defined __repr__s
        so seems to be ok"""
#%%     appendValue_ToNestedDictPathTests
class appendValue_ToNestedDictPathForSimpleDictionaryTests(BaseTestClass):
    def setUp(self):
        self.item1 = {'a': {'a1': {'b1': []},
                     'a2': {'b1': [], 'b2': [], 'b3': [], 'b4': []},
                     'a3': [],
                     'a4': []}}
        self.avtl=appendValue_ToNestedDictPath

    def test1(self):
        self.setUp()
        self.avtl(self.item1, ['a', 'a1', 'b1'], 5)
        self.assertEqual(self.item1,{'a': {'a1': {'b1': [5]}, 'a2': {'b1': [], 'b2': [], 'b3': [], 'b4': []}, 'a3': [], 'a4': []}})

    def test2(self):
        self.setUp()
        self.avtl(self.item1, ['a', 'a1', 'b1'], 6)
        self.avtl(self.item1, ['a', 'a1', 'b1'], 5)
        self.assertEqual(self.item1,{'a': {'a1': {'b1': [6,5]}, 'a2': {'b1': [], 'b2': [], 'b3': [], 'b4': []}, 'a3': [], 'a4': []}})

    def testDoesntLeadToAList(self):
        self.setUp()
        with self.assertRaises(AssertionError) as context:
            self.avtl(self.item1, ['a', 'a1'], 4)
        self.assertEqual(str(context.exception), "['a', 'a1'] doesn't lead to a list")

    def testNoKeyWithSomeName(self):
        self.setUp()
        with self.assertRaises(AssertionError) as context:
            self.avtl(self.item1, ['a', 'a1','b2'], 4)
        self.assertEqual(str(context.exception), "b2 is not in ['a', 'a1', 'b2']")

    def testNotADict(self):
        self.setUp()
        with self.assertRaises(AssertionError) as context:
            self.avtl(self.item1, ['a', 'a4','b2','xs'], 4)
        self.assertEqual(str(context.exception), "['a', 'a4', 'b2'] is not a dict or DotDict")

    def testKeyNotInDict(self):
        self.setUp()
        with self.assertRaises(AssertionError) as context:
            self.avtl(self.item1, ['a', 'a1','b2','b3'], 4)
        self.assertEqual(str(context.exception), "b2 is not in ['a', 'a1']")
#%%     fillBatchStructWithDataTests
class fillBatchStructWithDataTests(BaseTestClass):
    def setUp(self):
        self.item1={'a':{'a1':[2,2.4],
                         'a2':{'b1':3.4,'b2':{},'b3':4,'b4':True},
                         'a3':np.array([[3.1,4.1],[5.1,6.1]]),
                         'a4':{'z1':True},
                         'a5':{'m2':bytearray(i*15 for i in range(10)),
                               'm3':bytes(i*12 for i in range(10)),
                               'm4':(4,3.2,6.7),
                               'm5':torch.tensor(np.array([[3.1,4.1],[5.1,6.1]])),
                               'm6':set([12,43]),
                               'm7':'dsdnk',
                               'm8':None,
                               },
                    'df':{'df':pd.DataFrame({'y1': [1, 2, 3], 'y2': [4, 5, 6]}),
                          'series':pd.Series(np.array([3.1,4.1]))},
                     }}
        self.item2={'a':{'a1':[6.1,7.3],
                         'a2':{'b1':4.11,'b2':{},'b3':53,'b4':False},
                         'a3':np.array([[9.2,14.3],[6.1,1.1]]),
                         'a4':{'z1':False},
                         'a5':{'m2':bytearray(i*9 for i in range(10)),
                               'm3':bytes(i*13 for i in range(10)),
                               'm4':(4,1.3,7.8),
                               'm5':torch.tensor(np.array([[77.6,8.5],[7.2,7.3]])),
                               'm6':set([13,24]),
                               'm7':'fvb r',
                               'm8':None,
                               },
                    'df':{'df':pd.DataFrame({'y1': [8,7,16], 'y2': [14, 35, 61]}),
                          'series':pd.Series(np.array([6.5,7.2]))},
                     }}
        "#ccc dictToFillRes is concat of item1 and item2 items"
        self.dictToFillRes={'a':{'a1':[[6.1,7.3],[2,2.4]],
                                 'a2':{'b1':[4.11, 3.4],'b2':[{}, {}],'b3':[53, 4],'b4':[False, True]},
                                 'a3':[np.array([[9.2,14.3],[6.1,1.1]]), np.array([[3.1,4.1],[5.1,6.1]])],
                                 'a4':{'z1':[False, True]},
                                 'a5':{'m2':[bytearray(i*9 for i in range(10)), bytearray(i*15 for i in range(10))],
                                       'm3':[bytes(i*13 for i in range(10)), bytes(i*12 for i in range(10))],
                                       'm4':[(4,1.3,7.8), (4,3.2,6.7)],
                                       'm5':[torch.tensor(np.array([[77.6,8.5],[7.2,7.3]])), torch.tensor(np.array([[3.1,4.1],[5.1,6.1]]))],
                                       'm6':[set([13,24]), set([12,43])],
                                       'm7':['fvb r', 'dsdnk'],
                                       'm8':[None, None],
                                       },
                         'df':{'df':[pd.DataFrame({'y1': [8,7,16], 'y2': [14, 35, 61]}), pd.DataFrame({'y1': [1, 2, 3], 'y2': [4, 5, 6]})],
                               'series':[pd.Series(np.array([6.5,7.2])), pd.Series(np.array([3.1,4.1]))]},
                     }}
        self.item3=[4,6,8]
        self.item4=[7,9,6]
        self.nonDictionaryRes=[[4,6,8], [7,9,6]]

    def toTensorFunc(self, obj):
        obj = torch.tensor(obj)
        if obj.dtype == torch.float16 or obj.dtype == torch.float64:
            obj = obj.to(torch.float32)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return obj.to(device)

    def tensorSetUp(self):
        toTensorFunc=self.toTensorFunc

        self.dictToFillTensorRes={'a':{'a1':toTensorFunc([[6.1,7.3],[2,2.4]]),
                                 'a2':{'b1':toTensorFunc([4.11, 3.4]),'b2':[{}, {}],'b3':toTensorFunc([53, 4]),'b4':toTensorFunc([False, True])},
                                 'a3':toTensorFunc([np.array([[9.2,14.3],[6.1,1.1]]), np.array([[3.1,4.1],[5.1,6.1]])]),
                                 'a4':{'z1':toTensorFunc([False, True])},
                                 'a5':{'m2':toTensorFunc([bytearray(i*9 for i in range(10)), bytearray(i*15 for i in range(10))]),
                                       'm3':[bytes(i*13 for i in range(10)), bytes(i*12 for i in range(10))],
                                       'm4':toTensorFunc([(4,1.3,7.8), (4,3.2,6.7)]),
                                       'm5':toTensorFunc(torch.stack([torch.tensor(np.array([[77.6,8.5],[7.2,7.3]])), torch.tensor(np.array([[3.1,4.1],[5.1,6.1]]))])),
                                       'm6':[set([13,24]), set([12,43])],
                                       'm7':['fvb r', 'dsdnk'],
                                       'm8':[None, None],
                                       },
                         'df':{'df':torch.stack([toTensorFunc(pd.DataFrame({'y1': [8,7,16], 'y2': [14, 35, 61]}).values),#first converted to nparray then tensor
                                     toTensorFunc(pd.DataFrame({'y1': [1, 2, 3], 'y2': [4, 5, 6]}).values)]),
                               'series':torch.stack([toTensorFunc(pd.Series(np.array([6.5,7.2]))), toTensorFunc(pd.Series(np.array([3.1,4.1])))])},
                     }}

    def setUpAllTypesCheck(self):
        toTensorFunc=self.toTensorFunc

        self.inputs = {
            'df':pd.DataFrame({'y1': [i for i in range(10, 20)], 'y2': [i for i in range(20, 30)]}),
            'npDict':NpDict(pd.DataFrame({'y1': [i for i in range(10, 20)], 'y2': [i for i in range(20, 30)]})),
            'series':pd.Series(np.array([i for i in range(10, 20)])),
            'npArray':np.array([i for i in range(10, 20)]),
            'tensor':torch.tensor(np.array([i for i in range(10, 20)])),
            'npArray2d':np.array([[i,i+3] for i in range(10, 20)]),
            'tensor2d':torch.tensor(np.array([[i,i+3] for i in range(10, 20)])),
            'byteArray':bytearray(i for i in range(10, 20)),
            'bytes':bytes(i for i in range(10, 20)),
            'list':[i+.1 for i in range(10, 20)],
            'list2Elements':[[i+.1, i+3.1] for i in range(10, 20)],
            'tuple':[i+.1 for i in range(10, 20)],
            'set':set(i+.1 for i in range(10, 20)),
            'listStr':['fvb r' for i in range(10, 20)],
            'listBool':[bool(i%5) for i in range(10, 20)],
            'listNone':[None for i in range(10, 20)],
            'dict':{'s':1},
            'dotDict':DotDict({'s':1}),
            }
        self.inputsRes = {
            'df':toTensorFunc(pd.DataFrame({'y1': [i for i in range(10, 20)], 'y2': [i for i in range(20, 30)]}).values),
            'npDict':toTensorFunc(NpDict(pd.DataFrame({'y1': [i for i in range(10, 20)], 'y2': [i for i in range(20, 30)]})).df.values),
            'series':toTensorFunc(pd.Series(np.array([i for i in range(10, 20)]))),
            'npArray':toTensorFunc(np.array([i for i in range(10, 20)])),
            'tensor':toTensorFunc(torch.tensor(np.array([i for i in range(10, 20)]))),
            'npArray2d':toTensorFunc(np.array([[i,i+3] for i in range(10, 20)])),
            'tensor2d':toTensorFunc(torch.tensor(np.array([[i,i+3] for i in range(10, 20)]))),
            'byteArray':toTensorFunc(bytearray(i for i in range(10, 20))),
            'bytes':[bytes(i for i in range(10, 20))],
            'list':toTensorFunc([i+.1 for i in range(10, 20)]),
            'list2Elements':toTensorFunc([[i+.1, i+3.1] for i in range(10, 20)]),
            'tuple':toTensorFunc([i+.1 for i in range(10, 20)]),
            'set':[{i+.1 for i in range(10, 20)}],#kkk working correct for set because of append to BatchStruct .values
            'listStr':[['fvb r' for i in range(10, 20)]],#kkk these extra [ ] around is because of appendValue_ToNestedDictPath
            'listBool':toTensorFunc([bool(i%5) for i in range(10, 20)]),
            'listNone':[[None for i in range(10, 20)]],
            'dict':{'s':toTensorFunc([1])},#[1] is is because of appendValue_ToNestedDictPath
            'dotDict':[DotDict({'s':1})],#kkk didnt work for dotDict
            }

    def testBatchStructTemplate_forADictItem(self):
        self.setUp()
        dictToFill=BatchStructTemplate(self.item1)
        dictToFill.fillWithData(self.item2)
        dictToFill.fillWithData(self.item1)
        self.assertEqual(str(dictToFill.getBatchStructValues()), str(self.dictToFillRes))
        """#ccc seems to be a not secure way, but because we have verified types
        which are either default python types or torch, np, pd types so seems to be ok"""

    def testBatchStructTemplate_forA_NonDictItem(self):
        self.setUp()
        dictToFill=BatchStructTemplate(self.item3)
        dictToFill.fillWithData(self.item3)
        dictToFill.fillWithData(self.item4)
        self.assertEqual(dictToFill.getBatchStructValues(), self.nonDictionaryRes)

    def testGetDictStructTensors(self):
        self.setUp()
        self.tensorSetUp()
        dictToFill=BatchStructTemplate(self.item1)
        dictToFill.fillWithData(self.item2)
        dictToFill.fillWithData(self.item1)
        self.assertEqual(str(dictToFill.getBatchStructTensors()), str(self.dictToFillTensorRes))

    def testAllTypesCheck(self):
        self.setUpAllTypesCheck()
        dictToFill=BatchStructTemplate(self.inputs)
        dictToFill.fillWithData(self.inputs)
        self.assertEqual(str(dictToFill.getBatchStructTensors()), str(self.inputsRes))
#%% run test
if __name__ == '__main__':
    unittest.main()