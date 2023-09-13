import os
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tests.baseTest import BaseTestClass
import unittest
#%%
from dataPrep.datasetNDataloader import returnDictStruct, appendValueToNestedDictPath
from dataPrep.datasetNDataloader import returnDictStruct_Non_ReturnDictStruct_Objects as rdsObjFunc
import torch
import pandas as pd
import numpy as np
#%% batch data tests
class returnDictStructTests(BaseTestClass):
    def setUp(self):
        self.item1={'a':{'a1':[2],
                         'a2':{'b1':[2],'b2':{},'b3':4,'b4':True},
                         'a3':4,'a4':True}}
        self.item1Res={'a':{'a1':rdsObjFunc([2]),
                         'a2':{'b1':rdsObjFunc([2]),'b2':rdsObjFunc({}),'b3':rdsObjFunc(4),'b4':rdsObjFunc(True)},
                         'a3':rdsObjFunc(4),'a4':rdsObjFunc(True)}}

    def test(self):
        self.setUp()
        assert str(returnDictStruct(self.item1))==str(self.item1Res)
        """#ccc seems to be a not secure way
        but because we have only returnDictStruct and returnDictStruct_Non_ReturnDictStruct_Objects types with defined __repr__s
        so seems to be ok"""
#%% appendValueToNestedDictPathTests
class appendValueToNestedDictPathForSimpleDictionaryTests(BaseTestClass):
    def setUp(self):
        self.item1 = {'a': {'a1': {'b1': []},
                     'a2': {'b1': [], 'b2': [], 'b3': [], 'b4': []},
                     'a3': [],
                     'a4': []}}
        self.avtl=appendValueToNestedDictPath

    def test1(self):
        self.setUp()
        self.avtl(self.item1, ['a', 'a1', 'b1'], 5)
        assert self.item1=={'a': {'a1': {'b1': [5]}, 'a2': {'b1': [], 'b2': [], 'b3': [], 'b4': []}, 'a3': [], 'a4': []}}
        
    def test2(self):
        self.setUp()
        self.avtl(self.item1, ['a', 'a1', 'b1'], 6)
        self.avtl(self.item1, ['a', 'a1', 'b1'], 5)
        assert self.item1=={'a': {'a1': {'b1': [6,5]}, 'a2': {'b1': [], 'b2': [], 'b3': [], 'b4': []}, 'a3': [], 'a4': []}}

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
        self.assertEqual(str(context.exception), "['a', 'a4', 'b2'] is not a dict or NpDict")

    def testKeyNotInDict(self):
        self.setUp()
        with self.assertRaises(AssertionError) as context:
            self.avtl(self.item1, ['a', 'a1','b2','b3'], 4)
        self.assertEqual(str(context.exception), "b2 is not in ['a', 'a1']")
#%% fillDataWithDictStructTests
class fillDataWithDictStructTests(BaseTestClass):
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

    def tensorSetUp(self):
        def toTensorFunc(obj):
            obj = torch.tensor(obj)
            if obj.dtype == torch.float16 or obj.dtype == torch.float64:
                obj = obj.to(torch.float32)
            return obj.to(self.device)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dictToFillTensorRes={'a':{'a1':toTensorFunc([[6.1,7.3],[2,2.4]]),
                                 'a2':{'b1':toTensorFunc([4.11, 3.4]),'b2':[{}, {}],'b3':toTensorFunc([53, 4]),'b4':toTensorFunc([False, True])},
                                 'a3':toTensorFunc([np.array([[9.2,14.3],[6.1,1.1]]), np.array([[3.1,4.1],[5.1,6.1]])]),
                                 'a4':{'z1':toTensorFunc([False, True])},
                                 'a5':{'m2':toTensorFunc([bytearray(i*9 for i in range(10)), bytearray(i*15 for i in range(10))]),
                                       'm3':[bytes(i*13 for i in range(10)), bytes(i*12 for i in range(10))],
                                       'm4':toTensorFunc([(4,1.3,7.8), (4,3.2,6.7)]),
                                       'm5':torch.stack([torch.tensor(np.array([[77.6,8.5],[7.2,7.3]])), torch.tensor(np.array([[3.1,4.1],[5.1,6.1]]))]).to(torch.float32).to(self.device),
                                       'm6':[set([13,24]), set([12,43])],
                                       'm7':['fvb r', 'dsdnk'],
                                       'm8':[None, None],
                                       },
                         'df':{'df':torch.stack([toTensorFunc(pd.DataFrame({'y1': [8,7,16], 'y2': [14, 35, 61]}).values),#first converted to nparray then tensor
                                     toTensorFunc(pd.DataFrame({'y1': [1, 2, 3], 'y2': [4, 5, 6]}).values)]),
                               'series':torch.stack([toTensorFunc(pd.Series(np.array([6.5,7.2]))), toTensorFunc(pd.Series(np.array([3.1,4.1])))])},
                     }}

    def testWithDictionaryDictStruct(self):
        self.setUp()
        dictToFill=returnDictStruct(self.item1)
        dictToFill.fillDataWithDictStruct(self.item2)
        dictToFill.fillDataWithDictStruct(self.item1)
        assert str(dictToFill.getDictStructValues())==str(self.dictToFillRes)
        """#ccc seems to be a not secure way, but because we have verified types
        which are either default python types or torch, np, pd types so seems to be ok"""

    def testWithNonDictionaryDictStruct(self):
        self.setUp()
        dictToFill=returnDictStruct(self.item3)
        dictToFill.fillDataWithDictStruct(self.item3)
        dictToFill.fillDataWithDictStruct(self.item4)
        assert dictToFill.getDictStructValues()==self.nonDictionaryRes

    def testGetDictStructTensors(self):
        self.setUp()
        self.tensorSetUp()
        dictToFill=returnDictStruct(self.item1)
        dictToFill.fillDataWithDictStruct(self.item2)
        dictToFill.fillDataWithDictStruct(self.item1)
        assert str(dictToFill.getDictStructTensors())==str(self.dictToFillTensorRes)
#%%
if __name__ == '__main__':
    unittest.main()