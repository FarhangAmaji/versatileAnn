import os
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tests.baseTest import BaseTestClass
import unittest
#%%
from dataPrep.utils import returnDictStruct, appendValueToListForNestedDictPath, fillDataWithDictStruct
#%% batch data tests
class returnDictStructTests(BaseTestClass):
    def setUp(self):
        self.item1={'a':{'a1':[2],
                         'a2':{'b1':[2],'b2':{},'b3':4,'b4':True},
                         'a3':4,'a4':True}}
        self.item1TYPETypeRes={'a': {'a1': "<class 'list'>",
                                      'a2': {'b1': "<class 'list'>", 'b2': 'empty', 'b3': "<class 'int'>", 'b4': "<class 'bool'>"},
                                      'a3': "<class 'int'>", 'a4': "<class 'bool'>"}}
        self.item1TYPEEmptyList={'a': {'a1': [], 'a2': {'b1': [], 'b2': [], 'b3': [], 'b4': []}, 'a3': [], 'a4': []}}

    def testReturnDictStructOfTYPESType(self):
        self.setUp()
        returnDictStructOfTYPEType=returnDictStruct(self.item1,'types')
        assert returnDictStructOfTYPEType.dictStruct==self.item1TYPETypeRes
    
    def testReturnDictStructOfTYPEEmptyList(self):
        self.setUp()
        returnDictStructOfTYPEType=returnDictStruct(self.item1,'emptyList')
        assert returnDictStructOfTYPEType.dictStruct==self.item1TYPEEmptyList
#%% appendValueToListForNestedDictPathTests
class appendValueToListForNestedDictPathTests(BaseTestClass):
    def setUp(self):
        self.item1 = {'a': {'a1': {'b1': []},
                     'a2': {'b1': [], 'b2': [], 'b3': [], 'b4': []},
                     'a3': [],
                     'a4': []}}
        self.avtl=appendValueToListForNestedDictPath

    def test1(self):
        self.avtl(self.item1, ['a', 'a1', 'b1'], 5)
        assert self.item1=={'a': {'a1': {'b1': [5]}, 'a2': {'b1': [], 'b2': [], 'b3': [], 'b4': []}, 'a3': [], 'a4': []}}
        
    def test2(self):
        self.avtl(self.item1, ['a', 'a1', 'b1'], 6)
        self.avtl(self.item1, ['a', 'a1', 'b1'], 5)
        assert self.item1=={'a': {'a1': {'b1': [6,5]}, 'a2': {'b1': [], 'b2': [], 'b3': [], 'b4': []}, 'a3': [], 'a4': []}}

    def testDoesntLeadToAList(self):
        with self.assertRaises(AssertionError) as context:
            self.avtl(self.item1, ['a', 'a1'], 4)
        self.assertEqual(str(context.exception), "['a', 'a1'] doesnt lead to a list")

    def testNoKeyWithSomeName(self):
        with self.assertRaises(AssertionError) as context:
            self.avtl(self.item1, ['a', 'a1','b2'], 4)
        self.assertEqual(str(context.exception), "b2 is not in ['a', 'a1', 'b2']")

    def testNotADict(self):
        with self.assertRaises(AssertionError) as context:
            self.avtl(self.item1, ['a', 'a4','b2','xs'], 4)
        self.assertEqual(str(context.exception), "['a', 'a4', 'b2'] is not a dictionary")

    def testKeyNotInDict(self):
        with self.assertRaises(AssertionError) as context:
            self.avtl(self.item1, ['a', 'a1','b2','b3'], 4)
        self.assertEqual(str(context.exception), "b2 is not in ['a', 'a1']")
#%% fillDataWithDictStructTests
class fillDataWithDictStructTests(BaseTestClass):
    def setUp(self):
        self.item1={'a':{'a1':[2],
                         'a2':{'b1':[2],'b2':{},'b3':4,'b4':True},
                         'a3':4,'a4':True}}
        self.item2={'a':{'a1':[4],
                         'a2':{'b1':[3],'b2':{},'b3':11,'b4':False},
                         'a3':41,'a4':True}}
        self.dictToFillRes={'a': {'a1': [[2], [4]],
                                  'a2': {'b1': [[2], [3]], 'b2': [{}, {}], 'b3': [4, 11], 'b4': [True, False]},
                                  'a3': [4, 41], 'a4': [True, True]}}
    def test(self):
        dictStruct = returnDictStruct(self.item1).dictStruct
        dictToFill = dictStruct.copy()
        fillDataWithDictStruct(self.item1, dictToFill, path=[])
        fillDataWithDictStruct(self.item2, dictToFill, path=[])
        assert dictToFill==self.dictToFillRes
#%%
if __name__ == '__main__':
    unittest.main()