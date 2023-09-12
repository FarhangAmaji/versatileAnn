import os
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tests.baseTest import BaseTestClass
import unittest
#%%
from dataPrep.utils import returnDictStruct, appendValueToListForNestedDictPath
from utils.vAnnGeneralUtils import equalDfs
import pandas as pd
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
        returnDictStructOfTYPEType=returnDictStruct('types',self.item1)
        assert returnDictStructOfTYPEType.dictStruct==self.item1TYPETypeRes
    
    def testReturnDictStructOfTYPEEmptyList(self):
        self.setUp()
        returnDictStructOfTYPEType=returnDictStruct('emptyList',self.item1)
        assert returnDictStructOfTYPEType.dictStruct==self.item1TYPEEmptyList
#%%
if __name__ == '__main__':
    unittest.main()