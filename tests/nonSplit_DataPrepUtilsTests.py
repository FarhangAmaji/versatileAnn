import os
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tests.baseTest import BaseTestClass
import unittest
from dataPrep.utils import subtractFromIndexes, simpleSplit
#%% subtractFromIndexesTests
class subtractFromIndexesTests(BaseTestClass):
    def setUp(self):
        self.indexes=[i for i in range(100)]
        self.isAnyConditionApplied=False

    def assertLastInds(self, trainRatio,valRatio,
                       trainSeqLen, valSeqLen, testSeqLen):
        "#ccc we are subtracting some len from indexes so the last index of sets(train, val, test)"
        "... doesnt exceed the last index of original indexes"
        newIndexes=subtractFromIndexes(self.indexes, trainRatio,valRatio, trainSeqLen,
                            valSeqLen, testSeqLen, self.isAnyConditionApplied)
        trainIndexes, valIndexes, testIndexes=simpleSplit(newIndexes, trainRatio, valRatio)
        self.assertTrue(trainIndexes[-1]+trainSeqLen-1<=self.indexes[-1])
        self.assertTrue(valIndexes[-1]+valSeqLen-1<=self.indexes[-1])
        self.assertTrue(testIndexes[-1]+testSeqLen-1<=self.indexes[-1])

    def test1(self):
        trainRatio = 0.7
        valRatio = 0.2
        trainSeqLen = 37
        valSeqLen = 16
        testSeqLen = 5
        self.assertLastInds(trainRatio,valRatio, trainSeqLen, valSeqLen, testSeqLen)
                                
    def test2(self):
        trainRatio = 0.7
        valRatio = 0.2
        trainSeqLen = 37
        valSeqLen = 5
        testSeqLen = 16
        self.assertLastInds(trainRatio,valRatio, trainSeqLen, valSeqLen, testSeqLen)

    def test3(self):
        trainRatio = 0.7
        valRatio = 0.2
        trainSeqLen = 16
        valSeqLen = 37
        testSeqLen = 5
        self.assertLastInds(trainRatio,valRatio, trainSeqLen, valSeqLen, testSeqLen)

    def test4(self):
        trainRatio = 0.7
        valRatio = 0.2
        trainSeqLen = 16
        valSeqLen = 5
        testSeqLen = 37
        self.assertLastInds(trainRatio,valRatio, trainSeqLen, valSeqLen, testSeqLen)

    def test5(self):
        trainRatio = 0.7
        valRatio = 0.2
        trainSeqLen = 5
        valSeqLen = 37
        testSeqLen = 16
        self.assertLastInds(trainRatio,valRatio, trainSeqLen, valSeqLen, testSeqLen)

    def test6(self):
        trainRatio = 0.7
        valRatio = 0.2
        trainSeqLen = 5
        valSeqLen = 16
        testSeqLen = 37
        self.assertLastInds(trainRatio,valRatio, trainSeqLen, valSeqLen, testSeqLen)

    def test7(self):
        #kkk could have checked other combos
        for tr in [.5,.6]:
            trainRatio = tr
            valRatio = 0.2
            trainSeqLen = 5
            valSeqLen = 37
            testSeqLen = 16
            self.assertLastInds(trainRatio,valRatio, trainSeqLen, valSeqLen, testSeqLen)
#%% run test
if __name__ == '__main__':
    unittest.main()