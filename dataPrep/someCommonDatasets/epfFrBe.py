"""
this is data preparation steps of hourly electricity price forecasts (EPF) for France and Belgium markets
the data exists in data\datasets EPF_FR_BE.csv, EPF_FR_BE_futr.csv and EPF_FR_BE_static.csv files
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.dataPrepUtils import getDatasetFiles, NormalizerStack, SingleColsStdNormalizer, MultiColStdNormalizer
#%%
futureExogenousCols = ['genForecast', 'weekDay']
historyExogenousCols = ['systemLoad']
staticExogenousCols = ['market0', 'market1']

historyDf=getDatasetFiles('EPF_FR_BE.csv')
list(map(getDatasetFiles,['EPF_FR_BE.csv', 'EPF_FR_BE_futr.csv', 'EPF_FR_BE_static.csv']))#kkk EPF_FR_BE_futr has 24 nans
yCols=['yFR', 'yBE']
#%% shit
yCols=['yFR', 'yBE']
NormalizerStack(SingleColsStdNormalizer(['genForecast','systemLoad']),
                MultiColStdNormalizer(yCols))
#%% shit
import pandas as pd
df = pd.DataFrame({'Col1': range(0, 11),'Col2': range(30, 41),'Col3': range(40, 51),'Col4': range(80, 91)})
normalizerStack=NormalizerStack(SingleColsStdNormalizer(['Col1','Col2']),
                MultiColStdNormalizer(['Col3','Col4']))
#%%
print('TEST:try fit df\n')
normalizerStack.fitNTransform(df)
print('TEST:try refit cols again\n')
normalizerStack.fitNTransform(df)
#%%
print('TEST:try inverse transfrom cols \n')
df['Col1']=normalizerStack.inverseTransformCol(df, 'Col1')
df['Col4']=normalizerStack.inverseTransformCol(df, 'Col4')
print('TEST:try inverse inverse transfrom cols \n')
df['Col1']=normalizerStack.inverseTransformCol(df, 'Col1')
print('TEST:try inverse inverse df with retransforming cols again\n')
normalizerStack.inverseTransform(df)
#%%
from typing import List, Union

def sum_numbers(numbers: int) -> int:
    return numbers
result = sum_numbers('1')

# Usage
# result = sum_numbers([1, 2, '3'])
print(result)  # This will print 3, which is the sum of 1 and 2 (the integer values)

#%% shit
import pandas as pd

def split_and_combine(df, pastCols, renameCol):
    # Create two DataFrames, one for 'y1' and one for 'y2'
    df_y1 = df[pastCols + ['c1', 'c2', 'c3']].copy()
    df_y2 = df[pastCols + ['c1', 'c2', 'c3']].copy()
    
    # Rename the columns to 'y' and add a 'yType' column
    df_y1.rename(columns={pastCols[0]: renameCol, pastCols[1]: 'yType'}, inplace=True)
    df_y2.rename(columns={pastCols[1]: renameCol, pastCols[0]: 'yType'}, inplace=True)
    
    # Add 'yType' values
    df_y1['yType'] = 'y1'
    df_y2['yType'] = 'y2'
    
    # Concatenate the two DataFrames
    result_df = pd.concat([df_y1, df_y2], ignore_index=True)
    
    return result_df

# Create a sample DataFrame
data = {'y1': [1, 2, 3],
        'y2': [4, 5, 6],
        'c1': ['A', 'B', 'C'],
        'c2': ['X', 'Y', 'Z'],
        'c3': ['P', 'Q', 'R']}

df = pd.DataFrame({'y1': [1, 2, 3],
        'y2': [4, 5, 6],
        'c1': ['A', 'B', 'C'],
        'c2': ['X', 'Y', 'Z'],
        'c3': ['P', 'Q', 'R']})

# Specify the columns to combine and rename
pastCols = ['y1', 'y2']
renameCol = 'y'

# Call the function
result_df = split_and_combine(df, pastCols, renameCol)

print(result_df)
#%% 
def splitToNSeries(df, pastCols, renameCol):
    processedData=pd.DataFrame({})
    otherCols= [col for col in df.columns if col not in pastCols]
    for i,pc in enumerate(pastCols):
        thisSeriesDf=df[otherCols+[pc]]
        thisSeriesDf=thisSeriesDf.rename(columns={pc:renameCol})
        thisSeriesDf[renameCol+'Type']=pc
        processedData = pd.concat([processedData,thisSeriesDf]).reset_index(drop=True)
    return processedData
#%% to added to prep code
splitDefaultCondition='__possibleStartPoint__ == 1'
def addSequentAndAntecedentIndexes(indexes, sequentsToBeAdded=0, antecedentsToBeAdded=0):
    newIndexes = set()
    
    # Add sequent indexes
    if sequentsToBeAdded>0:
        for num in indexes:
            newIndexes.update(range(num + 1, num + sequentsToBeAdded))#kkk check to be correct
    
    # Add antecedent indexes
    if antecedentsToBeAdded>0:
        for num in indexes:
            newIndexes.update(range(num - antecedentsToBeAdded, num))#kkk check to be correct
    
    newIndexes.difference_update(indexes)  # Remove existing elements from the newIndexes set
    indexes = np.concatenate((indexes, np.array(list(newIndexes))))
    indexes.sort()
    return indexes

def splitTrainValTest(df, trainRatio, valRatio, trainBackLenToAdd=0, valBackLenToAdd=None, testBackLenToAdd=None, trainForeLenToAdd=0, valForeLenToAdd=None, testForeLenToAdd=None, shuffle=True, conditions=[splitDefaultCondition]):
    trainRatio, valRatio=round(trainRatio,3), round(valRatio,3)#kkk whole code
    testRatio=round(1-trainRatio-valRatio,3)
    assert sum([trainRatio, valRatio, testRatio])==1, 'sum of train, val and test ratios must be 1'
    if trainRatio==0:
        print('no train data')
    if valRatio==0:
        print('no val data')
    if testRatio==0:
        print('no test data')
    
    if valBackLenToAdd==None:
        valBackLenToAdd = trainBackLenToAdd
    if testBackLenToAdd==None:
        testBackLenToAdd = trainBackLenToAdd

    if valForeLenToAdd==None:
        valForeLenToAdd = trainForeLenToAdd
    if testForeLenToAdd==None:
        testForeLenToAdd = trainForeLenToAdd
    
    isCondtionsApplied=False
    filteredDf = df.copy()
    doQueryNTurnIsCondtionsApplied: lambda df,con,ica:(df.query(con),True)
    for condition in conditions:
        if condition==splitDefaultCondition:
            try:
                filteredDf, isCondtionsApplied = doQueryNTurnIsCondtionsApplied(filteredDf, condition, isCondtionsApplied)
            except:
                pass
        else:
            filteredDf, isCondtionsApplied = doQueryNTurnIsCondtionsApplied(filteredDf, condition, isCondtionsApplied)
    
    indexes=np.array(df.index)
    if shuffle:
        np.random.shuffle(indexes)
    
    trainIndexes=indexes[:int(trainRatio*len(indexes))]
    valIndexes=indexes[int(trainRatio*len(indexes)):int((trainRatio+valRatio)*len(indexes))]
    testIndexes=indexes[int((trainRatio+valRatio)*len(indexes)):]
    
    trainIndexes=addSequentAndAntecedentIndexes(trainIndexes, sequentsToBeAdded=trainForeLenToAdd, antecedentsToBeAdded=trainBackLenToAdd)
    valIndexes=addSequentAndAntecedentIndexes(valIndexes, sequentsToBeAdded=valForeLenToAdd, antecedentsToBeAdded=valBackLenToAdd)
    testIndexes=addSequentAndAntecedentIndexes(testIndexes, sequentsToBeAdded=testForeLenToAdd, antecedentsToBeAdded=testBackLenToAdd)


    train=data.loc[trainIndexes]
    val=data.loc[valIndexes]
    test=data.loc[testIndexes]
    return train, val, test
df = pd.DataFrame({'y1': [1, 2, 3],
        'y2': [4, 5, 6],
        'c1': ['A', 'B', 'C'],
        'c2': ['X', 'Y', 'Z'],
        'c3': ['P', 'Q', 'R']})
splitTrainValTest(df, .6754, .143)
#%% 
    ### nhits
    trainLen=int(trainRatio * len(dfNormalized))
    trainPlusValLen=int((trainRatio+valRatio) * len(dfNormalized))
    assert trainLen>self.backcastLen,'the trainLen should be bigger than backcastLen'
    trainData = dfNormalized.loc[:trainLen]
    valData = dfNormalized.loc[trainLen-self.backcastLen:trainPlusValLen].reset_index(drop=True)
    testData = dfNormalized.loc[trainPlusValLen-self.backcastLen:].reset_index(drop=True)
    
    ### deepar
    indexes=np.array(data[data['possibleStartPoint']==1].index)#kkk this a condition for possible
    np.random.shuffle(indexes)
    data = data.drop(['possibleStartPoint'], axis=1)

    trainIndexes=indexes[:int(trainRatio*len(indexes))]
    valIndexes=indexes[int(trainRatio*len(indexes)):int((trainRatio+valRatio)*len(indexes))]
    testIndexes=indexes[int((trainRatio+valRatio)*len(indexes)):]

    #kkk this is unnecessary because by having the whole data and indexes we would create batches
    def addSequentIndexes(indexes, sequentsToBeAdded):
        newIndexes = set()
        for num in indexes:
            newIndexes.update(range(num + 1, num + sequentsToBeAdded))
        newIndexes.difference_update(indexes)  # Remove existing elements from the newIndexes set
        indexes = np.concatenate((indexes, np.array(list(newIndexes))))
        indexes.sort()
        return indexes

    trainIndexes2=addSequentIndexes(trainIndexes, backcastLen)
    valIndexes2=addSequentIndexes(valIndexes, backcastLen)
    testIndexes2=addSequentIndexes(testIndexes, backcastLen)


    train=data.loc[trainIndexes2]
    val=data.loc[valIndexes2]
    test=data.loc[testIndexes2]
#%% shit
import pandas as pd
import numpy as np
import timeit

# Create a DataFrame with 8 rows and 100,000 random values
data = {'col1': np.random.rand(100000),
        'col2': np.random.rand(100000),
        'col3': np.random.rand(100000),
        'col4': np.random.rand(100000)}

df = pd.DataFrame(data)

# Approach 1: Using iteration and .loc
def approach4():
    for i in range(len(df)):
        df.loc[i, 'col1'] = df.loc[i, 'col2'] + df.loc[i, 'col3'] + df.loc[i, 'col4']

# Approach 2: Using df.at
def approach3():
    for i in range(len(df)):
        df.at[i, 'col1'] = df.at[i, 'col2'] + df.at[i, 'col3'] + df.at[i, 'col4']

# Approach 3: Using dictionary and .values
def approach2():
    for i in range(len(df)):
        df['col1'][i] = df['col2'][i]+df['col3'][i]+df['col4'][i]
def approach1():
    dic = {col:df[col].values for col in df.columns}

    for i in range(len(df)):
        dic['col1'][i] = dic['col2'][i]+dic['col3'][i]+dic['col4'][i]

# Measure execution time for each approach
time_approach1 = timeit.timeit(approach1, number=2)
print("Execution Time (Approach 1):", time_approach1)
time_approach2 = timeit.timeit(approach2, number=2)
print("Execution Time (Approach 2):", time_approach2)
time_approach3 = timeit.timeit(approach3, number=2)
print("Execution Time (Approach 3):", time_approach3)
time_approach4 = timeit.timeit(approach3, number=2)
print("Execution Time (Approach 3):", time_approach4)
#%%
data = np.random.rand(30, 5)

# Define custom index values
custom_index = range(100, 130)

# Create the DataFrame with custom indexes
df = pd.DataFrame(data, index=custom_index)
dic = {col:df[col].values for col in df.columns}
def dfToNpDict(df):
    return {col:df[col].values for col in df.columns}

def npDictToDfForCol(df, dic, col):
    assert col in dic.keys(),f'{col} is not in dictionary cols'
    assert col in df.columns,f'{col} is not in dataframe cols'
    assert len(dic[col])==len(df[col]),f'{col} lens in dataframe and dictionary are equal'
    df[col]=dic[col]
    
def npDictToDf(df, dic):
    for col in dic.keys():
        npDictToDfForCol(df, dic, col)
zz=df.copy()
npDictToDf(df, dic)
#%% shit sucks
from abc import ABC, abstractmethod

class WorkflowPipeline(ABC):
    def __init__(self):
        self.results = {}

    def run(self, method_order):
        for method_name in method_order:
            method = getattr(self, method_name)
            if callable(method) and hasattr(method, 'workflow_step'):
                args, kwargs = method.workflow_step(self.results)
                result = method(*args, **kwargs)
                self.results[method_name] = result

    @abstractmethod
    def method1(self):
        pass

    @abstractmethod
    def method2(self):
        pass

    # Add more methods as needed

def workflow_step(*args, **kwargs):
    def decorator(func):
        func.workflow_step = lambda results: (args, kwargs)
        return func
    return decorator

class CustomWorkflow(WorkflowPipeline):
    @workflow_step(5, b=10)
    def method1(self, a, b):
        return a + b

    @workflow_step("Hello, ", name="John")
    def method2(self, greeting, name):
        return greeting + name

    @workflow_step()
    def method3(self):
        return "This method has no inputs."

if __name__ == "__main__":
    workflow = CustomWorkflow()
    method_order = ["method2", "method1", "method3"]  # Specify the desired order
    workflow.run(method_order)
    
    print("Results:")
    print(workflow.results)
#%% shit
from sklearn.preprocessing import LabelEncoder
import numpy as np
# Create a LabelEncoder and fit it to some data
encoder = LabelEncoder()
data_to_fit = ['apple', 'banana', 'cherry']
encoder.fit(data_to_fit)

# Attempt to transform data with a label not seen during fitting
data_to_transform = ['cherry', 'banana']
encoded_data = encoder.transform(data_to_transform)
encoder.inverse_transform(['cherry', 'banana'])
encoder.inverse_transform([0,1.4])
#%% shit
from sklearn.preprocessing import LabelEncoder
import numpy as np

class LblEncoder:
    #qqq what happens if in transform or inverse_transform LabelEncoder receives a data which is not in its classes?!!: it would raise error
    #kkk it cant handle None or np.nan or other common missing values
    #kkk do I need , colShape=1
    #kkk I want to skip if its already transformed or inverse transformed

    def __init__(self, name=None):
        self.name = name
        self.encoder = LabelEncoder()

    @property
    def isFitted(self):
        return hasattr(self.encoder, 'classes_') and self.encoder.classes_ is not None

    def fit(self, dataToFit, colShape=1):
        if not self.isFitted:
            # Check if there are integer labels
            if any(isinstance(label, int) for label in dataToFit):
                raise ValueError("Integer labels detected. Use makeIntLabelsString to convert them to string labels.")
            self.encoder.fit(dataToFit.values.reshape(-1, colShape))
        else:
            print(f'LabelEncoder {self.name} is already fitted')

    def transform(self, dataToFit, colShape=1):
        if not self.isFitted:
            print(f'LabelEncoder {self.name} skipping transform: is not fitted; fit it first')
            return dataToFit

        dataToFit = dataToFit.values.reshape(-1, colShape)

        # Check if the data contains non-integer values; if so, transform it
        if not np.issubdtype(dataToFit.dtype, np.integer):
            return self.encoder.transform(dataToFit)
        else:
            print(f'LabelEncoder {self.name} skipping transform: data already seems transformed.')
            return dataToFit

    def inverseTransform(self, dataToInverseTransformed, colShape=1):
        if not self.isFitted:
            print(f'LabelEncoder {self.name} is not fitted; cannot inverse transform.')
            return dataToInverseTransformed

        dataToInverseTransformed = dataToInverseTransformed.values.reshape(-1, colShape)

        # Check if the data contains non-integer values; if so, inverse transform it
        if not np.issubdtype(dataToInverseTransformed.dtype, np.integer):
            return self.encoder.inverse_transform(dataToInverseTransformed)
        else:
            print(f'LabelEncoder {self.name} skipping inverse transform: data already seems inverse transformed.')
            return dataToInverseTransformed
#%% shit
from sklearn.preprocessing import LabelEncoder

# Create a LabelEncoder instance
encoder = LabelEncoder()

# Sample data with integer labels
labels = [1, 2, 3, 1, 2]

# Fit the encoder to the integer labels
encoder.fit(labels)

# Transform the integer labels
transformed_labels = encoder.transform(labels)

# Inverse transform to get back the original labels
original_labels = encoder.inverse_transform(transformed_labels)

print("Original labels:", labels)
print("Transformed labels:", transformed_labels)
print("Inverse transformed labels:", original_labels)
#%% shit
import pandas as pd

def makeIntLabelsString(df, colName):
    # Get unique values in the specified column
    unique_values = df[colName].unique()

    # Create a mapping dictionary from integers to labels
    int_to_label_mapping = {int_val: f'{colName}{label}' for label, int_val in enumerate(unique_values)}

    # Replace integer values with labels in the specified column
    df[colName] = df[colName].map(int_to_label_mapping)

    return df

# Example usage:
df = pd.DataFrame({'A': [1, 2, 2, 3, 1, 3]})

print("Original DataFrame:")
print(df)

df = makeIntLabelsString(df, 'A')

print("\nDataFrame with String Labels:")
print(df)
#%% wrapper shit
def my_wrapper(func):
    def inner(*args, **kwargs):
        # Add your custom code here
        result = func(*args, **kwargs)
        # More custom code if needed
        return result
    return inner
def original_function(x, y):
    """This is a sample function."""
    return x + y
wrapped_function = my_wrapper(original_function)

result = wrapped_function(10, 5)
print(result)  # Output: 15

# Accessing the original function's docstring
print(wrapped_function.__doc__)  # Output: This is a sample function.
#%% 
from functools import wraps

def my_wrapper(func):
    @wraps(func)
    def inner(*args, **kwargs):
        # Add your custom code here
        result = func(*args, **kwargs)
        # More custom code if needed
        return result
    return inner
wrapped_function = my_wrapper(original_function)

print(wrapped_function.__doc__)
#%% shit combinations 1
import pandas as pd
import itertools

# Create a sample DataFrame
data = {
    'A': ['A1', 'A2', 'A3', 'A4'],
    'B': ['B1', 'B2', 'B3', 'B4']
}

df = pd.DataFrame(data)

# Get unique combinations of two columns
columns_to_combine = ['A', 'B']
combinations = set(itertools.combinations(df[columns_to_combine].values, 2))

# Convert the combinations to a list if needed
unique_combinations = list(combinations)

# Display the unique combinations
for combo in unique_combinations:
    print(combo)

#%% shit combinations 2
import pandas as pd

df = pd.DataFrame(data = {
    'A': ['A1', 'A2', 'A3', 'A4', 'A1','A3'],
    'B': ['B1', 'B2', 'B4', 'B4', 'B1','B2'],
    'C': ['C1', 'C4', 'C4', 'C4', 'C1','C2'],
    'col1': [3, 3, 0, 0, 1, 4],
    'col2': [0, 3, 0, 1, 0, 2],
    'col3': [2, 1, 0, 3, 4, 0]
})

# Get unique combinations of two columns using groupby
unique_combinations = df.groupby(['A', 'B']).size().reset_index().rename(columns={0: 'count'})

# Display the unique combinations
print(unique_combinations)
#%% shit combinations 3
import pandas as pd
class Combo:
    def __init__(self, defDict, mainGroupColNames):
        assert isinstance(defDict, dict) and all(key in defDict for key in mainGroupColNames), "defDict format is invalid."
        
        for key in defDict:
            if key not in mainGroupColNames:
                raise ValueError(f"'{key}' is not a valid column name in mainGroupColNames.")
        
        for col in mainGroupColNames:
            if col not in defDict:
                raise ValueError(f"'{col}' is missing in combo definition.")
        
        self.defDict=defDict

    def shortRepr_(self):
        return '_'.join(self.defDict.values())
    
    def __repr__(self):
        return str(self.defDict)
#%% MainGroupBaseNormalizer
class MainGroupBaseNormalizer:
    def __init__(self, df, mainGroupColNames):
        self.mainGroupColNames = mainGroupColNames
        self.uniqueCombos = self._getUniqueCombinations(df)

    def uniqueCombosShortReprs(self):
        return [combo.shortRepr_() for combo in self.uniqueCombos]

    def findMatchingShortReprCombo(self, combo):
        for uniqueCombo in self.uniqueCombos:
            if combo == uniqueCombo.shortRepr_():
                return uniqueCombo
        return None

    def uniqueCombosDictReprs(self):
        return [combo.defDict for combo in self.uniqueCombos]

    def findMatchingDictReprCombo(self, combo):
        for uniqueCombo in self.uniqueCombos:
            if combo == uniqueCombo.defDict:
                return uniqueCombo
        return None

    def comboInUniqueCombos(self, combo):
        if isinstance(combo, Combo):
            if combo in self.uniqueCombos:
                return combo
        elif isinstance(combo, str):
            if self.findMatchingShortReprCombo(combo):
                return self.findMatchingShortReprCombo(combo)
        elif isinstance(combo, dict):
            if self.findMatchingDictReprCombo(combo):
                return self.findMatchingDictReprCombo(combo)
        return

    def _getUniqueCombinations(self, df):
        columnsToGroup = df[self.mainGroupColNames]
        uniqueCombos  = df.groupby(self.mainGroupColNames).size().reset_index().rename(columns={0: 'count'})
        uniqueCombos  = uniqueCombos.rename(columns={0: 'count'})
        comboObjs = []
        for index, row in uniqueCombos.iterrows():
            comboDict = {col: row[col] for col in self.mainGroupColNames}
            combo = Combo(comboDict, self.mainGroupColNames)
            comboObjs.append(combo)
        
        return comboObjs

    def getRowsByCombination(self, df, combo):
        comboObj=self.comboInUniqueCombos(combo)
        assert comboObj, "Combo is not in uniqueCombos"
        tempDf=df[(df[self.mainGroupColNames] == comboObj.defDict).all(axis=1)]
        
        # this is to correct dtypes
        npDict=NpDict(tempDf)
        tempDf=npDict.toDf(resetDtype=True)
        return tempDf
#%%
# Create a sample DataFrame
df = pd.DataFrame(data = {
    'A': ['A1', 'A2', 'A3', 'A4', 'A1','A3'],
    'B': ['B1', 'B2', 'B4', 'B4', 'B1','B2'],
    'C': ['C1', 'C4', 'C4', 'C4', 'C2','C2'],
    'col1': [3, 3, 0, 0, 1, 4],
    'col2': [0, 3, 0, 1, 0, 2],
    'col3': [2, 1, 0, 3, 4, 0]})


# Create an instance of the MainGroupBaseNormalizer class
#test
main_group_cols = ["A", "B"]#test
main_group_cols = ["A", "B","C"]#test
finder = MainGroupBaseNormalizer(df, main_group_cols)

# Get the unique combinations
uniqueCombinations = finder.uniqueCombos
print("Unique Combinations:")
print(uniqueCombinations)

# Example usage to get rows with a specific combination
combo_to_find = {'A': 'A1', 'B': 'B1'}
combo_to_find = {'A': 'A1', 'B': 'B1','C':'C1'}
result = finder.getRowsByCombination(df, combo_to_find)
combo_to_find = 'A1_B1_C1'#test
combo_to_find = Combo({'A': 'A1', 'B': 'B1','C':'C1'}, finder.mainGroupColNames)#test
print("\nRows with Combination:", combo_to_find)
print(result)
#%% MainGroupBaseSingleColsStdNormalizer
class MainGroupBaseSingleColsStdNormalizer(MainGroupBaseNormalizer):
    def __init__(self, classType, df, mainGroupColNames, colNames:list):
        super().__init__(df, mainGroupColNames)
        self.colNames=colNames
        self.container={}
        for col in colNames:
            self.container[col]={}
            for combo in self.uniqueCombos:
                self.container[col][combo.shortRepr_()]=classType([col])

    def fitNTransform(self, df):
        for col in self.colNames:
            for combo in self.uniqueCombos:
                dfToFit=self.getRowsByCombination(df, combo)
                inds=dfToFit.index
                dfToFit=dfToFit.reset_index(drop=True)
                self.container[col][combo.shortRepr_()].fitNTransform(dfToFit)
                dfToFit.index=inds
                df.loc[inds,col]=dfToFit

    def inverseTransformCol(self, df, col):
        for combo in self.uniqueCombos:
            dfToFit=self.getRowsByCombination(df, combo)
            inds=dfToFit.index
            dfToFit=dfToFit.reset_index(drop=True)
            invRes=self.container[col][combo.shortRepr_()].inverseTransformCol(dfToFit[col], col)
            df.loc[inds,col]=invRes

    def ultimateInverseTransformCol(self, df, col):
        for combo in self.uniqueCombos:
            dfToFit=self.getRowsByCombination(df, combo)
            inds=dfToFit.index
            dfToFit=dfToFit.reset_index(drop=True)
            invRes=self.container[col][combo.shortRepr_()].ultimateInverseTransformCol(dfToFit, col)
            #kkk only difference between this and inverseTransformCol is that here we cant pass dfToFit[col]
            #...maybe we correcting this we may can write both funcs in a helper(base) func which only takes ultimateInverseTransformCol or inverseTransformCol
            df.loc[inds,col]=invRes

class MainGroupSingleColsStdNormalizer(MainGroupBaseNormalizer):
    def __init__(self, df, mainGroupColNames, colNames:list):
        super().__init__(SingleColsStdNormalizer, df, mainGroupColNames, colNames)

class MainGroupSingleColsLblEncoder(MainGroupBaseNormalizer):
    "this the lblEncoder version of MainGroupSingleColsStdNormalizer; its rarely useful, but in some case maybe used"
    def __init__(self, df, mainGroupColNames, colNames:list):
        super().__init__(SingleColsLblEncoder, df, mainGroupColNames, colNames)
#%% MainGroupSingleColsStdNormalizer
class MainGroupSingleColsStdNormalizer(MainGroupBaseNormalizer):
    def __init__(self, df, mainGroupColNames, colNames:list):
        super().__init__(df, mainGroupColNames)
        self.colNames=colNames
        self.container={}
        for col in colNames:
            self.container[col]={}
            for combo in self.uniqueCombos:
                self.container[col][combo.shortRepr_()]=SingleColsStdNormalizer([col])

    def fitNTransform(self, df):
        for col in self.colNames:
            for combo in self.uniqueCombos:
                dfToFit=self.getRowsByCombination(df, combo)
                inds=dfToFit.index
                dfToFit=dfToFit.reset_index(drop=True)
                self.container[col][combo.shortRepr_()].fitNTransform(dfToFit)
                dfToFit.index=inds
                df.loc[inds,col]=dfToFit

    def inverseTransformCol(self, df, col):
        for combo in self.uniqueCombos:
            dfToFit=self.getRowsByCombination(df, combo)
            inds=dfToFit.index
            dfToFit=dfToFit.reset_index(drop=True)
            invRes=self.container[col][combo.shortRepr_()].inverseTransformCol(dfToFit[col], col)
            df.loc[inds,col]=invRes

    def ultimateInverseTransformCol(self, df, col):
        for combo in self.uniqueCombos:
            dfToFit=self.getRowsByCombination(df, combo)
            inds=dfToFit.index
            dfToFit=dfToFit.reset_index(drop=True)
            invRes=self.container[col][combo.shortRepr_()].ultimateInverseTransformCol(dfToFit, col)
            #kkk only difference between this and inverseTransformCol is that here we cant pass dfToFit[col]
            #...maybe we correcting this we may can write both funcs in a helper(replicate) func which only takes ultimateInverseTransformCol or inverseTransformCol
            df.loc[inds,col]=invRes
#%% fitNTransform test
runcell(0, 'F:/projects/public github projects/private repos/versatileAnnModule/dataPreparation/utils/dataPrepUtils.py')
runcell('normalizers', 'F:/projects/public github projects/private repos/versatileAnnModule/dataPreparation/utils/dataPrepUtils.py')
runcell('shit combinations 3', 'F:/projects/public github projects/private repos/versatileAnnModule/dataPreparation/someCommonDatasets/epfFrBe.py')
runcell('MainGroupBaseNormalizer', 'F:/projects/public github projects/private repos/versatileAnnModule/dataPreparation/someCommonDatasets/epfFrBe.py')
runcell('MainGroupSingleColsStdNormalizer', 'F:/projects/public github projects/private repos/versatileAnnModule/dataPreparation/someCommonDatasets/epfFrBe.py')

df = pd.DataFrame(data = {
    'A': ['A1', 'A2', 'A3', 'A4', 'A1','A3'],
    'B': ['B1', 'B2', 'B4', 'B4', 'B1','B2'],
    'C': ['C1', 'C4', 'C4', 'C4', 'C1','C2'],
    'col1': [3, 3, 0, 0, 1, 4],
    'col2': [0, 3, 0, 1, 0, 2],
    'col3': [2, 1, 0, 3, 4, 0]})
mag=MainGroupSingleColsStdNormalizer(df, ['A','B'], ['col1','col2'])
mag.container
mag.fitNTransform(df)
#%%
mag.inverseTransformCol(df, 'col1')
mag.ultimateInverseTransformCol(df, 'col1')
#%% MainGroupSingleColsLblEncoder
SingleColsLblEncoder#kkk
#kkk may only pass SingleColsLblEncoder
class MainGroupSingleColsLblEncoder(MainGroupBaseNormalizer):
    "this the lblEncoder version of MainGroupSingleColsStdNormalizer; its rarely useful, but in some case maybe used"#kkk
    def __init__(self, df, mainGroupColNames, colNames:list):
        super().__init__(df, mainGroupColNames)
        self.colNames=colNames
        self.container={}
        for col in colNames:
            self.container[col]={}
            for combo in self.uniqueCombos:
                self.container[col][combo.shortRepr_()]=SingleColsLblEncoder([col])

    def fitNTransform(self, df):
        for col in self.colNames:
            for combo in self.uniqueCombos:
                dfToFit=self.getRowsByCombination(df, combo)
                inds=dfToFit.index
                dfToFit=dfToFit.reset_index(drop=True)
                self.container[col][combo.shortRepr_()].fitNTransform(dfToFit)
                dfToFit.index=inds
                df.loc[inds,col]=dfToFit#.loc[inds,col]#kkk added '.loc[inds,col]' to dfToFit

    def inverseTransformCol(self, df, col):
        for combo in self.uniqueCombos:
            dfToFit=self.getRowsByCombination(df, combo)
            inds=dfToFit.index
            dfToFit=dfToFit.reset_index(drop=True)
            invRes=self.container[col][combo.shortRepr_()].inverseTransformCol(dfToFit[col], col)
            df.loc[inds,col]=invRes

    def ultimateInverseTransformCol(self, df, col):
        for combo in self.uniqueCombos:
            dfToFit=self.getRowsByCombination(df, combo)
            inds=dfToFit.index
            dfToFit=dfToFit.reset_index(drop=True)
            invRes=self.container[col][combo.shortRepr_()].ultimateInverseTransformCol(dfToFit, col)
            df.loc[inds,col]=invRes
#%% MainGroupSingleColsLblEncoder fitNTransform test
runcell(0, 'F:/projects/public github projects/private repos/versatileAnnModule/dataPreparation/utils/dataPrepUtils.py')
runcell('general vars', 'F:/projects/public github projects/private repos/versatileAnnModule/dataPreparation/utils/dataPrepUtils.py')
runcell('utils misc', 'F:/projects/public github projects/private repos/versatileAnnModule/dataPreparation/utils/dataPrepUtils.py')
runcell('normalizers: base normalizers', 'F:/projects/public github projects/private repos/versatileAnnModule/dataPreparation/utils/dataPrepUtils.py')
runcell('normalizers: NormalizerStack', 'F:/projects/public github projects/private repos/versatileAnnModule/dataPreparation/utils/dataPrepUtils.py')
runcell('normalizers: SingleColsNormalizers', 'F:/projects/public github projects/private repos/versatileAnnModule/dataPreparation/utils/dataPrepUtils.py')
runcell('shit combinations 3', 'F:/projects/public github projects/private repos/versatileAnnModule/dataPreparation/someCommonDatasets/epfFrBe.py')
runcell('MainGroupBaseNormalizer', 'F:/projects/public github projects/private repos/versatileAnnModule/dataPreparation/someCommonDatasets/epfFrBe.py')
runcell('MainGroupSingleColsLblEncoder', 'F:/projects/public github projects/private repos/versatileAnnModule/dataPreparation/someCommonDatasets/epfFrBe.py')
df = pd.DataFrame(data = {
    'A': ['A1', 'A2', 'A3', 'A4', 'A1','A3','A2'],
    'B': ['B1', 'B2', 'B4', 'B4', 'B1','B2','B2'],
    'C': ['C1', 'C4', 'C4', 'C4', 'C1','C2','C3'],
    'col1': [3, 3, 0, 0, 1, 4, 4],
    'col2': ['a', 'v', 'a', 'o', 'o', 'v','z'],
    'col3': [2, 1, 0, 3, 4, 0,4]})
mag=MainGroupSingleColsLblEncoder(df, ['A','B'], ['col1','col2'])
mag.container
#%%
mag.fitNTransform(df)
vars(mag.container['col1']['A2_B2'])
#%% 
mag.inverseTransformCol(df, 'col1')
mag.inverseTransformCol(df, 'col2')
#%% 
mag.ultimateInverseTransformCol(df, 'col1')
mag.ultimateInverseTransformCol(df, 'col2')
#%%
df1 = pd.DataFrame(data = {
    'col1': [3, 3, 0, 0, 1, 4],
    'col2': [0, 3, 0, 1, 0, 2]})
df2 = pd.DataFrame(data = {
    'col1': [None for _ in range(6)],
    'col2': [None for _ in range(6)]})
equalDfs(df1,df2)
