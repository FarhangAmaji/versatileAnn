# ---- LStl3
import pandas as pd

# Sample DataFrames with provided indices
df1 = pd.DataFrame({'value': [True, False, True, True, False, False, True]}, index=[1, 2, 3, 4, 5, 6, 7])
df2 = pd.DataFrame({'value': [False, True, False, True, False]}, index=[3, 5, 6, 7, 8])

# Concatenate the DataFrames
concatDf = pd.concat([df1, df2])

# Print the original concatenated DataFrame
print("Original Concatenated DataFrame:")
print(concatDf)

# Apply the given lines of code
tsStartPointColName = 'value'
concatDf[tsStartPointColName] = concatDf.groupby(concatDf.index)[tsStartPointColName].transform('any')
concatDf = concatDf[~concatDf.index.duplicated(keep='first')]

# Print the resulting DataFrame after applying the lines of code
print("\nResulting DataFrame:")
print(concatDf)
# ----