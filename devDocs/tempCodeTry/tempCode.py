import pandas as pd
import numpy as np
np.unique([4,6,1,5,7,'a',1.4])
from sklearn.preprocessing import StandardScaler, LabelEncoder
zs=LabelEncoder()
zs.fit(['d'])
zs.fit([4,6,1,5,7,'a',1.4])
zs.transform(['a'])
x=pd.Series([5,6])
zs.classes_
x.values
