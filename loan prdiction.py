import pandas as pd
import numpy as np
from multipledispatch import dispatch
df = pd.read_csv("train.csv")

from sklearn.impute import SimpleImputer

def replace_missing(data, *args):
    if len(args) == 3:
        missingvalues = SimpleImputer(missing_values=args[0], strategy=args[1], fill_value=args[2], verbose=0)
        missingvalues = missingvalues.fit([data])
        [data] = missingvalues.transform([data])

    if len(args) == 2:
        missingvalues = SimpleImputer(missing_values=args[0], strategy=args[3], verbose=0)
        missingvalues = missingvalues.fit([data])
        [data] = missingvalues.transform([data])

    return data

df['Dependents'] = replace_missing(df['Dependents'], np.nan,'constant', '0')

df['Dependents']=replace_missing(df['Dependents'], '3+', 'constant', '4')

df['Dependents'] = df['Dependents'].astype(int)