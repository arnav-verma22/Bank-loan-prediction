import pandas as pd
import numpy as np


df = pd.read_csv("train.csv")

from sklearn.impute import SimpleImputer

def replace_missing(data, *args):
    if len(args) == 3:
        missingvalues = SimpleImputer(missing_values=args[0], strategy=args[1], fill_value=args[2], verbose=0)
        missingvalues = missingvalues.fit([data])
        [data] = missingvalues.transform([data])

    if len(args) == 2:
        missingvalues = SimpleImputer(missing_values=args[0], strategy=args[1], verbose=0)
        missingvalues = missingvalues.fit([data])
        [data] = missingvalues.transform([data])

    return data

df['Dependents'] = replace_missing(df['Dependents'], np.nan,'constant', '0')

df['Dependents']=replace_missing(df['Dependents'], '3+', 'constant', '4')

df['Dependents'] = df['Dependents'].astype(int)

df.reset_index()

'''df['Gender']=replace_missing(df['Gender'], np.nan, 'most_frequent')

missingvalues = SimpleImputer(missing_values=np.nan, strategy='median')
missingvalues = missingvalues.fit([df['Loan_Amount_Term']])
[df['Loan_Amount_Term']] = missingvalues.transform([df['Loan_Amount_Term']])


frequent = ['Gender', 'Married', 'Loan_Amount_Term', 'Self_Employed', 'Credit_History', 'Loan_Amount']'''

for i in df.columns:
    if df[i].isnull().values.any() == True:
        df[i] = replace_missing(df[i], np.nan, 'constant',df[i].mode()[0])

