import pandas as pd
import numpy as np
df = pd.read_csv("train.csv")

from sklearn.impute import SimpleImputer
missingvalues = SimpleImputer(missing_values =np.nan, strategy="constant",fill_value='0', verbose = 0)
missingvalues = missingvalues.fit([df['Dependents']])
[df['Dependents']]=missingvalues.transform([df['Dependents']])

missingvalues = SimpleImputer(missing_values ='3+', strategy="constant",fill_value='4', verbose = 0)
missingvalues = missingvalues.fit([df['Dependents']])
[df['Dependents']]=missingvalues.transform([df['Dependents']])

df['Dependents'] = df['Dependents'].astype(int)