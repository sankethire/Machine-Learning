
#Data Preprocessing

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing dataset
dataset = pd.read_csv('Data.csv')
#print(dataset)

X = dataset.iloc[:,:-1].values
#print(x)

y = dataset.iloc[:,3].values
#print(y)

# Taking care of missing data
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean', verbose=0)
imputer = imputer.fit(X[:,1:3])
X[:,1:3] = imputer.transform(X[:,1:3])
# print(X)
# print("-----------------------------------")
# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
# print(X)
# print("-----------------------------------")
ct = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(), [0])],    # The column numbers to be transformed (here is [0] but can be [0, 1, 3])
    remainder='passthrough'                         # Leave the rest of the columns untouched
)
X_float = np.array(ct.fit_transform(X), dtype=np.float)
# print(X_float)
# print("----------------------------------")
X = np.array(ct.fit_transform(X), dtype=np.int)
X = X.astype(type('int', (int,), {}))
# print(X)

# print("----------------------------------")
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)
# print(y)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# print("----------------------------------")
# print("----------------------------------")
# print(X_train)
# print("----------------------------------")
# print(X_test)
# print("----------------------------------")
# print(y_train)
# print("----------------------------------")
# print(y_test)
# print("----------------------------------")
# print("----------------------------------")

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train.reshape(-1,1))


print(X_train)
print("----------------------------------")
print(X_test)
print("----------------------------------")
print(y_train)
print("----------------------------------")
print(y_test)
print("----------------------------------")
