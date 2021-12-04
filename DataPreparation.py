# -*- coding: utf-8 -*-
"""
Created on Sat Sep 11 08:49:00 2021

@author: User
"""

import math
import matplotlib.pyplot as plt
import keras
import pandas as pd
import numpy as np

df=pd.read_csv('checknull.csv')

df.isnull().values.any()
df.info()
df = df.dropna()

#df.hist()

from sklearn.impute import KNNImputer
X = df[['Date',"Price"]].copy() 

missing = df[df["Price"].isna()].index

knn_imputer = KNNImputer(n_neighbors=3)
filled_df = pd.DataFrame(data=knn_imputer.fit_transform(X),
                         columns=['Date',"Price"])
filled_df.loc[missing]