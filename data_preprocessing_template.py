# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 09:53:19 2019

@author: abrar
"""
#importing libs
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd

#importing the ataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 3].values 

#replacing empty values
from sklearn.preprocessing import Imputer 
imputer = Imputer(missing_values = 'NaN', strategy = 'mean' , axis=0)
imputer.fit (X[:,1:3])
X[:,1:3] = imputer.transform(X[:,1:3])

#encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:,0]=labelencoder_X.fit_transform (X[:,0])
onehotencoder = OneHotEncoder(categorical_features = [0] )
X = onehotencoder.fit_transform(X).toarray()
labelencoder_Y = LabelEncoder()
Y=labelencoder_Y.fit_transform (Y)

#spliting into test and train
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train,Y_test = train_test_split(X,Y, test_size =0.2, random_state = 0)
