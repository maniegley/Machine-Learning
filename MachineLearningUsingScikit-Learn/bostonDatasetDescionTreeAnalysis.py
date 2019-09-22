#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 11:38:00 2019

@author: vader
"""

import sklearn.datasets as datasets
import sklearn.model_selection
import numpy as np

np.random.seed(100)

boston = datasets.load_boston()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(boston.data, boston.target,  random_state = 30)
print(X_train.shape)
print(X_test.shape)


from sklearn.tree import DecisionTreeRegressor

dt_reg = DecisionTreeRegressor()   

dt_reg = dt_reg.fit(X_train, y_train) 
print(dt_reg.score(X_train, y_train))
print(dt_reg.score(X_test, y_test))
print(dt_reg.predict(X_test[0:2, :], check_input=True))


for i in range(2,6):
    dt_reg = DecisionTreeRegressor(max_depth=i)   
    dt_reg = dt_reg.fit(X_train, y_train) 
    score = dt_reg.score(X_test, y_test)
    if(dt_reg.score(X_test, y_test) > 0.69):
        print(i)

