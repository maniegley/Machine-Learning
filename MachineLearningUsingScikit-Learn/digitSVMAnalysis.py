#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 12:07:50 2019

@author: vader
"""

import sklearn.datasets as datasets
import sklearn.model_selection

digits = datasets.load_digits()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target,  random_state = 30)
print(X_train.shape)
print(X_test.shape)

from sklearn.svm import SVC
svm_clf = SVC()
svm_clf = svm_clf.fit(X_train, y_train)
print(svm_clf.score(X_test, y_test))



import sklearn.preprocessing as preprocessing

standardizer = preprocessing.StandardScaler()
standardizer = standardizer.fit(digits.data)
digits_standardized = standardizer.transform(digits.data)



from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(digits_standardized.data, digits.target,  random_state = 30)
svm_clf2 = SVC()

svm_clf2 = svm_clf2.fit(X_train, y_train)
print(svm_clf2.score(X_test, y_test))