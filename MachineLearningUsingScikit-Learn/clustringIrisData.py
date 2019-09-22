#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 12:41:24 2019

@author: vader
"""
import matplotlib.pyplot as plt
import sklearn.datasets as datasets
import sklearn.model_selection
import sklearn.metrics as metrics
ax = plt.axes()

iris = datasets.load_iris()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state = 30)
# ------------K means Clustering
from sklearn.cluster import KMeans

km_cls = KMeans(n_clusters=2)

km_cls = km_cls.fit(X_train) 

#km_cls.predict(X_test)

print(metrics.homogeneity_score(km_cls.predict(X_test), y_test))


# -----------------Agglomerative Clustering -------

from sklearn.cluster import AgglomerativeClustering
agg_cls = AgglomerativeClustering(n_clusters=3)
agg_cls = agg_cls.fit(X_train)
print(metrics.homogeneity_score(agg_cls.fit_predict(X_test), y_test))
plt.subplot(2,1,1)
ax.plot(X_train, y_train, 'ro')

plt.subplot(2,1,2)
ax.plot(agg_cls.fit_predict(X_train), y_train)
plt.show()