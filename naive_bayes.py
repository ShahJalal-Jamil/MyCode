from sklearn import metrics
from keras.losses import mean_squared_error
from pip._internal.utils import logging
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelBinarizer
import pandas as pd
import re
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
import numpy

# fix random seed for reproducibility
from keras_preprocessing.text import Tokenizer

numpy.random.seed(7)

import pre_processing_data as preProcess

X, Y, XTest, YTest, data, t = preProcess.preProcessData(True)
"""
Decision Tree Configure
"""

from sklearn.naive_bayes import GaussianNB

model = GaussianNB()

model.fit(X, Y.values.ravel())
predicted = model.predict(data)
print(predicted)
#
# print(metrics.classification_report(YTest, predicted))
#
# print(metrics.confusion_matrix(YTest, predicted))
#
# from sklearn.metrics import accuracy_score
#
# print(accuracy_score(YTest, predicted) * 100, "%")


# import hierarchical clustering libraries
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering

# create dendrogram
dendrogram = sch.dendrogram(sch.linkage(X, method='ward'))  # create clusters
hc = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='ward')  # save clusters for chart
y_hc = hc.fit_predict(XTest)
print(y_hc)


print(metrics.classification_report(YTest, y_hc))

print(metrics.confusion_matrix(YTest, y_hc))

from sklearn.metrics import accuracy_score

print(accuracy_score(YTest, y_hc) * 100, "%")
