from keras.losses import mean_squared_error
from pip._internal.utils import logging
from sklearn import metrics
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

import get_data_from_disk as preProcess

X, Y, XTest, YTest, dataset, t = preProcess.preProcessData(True)
"""
Decision Tree Configure
"""

# Run this program on your local python
# interpreter, provided you have installed
# the required libraries.

# Importing the required packages
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report


# Function to perform training with giniIndex.
def train_using_gini(X_train, X_test, y_train):
    # Creating the classifier object
    clf_gini = DecisionTreeClassifier(criterion="gini",
                                      random_state=100, max_depth=3, min_samples_leaf=5)

    # Performing training
    clf_gini.fit(X_train, y_train)
    return clf_gini


# Function to perform training with entropy.
def tarin_using_entropy(X_train, X_test, y_train):
    # Decision tree with entropy
    clf_entropy = DecisionTreeClassifier(
        criterion="entropy", random_state=100,
        max_depth=3, min_samples_leaf=5)

    # Performing training
    clf_entropy.fit(X_train, y_train)
    return clf_entropy


# Function to make predictions
def prediction(X_test, clf_object):
    # Predicton on test with giniIndex
    y_pred = clf_object.predict(X_test)
    print("Predicted values:")
    print(y_pred)

    # y_predN = clf_object.predict(data_text)
    # print("Predicted values:")
    # print(y_predN)

    return y_pred


# Function to calculate accuracy
def cal_accuracy(y_test, y_pred):
    # print("Confusion Matrix: ",
    #       confusion_matrix(y_test, y_pred))

    print("Accuracy : ",
          accuracy_score(y_test.values.argmax(axis=1), y_pred.argmax(axis=1)) * 100)

    # print("Report : ",
    #       classification_report(y_test, y_pred))


clf_gini = train_using_gini(X, XTest, Y)
clf_entropy = tarin_using_entropy(X, XTest, Y)

# Operational Phase
print("Results Using Gini Index:")

# Prediction using gini
y_pred_gini = prediction(XTest, clf_gini)
cal_accuracy(YTest, y_pred_gini)

print(y_pred_gini)

print("Results Using Entropy:")
# Prediction using entropy
y_pred_entropy = prediction(XTest, clf_entropy)
cal_accuracy(YTest, y_pred_entropy)

target_names = ['0', '1']
print(metrics.classification_report(YTest, y_pred_entropy))
print(metrics.confusion_matrix(YTest, y_pred_entropy))
