from keras.losses import mean_squared_error
from sklearn.tree import DecisionTreeClassifier
from pip._internal.utils import logging
from sklearn import model_selection, metrics
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import LabelBinarizer
import pandas as pd
import re
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
import numpy

# fix random seed for reproducibility
from keras_preprocessing.text import Tokenizer
from sklearn.svm import SVC

import pre_processing_data as preProcess

X, Y, XTest, YTest,dataText, t = preProcess.preProcessData(True)

seed = 7
kfold = model_selection.KFold(n_splits=10, random_state=seed)
# create the sub models
estimators = []

model1 = LogisticRegression()
estimators.append(('logistic', model1))

model3 = SVC()
estimators.append(('svm', model3))





# create the ensemble model
ensemble = VotingClassifier(estimators)
results = model_selection.cross_val_score(ensemble, X, Y.values.ravel(), cv=kfold)
print(results.mean())


ensemble.fit(X,Y.values.ravel())
predicted = ensemble.predict(XTest)

print(metrics.classification_report(YTest, predicted))
print(metrics.confusion_matrix(YTest, predicted))