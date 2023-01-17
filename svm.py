from keras.losses import mean_squared_error
from keras.utils import to_categorical
from pip._internal.utils import logging
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelBinarizer
import pandas as pd
import re
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
import numpy
import keras
# fix random seed for reproducibility
from keras_preprocessing.text import Tokenizer

numpy.random.seed(7)

import get_data_from_disk as preProcess

X, Y, XTest, YTest, data, t = preProcess.preProcessDataUnsupervised(True)
from sklearn import svm, metrics

X = to_categorical(X)
Y = to_categorical(Y)
# Assumed you have, X (predictor) and Y (target) for training data set and x_test(predictor) of test_dataset
# Create SVM classification object
model = svm.SVC(kernel='linear', C=1.0, gamma=1)
# there is various option associated with it, like changing kernel, gamma and C value. Will discuss more # about it in next section.Train the model using the training sets and check score
model.fit(X, Y)
model.score(X, Y)
# Predict Output
predicted = model.predict(XTest)

print(predicted)
from sklearn.metrics import accuracy_score

print(accuracy_score(YTest, predicted) * 100, "%")
# print(model.predict(data_text))

print(metrics.classification_report(YTest, predicted))
print(metrics.confusion_matrix(YTest, predicted))
