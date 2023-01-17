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

# fix random seed for reproducibility
from keras_preprocessing.text import Tokenizer

numpy.random.seed(7)

import get_data_from_disk as preProcess

X, Y, XTest, YTest, dataText, t = preProcess.preProcessData(True)
word_index = t.word_index
"""
Logistic Regression Configure
"""


def multiply(numbers):
    total = 1
    for x in numbers:
        total *= x
    return total


def addAll(final, numbers):
    total = final
    for x in numbers:
        total += x
    return total


def getPredictValue(text):
    text = preProcess.retieveEmo(text)
    list_text = list()
    list_text.append(text)
    data_text = t.texts_to_matrix(list_text, mode='count')
    return data_text

X = to_categorical(X)
Y = to_categorical(Y)

print(X, Y)
model = LogisticRegression(multi_class="ovr")
model.fit(X, Y)
y_pred = model.predict(XTest)
# print(y_pred)
# print(YTest)

from sklearn import metrics

accuracy = metrics.accuracy_score(YTest, y_pred)
print(accuracy * 100, '%')

print(metrics.classification_report(YTest, y_pred))
print(metrics.confusion_matrix(YTest, y_pred))

print(XTest)

list_predict = list()
for row in y_pred:
    list_predict.append(row)

list_predict = numpy.asarray(list_predict)
YPredict = pd.DataFrame(list_predict)
# print(YPredict)

ne = (YPredict != YTest).any(1)

int
track = 0
for nRow in ne:
    if nRow == True or False:
        df = XTest.iloc[[track]]
        for row in df.iterrows():
            trackRowId = 0
            listPr = list()
            listPrEmo = list()
            for rowNew in row[1]:
                if rowNew >= 1:
                    value = list(word_index.keys())[list(word_index.values()).index(trackRowId)]
                    # print(value)
                    if (preProcess.is_ascii(value)):
                        print(value)
                        valueNew = value.replace("'", "_")
                        valueNew = ''.join((':', valueNew, ':'))
                        # print(valueNew)
                        if (preProcess.emo_is_character(valueNew)):
                            print(value)
                            predict = model.predict(getPredictValue(value))
                            print(predict[0])
                            if (predict[0] == 1):
                                listPrEmo.append(2)
                            else:
                                listPrEmo.append(-2)
                    else:
                        predict = model.predict(
                            getPredictValue(list(word_index.keys())[list(word_index.values()).index(trackRowId)]))
                        # print(predict[0])
                        if (predict[0] == 1):
                            listPr.append(1)
                        else:
                            listPr.append(-1)

                trackRowId = trackRowId + 1
            # print(track, multiply(listPr))
            final_result = multiply(listPr)
            final_result = addAll(final_result, listPrEmo)
            if final_result < 0:
                final_result = 0
            else:
                final_result = 1
            y_pred[track] = final_result

    track = track + 1

print(dataText)
df = pd.DataFrame(dataText)
for row in df.iterrows():
    trackRowId = 0
    listPr = list()
    listPrEmo = list()
    for rowNew in row[1]:
        if rowNew >= 1:
            value = list(word_index.keys())[list(word_index.values()).index(trackRowId)]
            print(value)
            print(preProcess.is_ascii(value))
            if (preProcess.is_ascii(value)):
                print(value)
                valueNew = value.replace("'", "_")
                valueNew = ''.join((':', valueNew, ':'))
                print(valueNew)
                print(preProcess.emo_is_character(valueNew))
                if (preProcess.emo_is_character(valueNew)):
                    # print(value)
                    predict = model.predict(getPredictValue(value))
                    print(predict[0])
                    if (predict[0] == 1):
                        listPrEmo.append(2)
                    else:
                        listPrEmo.append(-2)
            else:
                predict = model.predict(
                    getPredictValue(list(word_index.keys())[list(word_index.values()).index(trackRowId)]))
                print(predict[0])
                if (predict[0] == 1):
                    listPr.append(1)
                else:
                    listPr.append(-1)

        trackRowId = trackRowId + 1
    print(multiply(listPr))
    print(listPrEmo)
    final_result = multiply(listPr)
    final_result = addAll(final_result, listPrEmo)
    if final_result < 0:
        final_result = 0
    else:
        final_result = 1
    print(final_result)

from sklearn import metrics

accuracy = metrics.accuracy_score(YTest, y_pred)
print(accuracy * 100, '%')

print(metrics.classification_report(YTest, y_pred))
print(metrics.confusion_matrix(YTest, y_pred))

print(model.predict(dataText))
