from keras.losses import mean_squared_error
from pip._internal.utils import logging
from sklearn import metrics
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

numpy.random.seed(7)

import pre_processing_data as preProcess

X, Y, XTest, YTest, word_index, encoder = preProcess.preProcessDataNN()

"""
Neural network configure.
"""

num_labels = len(encoder.classes_)
vocab_size = len(word_index) + 1
batch_size = 30

model = Sequential()
model.add(Dense(512, kernel_initializer='normal', input_shape=(vocab_size,)))
model.add(Activation('relu'))
model.add(Dropout(0.3))
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.3))
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.3))
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.3))
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.3))
model.add(Dense(1))
model.add(Activation('sigmoid'))
model.summary()

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

history = model.fit(X, Y,
                    batch_size=batch_size,
                    epochs=30,
                    verbose=1,
                    validation_split=0.1)

"""
Evaluate neural network model
"""

score = model.evaluate(XTest, YTest,
                       batch_size=batch_size, verbose=1)

predicted = model.predict(XTest)

print(predicted)

list_predict = list()
for row in predicted:
    for item in row:
        if (item > 0.5):
            list_predict.append(1)
        else:
            list_predict.append(0)

list_predict = numpy.asarray(list_predict)
YPredict = pd.DataFrame(list_predict)

print('Test accuracy:', score[1])

print(metrics.classification_report(YTest, YPredict))
print(metrics.confusion_matrix(YTest, YPredict))

# print(accuracy_score(YTest, score) * 100, "%")

# text_labels = encoder.classes_
# print(text_labels)
# # calculate predictions
# predictions = model.predict(data_text)
# # round predictions
# print(predictions)
# rounded = [round(x[0]) for x in predictions]
# print(rounded)
