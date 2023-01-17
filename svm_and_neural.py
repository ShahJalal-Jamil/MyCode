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

'''
Load  Training Data From Excel Sheet. Make two list of Comments and Class.
'''

file = "D:/Test/training_t_c.xlsx"
x1 = pd.ExcelFile(file)

# Print the sheet names
# print(x1.sheet_names)

# Load a sheet into a DataFrame by name: df1
sheet = x1.parse('Sheet1')
list_n = sheet['Comments']
list_class = sheet['Class']

'''

Pre processing the training data. So identify the hashtag, emoji's and plain comments.
list_main contains feature data and list_class_new contains class data.
'''


def isInAlphabeticalOrder(word):
    for i in range(len(word) - 1):
        if word[i] > word[i + 1]:
            return False
    return True


list_of_words = list()

import emoji as emo
import emoji_list as emoList


def char_is_emoji(character):
    return character in emo.UNICODE_EMOJI


def char_is_hashTag(character):
    return re.match(r'#', character)


def is_ascii(s):
    return all(ord(c) < 128 for c in s)


def retieveEmo(sentence):
    list_sentence = sentence.split()
    for word in list_sentence:
        retWord = split_count(word)
        list_sentence[list_sentence.index(word)] = retWord

    final_sentence = ' '.join(list_sentence)
    # print(final_sentence)
    return final_sentence


import regex


def split_count(text):
    emoji_counter = 0
    data = regex.findall(r'\X', text)
    chunks = []
    for word in data:
        if any(char in emo.UNICODE_EMOJI for char in word):
            emoji_counter += 1
            emoCode = emo.demojize(word)
            emoCode = emoCode.replace("_", "''")
            print(emoCode)
            chunks.append(str(' ') + emoCode + str(' '))
        else:
            chunks.append(word)
    # print(chunks)
    result = ''.join(chunks)
    return result


def retieveEmoFromWord(word):
    word_item = word.split()


list_main = list()
for row in list_n:
    list_main.append(row)

# print(list_main)

list_Class_new = list()
for row in list_class:
    list_Class_new.append(row)

# print(list_Class_new)


# Remove the sentence ending statement for better training
list_final = list()
for row in list_main:
    # print(row)
    if ("।" in row):
        row = row.replace("।", " ")
        row = retieveEmo(row)
        list_final.append(row)
    else:
        list_final.append(row)
print(list_final)

'''
Divide feature data into two parts. So 80% data will be used as train data and
other 20% as close test data.
'''

train_size = int(len(list_final) * .8)

x_Train = list_final[0:train_size]
x_Test = list_final[train_size:len(list_final)]

print(len(x_Train))
print(len(x_Test))

y_Train = list_Class_new[0:train_size]
y_Test = list_Class_new[train_size:len(list_Class_new)]

print(len(y_Train))
print(len(y_Test))

"""
Tokenize the native bangla sentence preparing  for  neural network.
"""

max_words = 3000
# print(docs)
# create the tokenizer


t = Tokenizer()
# fit the tokenizer on the documents
t.fit_on_texts(list_final)
dictionary = t.word_index
print(t.word_index)

X = t.texts_to_matrix(x_Train, mode='count')

# Data for close test
XTest = t.texts_to_matrix(x_Test, mode='count')
print(len(X), len(XTest))

# Data for open test
text = "এইটা  জোস জিনিশ "
text = retieveEmo(text)
list_text = list()
list_text.append(text)
data_text = t.texts_to_matrix(list_text, mode='count')
print(data_text)

"""
SVM Configure
"""

# X = pd.DataFrame(X)
# XTest = pd.DataFrame(XTest)
#
encoder = LabelBinarizer()
encoder.fit(list_Class_new)
Y = encoder.transform(y_Train)
YTest = encoder.transform(y_Test)

# Y = pd.DataFrame(Y)
# YTest = pd.DataFrame(YTest)

# Import Library
from sklearn import svm

# Assumed you have, X (predictor) and Y (target) for training data set and x_test(predictor) of test_dataset
# Create SVM classification object
model = svm.SVC(kernel='linear', C=1.0, gamma=1)
# there is various option associated with it, like changing kernel, gamma and C value. Will discuss more # about it in next section.Train the model using the training sets and check score
model.fit(X, Y[:, 0])
model.score(X, Y[:, 0])
# Predict Output
predicted = model.predict(XTest)

from sklearn.metrics import accuracy_score

print(accuracy_score(YTest, predicted) * 100, "%")

list_svm_output = list(predicted)
list_svm_final = list()
for svmOutput in list_svm_output:
    arr = list()
    arr.append(svmOutput)
    list_svm_final.append(arr)
list_svm_final = numpy.asarray(list_svm_final)

"""
Neural network configure.
"""
train_size = int(len(XTest[:, :]) * .8)
print(train_size)

XNewTrain = XTest[0:train_size - 1, :]
XNewTest = XTest[train_size - 1:len(XTest), :]

print(len(XNewTrain), len(XNewTest))
print(XNewTrain)

YNewTrain = list_svm_final[0:train_size - 1, :]
YNewTest = list_svm_final[train_size - 1:len(XTest), :]

print(len(YNewTrain), len(YNewTest))


# encoder = LabelBinarizer()
# encoder.fit(list_Class_new)
# Y = encoder.transform(y_Train)
# YTest = list_svm_final
# print(len(Y))

num_labels = len(encoder.classes_)
vocab_size = len(t.word_index) + 1
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

print('Test accuracy:', score[1])





# text_labels = encoder.classes_
# print(text_labels)
# # calculate predictions
# predictions = model.predict(data_text)
# # round predictions
# print(predictions)
# rounded = [round(x[0]) for x in predictions]
# print(rounded)
