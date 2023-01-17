from keras.losses import mean_squared_error
from sklearn.tree import DecisionTreeClassifier
from pip._internal.utils import logging
from sklearn import model_selection
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

X = pd.DataFrame(X)
XTest = pd.DataFrame(XTest)
Y = pd.DataFrame(Y)
YTest = pd.DataFrame(YTest)

seed = 7
kfold = model_selection.KFold(n_splits=10, random_state=seed)
# create the sub models
estimators = []

model1 = LogisticRegression()
estimators.append(('logistic', model1))

model2 = LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)
estimators.append(('cart', model2))

# create the ensemble model
ensemble = VotingClassifier(estimators)
results = model_selection.cross_val_score(ensemble, X, Y, cv=kfold)
print(results.mean())



