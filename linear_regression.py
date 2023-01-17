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
text = "খুবই ভাল৷"
text = retieveEmo(text)
list_text = list()
list_text.append(text)
data_text = t.texts_to_matrix(list_text, mode='count')
print(data_text)

"""
Linear Regression Configure
"""

# t = Tokenizer()
#
# t.fit_on_texts(list_Class_new)
# dictionary = t.word_index
# print(t.word_index)
#
#
# Y = t.texts_to_matrix(y_Train, mode='count')
# YTest = t.texts_to_matrix(y_Test, mode='count')
#
# print("Len")
# print(Y)

# dictionary_values = list(dictionary.values())
# newList = list()
# newList.append(0)
# newList = newList.append(sorted(dictionary_values))
# print(newList)

Stock_Market = {
    '0': [2017, 2017, 2017, 2017, 2017, 2017, 2017, 2017, 2017, 2017, 2017, 2017, 2016, 2016, 2016, 2016, 2016, 2016,
          2016, 2016, 2016, 2016, 2016, 2016],
    '1': [12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
    '2': [2.75, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.25, 2.25, 2.25, 2, 2, 2, 1.75, 1.75, 1.75, 1.75, 1.75, 1.75,
          1.75, 1.75, 1.75, 1.75, 1.75],
    '3': [5.3, 5.3, 5.3, 5.3, 5.4, 5.6, 5.5, 5.5, 5.5, 5.6, 5.7, 5.9, 6, 5.9, 5.8, 6.1, 6.2, 6.1, 6.1,
          6.1, 5.9, 6.2, 6.2, 6.1],
    '4': [1464, 1394, 1357, 1293, 1256, 1254, 1234, 1195, 1159, 1167, 1130, 1075, 1047, 965, 943, 958,
          971, 949, 884, 866, 876, 822, 704, 719]
}

# df = pd.DataFrame(Stock_Market, columns=['0', '1', '2', '3', '4'])
#
# print(df)
#
# # df = pd.DataFrame(X, columns=newList)
# print(df)
X = pd.DataFrame(X)
XTest = pd.DataFrame(XTest)

encoder = LabelBinarizer()
encoder.fit(list_Class_new)
Y = encoder.transform(y_Train)
YTest = encoder.transform(y_Test)
# print(Y)

Y = pd.DataFrame(Y)
YTest = pd.DataFrame(YTest)

print(X)
# print(XTest.shape)
# print(Y.shape)
# print(YTest.shape)

# for x in X:
#     print(x)
#     for xx in x:
#         if (xx > 0.0):
#             print(xx)


# df = pd.DataFrame(Stock_Market, columns=['0', '1', '2', '3', '4'])
#
# X = df[['2',
#         '3']]  # here we have 2 variables for multiple regression. If you just want to use one variable for simple linear regression, then use X = df['Interest_Rate'] for example.Alternatively, you may add additional variables within the brackets
# Y = df['4']
#
# print(X,Y)


print(XTest)
model = LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)
model.fit(X, Y)
# Make predictions using the testing set
diabetes_y_pred = model.predict(XTest)
print(diabetes_y_pred)

list_predict = list()
for row in diabetes_y_pred:
    for item in row:
        if (item > 0.5):
            list_predict.append(1)
        else:
            list_predict.append(0)

list_predict = numpy.asarray(list_predict)
YPredict = pd.DataFrame(list_predict)

from sklearn import metrics

accuracy = metrics.accuracy_score(YTest, YPredict)
print(accuracy * 100, '%')

# The coefficients
print('Coefficients: \n', model.coef_)
# prediction with sklearn
# New_Interest_Rate = 2.75
# New_Unemployment_Rate = 5.3
# # print('Predicted Stock Index Price: \n', model.predict([[New_Interest_Rate, New_Unemployment_Rate]]))

text_labels = encoder.classes_
print(text_labels)
ynew = model.predict(data_text)
print(ynew)

#
# df = pd.DataFrame(Stock_Market, columns=['0', '1', '2', '3', '4'])
#
# X = df[['2',
#         '3']]  # here we have 2 variables for multiple regression. If you just want to use one variable for simple linear regression, then use X = df['Interest_Rate'] for example.Alternatively, you may add additional variables within the brackets
# Y = df['4']
#
# print(X,Y)

# # with sklearn
# regr = LinearRegression()
# regr.fit(X, Y)
#
# print('Intercept: \n', regr.intercept_)
# print('Coefficients: \n', regr.coef_)
#
# # prediction with sklearn
# New_Interest_Rate = 2.75
# New_Unemployment_Rate = 5.3
# print('Predicted Stock Index Price: \n', regr.predict([[New_Interest_Rate, New_Unemployment_Rate]]))


# # Explained variance score: 1 is perfect prediction
# print('Variance score: %.2f' % r2_score(YTest, diabetes_y_pred))
# print('Intercept:', model.intercept_)

# # sum of square of residuals
# ssr = numpy.sum((diabetes_y_pred - YTest) ** 2)
#
# #  total sum of squares
# sst = numpy.sum((diabetes_y_pred - numpy.mean(YTest))**2)
#
# # R2 score
# r2_score = 1 - (ssr/sst)
#
# print(r2_score)
# import matplotlib.pyplot as plt
# # Plot outputs
# plt.scatter(XTest, YTest,  color='black')
# plt.plot(XTest, diabetes_y_pred, color='blue', linewidth=3)
#
# plt.xticks(())
# plt.yticks(())
#
# plt.show()


# import matplotlib.pyplot as plt
#
# plt.scatter(YTest,diabetes_y_pred)
# plt.show()
