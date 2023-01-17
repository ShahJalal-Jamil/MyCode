import nltk
from keras_preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelBinarizer

text = " আমি তোমাকে ভালবাসি. তুমি কি আমাকে ভালবাস"
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize

# list_naim = list()
# list_naim = sent_tokenize(text)
# print(list_naim)
# list_word = word_tokenize(text)
# print(list_word)

'''
Load  Training Data From Excel Sheet. Make two list of Comments and Class.
'''

import pandas as pd
file = "D:/Test/training.xlsx"
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


import re


# for row in list_n:
#     for i in re.split(r'(\s|\।|\,)', row):
#         text = i[0:len(i) - 1]
#         if re.match('^[a-zA-Z0-9]+$', text):
#             i = i.replace(".", "")
#         print(i)


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

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
import numpy

# fix random seed for reproducibility
from keras_preprocessing.text import Tokenizer

numpy.random.seed(7)
#
# dataset = numpy.loadtxt("data.csv", delimiter=",")
# # split into input (X) and output (Y) variables
# X = dataset[:, 0:8]
# Y = dataset[:, 8]
#
# print(X,Y)
#
# # create model
# model = Sequential()
# model.add(Dense(12, input_dim=8, activation='relu'))
# model.add(Dense(8, activation='relu'))
# model.add(Dense(1, activation='sigmoid'))
#
# # Compile model
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#
# # Fit the model
# model.fit(X, Y, epochs=150, batch_size=10)
#
# # evaluate the model
# scores = model.evaluate(X, Y)
# print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
#
# test = numpy.array([[45,117,92,0,0,34.1,0.337,38]])
#
# # calculate predictions
# predictions = model.predict(test)
# # round predictions
# rounded = [round(x[0]) for x in predictions]
# print(rounded)
#

docs = ['Hello ary ak',
        'Hi how are you man hello',
        'Hi am']

print(list_main)
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

train_size = int(len(list_final) * .8)

x_Train = list_final[0:train_size]
x_Test = list_final[train_size:len(list_final)]

print(len(x_Train))
print(len(x_Test))

y_Train = list_Class_new[0:train_size]
y_Test = list_Class_new[train_size:len(list_Class_new)]

print(len(x_Test))
print(len(y_Test))

# from sklearn.feature_extraction.text import CountVectorizer
# vectorizer = CountVectorizer(min_df=0, lowercase=False)
# vectorizer.fit(list_final)
# print(len(vectorizer.transform(list_final).toarray()))
# # list = list()
# # for word in docs:
# #     data = word.encode("utf-8")
# #     list.append(data)
max_words = 3000
# print(docs)
# create the tokenizer
t = Tokenizer()
# fit the tokenizer on the documents
t.fit_on_texts(x_Train)
dictionary = t.word_index
print(t.word_index)
X = t.texts_to_matrix(x_Train, mode='count')
text = "একটা ভুয়া জিনিশ শুধু ধোকাবাজি"
text = retieveEmo(text)
list_text = list()
list_text.append(text)
data_text = t.texts_to_matrix(list_text, mode='count')
print(data_text)

XTest = t.texts_to_matrix(x_Test, mode='count')
print("Hello")
print(len(X), len(XTest))
# print(len(list_final))
# print("Len"+str(len(list_n)))
# import json
# import keras
# import keras.preprocessing.text as kpt
# from keras.preprocessing.text import Tokenizer
# import numpy as np
#
# def convert_text_to_index_array(text):
#     # one really important thing that `text_to_word_sequence` does
#     # is make all texts the same length -- in this case, the length
#     # of the longest text in the set.
#     return [dictionary[word] for word in kpt.text_to_word_sequence(text)]
#
# allWordIndices = []
# # for each tweet, change each token to its ID in the Tokenizer's word_index
# for text in list_final:
#     wordIndices = convert_text_to_index_array(text)
#     allWordIndices.append(wordIndices)
#
# # now we have a list of all tweets converted to index arrays.
# # cast as an array for future usage.
# allWordIndices = np.asarray(allWordIndices)
#
# # create one-hot matrices out of the indexed tweets
# train_x = t.sequences_to_matrix(allWordIndices, mode='binary')
# print(train_x)
# print(len(train_x))
#
encoder = LabelBinarizer()
encoder.fit(y_Train)
Y = encoder.transform(y_Train)
YTest = encoder.transform(y_Test)
print(len(Y))

# # define 5 documents
# docs = ['well done!',
# 		'good work',
# 		'great effort',
# 		'nice work',
#         'nice time again',
# 		'excellent!',
#         'nice']
# # # create the tokenizer
# t = Tokenizer()
# # fit the tokenizer on the documents
# t.fit_on_texts(docs)
# # summarize what was learned
# print(t.word_counts)
# print(t.document_count)
# print(t.word_index)
# print(t.word_docs)
#
# encoded_docs = t.texts_to_matrix(docs, mode='count')
# print(encoded_docs)
#
# print(len(encoded_docs))
# print(len(docs))

# 20 news groups
num_labels = len(encoder.classes_)
vocab_size = len(t.word_index)+1
batch_size = 50

model = Sequential()
model.add(Dense(512, input_shape=(vocab_size,)))
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
model.add(Dense(num_labels))
model.add(Activation('softmax'))
model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

history = model.fit(X, Y,
                    batch_size=batch_size,
                    epochs=50,
                    verbose=1,
                    validation_split=0.1)

# evaluate the model
# scores = model.evaluate(x_Test, y_Test)
# print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))

score = model.evaluate(XTest, YTest,
                       batch_size=batch_size, verbose=1)

print('Test accuracy:', score[1])

text_labels = encoder.classes_
print(text_labels)
# calculate predictions
predictions = model.predict(data_text)
# round predictions
print(predictions)
rounded = [round(x[0]) for x in predictions]
print(rounded)

# for i in range(10):
#     prediction = model.predict(numpy.array([XTest[i]]))
#     predicted_label = text_labels[numpy.argmax(prediction[0])]
#     print(test_files_names.iloc[i])
#     print('Actual label:' + test_tags.iloc[i])
#     print("Predicted label: " + predicted_label)
#
# #
# text_labels = encoder.classes_
#
# for i in range(10):
#     prediction = model.predict(np.array([x_test[i]]))
#     predicted_label = text_labels[np.argmax(prediction[0])]
#     print(test_files_names.iloc[i])
#     print('Actual label:' + test_tags.iloc[i])
#     print("Predicted label: " + predicted_label)
#
# text = "জিনিশটা ভালো আছে"
# data_text = t.texts_to_matrix(text, mode='count')
# test = numpy.array(data_text)
#
# # calculate predictions
# predictions = model.predict(test)
# # round predictions
# rounded = [round(x[0]) for x in predictions]
# print(rounded)

# list_encode = list()
# for row in list_n:
#     list_sentence = list()
#     list_sentence = sent_tokenize(row)
#     # print(list_sentence)
#     for sentence in list_sentence:
#         retieveEmo(sentence)
#         list_word = list()
#         list_word = word_tokenize(sentence)
#
#         for word in list_word:
#             if re.match(r'\w', word.lower()) and not any(word.lower() in s for s in list_of_words):
#                 if ("।" in word.lower() and not is_ascii(word.lower())):
#                     word = word.replace("।", "")
#                     data = word.encode("utf-8")
#                     list_encode.append(word.lower())
#                 else:
#                     list_of_words.append(word.lower())
# list_of_words = list(list_of_words + list_encode)

# print(list_of_words)
#
# from nltk.corpus import stopwords
#
# stop_words = set(stopwords.words("english"))
# print(stop_words)
#
# filtered_sent = []
# for w in list_of_words:
#     if w not in stop_words:
#         filtered_sent.append(w)
# print("Tokenized Sentence:", list_of_words)
# print("Filterd Sentence:", filtered_sent)
#
# from nltk.probability import FreqDist
#
# frqList = FreqDist(filtered_sent)
# print(frqList)
# print(frqList.most_common())

# # Frequency Distribution Plot
# import matplotlib.pyplot as plt
#
# plt.rc('font', **{'sans-serif': 'Arial',
#                   'family': 'sans-serif'})
# frqList.plot(30, cumulative=False)
# plt.show()
