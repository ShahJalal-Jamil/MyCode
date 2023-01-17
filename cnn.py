from keras.callbacks import ModelCheckpoint
from keras.losses import mean_squared_error
from keras.utils import to_categorical
from pip._internal.utils import logging
from sklearn import metrics
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import r2_score, accuracy_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelBinarizer, StandardScaler, RobustScaler
import pandas as pd
import re
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Embedding, Bidirectional, LSTM, MaxPooling2D, Flatten, \
    MaxPooling1D, Conv1D
import numpy

# fix random seed for reproducibility
from keras_preprocessing.text import Tokenizer
from sklearn.svm import LinearSVC
from tensorflow.python.keras.models import load_model

numpy.random.seed(7)

import get_data_from_disk as preProcess

# X, Y, XTest, YTest, word_index, encoder = preProcess.preProcessDataNNPolOld()
# XCAT, YCAT, XTestCAT, YTestCAT, word_index, encoderCAT = preProcess.preProcessDataNNCAT()
XCAT, YCAT, XTestCAT, YTestCAT, encoderCAT, X, Y, XTest, YTest, word_index, encoder = preProcess.preProcessDataNNPol()

print("XCAT")
print(XCAT)
print("YCAT")
print(YCAT)
yy = YTest.values
# print(yy)
# print(YTest)
# print(XCAT)
# print(XTestCAT)
# scaller = StandardScaler()
# XCAT = scaller.fit_transform(XCAT)
# XTestCAT = scaller.fit_transform(XTestCAT)
# print(XCAT)
# print(XTestCAT)

from sklearn import metrics

# knn = KNeighborsClassifier(n_neighbors=3, metric='euclidean')
# knn.fit(XCAT, YCAT)
# y_pred = knn.predict(XTestCAT)
# SVC_pipeline = Pipeline([
#     ('clf', OneVsRestClassifier(MultinomialNB(
#         fit_prior=True, class_prior=None))),
# ])
SVC_pipeline = Pipeline([
    ('clf', OneVsRestClassifier(LinearSVC(), n_jobs=1)),
])
SVC_pipeline.fit(XCAT, YCAT)
y_pred = SVC_pipeline.predict(XTestCAT)

YPRED = encoderCAT.inverse_transform(y_pred)
list_predict = list()
for row in YPRED:
    list_predict.append(row)
print(list_predict)
Y1 = encoderCAT.transform(list_predict)
print(list_predict)
Y2 = pd.DataFrame(Y1)
print(Y2)
y_pred = Y2.values
YTestCAT = YTestCAT.values
y_pol_pred = list()
y_pol_test = list()
for i in range(len(y_pred)):
    y_pol_pred.append(numpy.argmax(y_pred[i]))
for i in range(len(YTestCAT)):
    y_pol_test.append(numpy.argmax(YTestCAT[i]))
from sklearn.metrics import accuracy_score

print("SVM")
print(accuracy_score(y_pol_pred, y_pol_test) * 100, "%")
print(metrics.classification_report(y_pol_pred, y_pol_test))
print(metrics.confusion_matrix(y_pol_pred, y_pol_test))
#
# LR_pipeline = Pipeline([
#     ('clf', OneVsRestClassifier(LogisticRegression(solver='sag'), n_jobs=1)),
# ])
# LR_pipeline.fit(XCAT, YCAT)
# y_pred = SVC_pipeline.predict(XTestCAT)
#
# YPRED = encoderCAT.inverse_transform(y_pred)
# list_predict = list()
# for row in YPRED:
#     list_predict.append(row)
# print(list_predict)
# Y1 = encoderCAT.transform(list_predict)
# print(list_predict)
# Y2 = pd.DataFrame(Y1)
# print(Y2)
# y_pred = Y2.values
# # YTestCAT = YTestCAT.values
# y_pol_pred = list()
# y_pol_test = list()
# for i in range(len(y_pred)):
#     y_pol_pred.append(numpy.argmax(y_pred[i]))
# for i in range(len(YTestCAT)):
#     y_pol_test.append(numpy.argmax(YTestCAT[i]))
# from sklearn.metrics import accuracy_score
#
# print("LR")
# print(accuracy_score(y_pol_pred, y_pol_test) * 100, "%")
# print(metrics.classification_report(y_pol_pred, y_pol_test))
# print(metrics.confusion_matrix(y_pol_pred, y_pol_test))
#
# # XCAT = to_categorical(XCAT)
# # YCAT = to_categorical(YCAT)
# # model = LogisticRegression(solver='liblinear')
# # SVC_pipeline.fit(XCAT, YCAT)
# # y_pred = SVC_pipeline.predict(XTestCAT)
#
from sklearn.ensemble import RandomForestClassifier

# # # Create a Gaussian Classifier
# clf = RandomForestClassifier()
# # Train the model using the training sets y_pred=clf.predict(X_test)
# clf.fit(XCAT, YCAT)
# y_pred = clf.predict(XTestCAT)
#
# YPRED = encoderCAT.inverse_transform(y_pred)
# list_predict = list()
# for row in YPRED:
#     list_predict.append(row)
# print(list_predict)
# Y1 = encoderCAT.transform(list_predict)
# print(list_predict)
# Y2 = pd.DataFrame(Y1)
# print(Y2)
# y_pred = Y2.values
# YTestCAT = YTestCAT.values
# y_pol_pred = list()
# y_pol_test = list()
# for i in range(len(y_pred)):
#     y_pol_pred.append(numpy.argmax(y_pred[i]))
# for i in range(len(YTestCAT)):
#     y_pol_test.append(numpy.argmax(YTestCAT[i]))
# from sklearn.metrics import accuracy_score
#
# print("RF")
# print(accuracy_score(y_pol_pred, y_pol_test) * 100, "%")
# print(metrics.classification_report(y_pol_pred, y_pol_test))
# print(metrics.confusion_matrix(y_pol_pred, y_pol_test))
#
# LR_pipeline = Pipeline([
#     ('clf', OneVsRestClassifier(MultinomialNB(fit_prior=True, class_prior=None), n_jobs=1)),
# ])
# LR_pipeline.fit(XCAT, YCAT)
# y_pred = SVC_pipeline.predict(XTestCAT)
#
# YPRED = encoderCAT.inverse_transform(y_pred)
# list_predict = list()
# for row in YPRED:
#     list_predict.append(row)
# print(list_predict)
# Y1 = encoderCAT.transform(list_predict)
# print(list_predict)
# Y2 = pd.DataFrame(Y1)
# print(Y2)
# y_pred = Y2.values
# YTestCAT = YTestCAT.values
# y_pol_pred = list()
# y_pol_test = list()
# for i in range(len(y_pred)):
#     y_pol_pred.append(numpy.argmax(y_pred[i]))
# for i in range(len(YTestCAT)):
#     y_pol_test.append(numpy.argmax(YTestCAT[i]))
# from sklearn.metrics import accuracy_score
#
# print("NB")
# print(accuracy_score(y_pol_pred, y_pol_test) * 100, "%")
# print(metrics.classification_report(y_pol_pred, y_pol_test))
# print(metrics.confusion_matrix(y_pol_pred, y_pol_test))

# Import Random Forest Model


# # # Create a Gaussian Classifier
# clf = RandomForestClassifier(n_estimators=100)
# # Train the model using the training sets y_pred=clf.predict(X_test)
# clf.fit(XCAT, YCAT)
# y_pred = clf.predict(XTestCAT)
# print("Pred")
# print(y_pred)
# # print("Actual")
# # print(XTestCAT.values)
#
# YPRED = encoderCAT.inverse_transform(y_pred)
# print(YPRED)
#
# list_predict = list()
# for row in YPRED:
#     list_predict.append(row)
# print(list_predict)
# Y1 = encoderCAT.transform(list_predict)
# print(list_predict)
# Y2 = pd.DataFrame(Y1)
# print(Y2)
#
# y_pred = Y2.values
# YTestCAT = YTestCAT.values
# y_pol_pred = list()
# y_pol_test = list()
# for i in range(len(y_pred)):
#     y_pol_pred.append(numpy.argmax(y_pred[i]))
# for i in range(len(YTestCAT)):
#     y_pol_test.append(numpy.argmax(YTestCAT[i]))
# for i in range(len(y_pred)):
#     if (y_pred[i][0] == YTestCAT[i][0] and y_pred[i][1] == YTestCAT[i][1] and y_pred[i][2] == YTestCAT[i][2]):
#         y_pol_pred.append(numpy.argmax(y_pred[i]))
#         y_pol_test.append(numpy.argmax(YTestCAT[i]))
#     else:
#         y_pol_test.append(numpy.argmax(YTestCAT[i]))
#         if (numpy.argmax(YTestCAT[i]) == 0):
#             y_pol_pred.append(1)
#         elif (numpy.argmax(YTestCAT[i]) == 1):
#             y_pol_pred.append(2)
#         else:
#             y_pol_pred.append(0)
#
# print(y_pol_pred)
# print(y_pol_test)
#
# from sklearn.metrics import accuracy_score
#
# print(accuracy_score(y_pol_pred, y_pol_test) * 100, "%")
# # print(model.predict(data_text))
#
# print(metrics.classification_report(y_pol_pred, y_pol_test))
# print(metrics.confusion_matrix(y_pol_pred, y_pol_test))
#
# # LogReg_pipeline = Pipeline([
# #     ('clf', OneVsRestClassifier(LogisticRegression(solver='sag'), n_jobs=1)),
# # ])
# #
# # SVC_pipeline = Pipeline([
# #                 ('clf', OneVsRestClassifier(LinearSVC(), n_jobs=1)),
# #             ])
# #
# # # XCAT = to_categorical(XCAT)
# # # YCAT = to_categorical(YCAT)
# # model = LogisticRegression(solver='liblinear')
# # SVC_pipeline.fit(XCAT, YCAT)
# # y_pred = SVC_pipeline.predict(XTestCAT)
#
# # Import Random Forest Model
# from sklearn.ensemble import RandomForestClassifier
#
# # # # Create a Gaussian Classifier
# # clf = RandomForestClassifier(n_estimators=100)
# # # Train the model using the training sets y_pred=clf.predict(X_test)
# # clf.fit(XCAT, YCAT)
# # y_pred = clf.predict(XTestCAT)
# print("Pred")
# print(y_pred)
# print("Actual")
# print(XTestCAT.values)
#
# YPRED = encoderCAT.inverse_transform(y_pred)
# print(YPRED)
#
# list_predict = list()
# for row in YPRED:
#     list_predict.append(row)
# print(list_predict)
# Y1 = encoderCAT.transform(list_predict)
# print(list_predict)
# Y2 = pd.DataFrame(Y1)
# print(Y2)
#
# y_pred = Y2.values
# # YTestCAT = YTestCAT.values
# y_pol_pred = list()
# y_pol_test = list()
# for i in range(len(y_pred)):
#     y_pol_pred.append(numpy.argmax(y_pred[i]))
# for i in range(len(YTestCAT)):
#     y_pol_test.append(numpy.argmax(YTestCAT[i]))
# # for i in range(len(y_pred)):
# #     if (y_pred[i][0] == YTestCAT[i][0] and y_pred[i][1] == YTestCAT[i][1] and y_pred[i][2] == YTestCAT[i][2]):
# #         y_pol_pred.append(numpy.argmax(y_pred[i]))
# #         y_pol_test.append(numpy.argmax(YTestCAT[i]))
# #     else:
# #         y_pol_test.append(numpy.argmax(YTestCAT[i]))
# #         if (numpy.argmax(YTestCAT[i]) == 0):
# #             y_pol_pred.append(1)
# #         elif (numpy.argmax(YTestCAT[i]) == 1):
# #             y_pol_pred.append(2)
# #         else:
# #             y_pol_pred.append(0)
#
# print(y_pol_pred)
# print(y_pol_test)
#
# from sklearn.metrics import accuracy_score
#
# print(accuracy_score(y_pol_pred, y_pol_test) * 100, "%")
# # print(model.predict(data_text))
#
# print(metrics.classification_report(y_pol_pred, y_pol_test))
# print(metrics.confusion_matrix(y_pol_pred, y_pol_test))
#
# # knn = KNeighborsClassifier(n_neighbors=3, metric='euclidean')
# # knn.fit(XCAT, YCAT)
# # y_pred = knn.predict(XTestCAT)
#
# # Import Random Forest Model
# from sklearn.ensemble import RandomForestClassifier
#
# # # Create a Gaussian Classifier
# clf = RandomForestClassifier(n_estimators=100)
# # Train the model using the training sets y_pred=clf.predict(X_test)
# clf.fit(XCAT, YCAT)
# y_pred = clf.predict(XTestCAT)
# print("Pred")
# print(y_pred)
# print("Actual")
# print(XTestCAT.values)
#
# YPRED = encoderCAT.inverse_transform(y_pred)
# print(YPRED)
#
# list_predict = list()
# for row in YPRED:
#     list_predict.append(row)
# print(list_predict)
# Y1 = encoderCAT.transform(list_predict)
# print(list_predict)
# Y2 = pd.DataFrame(Y1)
# print(Y2)
#
# y_pred = Y2.values
# # YTestCAT = YTestCAT.values
# y_pol_pred = list()
# y_pol_test = list()
# for i in range(len(y_pred)):
#     y_pol_pred.append(numpy.argmax(y_pred[i]))
# for i in range(len(YTestCAT)):
#     y_pol_test.append(numpy.argmax(YTestCAT[i]))
# # for i in range(len(y_pred)):
# #     if (y_pred[i][0] == YTestCAT[i][0] and y_pred[i][1] == YTestCAT[i][1] and y_pred[i][2] == YTestCAT[i][2]):
# #         y_pol_pred.append(numpy.argmax(y_pred[i]))
# #         y_pol_test.append(numpy.argmax(YTestCAT[i]))
# #     else:
# #         y_pol_test.append(numpy.argmax(YTestCAT[i]))
# #         if (numpy.argmax(YTestCAT[i]) == 0):
# #             y_pol_pred.append(1)
# #         elif (numpy.argmax(YTestCAT[i]) == 1):
# #             y_pol_pred.append(2)
# #         else:
# #             y_pol_pred.append(0)
#
# print(y_pol_pred)
# print(y_pol_test)
#
# from sklearn.metrics import accuracy_score
#
# print(accuracy_score(y_pol_pred, y_pol_test) * 100, "%")
# # print(model.predict(data_text))
#
# print(metrics.classification_report(y_pol_pred, y_pol_test))
# print(metrics.confusion_matrix(y_pol_pred, y_pol_test))
# #
# # # Place the DataFrames side by side
# # horizontal_stack = pd.concat([XTestCAT, Y2], axis=1)
# # print(horizontal_stack)
# #
# # # X, Y, XTest, YTest, word_index, encoder = preProcess.preProcessDataNNPol()
# # # print(X, Y, XTest, YTest)
# #
# # from sklearn import metrics
# #
# # knn = KNeighborsClassifier(n_neighbors=3, metric='euclidean')
# # knn.fit(X, Y)
# # y_pred = knn.predict(horizontal_stack)
# # # clf = RandomForestClassifier(n_estimators=100)
# # # # Train the model using the training sets y_pred=clf.predict(X_test)
# # # clf.fit(X, Y)
# # # y_pred = clf.predict(horizontal_stack)
# # YPRED = encoder.inverse_transform(y_pred)
# #
# # list_predict = list()
# # for row in YPRED:
# #     list_predict.append(row)
# # print(list_predict)
# # Y1 = encoder.transform(list_predict)
# # print(list_predict)
# # Y2 = pd.DataFrame(Y1)
# # print(Y2)
# #
# # y_pred = Y2.values
# # print(YPRED)
# # print("Pred")
# # print(y_pred)
# # YTest = YTest.values
# # print("Actual")
# # print(YTest)
# # # Converting predictions to label
# # y_pol_pred = list()
# # y_pol_test = list()
# # for i in range(len(y_pred)):
# #     y_pol_pred.append(numpy.argmax(y_pred[i]))
# # for i in range(len(YTest)):
# #     y_pol_test.append(numpy.argmax(YTest[i]))
# # # for i in range(len(y_pred)):
# # #     if (y_pred[i][0] == YTest[i][0] and y_pred[i][1] == YTest[i][1] and y_pred[i][2] == YTest[i][2]):
# # #         y_pol_pred.append(numpy.argmax(y_pred[i]))
# # #         y_pol_test.append(numpy.argmax(YTest[i]))
# # #     else:
# # #         y_pol_test.append(numpy.argmax(YTest[i]))
# # #         if (numpy.argmax(YTest[i]) == 0):
# # #             y_pol_pred.append(1)
# # #         elif (numpy.argmax(YTest[i]) == 1):
# # #             y_pol_pred.append(2)
# # #         else:
# # #             y_pol_pred.append(0)
# #
# # print(y_pol_pred)
# # print(y_pol_test)
# #
# # from sklearn.metrics import accuracy_score
# #
# # print(accuracy_score(y_pol_pred, y_pol_test) * 100, "%")
# # # print(model.predict(data_text))
# #
# # print(metrics.classification_report(y_pol_pred, y_pol_test))
# # print(metrics.confusion_matrix(y_pol_pred, y_pol_test))
# #
# # # # Converting predictions to label
# # # y_pol_pred = list()
# # # for i in range(len(y_pred)):
# # #     y_pol_pred.append(numpy.argmax(y_pred[i]))
# # # # Converting one hot encoded test label to label
# # # YTest = YTest.values
# # # y_pol_test = list()
# # # for i in range(len(YTest)):
# # #     y_pol_test.append(numpy.argmax(YTest[i]))
# # #
# # # print(y_cat_pred)
# # # print(y_cat_test)
# # # print(y_pol_pred)
# # # print(y_pol_test)
# # # t_pred = numpy.column_stack((y_cat_pred, y_pol_pred))
# # # t_test = numpy.column_stack((y_cat_test, y_pol_test))
# # # print(t_pred)
# # # print(t_test)
# # #
# # # y_test_non_category = [ numpy.argmax(t) for t in t_test ]
# # # y_predict_non_category = [ numpy.argmax(t) for t in t_pred ]
# # #
# # # # Converting predictions to label
# # # pred_pol = list()
# # # for i in range(len(t_pred)):
# # #     pred_pol.append(numpy.argmax(t_pred[i]))
# # # # Converting one hot encoded test label to label
# # #
# # # test_pol = list()
# # # for i in range(len(t_test)):
# # #     test_pol.append(numpy.argmax(t_test[i]))
# # #
# # # print(pred_pol)
# # # print(test_pol)
# # #
# # # from sklearn.metrics import accuracy_score
# # #
# # # print(accuracy_score(y_test_non_category, y_predict_non_category) * 100, "%")
# # # # print(model.predict(data_text))
# # #
# # # print(metrics.classification_report(y_test_non_category, y_predict_non_category))
# # # print(metrics.confusion_matrix(y_test_non_category, y_predict_non_category))
# #
#
# num_labels = len(encoder.classes_)
# vocab_size = len(word_index) + 1
# batch_size = 30
#
# sentiment_model = Sequential()
# sentiment_model.add(Dense(512, input_shape=(vocab_size,), activation='relu'))
# sentiment_model.add((Dense(128, activation='relu')))
# sentiment_model.add((Dense(64, activation='relu')))
# sentiment_model.add((Dense(32, activation='relu')))
# sentiment_model.add(Dense(5, activation='softmax'))
# sentiment_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# sentiment_model.fit(X, Y,
#                     batch_size=batch_size,
#                     epochs=5,
#                     verbose=1,
#                     validation_split=0.1)
#
# _, accuracy_test = sentiment_model.evaluate(XTest, YTest)
# print(accuracy_test)
# y_pred = sentiment_model.predict(XTest)
# print(y_pred)
# # YTest = YTest.values
#
# YPRED = encoder.inverse_transform(y_pred)
# print(YPRED)
# print(XTest)
#
# list_predict = list()
# for row in YPRED:
#     list_predict.append(row)
# print(list_predict)
# Y1 = encoder.transform(list_predict)
# print(list_predict)
# Y2 = pd.DataFrame(Y1)
# print(Y2)
# # Place the DataFrames side by side
# horizontal_stack = pd.concat([XTest, Y2], axis=1)
# print(horizontal_stack)
#
# X, Y, XTest, YTest, word_index, encoder = preProcess.preProcessDataNNPol()
# print(X, Y, XTest, YTest)
#
# # # Converting predictions to label
# # y_cat_pred = list()
# # for i in range(len(y_pred)):
# #     y_cat_pred.append(numpy.argmax(y_pred[i]))
# # # Converting one hot encoded test label to label
# #
# # y_cat_test = list()
# # for i in range(len(YTest)):
# #     y_cat_test.append(numpy.argmax(YTest[i]))
#
# # X, Y, XTest, YTest, word_index, encoder = preProcess.preProcessDataNN()
# #
# # print(X, Y, XTest, YTest)
# # yy = YTest.values
# # print(yy)
# # print(YTest)

num_labels = len(encoder.classes_)
vocab_size = len(word_index) + 1
batch_size = 30
# X1 = XCAT.values
# X2 = YCAT.values
# XCAT = to_categorical(X1)
# YCAT = to_categorical(X2)
# n_timesteps, n_features, n_outputs = X1.shape[0], X1.shape[1], X2.shape[1]
# print(n_timesteps, n_features, n_outputs)
# print(XCAT.shape)
# print(YCAT.shape)

mc = ModelCheckpoint('C:/Users/Naim/Downloads/best_model_b_28.h5', monitor='val_acc', mode='max',
                     save_best_only=True)  # Final
# sentiment_model = Sequential()
# sentiment_model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(n_features, 5)))
# sentiment_model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
# sentiment_model.add(MaxPooling1D(2))
# sentiment_model.add(Dropout(0.15))
# sentiment_model.add(Flatten())
# sentiment_model.add(Dense(32, activation='relu'))
# sentiment_model.add(Dropout(0.1))
# sentiment_model.add(Dense(n_outputs, activation='softmax'))
# sentiment_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'], )

# sentiment_model = Sequential()
# sentiment_model.add(Dense(256, input_shape=(vocab_size,), activation='relu'))
# sentiment_model.add((Dense(128, activation='relu')))
# sentiment_model.add((Dense(64, activation='relu')))
# sentiment_model.add(Dense(5, activation='softmax'))
# sentiment_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# sentiment_model.fit(XCAT, YCAT,
#                     batch_size=batch_size,
#                     epochs=20,
#                     verbose=1, callbacks=[mc],
#                     validation_split=0.1)
# X1 = XTestCAT.values
# XTestCAT = to_categorical(X1)

saved_model = load_model('C:/Users/Naim/Downloads/best_model_b_28.h5')
accuracy = saved_model.evaluate(XTestCAT, YTestCAT)
print(accuracy)
# _, accuracy_test = sentiment_model.evaluate(XTestCAT, YTestCAT)
# print(accuracy_test)
y_pred = saved_model.predict(XTestCAT)
print(y_pred)

# Converting predictions to label
y_pol_pred = list()
for i in range(len(y_pred)):
    y_pol_pred.append(numpy.argmax(y_pred[i]))
# Converting one hot encoded test label to label
# YTestCAT = YTestCAT.values
y_pol_test = list()
for i in range(len(YTestCAT)):
    y_pol_test.append(numpy.argmax(YTestCAT[i]))

print(y_pol_test)
print(y_pol_pred)

from sklearn.metrics import accuracy_score

print(accuracy_score(y_pol_pred, y_pol_test) * 100, "%")
# print(model.predict(data_text))

print(metrics.classification_report(y_pol_pred, y_pol_test))
print(metrics.confusion_matrix(y_pol_pred, y_pol_test))
#
# print(y_cat_pred)
# print(y_cat_test)
# print(y_pol_pred)
# print(y_pol_test)
# t_pred = numpy.column_stack((y_cat_pred, y_pol_pred))
# t_test = numpy.column_stack((y_cat_test, y_pol_test))
# print(t_pred)
# print(t_test)
#
# y_test_non_category = [ numpy.argmax(t) for t in t_test ]
# y_predict_non_category = [ numpy.argmax(t) for t in t_pred ]
#
# # Converting predictions to label
# pred_pol = list()
# for i in range(len(t_pred)):
#     pred_pol.append(numpy.argmax(t_pred[i]))
# # Converting one hot encoded test label to label
#
# test_pol = list()
# for i in range(len(t_test)):
#     test_pol.append(numpy.argmax(t_test[i]))
#
# print(pred_pol)
# print(test_pol)
#
# from sklearn.metrics import accuracy_score
#
# print(accuracy_score(y_test_non_category, y_predict_non_category) * 100, "%")
# # print(model.predict(data_text))
#
# print(metrics.classification_report(y_test_non_category, y_predict_non_category))
# print(metrics.confusion_matrix(y_test_non_category, y_predict_non_category))
#
# model = Sequential([
#     Embedding(vocab_size, 64),
#     Bidirectional(LSTM(64)),
#     Dense(64, activation='relu'),
#     Dense(5)
# ])
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# model.fit(X, Y, batch_size=batch_size,
#                     epochs=2,
#                     verbose=1,
#                     validation_split=0.1)
# test_loss, test_acc = model.evaluate(YTest)
# print('Test Loss: {}'.format(test_loss))
# print('Test Accuracy: {}'.format(test_acc))
#
#
#
# """
# Neural network configure.
# """
#
# num_labels = len(encoder.classes_)
# vocab_size = len(word_index) + 1
# batch_size = 30
#
# model = Sequential()
# model.add(Dense(512, kernel_initializer='normal', input_shape=(vocab_size,)))
# model.add(Activation('relu'))
# model.add(Dropout(0.3))
# model.add(Dense(512))
# model.add(Activation('relu'))
# model.add(Dropout(0.3))
# model.add(Dense(256))
# model.add(Activation('relu'))
# model.add(Dropout(0.3))
# model.add(Dense(128))
# model.add(Activation('relu'))
# model.add(Dropout(0.3))
# model.add(Dense(64))
# model.add(Activation('relu'))
# model.add(Dropout(0.3))
# model.add(Dense(1))
# model.add(Activation('sigmoid'))
# model.summary()
#
# model.compile(loss='binary_crossentropy',
#               optimizer='adam',
#               metrics=['accuracy'])
#
# history = model.fit(X, Y,
#                     batch_size=batch_size,
#                     epochs=30,
#                     verbose=1,
#                     validation_split=0.1)
#
# """
# Evaluate neural network model
# """
#
# score = model.evaluate(XTest, YTest,
#                        batch_size=batch_size, verbose=1)
#
# predicted = model.predict(XTest)
#
# print(predicted)
#
# list_predict = list()
# for row in predicted:
#     for item in row:
#         if (item > 0.5):
#             list_predict.append(1)
#         else:
#             list_predict.append(0)
#
# list_predict = numpy.asarray(list_predict)
# YPredict = pd.DataFrame(list_predict)
#
# print('Test accuracy:', score[1])
#
# print(metrics.classification_report(YTest, YPredict))
# print(metrics.confusion_matrix(YTest, YPredict))
#
# # print(accuracy_score(YTest, score) * 100, "%")
#
# # text_labels = encoder.classes_
# # print(text_labels)
# # # calculate predictions
# # predictions = model.predict(data_text)
# # # round predictions
# # print(predictions)
# # rounded = [round(x[0]) for x in predictions]
# # print(rounded)
