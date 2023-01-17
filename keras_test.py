from keras.models import Sequential
from keras.layers import Dense
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

docs = ['ভাল আছেন, আপনি?',
        'কেম্ন আছেন']
list = list()
for word in docs:
    data = word.encode("utf-8")
    list.append(data)

#print(docs)
# create the tokenizer
t = Tokenizer()
# fit the tokenizer on the documents
t.fit_on_texts(docs)
print(t.word_index)
encoded_docs = t.texts_to_matrix(docs, mode='count')
print(encoded_docs)