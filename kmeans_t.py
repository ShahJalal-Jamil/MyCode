import numpy
import pandas as pd

import numpy as np
from matplotlib.mlab import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

np.random.seed(7)

import pre_processing_data as preProcess

X, Y, XTest, YTest, dataText, t = preProcess.preProcessData(True)
word_index = t.word_index

# from pandas import DataFrame
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
#
# Data = {
#     'x': [25, 34, 22, 27, 33, 33, 31, 22, 35, 34, 67, 54, 57, 43, 50, 57, 59, 52, 65, 47, 49, 48, 35, 33, 44, 45, 38,
#           43, 51, 46],
#     'y': [79, 51, 53, 78, 59, 74, 73, 57, 69, 75, 51, 32, 40, 47, 53, 36, 35, 58, 59, 50, 25, 20, 14, 12, 20, 5, 29, 27,
#           8, 7]
#     }
#
# df = DataFrame(Data, columns=['x', 'y'])
#
# kmeans = KMeans(n_clusters=3).fit(df)
# centroids = kmeans.cluster_centers_
# print(centroids)


from sklearn import datasets
from sklearn.cluster import KMeans, MeanShift, DBSCAN

#
# X = np.array([[5, 3],
#               [10, 15],
#               [15, 12],
#               [24, 10],
#               [30, 45],
#               [85, 70],
#               [71, 80],
#               [60, 78],
#               [55, 52],
#               [80, 91], ])
#
# Y = np.array([[71, 80],
#               [61, 72],
#               ])
# Declaring Model
model = KMeans(n_clusters=2, init='k-means++', n_init=1, random_state=0, max_iter=10000)
# Fitting Model
culster = model.fit_predict(X)
centroids = model.cluster_centers_
# Prediction on the entire data
print(centroids)

from sklearn import metrics

accuracy = metrics.accuracy_score(Y, culster)
print(accuracy * 100, '%')
print(metrics.classification_report(Y, culster))
print(metrics.confusion_matrix(Y, culster))

import pandas as pd

list_predict = list()
for row in culster:
    list_predict.append(row)

list_predict = numpy.asarray(list_predict)
YPredict = pd.DataFrame(list_predict)

# print(YTest)
ne = (YPredict != Y).any(1)
# print(ne)
XTextMisClassify = pd.DataFrame()
YTextMisClassify = pd.DataFrame()
YClassify = pd.DataFrame()
int
track = 0
for nRow in ne:
    if nRow == True:
        XTextMisClassify = XTextMisClassify.append(X.iloc[[track]])
        YTextMisClassify = YTextMisClassify.append(Y.iloc[[track]])
    else:
        YClassify = YClassify.append(Y.iloc[[track]])
    track = track + 1

    # for row in df.iterrows():
    #     trackRowId = 0
    #     listPr = list()
    #     listPrEmo = list()
    #     for rowNew in row[1]:
    #         if rowNew >= 1:
    #             print(track, rowNew)

# print(XTextMisClassify)
# print(YTextMisClassify)

# ms = MeanShift()
# cluster = ms.fit_predict(X)
# labels = ms.labels_
# cluster_centers = ms.cluster_centers_
# print(cluster_centers)
from sklearn.cluster import AgglomerativeClustering

cluster = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='ward')
all_predictions = cluster.fit_predict(XTextMisClassify)
from sklearn import metrics

accuracy = metrics.accuracy_score(YTextMisClassify, all_predictions)
print(accuracy * 100, '%')
print(metrics.classification_report(YTextMisClassify, all_predictions))
print(metrics.confusion_matrix(YTextMisClassify, all_predictions))
# print(YTextMisClassify, all_predictions)

import pandas as pd

list_predict = list()
for row in all_predictions:
    list_predict.append(row)

list_predict = numpy.asarray(list_predict)
# print(len(YTextMisClassify))
# print(len(YClassify))
i = 0
for row in YTextMisClassify.index:
    YTextMisClassify.set_value(row, 0, list_predict[i])
    i = i + 1
YFinal = YClassify.append(YTextMisClassify)
YFinal = YFinal.sort_index()
# print(Y)
# print(YFinal)

# print(len(YFinal))
# print(len(Y))
accuracy = metrics.accuracy_score(Y, YFinal)
print(accuracy * 100, '%')
print(metrics.classification_report(Y, YFinal))
print(metrics.confusion_matrix(Y, YFinal))


# print(YTextMisClassify)
# YPredictTemp = pd.DataFrame(list_predict)
# print(YPredictTemp)
# print(YTextMisClassify)
# print(len(YPredictTemp))
# print(len(YTextMisClassify))
# ne = (YPredictTemp != YTextMisClassify).any(1)
#
# YLAST = pd.DataFrame()
#
# int
# track = 0
# for nRow in ne:
#     if nRow == True:
#         YLAST = YLAST.append(YTextMisClassify.iloc[[track]])
#     track = track + 1
# YFinal = YPredict.append(YLAST)
#
# accuracy = metrics.accuracy_score(YFinal, Y)
# print(accuracy * 100, '%')

#
#
# import numpy as np
# import pandas as pd
# from sklearn.cluster import MeanShift
# from sklearn.datasets.samples_generator import make_blobs
# from matplotlib import pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
#
# # We will be using the make_blobs method
# # in order to generate our own data.
#
# clusters = [[2, 2, 2], [7, 7, 7], [5, 13, 13]]
#
# X, _ = make_blobs(n_samples=500, centers=X,
#                   cluster_std=0.60)
#
# # After training the model, We store the
# # coordinates for the cluster centers
# ms = MeanShift()
# ms.fit(X)
# cluster_centers = ms.cluster_centers_
# pred = ms.predict(XTextMisClassify)
# print(pred)
# accuracy = metrics.accuracy_score(YTest, pred)
# print(accuracy * 100, '%')
# # Finally We plot the data points
# # and centroids in a 3D graph.
# fig = plt.figure()
#
# ax = fig.add_subplot(111, projection='3d')
#
# ax.scatter(X[:, 0], X[:, 1], X[:, 2], marker='o')
#
# ax.scatter(cluster_centers[:, 0], cluster_centers[:, 1],
#            cluster_centers[:, 2], marker='x', color='red',
#            s=300, linewidth=5, zorder=10)
#
# plt.show()

#
# XTextMisClassify = pd.DataFrame()
# CLASSIFY = ()
# for i in range(len(list_predict)):
#     if (list_predict[i] != YTest[i]):
#         XTextMisClassify = XTest[i]
#     else:
#         CLASSIFY[i] = list_predict[i]
#
# print(CLASSIFY)
# print(XTextMisClassify)

# print(all_predictions)
#
# from sklearn import metrics
#
# accuracy = metrics.accuracy_score(YTest, all_predictions)
# print(accuracy * 100, '%')
# #
# # knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
# # knn.fit(X)
# # y_pred = knn.predict(XTest)
# # accuracy = metrics.accuracy_score(YTest, y_pred)
# # print(accuracy * 100, '%')
#
# #
# # # Importing Modules
# # from sklearn.datasets import load_iris
# # import matplotlib.pyplot as plt
# # from sklearn.cluster import DBSCAN
# # from sklearn.decomposition import PCA
# #
# Declaring Model
# dbscan = MeanShift(bandwidth=2).fit(X)
# dbscan = DBSCAN(min_samples=15, eps=3).fit(X)
# # Fitting
# all_predictions = dbscan.fit_predict(XTextMisClassify)
# print(all_predictions)
# from sklearn import metrics
#
# accuracy = metrics.accuracy_score(YTextMisClassify, all_predictions)
# print(accuracy * 100, '%')
# #
# # pca = PCA(3)
# # pca.fit(X)
# # ypred = pca.prdict(YTest)
# # from sklearn import metrics
# #
# # accuracy = metrics.accuracy_score(YTest, ypred)
# # print(accuracy * 100, '%')
#
# # import hierarchical clustering libraries
# import scipy.cluster.hierarchy as sch
# from sklearn.cluster import AgglomerativeClustering
# import matplotlib.pyplot as plt
#
# # create dendrogram
# dendrogram = sch.dendrogram(sch.linkage(X, method='ward'))  # create clusters
# hc = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='ward')  # save clusters for chart
# y_hc = hc.fit_predict(X)
# print(y_hc)
# plt.title('Clusters of Customers (Hierarchical Clustering Model)')
# plt.xlabel('Annual Income(k$)')
# plt.ylabel('Spending Score(1-100')
# plt.show()
#
# list_predict = list()
# for row in y_hc:
#     list_predict.append(row)
#
# list_predict = numpy.asarray(list_predict)
# YPredict = pd.DataFrame(list_predict)
#
# from sklearn import metrics
#
# print(YTest)
# print(YPredict)
# accuracy = metrics.accuracy_score(YTest, y_hc)
# print(accuracy * 100, '%')

# dbscan = DBSCAN(algorithm='auto', min_samples=15, eps=1000, leaf_size=10, metric='euclidean', )
# # Fitting
# all_predictions = dbscan.fit_predict(X)
# print(all_predictions)
# print(dbscan.core_sample_indices_)
# print(dbscan.labels_)
# from sklearn import metrics
#
# for i in all_predictions:
#     print(i)
# accuracy = metrics.accuracy_score(Y, all_predictions)
# print(accuracy * 100, '%')
# print(metrics.classification_report(Y, all_predictions))
# print(metrics.confusion_matrix(Y, all_predictions))
