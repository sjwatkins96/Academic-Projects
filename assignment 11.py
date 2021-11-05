#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  1 17:44:49 2021

@author: stephaniewatkins
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from scipy.stats import norm
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
from time import time
from IPython.display import display # Allows the use of display() for DataFrames
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn import datasets
#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics


seeds=pd.read_table('~/Desktop/DANN862/seeds.csv', sep=',')
seeds = seeds.dropna()
seeds=seeds.drop_duplicates()
seeds.head()
seeds.info()
seeds.describe(include='all')
print(seeds.shape)
print(seeds.isnull().sum())
figsize=plt.rcParams['figure.figsize']
seeds.hist(figsize=(20,16), color='r')
corr = seeds.corr()
# plot correlation matrix
fig = plt.figure(figsize=(7, 5.5))
mask = np.zeros_like(corr, dtype=np.bool) # create mask to cover the upper triangle
mask[np.triu_indices_from(mask)] = True
sns.heatmap(corr, annot=True, mask=mask, vmax=0.5,linewidths=0.1)
fig.suptitle('Attribute Correlation Matrix', fontsize=14)

# Dividing Data into train and test
train_data = seeds[['Area', 'Perimeter', 'Compactness', 'Len_Kernel', 'Wid_Kernel','Coeff_Assym', 'Len_Kernel_Groove','Classification']]
test_data = seeds[['Area', 'Perimeter', 'Compactness', 'Len_Kernel', 'Wid_Kernel','Coeff_Assym', 'Len_Kernel_Groove']]
X = seeds.iloc[:,0:7]
y = seeds['Classification']
seeds.columns
from sklearn.cluster import KMeans
from sklearn import metrics
kmeans = KMeans(n_clusters = 3, random_state = 130)
y_pred = kmeans.fit_predict(X)
metrics.homogeneity_score(y,y_pred)
metrics.completeness_score(y,y_pred)
metrics.adjusted_rand_score(y,y_pred)
metrics.silhouette_score(X,y_pred, metric='euclidean')
centers = kmeans.cluster_centers_
centers_pl = centers[:,1]
centers_pw = centers[:,1]
plt.figure(figsize=(12,6))
plt.subplot(121)
plt.scatter(X.Area, X.Perimeter, c=y)
plt.title('Seed Data')
plt.xlabel('Area')
plt.ylabel('Perimeter')
plt.subplot(122)
plt.scatter(X.Area, X.Perimeter, c=y_pred, label = 'Predicted')
plt.scatter(centers_pl, centers_pw, s=100, c='r', marker = 'x', label = 'Cluster Centers')
plt.title('Clustering results')
plt.xlabel('Area')
plt.ylabel('Perimeter')
plt.legend()
plt.subplots_adjust(wspace=0.2)
#Average Linkage Type
from sklearn.cluster import AgglomerativeClustering
Hier_ypred1 = AgglomerativeClustering(n_clusters = 3, affinity='euclidean',linkage='average').fit_predict(X)
metrics.homogeneity_score(y,Hier_ypred1)
metrics.completeness_score(y,Hier_ypred1)
metrics.adjusted_rand_score(y,Hier_ypred1)
metrics.silhouette_score(X, Hier_ypred1, metric='euclidean')
from scipy.cluster.hierarchy import dendrogram, linkage
Zavg = linkage(X,method = 'average')
plt.figure(figsize=(100,100))
den=dendrogram(Zavg, leaf_font_size=8)
#Complete Linkage Type
from sklearn.cluster import AgglomerativeClustering
Hier_ypred2 = AgglomerativeClustering(n_clusters = 3, affinity = 'euclidean', linkage = 'complete').fit_predict(X)
metrics.homogeneity_score(y, Hier_ypred2)
metrics.completeness_score(y,Hier_ypred2)
metrics.silhouette_score(X,Hier_ypred2,metric = 'euclidean')
#ward linkkage 
from sklearn.cluster import AgglomerativeClustering
Hier_ypred3 = AgglomerativeClustering(n_clusters=3, affinity = 'euclidean',linkage = 'ward').fit_predict(X)
metrics.homogeneity_score(y,Hier_ypred3)
metrics.completeness_score(y,Hier_ypred3)
metrics.adjusted_rand_score(y,Hier_ypred3)
metrics.silhouette_score(X, Hier_ypred3, metric = 'euclidean')

#Mean or average linkage clustering:
#average of dissimilarities is the distance  between two clusters
print("The mean/average linkage clustering results are below")
print("the homogeneity score is ", metrics.homogeneity_score(y,Hier_ypred1)) 
print("thecompleteness score score is ", metrics.completeness_score(y,Hier_ypred1)) 
print("the adjusted random score score is ", metrics.adjusted_rand_score(y,Hier_ypred1)) 
print("the silhoutte score is ", metrics.silhouette_score(X, Hier_ypred1, metric='euclidean'))
#max or complete linkage computes all pairwise dissimilarties between elenents in cluster 1 and in 2
#considers largest value (max) of dissimilarites and tends to produce more compact clusters
print("The maximum/complete linkage clustering results are below")
print("the homogeneity score is ", metrics.homogeneity_score(y,Hier_ypred2)) 
print("thecompleteness score score is ", metrics.completeness_score(y,Hier_ypred2)) 
print("the adjusted random score score is ", metrics.adjusted_rand_score(y,Hier_ypred2)) 
print("the silhoutte score is ", metrics.silhouette_score(X, Hier_ypred2, metric='euclidean'))

#wards min variance minimizes total within the cluster variance 
#at each step the pair of clusters with min between cluster distance are merged
print("The wards min variance linkage clustering results are below")
print("the homogeneity score is ", metrics.homogeneity_score(y,Hier_ypred3)) 
print("thecompleteness score score is ", metrics.completeness_score(y,Hier_ypred3)) 
print("the adjusted random score score is ", metrics.adjusted_rand_score(y,Hier_ypred3)) 
print("the silhoutte score is ", metrics.silhouette_score(X, Hier_ypred3, metric='euclidean'))
#based on the overall scoring the mean/average linkage clustering performed the best

#4
from sklearn.cluster import DBSCAN
from sklearn import datasets
from sklearn.cluster import KMeans
import numpy as np
X.head()
y.head()
metrics.homogeneity_score(y, y_pred)
metrics.completeness_score(y, y_pred)
metrics.adjusted_rand_score(y,y_pred)
metrics.silhouette_score(X, y_pred, metric = 'euclidean')
result = []
epses = [0.4, 0.6, 0.8]
min_samples=[5,10,15]
range_n_clusters = [2, 3, 4, 5, 6]
for v in epses:
    for n in range_n_clusters:
        y_pred_temp = DBSCAN(eps = v, min_samples = n).fit_predict(X)
        score = metrics.silhouette_score(X,y_pred_temp, metric = 'euclidean') 
        result.append((v,n,score)) 
result
#  (0.8, 6, 0.2266848593143534)] was the best value 




