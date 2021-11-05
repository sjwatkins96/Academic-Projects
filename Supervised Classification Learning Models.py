#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 15 12:06:07 2021

@author: stephaniewatkins
"""

import pandas as pd


#data exploration
bc=pd.read_csv('~/Desktop/DANN862/breastcancer.csv', sep=',')
bc.columns
bc.shape
print(bc.isnull().sum())
bc.describe()
bc.info()
print(bc.describe())
print(bc.corr())
#30% data
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import linear_model
from sklearn import tree
from sklearn import naive_bayes
from sklearn.neural_network import MLPClassifier 
from sklearn.preprocessing import MinMaxScaler
X = bc.iloc[:, :9]
y = bc.Classification
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 123)

logr = linear_model.LogisticRegression(max_iter=10000)
logr.fit(X_train, y_train)
y_train_logr = logr.predict(X_train)
print('logr train accuracy is ' , metrics.accuracy_score(y_train, y_train_logr))
y_test_logr = logr.predict(X_test)
print(' logr test accuracy is ', metrics.accuracy_score(y_test, y_test_logr))
#naive bayes
nb = naive_bayes.GaussianNB()
nb.fit(X_train, y_train)
y_train_nb = nb.predict(X_train)
print( 'naive bayes train accuracy is ' , metrics.accuracy_score(y_train, y_train_nb))
y_test_nb= nb.predict(X_test)
print( ' naive bayes test accuracy is ', metrics.accuracy_score(y_test, y_test_nb))
#decision tree
dt= tree.DecisionTreeClassifier()
dt.fit(X_train, y_train)
y_train_dt = dt.predict(X_train)
print( ' decision tree train accuracy is ' , metrics.accuracy_score(y_train, y_train_dt))
y_test_dt = dt.predict(X_test)
print (' decision tree test accuracy is ', metrics.accuracy_score(y_test, y_test_dt))
#neural nework
scalar = MinMaxScaler()
X_train_scaled = scalar.fit_transform(X_train)
X_test_scaled = scalar.transform(X_test)
nn = MLPClassifier(hidden_layer_sizes = (10,5), max_iter=10000)
nn.fit(X_train_scaled, y_train)
y_train_nn = nn.predict(X_train_scaled)
print ('   neural network train accuracy is ',  metrics.accuracy_score(y_train, y_train_nn))
y_test_nn = nn.predict(X_test_scaled)
print( ' neural network test accuracy is ' , metrics.accuracy_score(y_test, y_test_nn))

#Based on the output of the test accuracy results, the neural network performed the best test predicition accuracy therefore neural network was the best performing model

from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train_scaled, y_train)
importances = pd.DataFrame(data={
    'Attribute': X_train.columns,
    'Importance': model.coef_[0]
})
importances = importances.sort_values(by='Importance', ascending=False)
import matplotlib.pyplot as plt
plt.bar(x=importances['Attribute'], height=importances['Importance'], color='#087E8B')
plt.title('Feature importances obtained from coefficients', size=20)
plt.xticks(rotation='vertical')
plt.show()
#from the plot, glucose has the most importance



