#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 18:00:09 2021

@author: stephaniewatkins
"""

import pandas as pd
import matplotlib.pyplot as plt

#data exploration
bc=pd.read_csv('~/Desktop/DANN862/breastcancer.csv', sep=',')
bc.head()
bc.columns
bc.shape
print(bc.isnull().sum())
bc.describe()
bc.info()
print(bc.describe())
print(bc.corr())
bc.describe()
plt.style.use('classic')
colormap=bc.Classification.factorize()[0]
pd.plotting.scatter_matrix(bc, c = colormap, diagonal = 'kde')

#2 
import warnings #because F-test was showing " 0" as warning for linear model
warnings.filterwarnings('ignore')
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
x = bc.iloc[:,0:9]
y = bc.Classification
xtrain, xtest, ytrain, ytest = train_test_split(x,y,test_size=0.3, random_state=2)
svm_poly = svm.SVC(kernel = 'poly' , degree=2)
svm_poly.fit(xtrain, ytrain)
svm_poly_pred_train = svm_poly.predict(xtrain)
svm_poly_pred_test = svm_poly.predict(xtest)
print('SVM ploy train accuracity is ', accuracy_score(svm_poly_pred_train, ytrain))
print(classification_report(svm_poly_pred_train,ytrain))
print('SVM poly bow test accuracy is ', accuracy_score(svm_poly_pred_test,ytest))
print(classification_report(svm_poly_pred_train,ytrain))


svm_rbf = svm.SVC(kernel = 'rbf')
svm_rbf.fit(xtrain,ytrain)
svm_rbf_pred_train = svm_rbf.predict(xtrain)
svm_rbf_pred_test = svm_rbf.predict(xtest)
print('SVM rbf train accuracy is ', accuracy_score(svm_rbf_pred_train, ytrain))
print(classification_report(svm_rbf_pred_train,ytrain))
print('SVM rbf test accuracy is ', accuracy_score(svm_rbf_pred_test, ytest))
print(classification_report(svm_rbf_pred_test,ytest))

svm_lin = svm.SVC(kernel = 'linear')
svm_lin.fit(xtrain, ytrain)
svm_lin_pred_train = svm_lin.predict(xtrain)
svm_lin_pred_test = svm_lin.predict(xtest)
print('SVM linear train accuracy is ', accuracy_score(svm_lin_pred_train, ytrain))
print(classification_report(svm_lin_pred_train,ytrain))
print('SVM linear test accuracy is ', accuracy_score(svm_lin_pred_test, ytest))
print(classification_report(svm_lin_pred_test,ytest))

#3

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
RF = RandomForestClassifier(n_estimators = 100, random_state = 0)
RF.fit(xtrain,ytrain)
RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
max_depth=None, max_features='auto', max_leaf_nodes=None,
min_impurity_decrease=0.0, min_impurity_split=None,
min_samples_leaf=1, min_samples_split=2,
min_weight_fraction_leaf=0.0, n_estimators=98, n_jobs=None,
oob_score=False, random_state=0, verbose=0, warm_start=False)
RF_pred = RF.predict(xtest)
accuracy_score(RF_pred,ytest)
print(classification_report(ytest,RF_pred))
pd.DataFrame({'feature':bc.columns[1:10],'importance':RF.feature_importances_})
n_estimator = range(2,100,2)
accuracy = []
for i in n_estimator:
    RF = RandomForestClassifier(n_estimators=i, random_state =0)
    scores = cross_val_score(RF,xtrain, ytrain)
    accuracy.append(scores.mean())

plt.figure()
plt.plot(n_estimator, accuracy)
plt.title('Ensemble Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Number of base estimators in ensemble')

import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(10,10))
# Creating a bar plot
RF.fit(xtrain,ytrain)
sns.barplot(x=bc.columns[1:10], y=RF.feature_importances_)
# Add labels to your graph
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title("Visualizing Features")
plt.legend()
plt.show()
#BMI isbest n-estimator  


#4
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
n_estimator = range(1, 50, 1)
accuracy = []
for i in n_estimator:
    ada = AdaBoostClassifier(n_estimators=i, learning_rate = 0.005,
random_state=21)
    scores = cross_val_score(ada, xtrain, ytrain)
    accuracy.append(scores.mean())

plt.figure()
plt.plot(n_estimator, accuracy)
plt.title('Adaboost Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Number of base estimators')
    























