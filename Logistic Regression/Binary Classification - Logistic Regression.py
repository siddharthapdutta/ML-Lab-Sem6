# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 11:24:29 2020
Logistic Regression on Kid Dataset
Binary Classification
@author: Siddhartha
"""
#%%
# Importing Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import seaborn as sns
#%%
# Preparing Dataset
df = pd.read_csv('Datasets\Kid.csv')
X = df.iloc[:,2:] # Independent Variables
y = df.iloc[:,1] # Dependent Variable: Buy
#%%
# Performing Train-Test Split
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size = 0.33)
#%%
# Applying Logistic Regression for Binary Classification
classifier = LogisticRegression()
classifier.fit(X_train, Y_train)
#%%
# Calculating Y_Pred
y_pred = classifier.predict(X_test)
y_prob = classifier.predict_proba(X_test)
#%%
# Printing Metrics
print("Accuracy Score:", metrics.accuracy_score(Y_test,y_pred))
print("Precision Score:", metrics.precision_score(Y_test, y_pred))
print("Recall Score:", metrics.recall_score(Y_test, y_pred))
#%%
# Plotting Confusion Metrics
cm = metrics.confusion_matrix(Y_test,y_pred)
sns.heatmap(cm, annot=True)