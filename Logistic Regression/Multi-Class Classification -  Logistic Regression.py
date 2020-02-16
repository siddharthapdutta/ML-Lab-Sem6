# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 12:00:25 2020
Logistic Regression on Social Network Ads Dataset
Multi-Class Classification
@author: Siddhartha
"""
#%%
# Importing Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
import seaborn as sns
#%%
# Preparing Dataset
df = pd.read_csv('Datasets\Social_Network_Ads.csv')
X = df.iloc[:,1:4] # Independent Variables (Gender, Age, EstimatedSalary)
y = df.iloc[:,4] # Dependent Variable: Purchased

# Labelling Categorical Column: Gender
le = LabelEncoder()
X['Gender']= le.fit_transform(X['Gender']) 
#%%
# Applying StandardScaler
scaler = StandardScaler()
#X = scaler.fit_transform(X)
#%%
# Performing Train-Test Split
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size = 0.1, random_state=45)
#%%
# Applying Logistic Regression for Multi-Class Classification
classifier = LogisticRegression()
classifier.fit(X_train, Y_train)
#%%
# Calculating Y_Pred
y_pred = classifier.predict(X_test)
y_prob = classifier.predict_proba(X_test)
#%%
# Printing Metrics
print("Accuracy Score:", metrics.accuracy_score(Y_test,y_pred))
#%%
# Plotting Confusion Metrics
cm = metrics.confusion_matrix(Y_test,y_pred)
sns.heatmap(cm, annot=True)