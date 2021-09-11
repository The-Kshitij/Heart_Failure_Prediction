# -*- coding: utf-8 -*-
"""
Created on Sat Sep 11 09:16:50 2021

@author: Kshitij
"""

import pandas as pd;
import numpy as np;

from sklearn.impute import SimpleImputer;

from sklearn.compose import ColumnTransformer;
from sklearn.preprocessing import OneHotEncoder;

from sklearn.model_selection import train_test_split;

from sklearn.preprocessing import StandardScaler;

from sklearn.metrics import accuracy_score;
from sklearn.metrics import confusion_matrix;


#dicstionary to store the accuracies
acc = {};

dataset = pd.read_csv("D:\StudyMat\Datasets\heart.csv");

#storing the dependent and independent variables
x = dataset.iloc[:,:-1].values;
y = dataset.iloc[:,-1].values;

#encoding the categorical data
ct = ColumnTransformer(transformers=[("encoder",OneHotEncoder(),[1,2,6,8,10])],remainder="passthrough");
x = np.array(ct.fit_transform(x));

#this data doesn't have any missing values

#splitting the dataset
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=1);

#scaling the data for efficiency
x_scaler = StandardScaler();
x_scaler.fit(x_train[:,14:]);
x_train[:,14:] = x_scaler.transform(x_train[:,14:]);
x_test[:,14:] = x_scaler.transform(x_test[:,14:]);


from sklearn.linear_model import LogisticRegression;
lr = LogisticRegression();
lr.fit(x_train,y_train);
y_pred_lr = lr.predict(x_test);
s = accuracy_score(y_test,y_pred_lr)
result = "LogisticRegression: {}".format(s);
acc[s] = result;
print(result);
#print(confusion_matrix(y_test,y_pred_lr));

from sklearn.neighbors import KNeighborsClassifier;
kn = KNeighborsClassifier();
kn.fit(x_train,y_train);
y_pred_kn = kn.predict(x_test);
s = accuracy_score(y_test,y_pred_kn);
result = "KNeighborsClassifier: {}".format(s);
print(result);
acc[s] = result;
#print(confusion_matrix(y_test,y_pred_kn));

from sklearn.svm import SVC;
sv_l = SVC(kernel="linear",random_state=1);
sv_l.fit(x_train,y_train);
y_pred_sv_l = sv_l.predict(x_test);
s = accuracy_score(y_test,y_pred_sv_l);
result = "SV_l: {}".format(s);
print(result);
acc[s] = result;
#print(confusion_matrix(y_test,y_pred_sv_l));


sv_nl = SVC(kernel="rbf",random_state=1);
sv_nl.fit(x_train,y_train);
y_pred_sv_nl = sv_nl.predict(x_test);
s = accuracy_score(y_test,y_pred_sv_nl);
result = "SV_Nl: {}".format(s);
print(result);
acc[s] = result;
#print(confusion_matrix(y_test,y_pred_sv_nl));

from sklearn.tree import DecisionTreeClassifier;
dc = DecisionTreeClassifier(criterion="entropy",random_state=1);
dc.fit(x_train,y_train);
dc_pred_l = dc.predict(x_test);
s = accuracy_score(y_test, dc_pred_l);
result = "DecisionTreeClassifier: {}".format(s);
print(result);
acc[s] = result;
#print(confusion_matrix(y_test,dc_pred_l));

from sklearn.ensemble import RandomForestClassifier;
rf = RandomForestClassifier(n_estimators=10,random_state=1);
rf.fit(x_train,y_train);
rf_pred = rf.predict(x_test);
s = accuracy_score(y_test, rf_pred);
result = "RandomForestClassifier: {}".format(s);
print(result);
acc[s] = result;
#print(confusion_matrix(y_test,rf_pred));


print("Winner winner chicker dinner: {}".format(acc[max(acc)]));