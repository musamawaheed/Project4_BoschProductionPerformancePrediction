import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report, confusion_matrix, roc_auc_score, accuracy_score
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.svm import OneClassSVM
from sklearn import tree

data = pd.read_csv("train_data_filled.csv")

X, y = data.iloc[:10000, 2:-1], data.iloc[:10000, -1]
# print(X.head())

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=123)

#data = pd.read_csv("train_data_filled.csv")
#X_test, y_test = data.iloc[10000:, 2:-1], data.iloc[10000:, -1]

# print(type(y_test))
y_test = np.array(y_test)
# print(type(y_test))

#select = SelectFromModel(RandomForestClassifier(n_estimators=100, random_state=42), threshold='median')
select = SelectFromModel(RandomForestClassifier(n_estimators=100, random_state=42), threshold=None)
select.fit(X_train, y_train)
X_train_selected = select.transform(X_train)
print("Shape of X_train:", X_train.shape)
print("Shape of X_train_selected:", X_train_selected.shape)
X_test_selected = select.transform(X_test)

# DECISION TREE
# clf = tree.DecisionTreeClassifier()
# clf.fit(X_train, y_train)
# pred = clf.predict(X_test)
# print("F1-Score of DT on all features: {:.5f}".format(f1_score(y_test, pred)))
# print("Accuracy of DT on all features: {:.5f}".format(clf.score(X_test, y_test)))
# clf.fit(X_train_selected, y_train)
# pred_selected = clf.predict(X_test_selected)
# print("F1-Score of DT on reduced features: {:.5f}".format(f1_score(y_test, pred_selected)))
# print("Accuracy of DT on reduced features: {:.5f}".format(clf.score(X_test_selected, y_test)))

# RANDOM FOREST
clf = RandomForestClassifier(n_estimators=1000)
clf.fit(X_train, y_train)
pred = clf.predict(X_test)
print(confusion_matrix(y_test, pred, labels=[0, 1]))
print(classification_report(y_test, pred))

clf.fit(X_train_selected, y_train)
pred_selected = clf.predict(X_test_selected)
print(confusion_matrix(y_test, pred_selected, labels=[0, 1]))
print(classification_report(y_test, pred_selected))

# SVM
# SVM = SVC(kernel='rbf', gamma='auto')
# SVM.fit(X_train, y_train)
# pred = SVM.predict(X_test)
# print("F1-Score of SVM on all features: {:.5f}".format(f1_score(y_test, pred)))
# print("Accuracy of SVM on all features: {:.5f}".format(SVM.score(X_test, y_test)))
# SVM.fit(X_train_selected, y_train)
# pred_selected = SVM.predict(X_test_selected)
# print("F1-Score of SVM on reduced features: {:.5f}".format(f1_score(y_test, pred_selected)))
# print("Accuracy of SVM on reduced features: {:.5f}".format(SVM.score(X_test_selected, y_test)))
