import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectPercentile
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import f1_score


# DOES NOT WORK WITH MISSING DATA
data = pd.read_csv("train_data_balanced_filled.csv")

X, y = data.iloc[:, 2:-1], data.iloc[:, -1]
# print(X.head())

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

select = SelectPercentile(percentile=20)
select.fit(X_train, y_train)
X_train_selected = select.transform(X_train)

print('X_train.shape is: {}'.format(X_train.shape))
print('X_train_selected.shape is: {}'.format(X_train_selected.shape))

# mask = select.get_support()
# print(mask)
# mask = mask[:100]
# plt.matshow(mask.reshape(1, -1), cmap='gray_r')
# plt.show()

X_test_selected = select.transform(X_test)
SVM = SVC(kernel='rbf', gamma='auto', C=10)
SVM.fit(X_train, y_train)
pred = SVM.predict(X_test)
print("F1-Score of SVM on all features: {:.5f}".format(f1_score(y_test, pred)))
print("Accuracy of SVM on all features: {:.5f}".format(SVM.score(X_test, y_test)))
SVM.fit(X_train_selected, y_train)
pred_selected = SVM.predict(X_test_selected)
print("F1-Score of SVM on reduced features: {:.5f}".format(f1_score(y_test, pred_selected)))
print("Accuracy of SVM on reduced features: {:.5f}".format(SVM.score(X_test_selected, y_test)))
