import sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report, confusion_matrix, f1_score, matthews_corrcoef
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.model_selection import RandomizedSearchCV


data = pd.read_csv("train_data_whole_reduced.csv")

X, y = data.iloc[:, :-1], data.iloc[:, -1]

X, X_pred, y, y_pred = train_test_split(X, y, test_size=0.2, random_state=123)

clf = XGBClassifier(max_depth=5, base_score=0.3, n_jobs=5)
#clf = XGBClassifier(objective='binary:logistic', colsample_bytree=0.3, colsample_bylevel=0.3, learning_rate=0.01, base_score=0.3, max_depth=100, reg_lambda=10, n_estimators=1000, gamma=1, n_jobs=5)
cv = StratifiedKFold(n_splits=3)
preds = np.ones(y.shape[0])

count = 1
for train_index, test_index in cv.split(X, y):
    # print(train_index)
    print("Fold", count)
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    clf.fit(X_train, y_train)
    pred = clf.predict_proba(X_test)[:, 1]
    preds[test_index] = pred
    print("ROC:", roc_auc_score(y_test, pred))
    #print("Accuracy:", accuracy_score(y_test, pred))
    #print(confusion_matrix(y_test, pred, labels=[0, 1]))
    #print(classification_report(y_test, pred))
    count += 1

# print(preds)
# print(len(preds))
# print(sum(preds))

thresholds = np.linspace(0.01, 0.99, 50)
mcc = np.array([f1_score(y, preds > thr) for thr in thresholds])
plt.plot(thresholds, mcc)
plt.title('Effect of decision threshold on mcc-score')
plt.ylabel('mcc score')
plt.xlabel('decision threshold')
best_threshold = thresholds[mcc.argmax()]
print(mcc.max())
# plt.show()

#best_threshold = 0.2

preds = (clf.predict_proba(X_pred)[:, 1] > best_threshold).astype(np.int8)
print('MCC:', matthews_corrcoef(y_pred, preds))
print("ROC:", roc_auc_score(y_pred, preds))
print("Accuracy:", accuracy_score(y_pred, preds))
print(confusion_matrix(y_pred, preds, labels=[0, 1]))
print(classification_report(y_pred, preds))


# parameters = {'base_score': [0.1, 0.3, 0.5, 0.7, 0.9], 'booster': ['gbtree', 'gblinear', 'dart'], 'colsample_bylevel': [0.5, 1], 'colsample_bytree': [0.5, 1], 'learning_rate': [0.1, 0.01], 'max_depth': [3, 5, 10, 20, 50, 100], 'n_estimators': [10, 100, 500, 1000], 'gamma': [0, 1, 10], 'reg_alpha': [0, 1, 5], 'reg_lambda': [0, 1, 5], 'scale_pos_weight': [0, 0, 5, 1, 5, 10, 50]}
# rand = RandomizedSearchCV(clf, parameters, cv=3, scoring='f1_macro', n_iter=30, random_state=5, n_jobs=10)
# rand.fit(X, y)
# print("Best parameters set found on development set:")
# print()
# print(rand.best_params_, rand.best_score_)
# print()
# print("Rand scores on development set:")
# print()
# means = rand.cv_results_['mean_test_score']
# stds = rand.cv_results_['std_test_score']
# for mean, std, params in zip(means, stds, rand.cv_results_['params']):
#     print("%0.3f (+/-%0.03f) for %r"
#           % (mean, std * 2, params))
