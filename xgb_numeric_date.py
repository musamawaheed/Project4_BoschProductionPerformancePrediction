import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import matthews_corrcoef, roc_auc_score
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import confusion_matrix


# feature manipulation and preprocessing
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_extraction import DictVectorizer

# sampling, grid search, and reporting
from sklearn.model_selection import StratifiedShuffleSplit
# from sklearn.cross_validation import StratifiedShuffleSplit
# from sklearn.grid_search import GridSearchCV
from sklearn.model_selection import GridSearchCV


import matplotlib.pyplot as plt
import seaborn as sns


def feat_imp(estimator, features):
    return pd.DataFrame(zip(features, estimator.feature_importances_), columns=["feature", "importance"])


df_date = pd.read_csv("train_date.csv", dtype=np.float32)
print(df_date)
df_numeric = pd.read_csv("train_numeric.csv", dtype=np.float32)
print(df_numeric)

df = pd.concat([df_date, df_numeric], axis=1)
df_train, df_test = train_test_split(df, test_size=0.2, random_state=123)

features = list(set(df.columns) - set(["Id", "Response"]))

df_train_small = df_train.sample(frac=0.5)
X_train = df_train_small[features].values
y_train = df_train_small["Response"].values
X_test = df_test[features].values
y_test = df_test["Response"].values

xgb_clf = XGBClassifier(n_jobs=8)
xgb_clf.fit(X_train, y_train)
important_indices = np.where(xgb_clf.feature_importances_ > 0.005)[0]
print(important_indices)

xgb_feat_imp = feat_imp(xgb_clf, features)
# use pre-written feature selection methods to determine
xgb_sfm = SelectFromModel(xgb_clf, threshold=0.005, prefit=True)
xgb_sfm_support = xgb_sfm.get_support()
xgb_sfm_features = list(map(lambda y: y[1], filter(lambda x: xgb_sfm_support[x[0]], enumerate(features))))
print(xgb_sfm_features)


xgb_clf = XGBClassifier(max_depth=5, base_score=0.005, n_jobs=8)

X_new_train = df_train[xgb_sfm_features].values
y_new_train = df_train["Response"].values
X_new_test = df_test[xgb_sfm_features].values
y_new_test = y_test

cv = StratifiedKFold(n_splits=3)
preds = np.ones(y_new_train.shape[0])
for train, test in cv.split(X_new_train, y_new_train):
    preds[test] = xgb_clf.fit(X_new_train[train], y_new_train[train]).predict_proba(X_new_train[test])[:, 1]
    print("ROC AUC: ", roc_auc_score(y_new_train[test], preds[test]))
print(roc_auc_score(y_new_train, preds))
# print(classification_report(y_true=y_new_train, y_pred=preds, digits=3))

thresholds = np.linspace(0.01, 0.99, 50)
mcc = np.array([matthews_corrcoef(y_new_train, preds > thr) for thr in thresholds])
plt.plot(thresholds, mcc)
best_threshold = thresholds[mcc.argmax()]
print(mcc.max())

preds = (xgb_clf.predict_proba(X_new_test)[:, 1] > best_threshold).astype(np.int8)
print(confusion_matrix(y_new_test, preds, labels=[0, 1]))
print(classification_report(y_true=y_new_test, y_pred=preds, digits=3))
print("MC:", matthews_corrcoef(y_new_test, preds))
