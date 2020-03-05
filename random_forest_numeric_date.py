import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.model_selection import train_test_split
# estimator imports
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import matthews_corrcoef, roc_auc_score
from sklearn.model_selection import cross_val_score, StratifiedKFold
# feature manipulation and preprocessing
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_extraction import DictVectorizer

# sampling, grid search, and reporting
from sklearn.model_selection import StratifiedShuffleSplit
# from sklearn.cross_validation import StratifiedShuffleSplit
# from sklearn.grid_search import GridSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
TRAIN_SZIE = 0.8
TEST_SIZE = 0.2


def nan_evaluation(df, axis=1, method="all"):
    methods = {
        "all": lambda x: np.all(x),
        "some": lambda x: np.any(x)
    }

    if axis == 1:
        return [col for col in df.columns if methods[method](df[col].isnull())]

    return [row for row in df.index if methods[method](df.iloc[row][1:-1].isnull())]


def combine_list(first_list, second_list):
    return first_list + list(set(second_list) - set(first_list))


def feat_imp(estimator, features):
    return pd.DataFrame(zip(features, estimator.feature_importances_), columns=["feature", "importance"])


df_date = pd.read_csv("train_date.csv", dtype=np.float32)
print(df_date)
df_numeric = pd.read_csv("train_numeric.csv", dtype=np.float32)
print(df_numeric)

df = pd.concat([df_date, df_numeric], axis=1)
df = df.fillna(9999999)
df_train, df_test = train_test_split(df, test_size=0.2, random_state=123)

features_train = list(set(df_train.columns) - set(["Id", "Response"]))
features_test = list(set(df_test.columns) - set(["Id", "Response"]))

if features_train != features_test:
    Print("WRONG")
else:
    features = features_test


df_train_small = df_train.sample(frac=0.5)
X_train = df_train_small[features_train].values
y_train = df_train_small["Response"].values

X_test = df_test[features_test].values
y_test = df_test["Response"].values


print("X_train.shape", X_train.shape, y_train.shape)
# print("X_train", X_train)
print("X_test.shape", X_test.shape, y_test.shape)

print("positive proportion in train: ", format((len(y_train[y_train == 1]) / len(y_train)) * 100))
print("negative proportion in train: ", format((len(y_train[y_train == 0]) / len(y_train)) * 100))
print("positive proportion in test: ", format((len(y_test[y_test == 1]) / len(y_test)) * 100))
print("negative proportion in test: ", format((len(y_test[y_test == 0]) / len(y_test)) * 100))

xt_clf = ExtraTreesClassifier(n_estimators=100, verbose=0, n_jobs=-1)
xt_clf.fit(X_train, y_train)


important_indices = np.where(xt_clf.feature_importances_ > 0.005)[0]
print("XT impoirtant indices(>0.005): ", important_indices)

xt_feat_imp = feat_imp(xt_clf, features)
# use pre-written feature selection methods to determine
xt_sfm = SelectFromModel(xt_clf, prefit=True)
xt_sfm_support = xt_sfm.get_support()
xt_sfm_features = list(map(lambda y: y[1], filter(lambda x: xt_sfm_support[x[0]], enumerate(features))))


# random forest feature selection
rf_clf = RandomForestClassifier(n_estimators=100, verbose=0, n_jobs=-1)
rf_clf.fit(X_train, y_train)


important_indices = np.where(rf_clf.feature_importances_ > 0.005)[0]
print("Random Forest impoirtant indices(>0.005): ", important_indices)

rf_feat_imp = feat_imp(rf_clf, features)
rf_sfm = SelectFromModel(rf_clf, prefit=True)
rf_sfm_support = rf_sfm.get_support()
rf_sfm_features = list(map(lambda y: y[1], filter(lambda x: rf_sfm_support[x[0]], enumerate(features))))


# GBM classifier
gb_clf = GradientBoostingClassifier(n_estimators=100, verbose=0)
gb_clf.fit(X_train, y_train)

important_indices = np.where(gb_clf.feature_importances_ > 0.005)[0]
print("GB impoirtant indices(>0.005): ", important_indices)
gb_feat_imp = feat_imp(gb_clf, features)
gb_sfm = SelectFromModel(gb_clf, prefit=True)

gb_sfm_support = gb_sfm.get_support()
gb_sfm_features = list(map(lambda y: y[1], filter(lambda x: gb_sfm_support[x[0]], enumerate(features))))
features_iter_1 = combine_list(xt_sfm_features, rf_sfm_features)
features_iter_1 = combine_list(features_iter_1, gb_sfm_features)

print("len(features_iter_1): ", len(features_iter_1))


X_new_train = df_train[features_iter_1].values
y_new_train = df_train["Response"].values
X_new_test = df_test[features_iter_1].values
y_new_test = y_test


print("GBClassifier Classification Report:")
gb_clf = GradientBoostingClassifier(max_depth=5, n_estimators=100, verbose=0)

cv = StratifiedKFold(n_splits=3)
preds = np.ones(y_new_train.shape[0])
for train, test in cv.split(X_new_train, y_new_train):
    preds[test] = gb_clf.fit(X_new_train[train], y_new_train[train]).predict_proba(X_new_train[test])[:, 1]
    print("ROC AUC: ", roc_auc_score(y_new_train[test], preds[test]))
print(roc_auc_score(y_new_train, preds))

thresholds = np.linspace(0.01, 0.99, 50)
mcc = np.array([matthews_corrcoef(y_new_train, preds > thr) for thr in thresholds])
plt.plot(thresholds, mcc)
best_threshold = thresholds[mcc.argmax()]
print(mcc.max())

preds = (gb_clf.predict_proba(X_new_test)[:, 1] > best_threshold).astype(np.int8)

print("MC:", matthews_corrcoef(y_new_test, preds))
print(classification_report(y_true=y_new_test, y_pred=preds, digits=3))


#=======================================================================
print("XTClassifier Classification Report:")
xt_clf = ExtraTreesClassifier(n_estimators=100, verbose=0, n_jobs=-1)
cv = StratifiedKFold(n_splits=3)
preds = np.ones(y_new_train.shape[0])
for train, test in cv.split(X_new_train, y_new_train):
    preds[test] = xt_clf.fit(X_new_train[train], y_new_train[train]).predict_proba(X_new_train[test])[:, 1]
    print("ROC AUC: ", roc_auc_score(y_new_train[test], preds[test]))
print(roc_auc_score(y_new_train, preds))

thresholds = np.linspace(0.01, 0.99, 50)
mcc = np.array([matthews_corrcoef(y_new_train, preds > thr) for thr in thresholds])
plt.plot(thresholds, mcc)
best_threshold = thresholds[mcc.argmax()]
print(mcc.max())

preds = (xt_clf.predict_proba(X_new_test)[:, 1] > best_threshold).astype(np.int8)

print("MC:", matthews_corrcoef(y_new_test, preds))
print(classification_report(y_true=y_new_test, y_pred=preds, digits=3))


#==========================================================
print("RFClassifier Classification Report:")
rf_clf = RandomForestClassifier(n_estimators=100, verbose=0, n_jobs=-1)
cv = StratifiedKFold(n_splits=3)
preds = np.ones(y_new_train.shape[0])
for train, test in cv.split(X_new_train, y_new_train):
    preds[test] = rf_clf.fit(X_new_train[train], y_new_train[train]).predict_proba(X_new_train[test])[:, 1]
    print("ROC AUC: ", roc_auc_score(y_new_train[test], preds[test]))
print(roc_auc_score(y_new_train, preds))

thresholds = np.linspace(0.01, 0.99, 50)
mcc = np.array([matthews_corrcoef(y_new_train, preds > thr) for thr in thresholds])
plt.plot(thresholds, mcc)
best_threshold = thresholds[mcc.argmax()]
print(mcc.max())

preds = (rf_clf.predict_proba(X_new_test)[:, 1] > best_threshold).astype(np.int8)

print("MC:", matthews_corrcoef(y_new_test, preds))
print(classification_report(y_true=y_new_test, y_pred=preds, digits=3))
