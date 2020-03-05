import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report, confusion_matrix, roc_auc_score, matthews_corrcoef, make_scorer, recall_score
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from mlxtend.plotting import plot_confusion_matrix
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV

#data = pd.read_csv("raw_data/bosch-production-line-performance/train_numeric.csv")
# data = data.iloc[:100000, :]
data = pd.read_csv("train_data_whole_reduced.csv")
data = data.iloc[:20000, :]

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
new_data = imputer.fit_transform(data)
df = pd.DataFrame(new_data, columns=data.columns, index=data.index)
data = df

X, y = data.iloc[:, 2:-1], data.iloc[:, -1]

X, X_pred, y, y_pred = train_test_split(X, y, test_size=0.1)

#print("Shape of X_train:", X_train.shape)

clf = RandomForestClassifier(n_estimators=500, n_jobs=5, max_depth=32)

cv = StratifiedKFold(n_splits=3)
count = 1
for train_index, test_index in cv.split(X, y):
    print("Fold", count)
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    #pred[pred > 0] = 0
    #pred[pred < 0] = 1
    #print("ROC:", roc_auc_score(y_test, pred))
    #print("Accuracy:", accuracy_score(y_test, pred))
    print(confusion_matrix(y_test, pred, labels=[0, 1]))
    #print(classification_report(y_test, pred))
    count += 1

preds = clf.predict(X_pred)
preds[preds > 0] = 0
preds[preds < 0] = 1
print("ROC:", roc_auc_score(y_pred, preds))
print("Accuracy:", accuracy_score(y_pred, preds))
print(confusion_matrix(y_pred, preds, labels=[0, 1]))
print(classification_report(y_pred, preds))


# print("ROC:", roc_auc_score(y_pred, preds))
# print("Accuracy:", accuracy_score(y_pred, preds))
# print(confusion_matrix(y_pred, preds, labels=[0, 1]))
# print(classification_report(y_pred, preds))


# PRINT CONFUSION MATRIX
# fig, ax = plot_confusion_matrix(cm, show_absolute=True, show_normed=True, colorbar=True)
# plt.show()

# http://rasbt.github.io/mlxtend/user_guide/plotting/plot_confusion_matrix/
# cm2 = np.array([[1, 6], [692, 1669]])
# fig2, ax2 = plot_confusion_matrix(cm2, show_absolute=True, show_normed=True, colorbar=True)
# plt.show()


scoring = {'MCC': make_scorer(matthews_corrcoef)}
# scoring = {'score': make_scorer(recall_score(average='macro'))}
n_est = [50, 100, 150, 200, 250, 300]
m_dep = [5, 10, 15, 20, 25]
gs = GridSearchCV(RandomForestClassifier(n_jobs=5), dict(n_estimators=n_est, max_depth=m_dep), scoring=scoring, refit='MCC', cv=3)
# gs = GridSearchCV(IsolationForest(n_jobs=5, behaviour='new'), dict(n_estimators=n_est, contamination=cont), scoring='f1_macro', refit='f1_macro', cv=3)

# print(y_train)
gs.fit(X, y)
results = gs.cv_results_
print(gs.best_params_, gs.best_score_)


def plot_grid_search(cv_results, grid_param_1, grid_param_2, name_param_1, name_param_2):
    # Get Test Scores Mean and std for each grid search
    scores_mean = cv_results['mean_test_MCC']
    scores_mean = np.array(scores_mean).reshape(len(grid_param_2), len(grid_param_1))

    scores_sd = cv_results['std_test_MCC']
    scores_sd = np.array(scores_sd).reshape(len(grid_param_2), len(grid_param_1))

    # Plot Grid search scores
    _, ax = plt.subplots(1, 1)

    # Param1 is the X-axis, Param 2 is represented as a different curve (color line)
    for idx, val in enumerate(grid_param_2):
        ax.plot(grid_param_1, scores_mean[idx, :], '-o', label=name_param_2 + ': ' + str(val))

    ax.set_title("Grid Search Scores", fontsize=20, fontweight='bold')
    ax.set_xlabel(name_param_1, fontsize=16)
    ax.set_ylabel('MCC-Score', fontsize=16)
    ax.legend(loc="best", fontsize=15)
    ax.grid('on')
    plt.show()


#plot_grid_search(results, n_est, m_dep, 'N Estimators', 'Max. Depth')
