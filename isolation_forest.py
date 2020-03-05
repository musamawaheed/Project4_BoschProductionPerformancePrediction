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

data = pd.read_csv("train_data_filled.csv")
#data = pd.read_csv("raw_data/bosch-production-line-performance/train_numeric.csv")
#data = data.iloc[:100000, :]

#data = data.fillna(9999999)
# imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
# new_data = imputer.fit_transform(data)
# df = pd.DataFrame(new_data, columns=data.columns, index=data.index)
# data = df

X, y = data.iloc[:, 2:-1], data.iloc[:, -1]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

select = SelectFromModel(RandomForestClassifier(n_estimators=100, random_state=42), threshold=None)
select.fit(X_train, y_train)
X_train_selected = select.transform(X_train)
print("Shape of X_train:", X_train.shape)
print("Shape of X_train_selected:", X_train_selected.shape)
X_test_selected = select.transform(X_test)

clf = IsolationForest(n_estimators=1000, contamination=0.15, max_samples='auto', behaviour='new', n_jobs=5)
#clf.fit(X_train, y_train)
#pred = clf.predict(X_test)
# pred[pred > 0] = 0
# pred[pred < 0] = 1
# print(len(pred))
# print(sum(pred))

# y_test = np.array(y_test)

# print(classification_report(y_test, pred, target_names=['OK', 'NOT-OK']))

clf.fit(X_train_selected, y_train)
pred_selected = clf.predict(X_test_selected)
pred_selected[pred_selected > 0] = 0
pred_selected[pred_selected < 0] = 1
print(len(pred_selected))
print(sum(pred_selected))


print(classification_report(y_test, pred_selected, target_names=['OK', 'NOT-OK']))
#print('MCC, pred:', matthews_corrcoef(y_test, pred))
print('MCC, pred_selected:', matthews_corrcoef(y_test, pred_selected))
cm = confusion_matrix(y_test, pred_selected, labels=[0, 1])
print(cm)
# fig, ax = plot_confusion_matrix(cm, show_absolute=True, show_normed=True, colorbar=True)
# plt.show()

# http://rasbt.github.io/mlxtend/user_guide/plotting/plot_confusion_matrix/
# cm2 = np.array([[1, 6], [692, 1669]])
# fig2, ax2 = plot_confusion_matrix(cm2, show_absolute=True, show_normed=True, colorbar=True)
# plt.show()


# scoring = {'MCC': make_scorer(matthews_corrcoef)}
# scoring = {'score': make_scorer(recall_score(average='macro'))}
# n_est = [50, 100, 150, 200]
# cont = [0.01, 0.05, 0.1, 0.2]
# gs = GridSearchCV(IsolationForest(n_jobs=5, behaviour='new'), dict(n_estimators=n_est, contamination=cont), scoring=scoring, refit='score', cv=3)
# gs = GridSearchCV(IsolationForest(n_jobs=5, behaviour='new'), dict(n_estimators=n_est, contamination=cont), scoring='f1_macro', refit='f1_macro', cv=3)

# y_train[y_train > 0] = -1
# y_train[y_train < 1] = 1
# print(y_train)
# gs.fit(X_train_selected, y_train)
# results = gs.cv_results_
# print(gs.best_params_, gs.best_score_)


def plot_grid_search(cv_results, grid_param_1, grid_param_2, name_param_1, name_param_2):
    # Get Test Scores Mean and std for each grid search
    scores_mean = cv_results['mean_test_score']
    scores_mean = np.array(scores_mean).reshape(len(grid_param_2), len(grid_param_1))

    scores_sd = cv_results['std_test_score']
    scores_sd = np.array(scores_sd).reshape(len(grid_param_2), len(grid_param_1))

    # Plot Grid search scores
    _, ax = plt.subplots(1, 1)

    # Param1 is the X-axis, Param 2 is represented as a different curve (color line)
    for idx, val in enumerate(grid_param_2):
        ax.plot(grid_param_1, scores_mean[idx, :], '-o', label=name_param_2 + ': ' + str(val))

    ax.set_title("Grid Search Scores", fontsize=20, fontweight='bold')
    ax.set_xlabel(name_param_1, fontsize=16)
    ax.set_ylabel('CV Average Score', fontsize=16)
    ax.legend(loc="best", fontsize=15)
    ax.grid('on')
    plt.show()


# plot_grid_search(results, n_est, cont, 'N Estimators', 'Contamination')
