import csv
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# data = pd.read_csv("raw_data/bosch-production-line-performance/train_numeric.csv")
# data = data.iloc[:100000, :]
data = pd.read_csv("train_data.csv")

X, y = data.iloc[:, 2:-1], data.iloc[:, -1]

data_dmatrix = xgb.DMatrix(data=X, label=y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

# xgbst = xgb.XGBClassifier(objective='binary:logistic', colsample_bytree=0.3, colsample_bylevel=0.3, learning_rate=0.01, max_depth=100, reg_lambda=10, n_estimators=1000, base_score=0.005, gamma=1, n_jobs=5)
xgbst = xgb.XGBClassifier(base_score=0.005, n_jobs=5)

xgbst.fit(X_train, y_train)

# print(xgbst.feature_importances_)
plt.hist(xgbst.feature_importances_[xgbst.feature_importances_ > 0])
important_indices = np.where(xgbst.feature_importances_ > 0.005)[0]
print(important_indices)
print(len(important_indices))
plt.title('Histogram of feature importance')
plt.ylabel('absolut number')
plt.xlabel('feature importance')
plt.show()
print(X_train.head())
important_indices = np.append(important_indices, data.shape[1] - 2)
print(important_indices)
print(len(important_indices))

new_data = pd.read_csv("raw_data/bosch-production-line-performance/train_numeric.csv", index_col=0, dtype=np.float32, usecols=important_indices)

print(new_data.head())
print(new_data.shape)

# new_data.to_csv('train_data_whole_reduced.csv')
