Eight code files are provided for this project. The results shown in the report can be reproduced by using those files in combination with the data (available at https://www.kaggle.com/c/bosch-production-line-performance/data) and the parameter-settings explained in the text.

Three files show the applied feature selection methods.
- feature_selection_modelbased.py
- feature_selection_univariate.py
- feature_selection_xgb.py

In same cases the feature selection files were taken to create new datasets that were then used by the models.

The files for the models are:
- isolation_forest.py
- random_forest_numeric.py
- random_forest_numeric_date.py
- xgb_numeric.py
- xgb_numeric_date.py