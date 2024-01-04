#### GBM (Gradient Boosting Machines)

import warnings
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import GridSearchCV, cross_validate, RandomizedSearchCV, validation_curve

pd.set_option("display.max_columns", None)
warnings.simplefilter(action="ignore",category=Warning)

df = pd.read_csv("diabetes.csv")

y = df.Outcome
X = df.drop(["Outcome"], axis = 1)

gbm_model = GradientBoostingClassifier(random_state=17).fit(X,y)

cv_results = cross_validate(gbm_model,
                           X,
                           y,
                           cv=5,
                           scoring= ["accuracy","f1","roc_auc"]  )

print(cv_results["test_accuracy"].mean())

print(cv_results["test_f1"].mean())

print(cv_results["test_roc_auc"].mean())

print(gbm_model.get_params())

gbm_params = {"learning_rate":[0.01,0.1],
            'max_depth': [3,8,10],
             'n_estimators': [100,500,1000],
             "subsample":[1,0.5,0.7]}

gbm_best_grid = GridSearchCV(gbm_model,
                             gbm_params,
                             cv=5,
                             n_jobs=-1,
                             verbose=1).fit(X,y)

print(gbm_best_grid.best_params_)

gbm_final = gbm_model.set_params(**gbm_best_grid.best_params_,random_state=17).fit(X,y)

cv_results = cross_validate(gbm_final,
                           X,
                           y,
                           cv=5,
                           scoring= ["accuracy","f1","roc_auc"] )
print(cv_results["test_accuracy"].mean())

print(cv_results["test_f1"].mean())

print(cv_results["test_roc_auc"].mean())


