##### Hyperparameter Optimization with RandomSearchCV

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

rf_model = RandomForestClassifier(random_state=17)

rf_random_params = {'max_depth': np.random.randint(5,50,10),
                  "max_features":[3,5,7,"auto","sqrt"],
                   'min_samples_split': np.random.randint(2,50,20),
                   'n_estimators': [int (x) for x in np.linspace(start=200,stop=1500,num=10)]}

rf_random = RandomizedSearchCV(estimator=rf_model,
                              param_distributions=rf_random_params,
                              n_iter=100,
                              cv=3,
                              verbose=True,
                              random_state=42,
                              n_jobs=-1)

rf_random.fit(X,y)

rf_random_final = rf_model.set_params(**rf_random.best_params_,random_state = 17).fit(X,y)

cv_results = cross_validate(rf_random_final,
                           X,
                           y,
                           cv=5,
                           scoring= ["accuracy","f1","roc_auc"])

print(cv_results["test_accuracy"].mean())

print(cv_results["test_f1"].mean())

print(cv_results["test_roc_auc"].mean())

## RandomSearchCV'den elde ettiğimiz hiperparametreleri GridSearchCV'de deneyebiliriz.

rf_params = {'max_depth': [30,35,45,50,65],
                  "max_features":["auto","sqrt"],
                   'min_samples_split': [2,5,7,11],
                   'n_estimators': [550,600,650,750,800]}

rf_best_grid = GridSearchCV(rf_random_final,
                             rf_params,
                             cv=5,
                             n_jobs=-1,
                             verbose=True).fit(X,y)

rf_final = rf_model.set_params(**rf_best_grid.best_params_,random_state = 17).fit(X,y)

cv_results = cross_validate(rf_final,
                           X,
                           y,
                           cv=5,
                           scoring= ["accuracy","f1","roc_auc"])

print(cv_results["test_accuracy"].mean())

print(cv_results["test_f1"].mean())

print(cv_results["test_roc_auc"].mean())

