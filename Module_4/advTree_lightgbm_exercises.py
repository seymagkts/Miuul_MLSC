##### LightGBM

import warnings
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import GridSearchCV, cross_validate, RandomizedSearchCV, validation_curve

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

pd.set_option("display.max_columns", None)
warnings.simplefilter(action="ignore",category=Warning)

df = pd.read_csv("diabetes.csv")

y = df.Outcome
X = df.drop(["Outcome"], axis = 1)

lightgbm_model = LGBMClassifier(random_state=17).fit(X,y)

cv_results = cross_validate(lightgbm_model,
                           X,
                           y,
                           cv=10,
                           scoring= ["accuracy","f1","roc_auc"]  )

print(cv_results["test_accuracy"].mean())

print(cv_results["test_f1"].mean())

print(cv_results["test_roc_auc"].mean())


lightgbm_params = {"learning_rate":[0.01,0.1],
                 'n_estimators': [100,300,500,1000],
                 "colsample_bytree":[1,0.5,0.7]}

lightgbm_best_grid = GridSearchCV(lightgbm_model,
                             lightgbm_params,
                             cv=5,
                             n_jobs=-1,
                             verbose=1).fit(X,y)

print(lightgbm_best_grid.best_params_)


lgbm_final = lightgbm_model.set_params(**lightgbm_best_grid.best_params_,random_state=17).fit(X,y)

cv_results = cross_validate(lgbm_final,
                           X,
                           y,
                           cv=10,
                           scoring= ["accuracy","f1","roc_auc"]  )

print(cv_results["test_accuracy"].mean())

print(cv_results["test_f1"].mean())

print(cv_results["test_roc_auc"].mean())


#### Hiperparametre yeni değerle
lightgbm_model = LGBMClassifier(random_state=17).fit(X,y)


lightgbm_params = {"learning_rate":[0.01,0.02,0.05,0.1],
                 'n_estimators': [200,300,350,400],
                 "colsample_bytree":[1,0.8,0.9]}

lightgbm_best_grid = GridSearchCV(lightgbm_model,
                             lightgbm_params,
                             cv=5,
                             n_jobs=-1,
                             verbose=1).fit(X,y)

print(lightgbm_best_grid.best_params_)

lgbm_final = lightgbm_model.set_params(**lightgbm_best_grid.best_params_,random_state=17).fit(X,y)

cv_results = cross_validate(lgbm_final,
                           X,
                           y,
                           cv=10,
                           scoring= ["accuracy","f1","roc_auc"]  )

print(cv_results["test_accuracy"].mean())

print(cv_results["test_f1"].mean())

print(cv_results["test_roc_auc"].mean())


#### Hiperparametre optimizasyonu sadece n_estimators için.
#### bunu yapmanın farkı daha büyük gözlem sayısına sahip verisetlerinde daha iyi ortaya cıkar.
lightgbm_model = LGBMClassifier(random_state=17,colsample_bytree= 0.9, learning_rate= 0.01).fit(X,y)

lightgbm_params = {'n_estimators': [200,300,500,1000,5000,8000,10000]}

lightgbm_best_grid = GridSearchCV(lightgbm_model,
                             lightgbm_params,
                             cv=5,
                             n_jobs=-1,
                             verbose=1).fit(X,y)

print(lightgbm_best_grid.best_params_)

lgbm_final = lightgbm_model.set_params(**lightgbm_best_grid.best_params_,random_state=17).fit(X,y)

cv_results = cross_validate(lgbm_final,
                           X,
                           y,
                           cv=10,
                           scoring= ["accuracy","f1","roc_auc"]  )

print(cv_results["test_accuracy"].mean())

print(cv_results["test_f1"].mean())

print(cv_results["test_roc_auc"].mean())

# değişkenlerin önem sırasına gore sıralar ve değerlerini verir
def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    print(feature_imp.sort_values("Value",ascending=False))
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show(block=True)
    if save:
        plt.savefig('importances.png')

plot_importance(lgbm_final,X)
