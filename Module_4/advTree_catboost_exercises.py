#### CatBoost

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

catboost_model = CatBoostClassifier(random_state=17,verbose=False)

cv_results = cross_validate(catboost_model,
                           X,
                           y,
                           cv=5,
                           scoring= ["accuracy","f1","roc_auc"]  )

print(cv_results["test_accuracy"].mean())

print(cv_results["test_f1"].mean())

print(cv_results["test_roc_auc"].mean())

print(catboost_model.get_params())

catboost_params = {'iterations': [200,500],
                  "learning_rate":[0.01,0.1],
                   'depth': [3,6]}

catboost_best_grid = GridSearchCV(catboost_model,
                             catboost_params,
                             cv=5,
                             n_jobs=-1,
                             verbose=True).fit(X,y)

print(catboost_best_grid.best_params_)

catboost_final = catboost_model.set_params(**catboost_best_grid.best_params_,random_state=17).fit(X,y)

cv_results = cross_validate(catboost_final,
                           X,
                           y,
                           cv=5,
                           scoring= ["accuracy","f1","roc_auc"] )

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

plot_importance(catboost_final,X)

