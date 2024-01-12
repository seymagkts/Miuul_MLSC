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

plot_importance(gbm_final,X)

def val_curve_params(model, X, y, param_name, param_range, scoring="roc_auc",cv=10):
    train_socre, test_score = validation_curve(model,
                                              X=X, y=y,
                                              param_name=param_name,
                                              param_range=param_range,
                                              cv=cv)
    mean_train_score = np.mean(train_socre, axis=1)
    mean_test_score = np.mean(test_score, axis=1)
    
    plt.plot(param_range, mean_train_score, label = "Training Score",color="b")
    plt.plot(param_range, mean_test_score, label = "Validation Score",color="g")
    
    plt.title(f"Validation Curve for {type(model).__name__}")
    plt.xlabel(f"Number of {param_name}")
    plt.ylabel(f"{scoring}")
    plt.tight_layout()
    plt.show(block=True)

gbm_params_arr = [["learning_rate",[0.01,0.1]],
                 ['max_depth', [3,8,10]],
                  ['n_estimators', [100,500,1000]],
                  ["subsample",[1,0.5,0.7]]]


for i in range(len(gbm_params_arr)):
    val_curve_params(gbm_final, X, y, gbm_params_arr[i][0], gbm_params_arr[i][1])
