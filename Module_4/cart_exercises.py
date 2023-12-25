### CART

# pip install pydotplus, skompiler, astor, joblib

import warnings
import joblib
import pydotplus
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeClassifier, export_graphviz, export_text
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate, validation_curve
from skompiler import skompile

pd.set_option("display.max_columns", None)
warnings.simplefilter(action="ignore", category=Warning)

##### 1. Exploratory Data Analysis


##### 2. Data Preprocessing & Feature Engineering


##### 3. Modeling using CART

df = pd.read_csv("diabetes.csv")

y = df.Outcome
X = df.drop(["Outcome"], axis=1)

cart_model = DecisionTreeClassifier(random_state=17).fit(X, y)

y_pred = cart_model.predict(X)

# confusion matrix
print(classification_report(y, y_pred))  ### overfit?
y_prob = cart_model.predict_proba(X)[:, 1]
roc_auc = roc_auc_score(y, y_prob)

print(roc_auc)

# holdout yöntemi
# train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=17)
cart_model = DecisionTreeClassifier(random_state=17).fit(X_train, y_train)

y_pred = cart_model.predict(X_train)
y_prob = cart_model.predict_proba(X_train)[:, 1]

print(classification_report(y_train, y_pred))
roc_auc = roc_auc_score(y_train, y_prob)
print(roc_auc)

# test - overfit +
y_pred = cart_model.predict(X_test)
y_prob = cart_model.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred))
roc_auc = roc_auc_score(y_test, y_prob)
print(roc_auc)

# CV ile başarı değerleme
cart_model = DecisionTreeClassifier(random_state=17).fit(X,
                                                         y)  # fit işlemi cros validate yaparken anlam ifade etmediği için yapıladabilir yapılmayadabilir.
cv_results = cross_validate(cart_model,
                            X,
                            y,
                            cv=5,
                            scoring=["accuracy", "f1", "roc_auc"])

cv_results["test_accuracy"].mean()
cv_results["test_f1"].mean()
cv_results["test_roc_auc"].mean()

##### 4. Hyperparameter Optimization with GridSearchCV

cart_model.get_params()
# overfitin önüne geçebilecek parametreler
# min_samples_split, bölme işlemini doğrudan etkiler
# max_depth

cart_params = {"max_depth": range(1, 11),
               "min_samples_split": range(2, 20)}  # ön tanımlı değerlere göre rangelar belirlenir

cart_best_grid = GridSearchCV(cart_model,
                              cart_params,
                              # scoring="roc_auc", parametre değerlerini verilen scoring değerini min edicek bicimde hesaplar
                              cv=5,
                              n_jobs=-1,
                              verbose=1).fit(X, y)

print(cart_best_grid.best_params_)
print(cart_best_grid.best_score_)

random_user = X.sample(1, random_state=35)
cart_best_grid.predict(random_user)

##### 5. Final Model

# final modeli kurmak için 2 alternatif kullanacağız

# 1 yeni parametrelerle bastan model kurmak
cart_final = DecisionTreeClassifier(**cart_best_grid.best_params_, random_state=17, ).fit(X, y)
cart_final.get_params()

# 2 kurulmus olan modele yeni parametreleri set etmek
# cart_final = cart_model.set_params(**cart_best_grid.best_params_).fit(X,y)

cv_results = cross_validate(cart_final,
                            X,
                            y,
                            cv=5,
                            scoring=["accuracy", "f1", "roc_auc"])

cv_results["test_roc_auc"].mean()
cv_results["test_accuracy"].mean()
cv_results["test_f1"].mean()

##### 6. Feature Importance

print(cart_final.feature_importances_)


# değişkenlerin önem sırasına gore sıralar ve değerlerini verir
def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    print(feature_imp.sort_values("Value", ascending=False))
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show(block=True)
    if save:
        plt.savefig('importances.png')


plot_importance(cart_final, X)  # num parametresine n değerini verirsen ilk n sıradakine kadar gösterir


##### 7. Analyzing Model Complexity with Learning Curves

# bu alan hiperparametre optimizasyonlarında elde edilen sonucun
# görselidir, ancak tek değere göre baktığımız için aynı sonuc
# cıkmayabilir. optimizasyonda eş anlı olarak parametrelere bakıyoruz.

def val_curve_params(model, X, y, param_name, param_range, scoring="roc_auc", cv=10):
    train_socre, test_score = validation_curve(model,
                                               X=X, y=y,
                                               param_name=param_name,
                                               param_range=param_range,
                                               cv=cv)
    mean_train_score = np.mean(train_socre, axis=1)
    mean_test_score = np.mean(test_score, axis=1)

    plt.plot(param_range, mean_train_score, label="Training Score", color="b")
    plt.plot(param_range, mean_test_score, label="Validation Score", color="g")

    plt.title(f"Validation Curve for {type(model).__name__}")
    plt.xlabel(f"Number of {param_name}")
    plt.ylabel(f"{scoring}")
    plt.tight_layout()
    plt.show(block=True)


val_curve_params(cart_final, X, y,
                 "max_depth",
                 range(1, 11),
                 scoring="f1")

cart_val_params = [["max_depth", range(1, 11)], ["min_samples_split", range(2, 20)]]

for i in range(len(cart_val_params)):
    val_curve_params(cart_final, X, y,
                     cart_val_params[i][0],
                     cart_val_params[i][1])

##### 8. Visualizing the Decision Tree

# conda install graphviz

import graphviz


def tree_graph(model, col_names, file_name):
    tree_str = export_graphviz(model, feature_names=col_names, filled=True, out_file=None)
    graph = pydotplus.graph_from_dot_data(tree_str)
    graph.write_png(file_name)


tree_graph(model=cart_final, col_names=X.columns, file_name="cart_final.png")

##### 9. Extracting Decision Rules

# karar kuralları
tree_rules = export_text(cart_final, feature_names=list(X.columns))

print(tree_rules)

##### 10. Extracting Python Codes of Decision Rules

import sklearn

print(sklearn.__version__)

# kararların python kodu hali
print(skompile(cart_final.predict).to("python/code"))

# kararların SQL kodu hali
print(skompile(cart_final.predict).to("sqlalchemy/sqlite"))

# kararların excel kodu hali
print(skompile(cart_final.predict).to("excel"))


##### 11. Prediction using Python Codes

def predict_with_rules(x):
    return ((((((0 if x[0] <= 7.5 else 1) if x[5] <= 30.949999809265137 else 0 if x[6] <=
    0.5005000084638596 else 0) if x[5] <= 45.39999961853027 else 1 if x[2] <=
    99.0 else 0) if x[7] <= 28.5 else (1 if x[5] <= 9.649999618530273 else 
    0) if x[5] <= 26.350000381469727 else (1 if x[1] <= 28.5 else 0) if x[1
    ] <= 99.5 else 0 if x[6] <= 0.5609999895095825 else 1) if x[1] <= 127.5
     else (((0 if x[5] <= 28.149999618530273 else 1) if x[4] <= 132.5 else 
    0) if x[1] <= 145.5 else 0 if x[7] <= 25.5 else 1 if x[7] <= 61.0 else 
    0) if x[5] <= 29.949999809265137 else ((1 if x[2] <= 61.0 else 0) if x[
    7] <= 30.5 else 1 if x[6] <= 0.4294999986886978 else 1) if x[1] <= 
    157.5 else (1 if x[6] <= 0.3004999905824661 else 1) if x[4] <= 629.5 else 0)
    )


random_user = X.sample(1, random_state=35)

predict_with_rules(random_user.values[0])

##### 12. Saving and Loading Model

# kurulan modelin çıktısı alınır, paylaşmına hazır hale getirmek
# baştan bütün kodları çalıştırıp tekrar aynı işlemleri yapmayı önler
# veritabanı ortamından cıkmadan işlemlerin devamlılığını sağlar

joblib.dump(cart_final, "cart_final.pkl")

cart_model_from_disc = joblib.load("cart_final.pkl")

x = [12, 13, 20, 23, 4, 55, 12, 7]

cart_model_from_disc.predict(pd.DataFrame(x).T)
