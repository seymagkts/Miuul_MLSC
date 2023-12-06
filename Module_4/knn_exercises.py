## KNN

#### Diabetes Dataset

##### 1. Exploratory Data Analysis
##### 2. Data Preprocessing & Feature Engineering
##### 3. Model & Prediction
##### 4. Model Evaluation
##### 5. Hyperparameter Optimization
##### 6. Final Model

import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
pd.set_option("display.max_columns", None)

### 1. Exploratory Data Analysis

df = pd.read_csv("diabetes.csv")

print(df.head())

print(df.shape)

print(df.describe().T)

df["Outcome"].value_countsounts()

### 2. Data Preprocessing & Feature Engineering

y = df["Outcome"]
X = df.drop(["Outcome"],axis=1)

## gradient descent ve uzaklık temelli yöntemlerde standartlaştırma önemlidir
X_scaled = StandardScaler().fit_transform(X)

X = pd.DataFrame(X_scaled,columns=X.columns)

### 3. Model & Prediction

## bir modeli eğitmek ayrı bir şey tahmin etmek ayrı bir şey

knn_model = KNeighborsClassifier().fit(X,y) # eğitmek

random_user=X.sample(1,random_state=45) # veriden rastgele seçim yapmak

knn_model.predict(random_user) # tek bir gözlem için tahmin etmek

### 4. Model Evaluation

# confusion matrix için y pred
y_pred = knn_model.predict(X)

# Auc için y prob
y_prob = knn_model.predict_proba(X)[:,1]

# basarı degerleme metrikleri
print(classification_report(y,y_pred))

# AUC değeri
roc_auc_score(y,y_prob)

# 5 katlı capraz dogrulama
# capraz doğrulama sonucları en güvenli sonuclardır
cv_results = cross_validate(knn_model, X,y,cv=5,scoring=("accuracy","f1","roc_auc"))

cv_results["test_accuracy"].mean()

cv_results["test_f1"].mean()

cv_results["test_roc_auc"].mean()

##### Başarı sonuçları nasıl arttırılır?
##### 1. örnek/gözlem boyutu arttırılabilir
##### 2. veri ön işleme işlemleri detaylandırılabilir
##### 3. özellik mühendisliği yeni değişkenler türetilebilir
##### 4. ilgili algoritma için optimizasyonlar yapılablir.

knn_model.get_params()

### 5. Hyperparameter Optimization

# en optimumum komsuluk sayısını ararız.

knn_model = KNeighborsClassifier()
knn_model.get_params()

knn_params = {"n_neighbors" : range(2,50)} # isim parametreyle aynı adda olmalı

knn_gs_best = GridSearchCV(knn_model,
                           knn_params,
                           cv=5,
                           n_jobs=-1,verbose=1).fit(X,y)
# 5 katlı doğrulama bu kez hatalar üzerinde islem yapar
# n_jobs -1 demek maksimumu işlemci kullanmak demektir
# verbose 1 demek rapor istemek demek
# GridSearhCV, verilen range değerleri ile baştan sona gezerek en min hatayı veren hipermarametre değerini best olarak cıkartır.


print(knn_gs_best.best_params_)
## GridSearchCV en az hatayı 17 komsuluk sayısında alacağımızı belirledi

### 6. Final Model

# sozluk değeri atamasını ** ile direkt yapabiliriz.

knn_final = knn_model.set_params(**knn_gs_best.best_params_).fit(X,y)

cv_results = cross_validate(knn_model,
                            X,
                            y,
                            cv=5,scoring=("accuracy","f1","roc_auc"))

cv_results["test_accuracy"].mean()

cv_results["test_f1"].mean()

cv_results["test_roc_auc"].mean()

random_user=X.sample(1)

knn_final.predict(random_user)

