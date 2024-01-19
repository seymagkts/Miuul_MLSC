##### Telco Customer Churn Makine Öğrenmesi

###### CustomerId : Müşteri İd’si
###### Gender : Cinsiyet
###### SeniorCitizen : Müşterinin yaşlı olup olmadığı (1, 0)
###### Partner : Müşterinin bir ortağı olup olmadığı (Evet, Hayır) ? Evli olup olmama
###### Dependents : Müşterinin bakmakla yükümlü olduğu kişiler olup olmadığı (Evet, Hayır) (Çocuk, anne, baba, büyükanne)
###### tenure : Müşterinin şirkette kaldığı ay sayısı
###### PhoneService : Müşterinin telefon hizmeti olup olmadığı (Evet, Hayır)
###### MultipleLines : Müşterinin birden fazla hattı olup olmadığı (Evet, Hayır, Telefon hizmeti yok)
###### InternetService : Müşterinin internet servis sağlayıcısı (DSL, Fiber optik, Hayır)
###### OnlineSecurity : Müşterinin çevrimiçi güvenliğinin olup olmadığı (Evet, Hayır, İnternet hizmeti yok)
###### OnlineBackup : Müşterinin online yedeğinin olup olmadığı (Evet, Hayır, İnternet hizmeti yok)
###### DeviceProtection : Müşterinin cihaz korumasına sahip olup olmadığı (Evet, Hayır, İnternet hizmeti yok)
###### TechSupport : Müşterinin teknik destek alıp almadığı (Evet, Hayır, İnternet hizmeti yok)
###### StreamingTV : Müşterinin TV yayını olup olmadığı (Evet, Hayır, İnternet hizmeti yok) Müşterinin, bir üçüncü taraf sağlayıcıdan televizyon programları yayınlamak için İnternet hizmetini kullanıp kullanmadığını gösterir
###### StreamingMovies : Müşterinin film akışı olup olmadığı (Evet, Hayır, İnternet hizmeti yok) Müşterinin bir üçüncü taraf sağlayıcıdan film akışı yapmak için İnternet hizmetini kullanıp kullanmadığını gösterir
###### Contract : Müşterinin sözleşme süresi (Aydan aya, Bir yıl, İki yıl)
###### PaperlessBilling : Müşterinin kağıtsız faturası olup olmadığı (Evet, Hayır)
###### PaymentMethod : Müşterinin ödeme yöntemi (Elektronik çek, Posta çeki, Banka havalesi (otomatik), Kredi kartı (otomatik))
###### MonthlyCharges : Müşteriden aylık olarak tahsil edilen tutar
###### TotalCharges : Müşteriden tahsil edilen toplam tutar
###### Churn : Müşterinin kullanıp kullanmadığı (Evet veya Hayır)

import warnings
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder,RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import GridSearchCV, cross_validate, RandomizedSearchCV, validation_curve
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import RidgeClassifier,LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

pd.set_option("display.max_columns", None)
warnings.simplefilter(action="ignore",category=Warning)

df = pd.read_csv("Telco-Customer-Churn.csv")

### Görev 1 : Keşifçi Veri Analizi
#### Adım 1: Numerik ve kategorik değişkenleri yakalayınız.


def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

check_df(df)


def grab_col_names(df, cat_th=10, car_th=20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        df: df
                Değişken isimleri alınmak istenilen df
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

    """


    # cat_cols, cat_but_car
    cat_cols = [col for col in df.columns if df[col].dtypes == "O"]
    num_but_cat = [col for col in df.columns if df[col].nunique() < cat_th and
                   df[col].dtypes != "O"]
    cat_but_car = [col for col in df.columns if df[col].nunique() > car_th and
                   df[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in df.columns if df[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {df.shape[0]}")
    print(f"Variables: {df.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car, num_but_cat

cat_cols, num_cols, cat_but_car, num_but_cat = grab_col_names(df)

#### Adım 2: Gerekli düzenlemeleri yapınız. (Tip hatası olan değişkenler gibi)

# TotalCharges sayısal bir değişken olmalı
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors='coerce')

# yes ise 1 no ise 0 numeriklestirme
df["Churn"] = df["Churn"].apply(lambda x : 1 if x == "Yes" else 0)

#### Adım 3: Numerik ve kategorik değişkenlerin veri içindeki dağılımını gözlemleyiniz.


cat_cols, num_cols, cat_but_car, num_but_cat = grab_col_names(df)

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")

    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)

for col in cat_cols:
    cat_summary(df,col)

def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]

    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)

for col in num_cols:
    num_summary(df,col,True)

####  Adım 4: Kategorik ve numerik değişkenler ile hedef değişken incelemesini yapınız.

def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")

for col in num_cols:
    target_summary_with_num(df, "Churn", col)

def target_summary_with_cat(dataframe, target, categorical_col):
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean()}), end="\n\n\n")

for col in cat_cols:
    target_summary_with_cat(df,"Churn",col)

#### Adım 5: Aykırı gözlem var mı inceleyiniz.

def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def check_outlier(dataframe, col_name, q1=0.05, q3=0.95):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name, q1, q3)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

for col in num_cols:
    print(col,check_outlier(df,col))

#### Adım 6: Eksik gözlem var mı inceleyiniz

df.isnull().sum()

### Görev 2 : Feature Engineering
#### Adım 1: Eksik ve aykırı gözlemler için gerekli işlemleri yapınız.


df["TotalCharges"] = df.TotalCharges.fillna(df["MonthlyCharges"] * 12)

#### Adım 2: Yeni değişkenler oluşturunuz.

# aylık ödeme
df["New_Avg_Charges"] = df.TotalCharges / (df.tenure + 1)

#
df["New_Gen_PhoneService"] = df.apply(lambda x:1 if ((x["gender"]=="Female") or
                                                       (x["PhoneService"]=="1")) else 0,axis=1)

# Herhangi bir streaming hizmeti alan kişiler
df["New_Flag_Any_Streaming"] = df.apply(lambda x:1 if ((x["StreamingTV"]=="Yes") or
                                                       (x["StreamingMovies"]=="Yes")) else 0,axis=1)

# Tenure  değişkeninden yıllık kategorik değişken oluşturma
df.loc[(df["tenure"]>=0) & (df["tenure"]<=12),"New_Tenure_Year"] = "New"
df.loc[(df["tenure"]>=12) & (df["tenure"]<=24),"New_Tenure_Year"] = "Jr"
df.loc[(df["tenure"]>=24) & (df["tenure"]<=36),"New_Tenure_Year"] = "Little Mid"
df.loc[(df["tenure"]>=36) & (df["tenure"]<=48),"New_Tenure_Year"] = "Mid"
df.loc[(df["tenure"]>=48) & (df["tenure"]<=60),"New_Tenure_Year"] = "Senior"
df.loc[(df["tenure"]>=60) & (df["tenure"]<=72),"New_Tenure_Year"] = "Expert"

# Kişinin toplam aldığı servis sayısı
df['New_TotalService'] = (df[["PhoneService", "InternetService", "OnlineSecurity",
                                       'OnlineBackup', 'DeviceProtection', 'TechSupport',
                                       'StreamingTV', "StreamingMovies"]] == "Yes").sum(axis=1)

#  güncel fiyatın ortalama fiyata göre artışı
df["New_Charges_Rating"] = df.New_Avg_Charges / df.MonthlyCharges

# Aylık sözleşmesi bulunan ve yaşlı olan müşteriler
df["New_Old_Month"] = df.apply(lambda x:1 if ((x["Contract"] == "Month-to-month") and
                                                      (x["SeniorCitizen"]==1)) else 0, axis=1)

#### Adım 3: Encoding işlemlerini gerçekleştiriniz.

cat_cols, num_cols, cat_but_car, num_but_cat = grab_col_names(df)

def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe


def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

binary_cols = [col for col in df.columns if df[col].dtypes == "O" and df[col].nunique() == 2]

for col in binary_cols:
    df = label_encoder(df,col)

ohe_cols = [col for col in df.columns if 10 >= df[col].nunique() > 2]

df = one_hot_encoder(df,ohe_cols,True)

#### Adım 4: Numerik değişkenler için standartlaştırma yapınız.

scaler = RobustScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

### Görev 3 : Modelleme
#### Adım 1: Sınıflandırma algoritmaları ile modeller kurup, accuracy skorlarını inceleyip. En iyi 4 modeli seçiniz.

y = df.Churn
X = df.drop(["Churn","customerID"],axis=1)

models = [('LR', LogisticRegression(random_state=12345)),
          ('KNN', KNeighborsClassifier()),
          ('CART', DecisionTreeClassifier(random_state=12345)),
          ('RF', RandomForestClassifier(random_state=12345)),
          ('GBM', GradientBoostingClassifier(random_state=12345)),
          ("XGBoost", XGBClassifier(random_state=12345)),
          ("LightGBM", LGBMClassifier(random_state=12345)),
          ("CatBoost", CatBoostClassifier(verbose=False,random_state=12345))]


for name, model in models:
    cv_results = cross_validate(model, X, y, cv=10, scoring=["accuracy", "f1", "roc_auc", "precision", "recall"])
    print(f"########## {name} ##########")
    print(f"Accuracy: {round(cv_results['test_accuracy'].mean(), 4)}")


#### Adım 2: Seçtiğiniz modeller ile hiperparametre optimizasyonu gerçekleştirin ve bulduğunuz hiparparametreler ile modeli tekrar kurunuz

#### GBM (GBM 0.8036)

gbm_model = GradientBoostingClassifier(random_state=12345)

gbm_model.get_params()

gbm_params = {'max_depth':  [2,3,4],
              "learning_rate":[0.01,0.1,0.08],
             'n_estimators': [40,45,50,100],
             'subsample':[1,3,5]}

gbm_best_grid = GridSearchCV(gbm_model,
                              gbm_params,
                               cv=10,
                             n_jobs=-1,
                             verbose=1).fit(X,y)

print(gbm_best_grid.best_params_)

gbm_final = gbm_model.set_params(**gbm_best_grid.best_params_,random_state = 12345).fit(X,y)

cv_results = cross_validate(gbm_model, X, y, cv=10, scoring=["accuracy", "f1", "roc_auc", "precision", "recall"])

cv_results["test_accuracy"].mean()

# 0.8044840022566087

#### Logistic Regression (LR 0.8068)

lr_params =  LogisticRegression(solver='saga',penalty='l2',max_iter=1000,random_state=12345).fit(X,y)

cv_results = cross_validate(lr_params, X, y, cv=5, scoring=["accuracy"])

cv_results["test_accuracy"].mean()

# 0.8064756193947996

#### LightGBM (LGBM 0.797)

lgbm_model = LGBMClassifier(random_state=12345)

lgbm_model.get_params()

lgbm_params = {"learning_rate":[0.01,0.02,0.04,0.1],
                 'n_estimators': [100,250,300,350,400],
                 "colsample_bytree":[1,0.8,0.9]}

lgbm_best_grid = GridSearchCV(lgbm_model,
                              lgbm_params,
                               cv=10,
                             n_jobs=-1,
                             verbose=1).fit(X,y)

print(lgbm_best_grid.best_params_)

lgbm_final = lgbm_model.set_params(**lgbm_best_grid.best_params_).fit(X,y)

cv_results = cross_validate(lgbm_final, X, y, cv=10, scoring=["accuracy"])

cv_results["test_accuracy"].mean()

# 0.8051950354609929

#### CatBoost (CatBoost 0.7991)

catboost_model = CatBoostClassifier(random_state=12345)

catboost_params = {	"iterations" : [500,600,650,7000],
                    "learning_rate": [0.01,0.1],
                    "depth" : [3,6,8,11]}

catboost_best_grid = GridSearchCV(catboost_model,
                              catboost_params,
                               cv=10,
                             n_jobs=-1,
                             verbose=True).fit(X,y)

print(catboost_best_grid.best_params_)

catboost_final = catboost_model.set_params(**catboost_best_grid.best_params_).fit(X,y)

cv_results = cross_validate(catboost_final, X, y, cv=10, scoring=["accuracy"])

cv_results["test_accuracy"].mean()

# 0.8084604690522245

##### Feature Importance

def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')

plot_importance(gbm_final, X)

plot_importance(lr_params, X)

plot_importance(lgbm_final, X)

plot_importance(catboost_final, X)

