### TELCO CHURN FEATURE ENGINEERING

## CustomerId : Müşteri İd’si
## Gender : Cinsiyet
## SeniorCitizen : Müşterinin yaşlı olup olmadığı (1, 0)
## Partner : Müşterinin bir ortağı olup olmadığı (Evet, Hayır) ? Evli olup olmama
## Dependents : Müşterinin bakmakla yükümlü olduğu kişiler olup olmadığı (Evet, Hayır) (Çocuk, anne, baba, büyükanne)
## tenure : Müşterinin şirkette kaldığı ay sayısı
## PhoneService : Müşterinin telefon hizmeti olup olmadığı (Evet, Hayır)
## MultipleLines : Müşterinin birden fazla hattı olup olmadığı (Evet, Hayır, Telefon hizmeti yok)
## InternetService : Müşterinin internet servis sağlayıcısı (DSL, Fiber optik, Hayır)
## OnlineSecurity : Müşterinin çevrimiçi güvenliğinin olup olmadığı (Evet, Hayır, İnternet hizmeti yok)
## OnlineBackup : Müşterinin online yedeğinin olup olmadığı (Evet, Hayır, İnternet hizmeti yok)
## DeviceProtection : Müşterinin cihaz korumasına sahip olup olmadığı (Evet, Hayır, İnternet hizmeti yok)
## TechSupport : Müşterinin teknik destek alıp almadığı (Evet, Hayır, İnternet hizmeti yok)
## StreamingTV : Müşterinin TV yayını olup olmadığı (Evet, Hayır, İnternet hizmeti yok) Müşterinin, bir üçüncü taraf sağlayıcıdan televizyon programları yayınlamak için İnternet hizmetini kullanıp kullanmadığını gösterir
## StreamingMovies : Müşterinin film akışı olup olmadığı (Evet, Hayır, İnternet hizmeti yok) Müşterinin bir üçüncü taraf sağlayıcıdan film akışı yapmak için İnternet hizmetini kullanıp kullanmadığını gösterir
## Contract : Müşterinin sözleşme süresi (Aydan aya, Bir yıl, İki yıl)
## PaperlessBilling : Müşterinin kağıtsız faturası olup olmadığı (Evet, Hayır)
## PaymentMethod : Müşterinin ödeme yöntemi (Elektronik çek, Posta çeki, Banka havalesi (otomatik), Kredi kartı (otomatik))
## MonthlyCharges : Müşteriden aylık olarak tahsil edilen tutar
## TotalCharges : Müşteriden tahsil edilen toplam tutar
## Churn : Müşterinin kullanıp kullanmadığı (Evet veya Hayır)


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import warnings

warnings.simplefilter(action="ignore")
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_rows', 20)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

df = pd.read_csv("Telco-Customer-Churn.csv")
df.head()

df.isnull().sum()

df.columns = [col.upper() for col in df.columns]

# TotalCharges sayısal bir değişken olmalı
df["TOTALCHARGES"] = pd.to_numeric(df["TOTALCHARGES"], errors='coerce')
# yes ise 1 no ise 0 numeriklestirme
df["CHURN"] = df["CHURN"].apply(lambda x: 1 if x == "Yes" else 0)


def check(dataframe, head=5):
    print(dataframe.shape)
    print("********************************************************************")
    print(dataframe.dtypes)
    print("********************************************************************")
    print(dataframe.info())
    print("********************************************************************")
    print(dataframe.describe().T)
    print("********************************************************************")
    print(dataframe.columns)
    print("********************************************************************")
    print(dataframe.head(head))
    print("********************************************************************")
    print(dataframe.tail(head))
    print("********************************************************************")
    print(dataframe.isnull().sum())
    print("********************************************************************")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)


check(df)


def grab_col_names(dataframe, cat_th=10, car_th=20):
    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car


cat_cols, num_cols, cat_but_car = grab_col_names(df)

# kategorik değişkenlerin sınıflarını ve oranlarını getirir, plot True ise görselleştirme yapar

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()


cat_summary(df, "CHURN")


## Numerik değişken analizi
## sayısal degiskenlerin çeyreklik değerlerini gösterir ve histogram grafiği oluşturur

def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)

for col in num_cols:
    num_summary(df, col, True)


## Hedef değişken analizi - numerik

def target_sum_with_num(dataframe, target, num_col):
    print(dataframe.groupby(target).agg({num_col: "mean"}), end="\n\n\n")


for col in num_cols:
    target_sum_with_num(df, "CHURN", col)

## Korelasyon analizi

df[num_cols].corr()

# Korelasyon Matrisi
f, ax = plt.subplots(figsize=[18, 13])
sns.heatmap(df[num_cols].corr(), annot=True, fmt=".2f", ax=ax, cmap="magma")
ax.set_title("Correlation Matrix", fontsize=20)
plt.show(block=True)

df.corrwith(df["CHURN"]).sort_values(ascending=False)


# Eksik değer tespiti ve doldurulması

def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")

    if na_name:
        return na_columns


na_columns = missing_values_table(df, na_name=True)

print(df[df["TOTALCHARGES"].isnull()]["TENURE"])
df["TOTALCHARGES"].fillna(0, inplace=True)

df.isnull().sum()

## Base Model
dff = df.copy()
cat_cols = [col for col in cat_cols if col not in ["CHURN"]]

def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

dff = one_hot_encoder(dff, cat_cols, drop_first=True)

y = dff["CHURN"]
X = dff.drop(["CHURN", "CUSTOMERID"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)

catboost_model = CatBoostClassifier(verbose=False, random_state=12345).fit(X_train, y_train)
y_pred = catboost_model.predict(X_test)

print(f"Accuracy: {round(accuracy_score(y_pred, y_test), 4)}")
print(f"Recall: {round(recall_score(y_pred, y_test), 4)}")
print(f"Precision: {round(precision_score(y_pred, y_test), 4)}")
print(f"F1: {round(f1_score(y_pred, y_test), 4)}")
print(f"Auc {round(roc_auc_score(y_pred, y_test), 4)}")


# Aykırı deger analizi

def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range

    return low_limit, up_limit


def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):  # var mı?
        return True
    else:
        return False


def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


for col in num_cols:
    print(col, check_outlier(df, col))
    if check_outlier(df, col):
        replace_with_thresholds(df, col)

# Ozellik cıkarımı

# Tenure  değişkeninden yıllık kategorik değişken oluşturma
df.loc[(df["TENURE"] >= 0) & (df["TENURE"] <= 12), "NEW_TENURE_YEAR"] = "0-1 Year"
df.loc[(df["TENURE"] >= 12) & (df["TENURE"] <= 24), "NEW_TENURE_YEAR"] = "1-2 Year"
df.loc[(df["TENURE"] >= 24) & (df["TENURE"] <= 36), "NEW_TENURE_YEAR"] = "2-3 Year"
df.loc[(df["TENURE"] >= 36) & (df["TENURE"] <= 48), "NEW_TENURE_YEAR"] = "3-4 Year"
df.loc[(df["TENURE"] >= 48) & (df["TENURE"] <= 60), "NEW_TENURE_YEAR"] = "4-5 Year"
df.loc[(df["TENURE"] >= 60) & (df["TENURE"] <= 72), "NEW_TENURE_YEAR"] = "5-6 Year"

# Kontratı 1 veya 2 yıllık müşterileri Engaged olarak belirtme
df["NEW_ENGAGED"] = df["CONTRACT"].apply(lambda x: 1 if x in ["Two year", "One year"] else 0)

# Herhangi bir destek, yedek veya koruma almayan kişiler
df["NEW_NOPROT"] = df.apply(lambda x: 1 if ((x["ONLINEBACKUP"] != "Yes") or
                                            (x["DEVICEPROTECTION"] != "Yes") or
                                            (x["TECHSUPPORT"] != "Yes")) else 0, axis=1)

# Aylık sözleşmesi bulunan ve genç olan müşteriler
df["NEW_YOUNG_NOT_ENGAGED"] = df.apply(lambda x: 1 if ((x["CONTRACT"] == "Month-to-month") and
                                                       (x["SENIORCITIZEN"] == 0)) else 0, axis=1)

# Kişinin toplam aldığı servis sayısı
df['NEW_TOTALSERVICE'] = (df[["PHONESERVICE", "INTERNETSERVICE", "ONLINESECURITY",
                              'ONLINEBACKUP', 'DEVICEPROTECTION', 'TECHSUPPORT',
                              'STREAMINGTV', "STREAMINGMOVIES"]] == "Yes").sum(axis=1)

# Herhangi bir streaming hizmeti alan kişiler
df["NEW_FLAG_ANY_STREAMING"] = df.apply(lambda x: 1 if ((x["STREAMINGTV"] == "Yes") or
                                                        (x["STREAMINGMOVIES"] == "Yes")) else 0, axis=1)

# Kişi otomatik ödeme yapıyor mu?
df["NEW_FLAG_AUTOPAYMENT"] = df["PAYMENTMETHOD"].apply(lambda x: 1 if x in ["Bank transfer (automatic)",
                                                                            "Credit card (automatic)"] else 0)

# ortalama aylık ödeme
df["NEW_AVG_CHARGES"] = df["TOTALCHARGES"] / (
            df["TENURE"] + 1)  # min 0 degeri varsa ve 1 eklemezsek null değer elde ederiz

# Güncel Fiyatın ortalama fiyata göre artışı
df["NEW_INCREASE"] = df["NEW_AVG_CHARGES"] / df["MONTHLYCHARGES"]

# Servis başına ücret
df["NEW_AVG_SERVICE_FEE"] = df["MONTHLYCHARGES"] / (df["NEW_TOTALSERVICE"] + 1)

### encoding

cat_cols, num_cols, cat_but_car = grab_col_names(df)

def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

binary_cols = [col for col in df.columns if df[col].dtypes == "O" and df[col].nunique() == 2]

for col in binary_cols:
    df = label_encoder(df, col)


def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

cat_cols = [col for col in cat_cols if col not in binary_cols and col not in ["CHURN"]]

df = one_hot_encoder(df, cat_cols, True)

#### Model

y = df["CHURN"]
X = df.drop(["CHURN", "CUSTOMERID"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)

catboost_model = CatBoostClassifier(verbose=False, random_state=12345).fit(X_train, y_train)
y_pred = catboost_model.predict(X_test)

print(f"Accuracy: {round(accuracy_score(y_pred, y_test), 4)}")
print(f"Recall: {round(recall_score(y_pred, y_test), 4)}")
print(f"Precision: {round(precision_score(y_pred, y_test), 4)}")
print(f"F1: {round(f1_score(y_pred, y_test), 4)}")
print(f"Auc {round(roc_auc_score(y_pred, y_test), 4)}")

def plot_feature_importance(importance, names, model_type):
    # Create arrays from feature importance and feature names
    feature_importance = np.array(importance)
    feature_names = np.array(names)

    # Create a DataFrame using a Dictionary
    data = {'feature_names': feature_names, 'feature_importance': feature_importance}
    fi_df = pd.DataFrame(data)

    # Sort the DataFrame in order decreasing feature importance
    fi_df.sort_values(by=['feature_importance'], ascending=False, inplace=True)

    # Define size of bar plot
    plt.figure(figsize=(15, 10))
    # Plot Searborn bar chart
    sns.barplot(x=fi_df['feature_importance'], y=fi_df['feature_names'])
    # Add chart labels
    plt.title(model_type + ' FEATURE IMPORTANCE')
    plt.xlabel('FEATURE IMPORTANCE')
    plt.ylabel('FEATURE NAMES')
    plt.show()

plot_feature_importance(catboost_model.get_feature_importance(), X.columns, 'CATBOOST')

























