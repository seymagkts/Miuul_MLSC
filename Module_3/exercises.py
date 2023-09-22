### FEATURE ENGINEERING & DATA PREPROCESSING
##### TITANIC

import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from datetime import date
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler

df = pd.read_csv("titanic.csv")
df.shape
df.head()

df.columns = [col.upper() for col in df.columns]

##### 1. Feature Engineering (Özellik Mühendisliği)

# CABIN BOOL
df["NEW_CABIN_BOOL"] = df["CABIN"].notnull().astype("int")

# NAME COUNT
df["NEW_NAME_COUNT"] = df["NAME"].str.len()

# NAME WORD COUNT
df["NEW_NAME_WORD_COUNT"] = df["NAME"].apply(lambda x: len(str(x).split(" ")))

# NAME DR
df["NEW_NAME_DR"] = df["NAME"].apply(lambda x:len([x for x in x.split() if x.startswith("Dr")]))

# NEW TITLE
df["NEW_TITLE"] = df.NAME.str.extract("([A-Za-z]+)\.", expand = False)

# FAMILY SIZE
df["NEW_FAMILY_SIZE"] = df["SIBSP"] + df["PARCH"]+1

# AGE P_CLASS
df["NEW_AGE_PCLASS"] = df["AGE"] * df["PCLASS"]

# IS ALONE 
df.loc[((df["SIBSP"] + df["PARCH"])>0),"NEW_IS_ALONE"] = "NO"
df.loc[((df["SIBSP"] + df["PARCH"])==0),"NEW_IS_ALONE"] = "YES"

# AGE LEVEL
df.loc[(df["AGE"] <18 ),"NEW_AGE_CAT"] = "young"
df.loc[(df["AGE"] >=18) & (df["AGE"] < 56),"NEW_AGE_CAT"] = "mature"
df.loc[(df["AGE"] >= 56),"NEW_AGE_CAT"] = "senior"

# SEX x AGE
df.loc[(df["SEX"]=="male") & (df["AGE"] <= 21 ),"NEW_SEX_CAT"] = "youngmale"
df.loc[(df["SEX"]=="male") & (df["AGE"] > 21 ) & (df["AGE"] <= 50),"NEW_SEX_CAT"] = "maturemale"
df.loc[(df["SEX"]=="male") & (df["AGE"] > 50 ),"NEW_SEX_CAT"] = "seniormale"
df.loc[(df["SEX"]=="female") & (df["AGE"] <= 21 ),"NEW_SEX_CAT"] = "youngfemale"
df.loc[(df["SEX"]=="female") & (df["AGE"] > 21 ) & (df["AGE"] <= 50),"NEW_SEX_CAT"] = "maturefemale"
df.loc[(df["SEX"]=="female") & (df["AGE"] > 50 ),"NEW_SEX_CAT"] = "seniorfemale"

# boxplot 0.25 ile 0.75lik eşik değerlerini bulma

def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range

    return low_limit, up_limit

# eşik değerlerine göre aykırı değer kontrolu

def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None): # var mı?
        return True
    else:
        return False

def grab_col_names(dataframe, cat_th=10, car_th=20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
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

# aykırı degerlerin kendilerine erismek için

def grab_outliers(dataframe, col_name, index=False):
    low, up = outlier_thresholds(dataframe, col_name)

    if dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].shape[0] > 10:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].head())
    else:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))])

    if index:
        outlier_index = dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].index
        return outlier_index

# aykırı degerleri silme

def remove_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    df_without_outliers = dataframe[~((dataframe[col_name] < low_limit) | (dataframe[col_name] > up_limit))]
    return df_without_outliers

# aykırı degerleri limitlere göre atama yapmak için (baskılama yöntemi)

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

# eksik veriye sahip kolonu seçme, eksik deger sayısı ve oranı, bunları df'ye cevir
# na_name True olursa eksik değere sahip degisken adları gelir

def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")

    if na_name:
        return na_columns


# eksik değerlerin bağımlı değisken ile ilişkisi

def missing_vs_target(dataframe, target, na_columns):
    temp_df = dataframe.copy() # gerçğinde oynama yapmamak icin

    for col in na_columns:
        temp_df[col + '_NA_FLAG'] = np.where(temp_df[col].isnull(), 1, 0)

    na_flags = temp_df.loc[:, temp_df.columns.str.contains("_NA_")].columns #tüm sutunları seç içinde _NA_ geçen sutunları getir

    for col in na_flags:
        print(pd.DataFrame({"TARGET_MEAN": temp_df.groupby(col)[target].mean(),
                            "Count": temp_df.groupby(col)[target].count()}), end="\n\n\n")


# label encoding 1 ve 0 uygulaması

def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

# one hot encoding uygulaması

def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe


# kategorik değişkenlerin sınıflarını ve oranlarını getirir, plot True ise görselleştirme yapar

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()

# rare kategoriler ile bağımlı değişken arasındaki ilişkinin belirlenmesi

def rare_analyser(dataframe, target, cat_cols):
    for col in cat_cols:
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                            "RATIO": dataframe[col].value_counts() / len(dataframe),
                            "TARGET_MEAN": dataframe.groupby(col)[target].mean()}), end="\n\n\n")


# rare encoding uygulaması

def rare_encoder(dataframe, rare_perc):
    temp_df = dataframe.copy()
    # verilen rare oranına göre seçme
    rare_columns = [col for col in temp_df.columns if temp_df[col].dtypes == 'O'
                    and (temp_df[col].value_counts() / len(temp_df) < rare_perc).any(axis=None)] 
    
    for var in rare_columns: 
        tmp = temp_df[var].value_counts() / len(temp_df) # sınıf oranı olusturur
        rare_labels = tmp[tmp < rare_perc].index # verilen orandan daha az olanların indexlerini tut
        temp_df[var] = np.where(temp_df[var].isin(rare_labels), 'Rare', temp_df[var]) # yerine Rare yaz

    return temp_df

## sayısal degiskenlerin çeyreklik değerlerini gösterir ve histogram grafiği oluşturur

# def num_summary(dataframe, numerical_col, plot=False):
#     quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
#     print(dataframe[numerical_col].describe(quantiles).T)

#     if plot:
#         dataframe[numerical_col].hist(bins=20)
#         plt.xlabel(numerical_col)
#         plt.title(numerical_col)
#         plt.show(block=True)

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

cat_cols, num_cols, cat_but_car = grab_col_names(df)
num_cols = [col for col in num_cols if "PASSENGERID" not in col]

##### 2. Aykırı Değerler

# aykırı değer kontrolu
for col in num_cols:
    print(col,check_outlier(df,col))

# eşit değerler ile aykırı değerleri değiştirilir
for col in num_cols:
    replace_with_thresholds(df,col)

for col in num_cols:
    print(col,check_outlier(df,col))

##### 3. Eksik Değerler

missing_values_table(df)

df.drop("CABIN",inplace=True,axis=1)

remove_cols = ["TICKET","NAME"]
df.drop(remove_cols,inplace=True,axis=1)

df["AGE"] = df["AGE"].fillna(df.groupby("NEW_TITLE")["AGE"].transform("median"))

df["NEW_AGE_PCLASS"] = df["AGE"] * df["PCLASS"]

df.loc[(df["AGE"] <18 ),"NEW_AGE_CAT"] = "young"
df.loc[(df["AGE"] >=18) & (df["AGE"] < 56),"NEW_AGE_CAT"] = "mature"
df.loc[(df["AGE"] >= 56),"NEW_AGE_CAT"] = "senior"

df.loc[(df["SEX"]=="male") & (df["AGE"] <= 21 ),"NEW_SEX_CAT"] = "youngmale"
df.loc[(df["SEX"]=="male") & (df["AGE"] > 21 ) & (df["AGE"] <= 50),"NEW_SEX_CAT"] = "maturemale"
df.loc[(df["SEX"]=="male") & (df["AGE"] > 50 ),"NEW_SEX_CAT"] = "seniormale"
df.loc[(df["SEX"]=="female") & (df["AGE"] <= 21 ),"NEW_SEX_CAT"] = "youngfemale"
df.loc[(df["SEX"]=="female") & (df["AGE"] > 21 ) & (df["AGE"] <= 50),"NEW_SEX_CAT"] = "maturefemale"
df.loc[(df["SEX"]=="female") & (df["AGE"] > 50 ),"NEW_SEX_CAT"] = "seniorfemale"

df = df.apply(lambda x: x.fillna(x.mode()[0]) if (x.dtype == "O" 
              and len(x.unique()) <= 10) else x, axis=0)

##### 4. Label Encoding

binary_cols = [col for col in df.columns if df[col].dtype not in [int,float] # tipi int ya da float değilse ve
              and df[col].nunique() == 2] # iki farklı sınıf varsa sec

for col in binary_cols:
    df = label_encoder(df,col)

##### 5. Rare Encoder

rare_analyser(df,"SURVIVED",cat_cols)
df = rare_encoder(df,0.01)
df["NEW_TITLE"].value_counts()

##### 6. One Hot Encoding

ohe_cols = [col for col in df.columns if 10>=df[col].nunique()>2] # sayılar ozneldir
ohe_cols
df = one_hot_encoder(df,ohe_cols)
df.head()

###### 6. Standart Scaler

scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])
df[num_cols].head()

###### 7. Model

y = df["SURVIVED"]
X = df.drop(["PASSENGERID","SURVIVED"], axis = 1)

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.30,random_state=17)

rf_model = RandomForestClassifier(random_state=46).fit(X_train,y_train)
y_pred= rf_model.predict(X_test)
accuracy_score(y_pred,y_test)

plot_importance(rf_model,X_train)

