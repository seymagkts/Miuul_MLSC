##### Makine Öğrenmesi ile Maaş Tahmini

##### AtBat: 1986-1987 sezonunda bir beyzbol sopası ile topa yapılan vuruş sayısı
##### Hits: 1986-1987 sezonundaki isabet sayısı
##### HmRun: 1986-1987 sezonundaki en değerli vuruş sayısı
##### Runs: 1986-1987 sezonunda takımına kazandırdığı sayı
##### RBI: Bir vurucunun vuruş yaptıgında koşu yaptırdığı oyuncu sayısı
##### Walks: Karşı oyuncuya yaptırılan hata sayısı
##### Years: Oyuncunun major liginde oynama süresi (sene)
##### CAtBat: Oyuncunun kariyeri boyunca topa vurma sayısı
##### CHits: Oyuncunun kariyeri boyunca yaptığı isabetli vuruş sayısı
##### CHmRun: Oyucunun kariyeri boyunca yaptığı en değerli sayısı
##### CRuns: Oyuncunun kariyeri boyunca takımına kazandırdığı sayı
##### CRBI: Oyuncunun kariyeri boyunca koşu yaptırdırdığı oyuncu sayısı
##### CWalks: Oyuncun kariyeri boyunca karşı oyuncuya yaptırdığı hata sayısı
##### League: Oyuncunun sezon sonuna kadar oynadığı ligi gösteren A ve N seviyelerine sahip bir faktör
##### Division: 1986 sonunda oyuncunun oynadığı pozisyonu gösteren E ve W seviyelerine sahip bir faktör
##### PutOuts: Oyun icinde takım arkadaşınla yardımlaşma
##### Assits: 1986-1987 sezonunda oyuncunun yaptığı asist sayısı
##### Errors: 1986-1987 sezonundaki oyuncunun hata sayısı
##### Salary: Oyuncunun 1986-1987 sezonunda aldığı maaş(bin uzerinden)
##### NewLeague: 1987 sezonunun başında oyuncunun ligini gösteren A ve N seviyelerine sahip bir faktör

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.max_rows', 20)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler

df = pd.read_csv("hitters.csv")

df.head()


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


# eksik değer tespiti ve doldurulması

def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")

    if na_name:
        return na_columns


na_columns = missing_values_table(df, na_name=True)

df["Salary"].fillna(df["Salary"].mean(), inplace=True)
df.isnull().sum()


# aykırı deger analizi

def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
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

# kategorik değişken analizi


def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")

    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)


for col in cat_cols:
    cat_summary(df, col, plot=True)


# numerik değişken analizi

def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]

    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)


for col in num_cols:
    num_summary(df, col, plot=True)


# hedef değişken analizi
def target_summary_with_cat(dataframe, target, categorical_col):
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean()}), end="\n\n\n")


for col in cat_cols:
    target_summary_with_cat(df, "Salary", col)


# korelasyon matrisi

def high_correlated_cols(dataframe, plot=False, corr_th=0.90):
    corr = dataframe.corr()
    cor_matrix = corr.abs()
    upper_triangle_matrix = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(np.bool))
    drop_list = [col for col in upper_triangle_matrix.columns if any(upper_triangle_matrix[col] > corr_th)]
    if plot:
        import seaborn as sns
        import matplotlib.pyplot as plt
        sns.set(rc={'figure.figsize': (15, 15)})
        sns.heatmap(corr, cmap="RdBu", annot=True)
        plt.show(block=True)
    return drop_list


high_correlated_cols(df, plot=True)

## Ozellik cıkarımı
# vurdugu isabetli vurusların tum vurmalarına oranı

new_num_cols = [col for col in num_cols if col not in ["Salary", "Years"]]
df[new_num_cols] = df[new_num_cols] + 1

df["CHits_CAtBat_Ratio"] = df["CHits"] / df["CAtBat"]

# en degerli vurus sayısı + kazandırdıgı sayı skoruna göre statu 
df.loc[((df["HmRun"] + df["Runs"]) >= 120) & ((df["HmRun"] + df["Runs"]) <= 150), "Num_Sta"] = "Expert"
df.loc[((df["HmRun"] + df["Runs"]) >= 70) & ((df["HmRun"] + df["Runs"]) <= 119), "Num_Sta"] = "Senior"
df.loc[((df["HmRun"] + df["Runs"]) >= 30) & ((df["HmRun"] + df["Runs"]) <= 69), "Num_Sta"] = "Middle"
df.loc[((df["HmRun"] + df["Runs"]) >= 0) & ((df["HmRun"] + df["Runs"]) <= 29), "Num_Sta"] = "Beginner"

# yıllık ortalamalar
df["NEW_CATBAT_MEAN"] = df["CAtBat"] / df["Years"]
df["NEW_CHITS_MEAN"] = df["CHits"] / df["Years"]
df["NEW_CHMRUN_MEAN"] = df["CHmRun"] / df["Years"]
df["NEW_CRUNS_MEAN"] = df["CRuns"] / df["Years"]
df["NEW_CRBI_MEAN"] = df["CRBI"] / df["Years"]
df["NEW_CWALKS_MEAN"] = df["CWalks"] / df["Years"]

# Bir sonraki lige oyuncu terfisi
df.loc[(df["League"] == "N") & (df["NewLeague"] == "N"), "NEW_PLAYER_PROGRESS"] = "StandN"
df.loc[(df["League"] == "A") & (df["NewLeague"] == "A"), "NEW_PLAYER_PROGRESS"] = "StandA"
df.loc[(df["League"] == "N") & (df["NewLeague"] == "A"), "NEW_PLAYER_PROGRESS"] = "Descend"
df.loc[(df["League"] == "A") & (df["NewLeague"] == "N"), "NEW_PLAYER_PROGRESS"] = "Ascend"

# Encode işlemleri
binary_cols = [col for col in df.columns if df[col].dtypes == "O" and df[col].nunique() == 2]


def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe


for col in binary_cols:
    df = label_encoder(df, col)

cat_cols = [col for col in cat_cols if col not in binary_cols and col not in ["Salary"]]


def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe


ohe_cols = [col for col in df.columns if 10 >= df[col].nunique() > 2]

df = one_hot_encoder(df, ohe_cols, True)

for col in num_cols:
    transformer = RobustScaler().fit(df[[col]])
    df[col] = transformer.transform(df[[col]])

### Model

X = df.drop(["Salary"], axis=1)
y = df["Salary"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=17)

reg_model = LinearRegression().fit(X_train, y_train)

print(reg_model.intercept_)

print(reg_model.coef_)

### Tahmin Başarısı

y_pred = reg_model.predict(X_train)
np.sqrt(mean_squared_error(y_train, y_pred))  # train rmse

reg_model.score(X_train, y_train)  # train Rkare

y_pred = reg_model.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))  # test rmse

reg_model.score(X_test, y_test)  # test Rkare

np.mean(
    np.sqrt(-cross_val_score(reg_model, X, y, cv=10, scoring="neg_mean_squared_error")))  # 10 katlı capraz dogrulama
