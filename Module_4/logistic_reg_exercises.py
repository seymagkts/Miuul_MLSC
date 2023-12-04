##### Diabetes Prediction with Logistic Regression

###### Değişkenler
###### Pregnancies: Hamilelik sayısı
###### Glucose: Glikoz
###### BloodPressue: Kan basıncı
###### SkinThickness: Cilt kalınlığı
###### Insulin: İnsülin
###### BMI: Beden kitle indeksi
###### DiabetesPedigreeFunction: Soyumuzdaki kişilere göre diyabet olma ihtimalimizi hesaplayan bir fonksiyon
###### Age: Yaş (yıl)
###### Outcome: Kişinin diyabet olup olmadığı bilgisi. Hastalığa sahip (1) ya da değil (0)


"""
1. Exploratory Data Analysis
2. Data Preprocessing
3. Model & Prediction
4. Model Evaluation
5. Model Validation: Holdout
6. Model Validation: 10-fold Cross Validation
7. Prediction for A New Observation

"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report, plot_roc_curve
from sklearn.model_selection import train_test_split,cross_validate

### aykırı değer analizi fonksiyonları

def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95): # teorikte q1 0.25 q3 0.75 verilir, bu degerler subjektiftir. sadece cok cok uc noktaları veriden ayrıstırmak için 0.05e 0.95 verdik
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range

    return low_limit, up_limit

def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None): # var mı?
        return True
    else:
        return False

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.max_rows', 20)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

df = pd.read_csv("diabetes.csv")

df.head()

### Exploratory Data Analysis

#  Target analizi
def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()

cat_summary(df,"Outcome", True)

def check(dataframe,head=5):
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

## Feature analizi
num_cols = [col for col in df.columns if "Outcome" not in col]

def plot_numerical_col(dataframe, numerical_col):
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True) # block true olursa üst üste gösterilen grafikler cakısmaz

for col in num_cols:
    plot_numerical_col(df,col)

def target_sum_with_num(dataframe,target,num_col):
    print(dataframe.groupby(target).agg({num_col:"mean"}),end="\n\n\n")

for col in num_cols:
    target_sum_with_num(df,"Outcome",col)

### Data Preprocessing

# aykırı değer
for col in num_cols:
    print(col,check_outlier(df,col))

replace_with_thresholds(df,"Insulin")

# değişken ölçeklendirme
### robust scaler bütün gözlem biriminin değerlerinden medyanı cıkarıp range (q3-q1) değerine böler.
### robust scaler aykırı değerlere dayanıklı/duyarsız oldugu icin bunu kullanıyoruz.
for col in num_cols:
    df[col] = RobustScaler().fit_transform(df[[col]])

df.head()

### Model & Prediction

y = df["Outcome"]
X = df.drop(["Outcome"],axis=1)

log_model = LogisticRegression().fit(X,y)

# w
print(log_model.coef_)

# b
print(log_model.intercept_)

y_pred = log_model.predict(X) # bagımsız degiskenlerden bagımlı degisken tahmin etme fonksiyonu

### Model Evaluation

def plot_confusion_matrix(y, y_pred):
    acc = round(accuracy_score(y,y_pred),2)
    cm = confusion_matrix(y,y_pred)
    sns.heatmap(cm,annot=True,fmt=".0f")
    plt.xlabel("y_pred")
    plt.ylabel("y")
    plt.title("Accuracy Score: {0}".format(acc),size=10)
    plt.show()

plot_confusion_matrix(y,y_pred)

print(classification_report(y,y_pred))

# ROC AUC
y_prob = log_model.predict_proba(X)[:,1] ## gerçekleşme olasılığının tahmin değerleri
roc_auc_score(y,y_prob)

# Accuracy: 0.78
# Precision: 0.74
# Recall: 0.58
# F1 Score: 0.65
# ROC AUC: 0.84

### Model Validation: Holdout

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2,random_state=17)
# random state hangi 80e 20lik verinin seçileceği ilgilidir.

log_model = LogisticRegression().fit(X_train,y_train)

y_pred = log_model.predict(X_test)

y_prob = log_model.predict_proba(X_test)[:,1]

print(classification_report(y_test,y_pred))

roc_auc_score(y_test,y_prob)

# Accuracy: 0.77
# Precision: 0.79
# Recall: 0.53
# F1 Score: 0.63
# ROC AUC: 0.88

plot_roc_curve(log_model,X_test,y_test)
plt.title = ("ROC curve")
plt.plot([0,1],[0,1],"g--")
plt.show()

### Model Validation: 10-fold Cross Validation

# veri bolsa train test ayırıp train uzerinde 10 katlı doğrulama yapılabilir veri azsa tüm veride
# cok daha güvenilir sonuclar verir
log_model = LogisticRegression().fit(X,y)
cv_results = cross_validate(log_model,
                           X,
                           y,
                           cv=5,
                           scoring=["accuracy","precision","recall","f1","roc_auc"])

# ortalama accuracy
cv_results["test_accuracy"].mean()

# ortalama precision
cv_results["test_precision"].mean()

# ortalama recall
cv_results["test_recall"].mean()

# ortalama f1
cv_results["test_f1"].mean()

# ortalama roc_auc
cv_results["test_roc_auc"].mean()

# Accuracy: 0.77
# Precision: 0.72
# Recall: 0.58
# F1 Score: 0.64
# ROC AUC: 0.83

### Prediction for A New Observation

random_user = X.sample(1,random_state=96)
random_user

log_model.predict(random_user)

