"""

PANDAS

"""

#################################################
# Seaborn kütüphanesi içerisinden Titanic veri setini tanımlayınız.
#################################################

import seaborn as sns
import pandas as pd
pd.set_option("display.max_columns", None)
df = sns.load_dataset('titanic')
df.head()
#sns.get_dataset_names()

#################################################
# Titanic veri setindeki kadın ve erkek yolcuların sayısını bulunuz.
#################################################

df["sex"].value_counts()["female"]
# len(df[df["sex"] == 'female'])

df["sex"].value_counts()["male"]
# len(df[df["sex"] == 'male'])

#################################################
# Her bir sutuna ait unique değerlerin sayısını bulunuz.
#################################################

df.nunique()

#################################################
# pclass değişkeninin unique değerlerinin sayısını bulunuz, pclass ve parch değişkenlerinin unique değerlerinin sayısını bulunuz.
#################################################

df["pclass"].nunique()
# df["pclass"].unique()

df[["pclass","parch"]].nunique()

#################################################
# embarked değişkeninin tipini kontrol ediniz. Tipini category olarak değiştiriniz ve tekrar kontrol ediniz.
#################################################

df["embarked"].dtype
df["embarked"] = df["embarked"].astype("category")
df.dtypes

#################################################
# embarked değeri C olanların tüm bilgelerini gösteriniz.
# embarked değeri S olmayanların tüm bilgelerini gösteriniz.
#################################################

df[df["embarked"] == "C"]

df[df["embarked"] != "S"]
# df[~(df["embarked"]=="S")]

#################################################
# Yaşı 30 dan küçük ve kadın olan yolcuların tüm bilgilerini gösteriniz.
#################################################

df[(df["age"] < 30) & (df["sex"]== "female")]

#################################################
# Fare'i 500'den büyük veya yaşı 70 den büyük yolcuların bilgilerini gösteriniz.
#################################################

df[(df["fare"]>500) | (df["age"]>70)]

#################################################
# Her bir değişkendeki boş değerlerin toplamını bulunuz.
#################################################

df.isnull().sum()

#################################################
# who değişkenini dataframe’den çıkarınız.
#################################################

df.drop("who",axis=1,inplace=True)

#################################################
# deck değikenindeki boş değerleri deck değişkenin en çok tekrar eden değeri (mode) ile doldurunuz.
#################################################

df["deck"].fillna(df.mode().deck[0])
# df["deck"].fillna(df.["deck"].mode()[0],inplace=True)

#################################################
# age değikenindeki boş değerleri age değişkenin medyanı ile doldurunuz.
#################################################

df["age"].fillna(df.median().age)

#################################################
# survived değişkeninin pclass ve cinsiyet değişkenleri kırılımınında sum, count, mean değerlerini bulunuz.
#################################################

df.groupby(["pclass","sex"]).agg({"survived":["sum","count","mean"]})

#################################################
# 30 yaşın altında olanlar 1, 30'a eşit ve üstünde olanlara 0 vericek bir fonksiyon yazın.
# Yazdığınız fonksiyonu kullanarak titanik veri setinde age_flag adında bir değişken oluşturunuz oluşturunuz.
# (apply ve lambda yapılarını kullanınız.
#################################################

func = lambda x: 1 if x < 30 else 0
df["age_flag"] = df["age"].apply(func)

#################################################
# Seaborn kütüphanesi içerisinden Tips veri setini tanımlayınız.
#################################################

db = sns.load_dataset("tips")

#################################################
# Time değişkeninin kategorilerine (Dinner, Lunch) göre total_bill değerlerinin toplamını, min, max ve ortalamasını bulunuz.
#################################################

db.groupby("time").agg({"total_bill":("sum","min","max","mean")})

#################################################
# Günlere ve time göre total_bill değerlerinin toplamını, min, max ve ortalamasını bulunuz.
#################################################

db.groupby(["day","time"]).agg({"total_bill":("sum","min","max","mean")})

#################################################
# Lunch zamanına ve kadın müşterilere ait total_bill ve tip değerlerinin day'e göre toplamını, min, max ve ortalamasını bulunuz.
#################################################

db_Female_lunch = db[(db["time"]=="Lunch") & (db["sex"]=="Female")]
db_Female_lunch.groupby(["day"]).agg({"total_bill":["sum","min","max","mean"],
                                     "tip":["sum","min","max","mean"]})
#################################################
# size'i 3'ten küçük, total_bill'i 10'dan büyük olan siparişlerin ortalaması nedir? (loc kullanınız)
#################################################

db.loc[(db["size"]< 3)
       & (db["total_bill"] > 10),
       "total_bill"].mean()

#################################################
# total_bill_tip_sum adında yeni bir değişken oluşturunuz. Her bir müşterinin ödediği totalbill ve tip in toplamını versiN.
#################################################

db["total_bill_tip_sum"] = db["total_bill"] + db["tip"]


#################################################
# total_bill_tip_sum değişkenine göre büyükten küçüğe sıralayınız ve ilk 30 kişiyi yeni bir dataframe'e atayınız.
#################################################

new_db = db["total_bill_tip_sum"].sort_values(ascending=False).head(30)

