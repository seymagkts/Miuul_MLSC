"""

RULES-BASED CLASSIFICATION

"""

#################################################
# persona.csv dosyasını okutunuz ve veri seti ile ilgili genel bilgileri gösteriniz.
#################################################

import pandas as pd
df = pd.read_csv("persona.csv")
df.info()
df.describe().T
df.shape
df.tail()
df.head()
df.isnull().any()
df.columns
df.index

#################################################
# Kaç unique SOURCE vardır? Frekansları nedir?
#################################################

df.SOURCE.unique()
df.SOURCE.value_counts()

#################################################
# Kaç unique PRICE vardır?
#################################################

df.PRICE.nunique()

#################################################
# Hangi PRICE'dan kaçar tane satış gerçekleşmiş?
#################################################

df.PRICE.value_counts()

#################################################
# Hangi ülkeden kaçar tane satış olmuş?
#################################################

df.COUNTRY.value_counts()

#################################################
# Ülkelere göre satışlardan toplam ne kadar kazanılmış?
#################################################

df.groupby("COUNTRY").agg({"PRICE":"sum"})

#################################################
# SOURCE türlerine göre satış sayıları nedir?
#################################################

df.SOURCE.value_counts()

#################################################
# Ülkelere göre PRICE ortalamaları nedir?
#################################################

df.groupby("COUNTRY").agg({"PRICE":"mean"})

#################################################
# SOURCE'lara göre PRICE ortalamaları nedir?
#################################################

df.groupby("SOURCE").agg({"PRICE":"mean"})

#################################################
# COUNTRY-SOURCE kırılımında PRICE ortalamaları nedir?
#################################################

df.groupby(["COUNTRY","SOURCE"]).agg({"PRICE":"mean"})

#################################################
# COUNTRY, SOURCE, SEX, AGE kırılımında ortalama kazançlar nedir?
#################################################

ort_kaz = df.groupby(["COUNTRY","SOURCE","SEX","AGE"]).agg({"PRICE":"mean"})

#################################################
# Çıktıyı PRICE’a göre sıralayınız.
# Önceki sorudaki çıktıyı daha iyi görebilmek için sort_values metodunu azalan olacak şekilde PRICE’a göre uygulayınız.
# Çıktıyı agg_df olarak kaydediniz.
#################################################

agg_df = ort_kaz.sort_values("PRICE",ascending=False)
agg_df.head()

#################################################
# Indekste yer alan isimleri değişken ismine çeviriniz.
# Üçüncü sorunun çıktısında yer alan PRICE dışındaki tüm değişkenler index isimleridir.
# Bu isimleri değişken isimlerine çeviriniz.
#################################################

agg_df = agg_df.reset_index()
agg_df.columns

#################################################
# Age değişkenini kategorik değişkene çeviriniz ve agg_df’e ekleyiniz.
# Age sayısal değişkenini kategorik değişkene çeviriniz.
# Aralıkları ikna edici şekilde oluşturunuz.
# Örneğin: ‘0_18', ‘19_23', '24_30', '31_40', '41_70'
#################################################

agg_df["AGE"] = agg_df["AGE"].astype("int64")
agg_df.dtypes

bol = [0,18,23,30,40,agg_df["AGE"].max()]

isimlendir = ["0_18","19_23","24_30","31_40","41_+"]

agg_df["AGE_CAT"]= pd.cut(agg_df.AGE,bins=bol,labels=isimlendir)

agg_df.head()

#################################################
# Yeni seviye tabanlı müşterileri (persona) tanımlayınız.
# Yeni seviye tabanlı müşterileri (persona) tanımlayınız ve veri setine değişken olarak ekleyiniz.
# Yeni eklenecek değişkenin adı: customers_level_based
# Önceki soruda elde edeceğiniz çıktıdaki gözlemleri bir araya getirerek
# customers_level_based değişkenini oluşturmanız gerekmektedir
#################################################

agg_df["customers_level_based"] = ["_".join(txt).upper() for txt in agg_df.drop(['AGE', 'PRICE'], axis=1).values]

agg_df = agg_df[['customers_level_based', 'PRICE']]
agg_df = agg_df.groupby('customers_level_based')['PRICE'].mean().reset_index()

#################################################
# Yeni müşterileri (personaları) segmentlere ayırınız.
# Yeni müşterileri (Örnek: USA_ANDROID_MALE_0_18) PRICE’a göre 4 segmente ayırınız.
# Segmentleri SEGMENT isimlendirmesi ile değişken olarak agg_df’e ekleyiniz.
# Segmentleri betimleyiniz (Segmentlere göre group by yapıp price mean, max, sum’larını alınız).
#################################################

agg_df["SEGMENT"] = pd.qcut(agg_df.PRICE,q=4,labels=["D","C","B","A"])
agg_df.head()

agg_df.groupby(["SEGMENT"]).agg({"PRICE":["sum","max","mean"]}).reset_index()

#################################################
# Yeni gelen müşterileri sınıflandırıp, ne kadar gelir getirebileceklerini tahmin ediniz.
# 33 yaşında ANDROID kullanan bir Türk kadını hangi segmente aittir ve ortalama ne kadar gelir kazandırması beklenir?
# 35 yaşında IOS kullanan bir Fransız kadını hangi segmente aittir ve ortalama ne kadar gelir kazandırması beklenir?
#################################################

new_user = "TUR_ANDROID_FEMALE_31_40"
agg_df[agg_df["customers_level_based"]== new_user]

new_user = "FRA_IOS_FEMALE_31_40"
agg_df[agg_df["customers_level_based"]== new_user]
