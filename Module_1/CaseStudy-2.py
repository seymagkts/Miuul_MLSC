"""

COMPREHENSION

"""

#################################################
# Comprehension yapısı kullanarak car_crashes verisindeki numeric
# değişkenlerin isimlerini büyük harfe çeviriniz ve başına NUM ekleyiniz.
#################################################

import seaborn as sns
df = sns.load_dataset("car_crashes")
df.columns
df.info
df.head()

["NUM_" + col.upper() if df[col].dtype != "O" else col.upper() for col in df.columns]
# ['NUM_' + col.upper() for col in df.columns if df[col].dtype != 'O' ]

#################################################
# List Comprehension yapısı kullanarak car_crashes verisinde
# isminde "no" barındırmayan değişkenlerin isimlerinin
# sonuna "FLAG" yazınız
#################################################

[col.upper() + '_FLAG' if 'no' not in col else col.upper() for col in df.columns]

#################################################
# List Comprehension yapısı kullanarak aşağıda verilen değişken
# isimlerinden FARKLI olan değişkenlerin isimlerini seçiniz ve
# yeni bir dataframe oluşturunuz.
#################################################

og_list = ['abbrev', 'no_previous']

new_col = [col for col in df.columns if col not in og_list]
new_df = df[new_col]
new_df.head()
