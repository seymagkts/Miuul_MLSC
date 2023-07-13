#### DATA STRUCTURES ####
#### VERI YAPILARI ####
### HIZLI GIRIS

# SAYILAR (NUMBERS)
# INT
x = 46
type(x)
float(x) # tip dönüşümü

# FLOAT
y = 10.3
type(y)
float(x * y / 10) # tip dönüşümü

# COMPLEX
z = 2j+1
type(z)

x * 3
z / 7
y ** 2

# KARAKTER DİZİLERİ (STRINGS)
# STR
x = "Hello ai era"
type(x)
print("John") # ekrana bilgi yazdırma
print('John')
val = """ cok 
    satırlı
    karakter
    dizisi"""
print(val[0]) # str indekler 0'dan başlar
print(val[0:5]) # slice islemi
"cok" in val # str içinde karakter sorgulama

# STR METODLARI ( CLASS YAPISI İCİNDE TANIMLANANLAR METOD, CLASS İÇERİSİNDE TANIMLANMAYANLAR FONKSİYONDUR )
dir(str) # str'ye ait tüm metod adları gelir
name = "john"
len(name) # karakter bilgisi verir, fonksiyondur
name.upper() # tüm harfleri büyük harf yapar, metoddur
name.lower() # tüm harfleri küçük harf yapar, metoddur
x.replace("a","y") # ilk verilen değeri ikincisiyle değiştirir
x.split() # parametreye göre bölme işlemi yapar, default parametre bosluktur
x.strip() # parametreye göre kırpma işlemi yapar, default parametre bosluktur
val.capitalize() # ilk harfi büyük harf yapar
"foo".startswith("f") # verilen parametreyle mi baslıyor sorgusu yapar

# BOOLEAN
type(True)
type(3==2)

# LIST
## Degistirilebilir
## Sıralıdır, indeks işlemleri yapılabilir
## Kapsayıcıdır
x = ["btc","eth","xrp"]
type(x)
not_nam = [1,2,3,True,"a","b",[1,2,3]]
not_nam[6][1]
not_nam[0] = 99

# LISTE METODLAR
dir(not_nam) # liste metodlarını getirir
len(not_nam) # boyut bilgisi dondurur
not_nam.append(100) # listeye eleman ekler
not_nam.pop(0) # indekse göre silme yapar
not_nam.insert(0,"on bir") # indekse göre ekleme yapar

# DICTIONARY
## Degistirilebilir
## Sırasızdır
## Kapsayıcıdır
### key - value
x = {"name":"Peter",
     "Age":36,
     "REG":["Regression",10]}
type(x)
"REG" in x # key sorgulama
x.get("REG")
x["REG"] = ["YSA",20] # value degistirme
x.keys() # tum keyler gelir
x.values() # tum valuelar gelir
x.items() # tüm çiftler tuple halinde çevrilir
x.update({"RF":5}) # güncelleme yapar

# TUPLE
## Değiştirilemez
## Sıralıdır
## Kapsayıcıdır
x = ("python","ml","ds")
type(x)
# önce listeye çevrilip sonra değişiklik uygulanır,
# sonra tekrar tuple'a dönüştürülerek değiştirilme işlemi uygulanabilir


# SET
## Değiştirilebilir
## Sırasız ve eşsizdir
## Kapsayıcıdır
x = {"python","ml","ds"}
set2 = set([1,2,3])
set3 = set([1,5,3])
type(x)
set2.difference(set3) # set2de olup set3te olmayanları verir set2 - set3
set2.symmetric_difference(set3) # iki kümede de birbirlerine göre olmayanları verir
set2.intersection(set3) # iki kümenin kesisimi set2 & set3
set2.union(set3) # iki kümenin birleşimi
set2.isdisjoint(set3) # iki kümenin kesisimi bos mu? T / F
set2.issubset(set3) # bir küme diğer kümenin alt kümesi mi? T / F
set2.issuperset(set3) # bir küme diğer kümeyi kapsıyor mu? T / F
