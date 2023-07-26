"""

PYTHON

"""

#################################################
# Verilen değerlerin veri yapılarını inceleyin.
#################################################

x = 8
y = 3.2
z = 8j + 18
a = "Hello World"
b = True
c = 23 < 22
l = [1,2,3,4]
d = {"Name":"Jake",
    "Age": 27,
     "Adress":"Downtown"}
t = ("Machine Learning", "Data Science")
s = {"Python","Machine Learning","Data Science"}

def tip_sorgu(data):
    print(data,"Turu: ",type(data))
    print("****")

tip_sorgu(x)
tip_sorgu(y)
tip_sorgu(z)
tip_sorgu(a)
tip_sorgu(b)
tip_sorgu(c)
tip_sorgu(l)
tip_sorgu(d)
tip_sorgu(t)
tip_sorgu(s)

#################################################
# Verilen string ifadenin tüm harflerini büyük harfe çeviriniz.
# Virgül ve nokta yerine space koyunuz, kelime kelime ayırınız.
#################################################

text = "The goal is to turn data into information," \
       "and information into insight."
text.upper().replace("."," ").replace(","," ").split()

#################################################
# Verilen listeye aşağıdaki adımları uygulayınız.
# Adım 1: Verilen listenin eleman sayısına bakınız.
# Adım 2: Sıfırıncı ve onuncu indeksteki elemanları çağırınız.
# Adım 3: Verilen liste üzerinden ["D", "A", "T", "A"] listesi oluşturunuz.
# Adım 4: Sekizinci indeksteki elemanı siliniz.
# Verilen listeye aşağıdaki adımları uygulayınız.
# Adım 5: Yeni bir eleman ekleyiniz.
# Adım 6: Sekizinci indekse "N" elemanını tekrar ekleyiniz.
#################################################

text = ['D','A','T','A','S','C','I','E','N','C','E']

print("Eleman sayisi:",len(text))

print(f"0. indeks: {text[0]}")
print(f"10. indeks: {text[10]}")

text[0:4]
text.pop(8)
text.append("X")
text.insert(8,'N')

#################################################
# Verilen sözlük yapısına aşağıdaki adımları uygulayınız.
# Adım 1: Key değerlerine erişiniz.
# Adım 2: Value'lara erişiniz.
# Adım 3: Daisy key'ine ait 12 değerini 13 olarak güncelleyiniz.
# Adım 4: Key değeri Ahmet value değeri [Turkey,24] olan yeni bir değer ekleyiniz.
# Adım 5: Antonio'yu dictionary'den siliniz
#################################################

dict = {"Christian":["America",18],
        "Daisy":["England",12],
        "Antonio":["Spain",22],
        "Dante":["Italy",25]}

dict.keys()
dict.values()
dict.update({'Daisy':['England',13]})
dict.update({'Ahmet':['Turkey',24]})
dict.pop("Antonio")

#################################################
# Argüman olarak bir liste alan, listenin içerisindeki tek ve çift sayıları
# ayrı listelere atayan ve bu listeleri return eden fonksiyon yazınız.
#################################################

l = [2,13,18,93,22]

def tek_cift(list):
    arr_tek = []
    arr_cift = []
    for eleman in list:
        if eleman % 2 == 0:
            arr_cift.append(eleman)
        else:
            arr_tek.append(eleman)
    return arr_cift,arr_tek

cift, tek = tek_cift(l)

# List comp.
def func(list):
    cift_list = [x for x in list if x % 2 == 0]
    tek_list = [x for x in list if x % 2 != 0]

    return cift_list, tek_list

cift, tek = func(l)

#################################################
# Aşağıda verilen listede mühendislik ve tıp fakültelerinde dereceye
# giren öğrencilerin isimleri bulunmaktadır. Sırasıyla ilk üç öğrenci mühendislik fakültesinin başarı sırasını temsil ederken
# son üç öğrenci de tıp fakültesi öğrenci sırasına aittir. Enumarate
# kullanarak öğrenci derecelerini fakülte özelinde yazdırınız.
#################################################

ogrenciler = ['Ali','Veli','Ayse','Talat','Zeynep','Ece']

for indeks, eleman in enumerate(ogrenciler,1):
    if indeks<4:
        print(f"Muhendislik Fakultesi {indeks}. ogrenci: {eleman}")
    else:
        print(f"Tip Fakultesi {indeks-3}. ogrenci: {eleman}")

#################################################
# Aşağıda 3 adet liste verilmiştir. Listelerde sırası ile bir dersin kodu, kredisi ve kontenjan bilgileri yer almaktadır.
# Zip kullanarak ders bilgilerini bastırınız.
#################################################

ders_kodu =["CMP1005","PSY1001","HUK1005","SEN2204"]
kredi = [3,4,2,4]
kontenjan = [30,75,150,25]

for kod, kredi, kontenjan in zip(ders_kodu,kredi,kontenjan):
    print(f"Kredisi {kredi} olan {kod} kodlu dersin kontenjani {kontenjan} kisidir.")

#################################################
# Aşağıda 2 adet set verilmiştir. Sizden istenilen eğer 1. küme 2. kümeyi kapsiyor ise ortak elemanlarını eğer kapsamıyor ise 2. kümenin
# 1. kümeden farkını yazdıracak fonksiyonu tanımlamanız beklenmektedir
#################################################

kume1 = set(['data','python'])
kume2 = set(['data','function','qcut','lambda','python','miuul'])

def kume(kume1, kume2):
    if kume1.issuperset(kume2):
        print(kume1.intersection(kume2))
    else:
        print(kume2.difference(kume1))

kume(kume1,kume2)

