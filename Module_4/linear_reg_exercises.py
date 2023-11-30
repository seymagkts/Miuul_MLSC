##### Sales Prediction with Linear Regression

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option("display.float_format", lambda x: "%.2f" % x)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_score

##### Simple Linear Regression with OLS Using Scikit-Learn

df = pd.read_csv("advertising.csv")

print(df.shape)

print(df.describe().T)

##### Model

X = df[["TV"]]
y = df[["sales"]]

reg_model = LinearRegression().fit(X, y)

## b + w*TV
# tv katsayısı (w)
print(reg_model.coef_[0][0])

# sabit (b)
print(reg_model.intercept_[0])

##### Tahmin

# 150 birimlik TV harcaması olsa satıs ne olur?
print(reg_model.intercept_[0] + reg_model.coef_[0][0] * 150)

## modelin görselleştirilmesi
vis = sns.regplot(x=X, y=y, scatter_kws={"color": "g", "s": 9},
                  ci=False, color="b")  # model tahmin değerleri mavi, gerçek değerler yesil
vis.set_title(f"Model Denklemi: Sales = {round(reg_model.intercept_[0], 2)} + TV*{round(reg_model.coef_[0][0], 2)}")
vis.set_ylabel("Satıs Sayısı")
vis.set_xlabel("TV Harcamaları")
plt.xlim(-10, 310)
plt.ylim(bottom=0)
plt.show()

##### Tahmin Başarısı

# MSE
y_pred = reg_model.predict(X)  # tahmin edilen değerler
mean_squared_error(y, y_pred)  # gerçek değerler ve tahmin değerleri prmteler

y.mean()  # gercek bagımlı değişken değerlerin ort

y.std()  # gercek bagımlı değişken degerlerin std sapması, degerler 9-19 arasında değişiyor

# RMSE
np.sqrt(mean_squared_error(y, y_pred))

# MAE
mean_absolute_error(y, y_pred)

# R-KARE, veri setindeki bagımsız degisklerin bagımlı degiskeni acıklama yuzdesidir
print(reg_model.score(X, y))

##### *Değişken sayısı arttıkca R-kare şişebilir. bu yuzden duzeltilmiş r karenin de değeri göz onune alınmalıdır.
###### *Katsayıların anlamlılığı, modelin anlamlılığı ile ilgilenmiyoruz, optimizasyon, makine öğrenmesi açısından yüksek tahmin başarısıyla ilgileniyoruz. doğrusal formda tahmin etmek.
##### *Gelişmiş regresyon problemleri çözmek için doğrusal değil ağaca dayalı yöntemleri daha cok kullanacağız.

##### Multiple Linear Regression

X = df.drop("sales", axis=1)
y = df[["sales"]]

##### Model

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

print(X_test.shape)
print(y_train.shape)

reg_model = LinearRegression().fit(X_train, y_train)

# b
print(reg_model.intercept_[0])

# w
print(reg_model.coef_[0])


##### Tahmin

# regresyon cıktısı
print(reg_model.intercept_[0] + reg_model.coef_[0][0] * 30 + reg_model.coef_[0][1] * 10 + reg_model.coef_[0][2] * 40)

veri_deneme = [[30], [10], [40]]
veri_deneme = pd.DataFrame(veri_deneme).T
reg_model.predict(veri_deneme)  # tahmin

###### Tahmin Başarısı

y_pred = reg_model.predict(X_train)
np.sqrt(mean_squared_error(y_train, y_pred))  # train RMSE

reg_model.score(X_train, y_train)  # train R-Kare

y_pred = reg_model.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))  # test RMSE

reg_model.score(X_test, y_test)  # test R-Kare

# 10 katlı cv RMSE
np.mean(np.sqrt(-cross_val_score(reg_model, X, y, cv=10, scoring="neg_mean_squared_error")))


# crossun basında - olmasının sebebi negatif sonuclar vermesi, negatif hata olmayacağından - ile carpılarak pozitif sayılar elde edilir.
# veri setinin boyutu az oldugundan capraz dogrulamaya bakmak daha mantıklı olabilir

### BONUS

##### Simple Linear Regression with Gradient Descent from Scratch

# Cost function MSE
def cost_function(Y, b, w, X):
    num_of_ob = len(Y)
    sse = 0

    for i in range(0, num_of_ob):
        y_pred = b + w * X[i]  # bagimli tahmin degerleri
        y_sum = (Y[i] - y_pred) ** 2
        sse += y_sum

    mse = sse / num_of_ob
    return mse


# gradient descent
def update_weights(Y, b, w, X, learning_rate):
    num_of_ob = len(Y)
    b_derivative_sum = 0
    w_derivative_sum = 0

    for i in range(0, num_of_ob):
        y_pred = b + w * X[i]  # bagimli tahmin degerleri
        y = Y[i]  # bagimli gercek degerleri
        b_derivative_sum += (y_pred - y)  # b ye gore kismi turev
        w_derivative_sum += (y_pred - y) * X[i]  # w ye gore kismi turev

    new_b = b - (learning_rate * b_derivative_sum * 1 / num_of_ob)
    new_w = w - (learning_rate * w_derivative_sum * 1 / num_of_ob)
    return new_b, new_w


# train fonksiyonu
def train(Y, initial_b, initial_w, X, learning_rate, num_iters):
    print('b:{} w:{} MSE:{}'.format(initial_b, initial_w, cost_function(Y, initial_b, initial_w, X)))

    b = initial_b
    w = initial_w
    cost_h = []

    for i in range(num_iters):
        b, w = update_weights(Y, b, w, X, learning_rate)
        mse = cost_function(Y, b, w, X)
        cost_h.append(mse)
        if i % 500 == 0:
            print('iter:{} b:{:.2f}, w:{:.2f}, MSE:{:.2f}'.format(i, b, w, mse))

    print('after {} iterations b: {:.2f}, w: {:.2f}, MSE: {:.2f}'.format(num_iters, b, w, cost_function(Y, b, w, X)))
    return cost_h, b, w


df = pd.read_csv('advertising.csv')
X = df['radio']
Y = df['sales']

# hyperparameters

initial_b = 0.001
initial_w = 0.001
learning_rate = 0.001
num_iters = 10000

cost_h, b, w = train(Y, initial_b, initial_w, X, learning_rate, num_iters)
