import pandas as pd
from statistics import mean, pvariance
from scipy import stats
import matplotlib.pyplot as plt
#%matplotlib inline   # jupyter notebook 内では必要

data = pd.read_csv("Davis.csv")
data

plt.hist(data['weight']);  # ;をつけると変なのが出ない

mu = mean(data['weight'])  # 標本平均
sigma = pvariance(data["weight"])  # 標本分散

# 異常度の定義
import math
a = ((data["weight"] - mu)/(math.sqrt(sigma)))**2

# 閾値の設定
th = stats.chi2.ppf(q=0.99, df=1) # dfは自由度
# 以下のようにしても同じ
th2 = stats.chi2.isf(q=0.01, df=1)

plt.figure(figsize=(10, 10))
plt.plot(a, 'o')
plt.xlabel("index number")
plt.ylabel("anomaly score")
plt.plot([0, 200], [th,th],  'r--')
plt.show()

anomaly_list=[]
for i in range(len(a)):
    if a[i] >= th:
        anomaly_list.append(i)

anomaly_list # 異常なindex値を表示

# 2.4多変量正規分布に基づく異常検知

X = data[["weight", "height"]]

plt.figure(figsize=(8, 8))
plt.xlabel("weight")
plt.ylabel("height")
plt.plot(X["weight"], X["height"], "bo")
plt.show()

import numpy as np
m = np.mean(X, axis = 0)
xc = X - m   # (200, 2)
S = np.dot(xc.T, xc)/X.shape[0]  # (200, 200)

S_inv = np.linalg.inv(S)

N = X.shape[0] 
a = np.zeros(N) # 初期化
# xc.iloc[i, :]でデータフレーム構造のi行目を抽出する
# 異常度の計算
for i in range(N):
    a[i] = np.dot(np.array(xc.iloc[i,:]).reshape(2,1).T , S_inv).dot(np.array(xc.iloc[i,:]).reshape(2,1))
    
a


plt.figure(figsize=(10, 10))
plt.xlabel("index number")
plt.ylabel("anomaly score")
plt.plot(a, "bo")
plt.plot([0,200], [th, th], "r--")
plt.show()

anomaly_index =[]
for i in range(N):
    if a[i] >= th:
        anomaly_index.append(i)

anomaly_index  # 異常度を超えた標本の番号が入っている



# 2.6 マハラノビス=タグチ法

import pandas as pd
import matplotlib.pyplot as plt
#%matplotlib inline   # jupyter notebookでは必要

# カレントディレクトリが/Users/hayashidaichi/である事が前提条件

road_data = pd.read_csv("./R_dataset/road.csv")
road_data


import numpy as np

X = road_data.copy()
name_list = list(map(str, X.columns)) # X.columns で変数名を獲得
name_list.pop(0)  # Unnamed: 0をname_listから削除
drivers_number = X["drivers"].copy()

# Xの各列をdriversにより割る
for name in name_list:
    X[name] /= drivers_number

name_list.remove("drivers")   # name_listからdriversを削除
X = X.drop("drivers", axis=1)   # 変数名driversの列を削除する
for name in name_list:
    X[name] = np.log(X[name] + 1)   # ボックス=コックス変換: 突発的にあたいが大きくなるようなデータの変動を穏やかにする


Y = X.drop("Unnamed: 0", axis=1)  # データフレームXからUnnames: 0 という名前の列を消す  
Y = Y.as_matrix()   # convert DataFrame to array

m = np.mean(Y, axis=0)    # 列方向の平均を取る(標本平均ベクトル)
y_c = Y - m              # 中心化行列
N = Y.shape[0]   # データ数
S_x = np.dot(y_c.T, y_c)/N    # 標本共分散行列
S_inv = np.linalg(S_x)       # 標本共分散行列の逆行列

# 1変数あたりの異常度の計算
n = Y.shape[0]   # データ数
M = Y.shape[1]   # 変数の数
anomaly_score = np.zeros(n)
for i in range(N):
        anomaly_score[i] = np.dot(y_c[i, ].reshape(1,5), S_inv).dot(y_c[i,].reshape(1,5).T)/M   # １変数当たりの異常度を求めたいので，Mで割る


th = 1.5
plt.figure(figsize=(8,8))
plt.xlabel("index number")
plt.ylabel("anomaly score")
plt.plot(anomaly_score, 'bo')
plt.plot([0,n], [th,th], 'r--')
plt.show()

# 閾値を超えている都市名の取得
n = Y.shape[0]
for i in range(n):
        if anomaly_score[i] >= th:
                print(X.iloc[i,0])     # X.iloc[i,j]でデータフレーム構造の(i,j)列目を獲得する


# SN比解析
xc_prime = y_c[4, ]
sn1 = 10*np.log10(xc_prime**2/np.diag(S_x))
label = ["deaths", "popden", "rural", "temp", "fuel"]
x = np.array([1, 2, 3, 4, 5])
plt.figure(figsize=(8,8))
plt.bar(x, sn1, tick_label = label, align="center")
plt.show()


sn_1 = []
index = [1, 4, 8, 18, 25]
for i in index:
        s = 10*np.log10(y_c[i, ]**2/np.diag(S_x))
        sn_1.append(s)

plt.figure(figsize=(6,6))
plt.subplot(2, 3, 1)
plt.bar(x, sn_1[0], tick_label = label, align='center')
plt.title("Alaska")

plt.subplot(2, 3, 2)
plt.bar(x, sn_1[1], tick_label = label, align = "center")
plt.title("Calif")

plt.subplot(2, 3, 3)
plt.bar(x, sn_1[2], tick_label = label, align = "center")
plt.title("DC")

plt.subplot(2, 3, 4)
plt.bar(x, sn_1[3], tick_label = label , align = "center")
plt.title("Maine")

plt.subplot(2, 3, 5)
plt.bar(x, sn_1[4], tick_label = label, align="center")
plt.title("Mont")
plt.show()
