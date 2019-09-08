# EMアルゴリズム

# 訓練データに異常標本が混ざっている場合
mu_0 = 3
mu_1 = 0
sigma_0 = 0.5  # 標準偏差
sigma_1 = 3    # 標準偏差

import random
import numpy as np
random.seed(1)   # 乱数固定

N = 1000 # 標本数
pi_0, pi_1 = 0.6, 0.4
prob = [pi_0, pi_1]
attr = np.random.choice((0,1), size = N, replace = True, p = prob)   # 復元抽出, probは確率の重み

x = np.zeros(N)   # 観測値の初期化

from collections import Counter
n_0, n_1 = Counter(attr)[0], Counter(attr)[1]    # 0 と 1の出現回数    意味なかったかも

for i in range(N):
    if attr[i] == 0:
        x[i] = np.random.normal(mu_0, sigma_0)  # N(mu_0, sigma_0)従う正規分布
    else:
        x[i] = np.random.normal(mu_1, sigma_1)  # N(mu_1, sigma_1)に従う正規分布


# EMアルゴリズム
from scipy import stats
for i in range(100):
    pi_N0 = pi_0 * stats.norm.pdf(x, loc = mu_0, scale = sigma_0)   # scaleは標準偏差であることに注意
    pi_N1 = pi_1 * stats.norm.pdf(x, loc = mu_1, scale = sigma_1)
    q_0 = pi_N0/(pi_N0 + pi_N1)
    q_1 = pi_N1/(pi_N0 + pi_N1)
    pi_0 = sum(q_0)/N    # 0への帰属度
    pi_1 = sum(q_1)/N    # 1への帰属度
    mu_0 = sum(q_0 * x)/(sum(q_0))
    mu_1 = sum(q_1 * x) / (sum(q_1))
    sigma_0 = np.sqrt(sum(q_0 * (x - mu_0)**2)/(sum(q_0)))
    sigma_1 = np.sqrt(sum(q_1 * (x - mu_1)**2)/(sum(q_1)))


print("EMアルゴリズムによる期待値の推定値はmu_0: {},  mu_1:{} \n 標準偏差の推定値は sigma_0:{},  sigma_1:{}\n 確率の重みは pi_0:{} pi_1:{}"
    .format(mu_0, mu_1, sigma_0, sigma_1, pi_0, pi_1))


# 混合ガウス分布の図
import matplotlib.pyplot as plt
x = np.linspace(-5, 6, num=100)
plt.figure(figsize=(10,10))
plt.ylabel('probability density')
plt.xlabel('x')
plt.plot(x, stats.norm.pdf(x, loc = 3, scale=0.5), 'b--', label = 'signal component')
plt.plot(x, stats.norm.pdf(x, loc = 0, scale = 3), 'r', linestyle='dashdot', label = 'noise component')
plt.plot(x, stats.norm.pdf(x, loc = 3, scale = 0.5)+stats.norm.pdf(x, loc = 0, scale=3), 'k-', label = 'observation value')
plt.legend()
plt.show()

# kernel density estimation

import pandas as pd
data = pd.read_csv("./R_dataset/Davis.csv")
x = data[["weight", "height"]]
X = np.array(x)

#分布図 
import seaborn as sns
plt.figure(figsize=(8,8))
plt.xlim(30, 110)
plt.ylim(140, 200)
plt.ylabel("height")
plt.xlabel("weight")
sns.kdeplot(X, shade=True)
plt.show()

# kernel density estimation ---

# カーネル行列の作成
from sklearn.metrics.pairwise import rbf_kernel
import math
K = 1/((2 * math.pi) * 0.1 ) * rbf_kernel(X, gamma = 1/2)
K   # カーネル行列

# 異常度の計算
n = X.shape[0]
aa = np.sum(K, axis=0) - np.diag(K)
#aaの中に0となる値があるとlogの特異点になるので，それを1e-20に置き換える
aa[aa < 1e-20] = 1e-20
anomaly_score = - np.log(aa/(n-1))

threatened = 20

plt.figure(figsize=(9,9))
plt.xlabel("index number")
plt.ylabel("anomaly score")
plt.plot(anomaly_score, 'bo')
plt.plot([0,200], [20,20], 'r--')
plt.show()

index = []
for i in range(n):
    if anomaly_score[i] >= threatened:
        index.append(i)

print("異常と思われる標本番号は{}".format(index))
