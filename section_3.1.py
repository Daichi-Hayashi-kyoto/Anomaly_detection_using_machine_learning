# 3章 非正規データからの異常検知
import pandas as pd
from statistics import mean, pvariance, variance
from scipy import stats
import matplotlib.pyplot as plt
#%matplotlib inline          # jupyter notebookでは必要

data = pd.read_csv("./R_dataset/Davis.csv")

import numpy as np
import math

def Gamma_dist(x, k, s):
    x = np.array(x)
    return (x/s)**(k-1) *  np.exp(-(x/s))/(s*math.gamma(x))

mu = mean(data["weight"])   # 標本平均
sigma = pvariance(data["weight"])   # 標本分散

# モーメント法によるパラメータ推定
k_mo = mu**2/sigma
s_mo = sigma/mu

plt.figure(figsize=(8,8))
plt.hist(data["weight"], bins=30)
x = np.linspace(40, 120, num=1000)
plt.plot(x, Gamma_dist(x, k_mo, s_mo), 'r-')
plt.show()


# gamma分布による異常度の定義

n = data.shape[0]
anomaly_score = np.zeros(n)
for i in range(n):
    anomaly_score[i] = data["weight"][i]/s_mo - (k_mo - 1) * np.log(data["weight"][i]/s_mo)

threatened = sorted(anomaly_score, reverse= True)[1]

plt.plot(anomaly_score, 'bo')
plt.plot([0,200], [threatened, threatened], 'r-')
plt.show()

