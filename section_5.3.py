import numpy as np
import pandas as pd
from scipy import stats 
from statistics import mean, pvariance, variance
import matplotlib.pyplot as plt
import matplotlib.patches as pat
import japanize_matplotlib
#%matplotlib inline
plt.style.use("seaborn")

data = pd.read_csv("Cars93.csv")

# データのマスク
mask_set = ["Min.Price", "Price", "Max.Price", "MPG.city", "MPG.highway", "EngineSize", "Horsepower", "RPM", "Rev.per.mile", 
          "Fuel.tank.capacity", "Length", "Wheelbase", "Width", "Turn.circle", "Weight"]

df_1 = data[mask_set]
df_1.head()
 
# 中心化行列の作成
Xc = stats.zscore(df_1).T

Xc = pd.DataFrame(Xc)   # データフレーム型へ変更
Xc.columns=data["Make"]   # 列名称の変更
Xc

# 散布行列の作成と固有値分解
S = Xc.dot(Xc.T)
eigen_value, eigen_vec = np.linalg.eig(S)

# 散布行列に対する固有値の大きさをプロット
plt.figure(figsize = (10, 8))
plt.xlabel("index of eigen number")
plt.ylabel("eigen value")
plt.plot(eigen_value, "o-")
plt.show()

m = 2
x_2 = np.dot(eigen_vec[:, :m].T, Xc)  # 主部分空間の成分を計算

U_m = eigen_vec[:, :m]
E_m = np.identity(n = Xc.shape[1])

a_1 = np.sum(Xc * Xc, axis = 0) - np.sum(x_2 * x_2, axis = 0)    # 列方向にたす
A_1 = pd.DataFrame(a_1)
A_1.index = data["Make"]
A_1.columns = {"Anomaly_score"}
A_1.sort_values(by = "Anomaly_score", ascending = False)

# 主成分分析の可視化

z_n = np.dot(U_m.T, Xc)


