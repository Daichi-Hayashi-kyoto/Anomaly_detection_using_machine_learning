import numpy as np
import matplotlib.pyplot as plt
plt.style.use("seaborn")
%matplotlib inline
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

df = pd.read_csv("/Rdatasets/csv/MASS/UScrime.csv")
X = df.drop(["So", "y"], axis=1)   # データ行列
X.head()
y = df["y"]
N = y.shape[0]   # 47個

# モデルの設定
lm_ridge = Ridge()   # Ridge classを入れる

# 正則化パラメータのGrid Searchを行う
lambdas = list(np.linspace(0, 5, 50))    # 正則化パラメータの候補
param_grid = [{'alpha': lambdas}]

grid_search = GridSearchCV(lm_ridge, param_grid, cv = 3)
grid_search.fit(y, X)

# Ridge回帰の一般化One-LV法
class G_Cross_validation:
    import numpy as np
    import pandas as pd

    def __init__(self, lambdas):
        self.lambdas = lambdas

    def estimator_params(self, y, X, lambdas):
        self.y = y.reshape(-1, 1)
        self.X = X
        N = self.X.shape[0]    # データ数
        H_n = np.identity(N) - 1/N * np.dot(np.ones(N), np.ones(N).T)   # 中心化行列

        X_c = np.dot(self.X, H_n)
        y_c = np.dot(H_n, self.y)
        
        A = np.linalg.solve(np.dot(X_c, X_c.T) + lambdas * np.identity(N), X_c)   # A = (X_c X_c.T + lambdas * E)^(-1) * X_cになる
        H = np.dot(X_c.T, A)     

        e_gcv = 1/N * (np.dot(y_c.T, (np.identity(N) - H.T)).dot(np.identity(N) - H).dot(y_c))/(1- np.trace(H)/N)**2   # 評価
        return e_gcv

    def GridSearch(self, lambdas, y, X):
        self.lambdas = lambdas
        self.y = y
        self.X = X
        e_gcv_list = []
        lambda_list = []
        for i in self.lambdas:
            e_gcv_list.append(estimator_params(self.y, self.X, self.lambdas))
            lambda_list.append(self.lambdas)

        j = np.argmin(e_gcv_list)
        return lambda_list[j]

            



