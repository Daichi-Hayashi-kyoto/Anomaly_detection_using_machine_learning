import numpy as np

class ar:

    def __init__(self, train_data, test_data, order_list):

        '''
        入力次元は1次元．将来的には多次元にも対応していく．
        ARモデルによる異常検知
        y: 目的変数
        X: 特徴量
        order_list : 次数の候補
        '''

        self.y_train = train_data
        self.y_test = test_data
        self.order_list = order_list
        self.T = len(self.y_train)
        self.aic = None
    
    def train(self):

        '''
        窓スライドによる部分時系列の作成
        '''
        count = 0

        for r in self.order_list:

            self.N = self.T - r
            self.X = np.zeros((r, self.N))
            self.y = np.zeros((self.N, 1))
            x_t = np.zeros((r, 1))

            for i in range(self.N):
                
                x_t = self.y_train[i:i+r]
                self.X[:, i] = x_t
                self.y[i, :] = self.y_train[i+r]

            '''
            パラメータの推定
            '''
            xx_t = self.X @ self.X.T

            if count == 0:
                self.alpha = np.linalg.solve(xx_t, np.identity(r)) @ self.X @ self.y
                self.sigma = np.mean((self.y - self.alpha.T @ self.X)**2)
                self.aic = np.log(self.sigma) + 2 * r /self.N
                self.best_alpha = self.alpha
                self.best_sigma = self.sigma
                self.best_order = r

            else:
                self.alpha = np.linalg.solve(xx_t, np.identity(r)) @ self.X @ self.y
                self.sigma = np.mean((self.y - self.alpha.T @ self.X)**2)
                tmp_aic = np.log(self.sigma) + 2 * r /self.N
                
                if tmp_aic < self.aic:
                    self.aic = tmp_aic
                    self.best_alpha = self.alpha
                    self.best_sigma = self.sigma
                    self.best_order = r
            count += 1

        return self.aic, self.best_alpha, self.best_sigma, self.best_order


    def detection(self):
        
        self.train()
        r = self.best_order
        N = self.T - r
        anomaly_score = np.zeros(N)
        for i in range(N):
            y_real = self.y_test[r+i]
            X = self.y_test[i:r+i]
            y_pred = self.best_alpha.T @ X
            anomaly_score[i] = ((y_real - y_pred)**2)/self.best_sigma

        return anomaly_score