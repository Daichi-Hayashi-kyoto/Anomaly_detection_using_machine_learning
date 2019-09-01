import numpy as np
from dtw import dtw

'''
データ数が多いと時間がすごくかかる

'''


class Knn:
    
    def __init__(self, k, train_data, test_data, window_size):

        self.k = k
        self.train_data = train_data
        self.test_data = test_data
        self.window_size = window_size

        if not isinstance(self.train_data, np.ndarray):
            self.train_data = np.array(self.train_data)


        if not isinstance(self.test_data, np.ndarray):
            self.test_data = np.array(self.test_data)

        self.train_data = self.train_data.flatten()
        self.test_data = self.test_data.flatten()

    
    def train(self):

        '''
        スライド窓による部分時系列データの作成
        '''
        if self.window_size ==  None:
            self.train_vec = self.train_data
        
        else:
            self.train_vec = []
            for i in range(len(self.train_data) - self.window_size):
                self.train_vec.append(self.train_data[i:i + self.window_size])

            self.train_vec = np.array(self.train_vec)
            
        
        
    def detection(self, dist):

        '''
        distは用いる距離関数
        '''
        test_vec = []
        neighbor_distance = []
        
        if self.window_size == None:
            test_vec = self.test_data
        
        else:
            
            '''
            スライド窓による部分時系列データの作成
            '''
        
            for i in range(len(self.test_data) - self.window_size):
                test_vec.append(self.test_data[i:i + self.window_size])

            test_vec = np.array(test_vec)
            
        n = len(test_vec)

        if dist == 'Euclid':
            model = self.euclid_dist
            for i in range(n):
                distance = [model(value, test_vec[i]) for value in self.train_vec]
                distance.sort()
                neighbor_distance.append(np.mean(distance[:self.k]))
        
        elif dist == 'dtw':
            euclidean_norm = lambda x, y: np.abs(x - y)
            x = self.train_vec.reshape(-1, 1)
            distance = []
            for i in test_vec:
                d, _, _, _ = dtw(x, i.reshape(-1, 1), dist = euclidean_norm)
                neighbor_distance.append(d)

        else:
            print("Your select distance function is not supported")

        return neighbor_distance

    def euclid_dist(self, y, y_pred):
        return np.linalg.norm(y - y_pred)