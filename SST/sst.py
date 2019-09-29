'''
特異スペクトル変換の実装
'''
import numpy as np
import time

def create_matrix(data, start, end, window_size):

    '''
    スライド窓による行列の作成
    '''

    row_size = window_size
    column_size = end - start + 1
    matrix = np.empty((row_size, column_size))
    i = 0
    for t in range(start, end+1):
        matrix[:, i] = data[t-1:t-1+row_size]
        i += 1

    return matrix

def sst(data, window_size, m = 2, k = None, L = None):
    start_time = time.time()
    '''
    m : パターン数(主成分分析した時の採用ベクトル本数)
    k : 履歴行列の列サイズ
    L : ラグの大きさ
    data: numpy 配列を仮定
    '''

    '''
    if not isinstance(data, np.ndarray):   
        data = np.array(data)  

    '''

    if k is None:
        k = int(np.round(window_size/2))  


    if L is None:
        L = int(np.round(k))
        #L = int(np.round(k/2))

    # dataをsample queue に変えたら
    T = len(data)

    # 計算範囲
    start_calu = k + window_size
    end_calu = T - L + 1
    changing_scores = np.zeros(len(data))
    for t in range(start_calu, end_calu + 1):

        # 履歴行列
        start_traject = t - window_size - k + 1
        end_traject = t - window_size
        traject_matrix = create_matrix(data, start_traject, end_traject, window_size)

        # test matrix
        start_test = start_traject + L
        end_test = end_traject + L
        test_matrix = create_matrix(data, start_test, end_test, window_size)

        # 特異値分解
        U_traject, U_eigen_value, o = np.linalg.svd(traject_matrix, full_matrices = False)   # 一つ目の返り値だけがほしいのでこのようにする
        Q_test, _, o = np.linalg.svd(test_matrix, full_matrices = False)        # 上記と同様
        u_m = U_traject[:, :m]    # 列成分をm個抽出
        q_m = Q_test[:, :m]      # 列成分をm個抽出

        eigen_max = np.linalg.svd(u_m.T @ q_m, full_matrices = False, compute_uv=False)    # @は行列の積を表す演算
        changing_scores[t] = 1 -eigen_max[0]

    return changing_scores