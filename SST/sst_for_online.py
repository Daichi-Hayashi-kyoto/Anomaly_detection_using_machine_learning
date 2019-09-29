'''
逐次更新型の得意スペクトル変換
'''

import numpy as np
import panda as pd
import matplotlib.pyploto as plt
import seaborn
plt.style.use("seaborn")

# file path は適時変更してください.
data = pd.read_csv("./dockerfiles_rubisdb_1_20190818-1900_20190819-0900_5.csv")
df_1 = data["node_cpuusage"]    # 異常検知したいデータ

X = df_1.as_matrix()    # numpy 配列に変換


def create_matrix(data, start, end, window_size):

    '''
    行列の抽出
    '''

    row_size = window_size
    column_size = end - start + 1
    A = np.empty((row_size, column_size))
    i = 0
    for t in range(start, end+1):
        A[:, i] = data[t-1:t-1+row_size]   # len(data[t-1:t-1+row_size]) = row_sizeになるよ 
        i += 1

    return A


def sst(data, window_size, m , k = None, L = None):

    '''
    m : パターン数(主成分分析した時の採用ベクトル本数)
    k : 履歴行列の列サイズ
    L : ラグの大きさ
    data: numpy 配列を仮定
    '''

    # k, L に指定がなかったらこれを適用する
    if k is None:
        k = int(np.round(window_size/2))    

    if L is None:
        L = k
        #L = int(np.round(k/2))   

    T = len(data)

    # 計算範囲
    start_calu = k + window_size       # kを足すのは，
    end_calu = T - L + 1


    changing_scores = np.zeros(len(data))
    init_sample_size = int(np.round(len(data) * 0.02 ))    # 初めのsample_queue size
    sample_queue = data[:init_sample_size]                 # このとき，numpy 配列
    count = 0

    for t in range(start_calu, end_calu + 1):
        
        # 履歴行列
        start_traject = t - window_size - k + 1
        end_traject = t - window_size
        traject_matrix = create_matrix(sample_queue, start_traject, end_traject, window_size)

        # test matrix
        start_test = start_traject + L
        end_test = end_traject + L
        test_matrix = create_matrix(sample_queue, start_test, end_test, window_size)

        # 特異値分解
        U, _, o = np.linalg.svd(traject_matrix, full_matrices = False)   # 一つ目の返り値だけがほしいのでこのようにする
        Q, _, o = np.linalg.svd(test_matrix, full_matrices = False)        # 上記と同様
        u_m = U[:, :m]    # 列成分をm個抽出
        q_m = Q[:, :m]    # 列成分をm個抽出

        eigen_values = np.linalg.svd(u_m.T @ q_m, full_matrices = False, compute_uv = False)    # @ は行列の積演算
        changing_scores[t] = 1 - eigen_values[0]     # np.linalg.svd()の返り値は特異値の値が降順に並んでいる．最大特異値が欲しいので，このようにする．
        list(sample_queue)
        sample_queue.append(data[init_sample_size + count])
        count += 1
        sample_queue = np.array(sample_queue)

    return changing_scores

