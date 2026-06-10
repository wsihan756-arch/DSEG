import numpy as np
from sklearn import metrics
from scipy.optimize import linear_sum_assignment

def cluster_acc(y_true, y_pred):
    y_true = np.array(y_true).astype(np.int64)
    y_pred = np.array(y_pred).astype(np.int64)    
    # 类别总数（取两者最大值，防止KMeans分配了真实标签中不存在的编号）
    D = max(y_pred.max(), y_true.max()) + 1
    # 构建代价矩阵 (混淆矩阵的变体)
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
        
    # 匈牙利算法 求最大匹配 最小代价
    row, col = linear_sum_assignment(w.max() - w)
    acc = sum([w[i, j] for i, j in zip(row, col)]) * 1.0 / y_pred.size
    
    return acc

def evaluate_clustering(y_true, y_pred):

    acc = cluster_acc(y_true, y_pred)
    nmi = metrics.normalized_mutual_info_score(y_true, y_pred)
    ari = metrics.adjusted_rand_score(y_true, y_pred)
    
    return acc, nmi, ari

