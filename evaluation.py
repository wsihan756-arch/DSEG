import numpy as np
from sklearn import metrics
from scipy.optimize import linear_sum_assignment

def cluster_acc(y_true, y_pred):
    """Calculate ACC"""
    y_true = np.array(y_true).astype(np.int64)
    y_pred = np.array(y_pred).astype(np.int64)    
    
    # Total number of categories (take the larger of the two to prevent KMeans from assigning numbers that do not exist in the actual labels)
    D = max(y_pred.max(), y_true.max()) + 1
    
    # Construct the cost matrix (a variant of the confusion matrix)
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
        
    # Hungarian algorithm
    row, col = linear_sum_assignment(w.max() - w)
    acc = sum([w[i, j] for i, j in zip(row, col)]) * 1.0 / y_pred.size
    
    return acc

def evaluate_clustering(y_true, y_pred):
    """Calculate clustering metrics"""
    acc = cluster_acc(y_true, y_pred)
    nmi = metrics.normalized_mutual_info_score(y_true, y_pred)
    ari = metrics.adjusted_rand_score(y_true, y_pred)
    
    return acc, nmi, ari

