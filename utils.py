import numpy as np
import torch
import random
import os
import scipy.sparse as sp
from sklearn.neighbors import kneighbors_graph
class EMA():
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def register(self):
        """保存初始权重"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        """更新动量权重: shadow = decay * shadow + (1 - decay) * current"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        """将模型权重替换为EMA权重"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self):
        """恢复原始权重"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.backup[name])
        self.backup = {}

def set_seed(seed):
    """固定随机种子"""
    # 固定 Python 内置随机种子
    random.seed(seed)
    # 固定 Hash 随机种子
    os.environ['PYTHONHASHSEED'] = str(seed)
    # 固定 NumPy 随机种子
    np.random.seed(seed)
    # 固定 PyTorch CPU 随机种子
    torch.manual_seed(seed)
    # 固定 mps 随机种子
    torch.mps.manual_seed(seed)

def knn_graph(X, k):
    """基于原始特征 X 构建 KNN 真实邻接矩阵 A"""
    # 输入是张量 先转换为NumPy数组以便使用sklearn
    X_np = X.detach().cpu().numpy()

    n_samples = X_np.shape[0]
    k = min(k, n_samples) 
    # 使用 sklearn 构建 KNN 图
    # mode='connectivity' 返回 0/1 矩阵，代表是否连接
    # include_self=True 在对角线上加 1 添加自环
    A_sparse = kneighbors_graph(X_np, n_neighbors=k, mode='connectivity', include_self=True)
    
    # 对称化处理
    # A = max(A, A^T) 保证只要一方把另一方当做邻居，两者就相连
    A_sparse = A_sparse.maximum(A_sparse.T)
        
    # 将 scipy 的稀疏矩阵转换为 dense 格式，再转为 PyTorch Tensor
    return torch.FloatTensor(A_sparse.toarray())
    
def normalize(A):
    """计算图卷积网络所需的归一化拉普拉斯矩阵: D^{-1/2} A D^{-1/2}"""

    A = A.numpy() 
    # 计算度矩阵 D (即每一行的非零元素个数总和)
    d = np.array(A.sum(1))

    # 计算 D^{-1/2}
    D1 = np.power(d, -0.5).flatten()
    # 防止除以 0 产生无穷大 (孤立节点的情况)
    D1[np.isinf(D1)] = 0.
    
    # 构建对角矩阵 D^{-1/2}
    D2 = sp.diags(D1)
    
    # A_norm = D^{-1/2} * A * D^{-1/2}
    A_sparse = sp.csr_matrix(A)
    A_norm = A_sparse.dot(D2).transpose().dot(D2)
    
    return torch.FloatTensor(A_norm.toarray())

