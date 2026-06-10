import os
import numpy as np
import scipy.io as sio
import sklearn.preprocessing as skp
import torch
from torch.utils.data import Dataset
from scipy import sparse
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import MaxAbsScaler
def load_mat(args):
    """加载数据集"""
    data_x = []
    Y = None
    if args.dataset == "Mfeat":
        mat = sio.loadmat(os.path.join('dataset', "Mfeat.mat"))
        Y = mat['truelabel'][0][0].squeeze().astype("int64")
        for v in mat['data'][0]:
            x_v = v.astype("float32")
            x_v = x_v.T
            data_x.append(x_v)

    elif args.dataset == "synthetic3d":
        mat = sio.loadmat(os.path.join('dataset', "synthetic3d.mat"))
        Y = mat["Y"].squeeze().astype("int64")
        for v in mat['X'].T[0]:
            x_v = v.astype("float32")
            data_x.append(x_v)
    
    elif args.dataset == "Hdigit":
        mat = sio.loadmat(os.path.join('dataset', "Hdigit.mat"))
        Y = mat['truelabel'][0][0].squeeze().astype("int64")
        for v in mat['data'][0]:
            x_v = v.astype("float32")
            x_v = x_v.T
            data_x.append(x_v)

    elif args.dataset == "BDGP":
        mat = sio.loadmat(os.path.join('dataset', "BDGP.mat"))
        Y = mat["Y"].squeeze().astype("int64")
        for v in [mat["X1"], mat["X2"]]:
            x_v = v.astype("float32")
            data_x.append(x_v)

    elif args.dataset == "Fashion":
        mat = sio.loadmat(os.path.join('dataset', "Fashion.mat"))
        Y = mat["Y"].squeeze().astype("int64")
        for v in [mat["X1"], mat["X2"], mat["X3"]]:
            v = v.reshape(v.shape[0], -1)
            x_v = v.astype("float32")
            data_x.append(x_v)
        
    elif args.dataset == "MNIST_USPS":
        mat = sio.loadmat(os.path.join('dataset', "MNIST_USPS.mat"))
        Y = mat["Y"].squeeze().astype("int64")
        for v in [mat["X1"], mat["X2"]]:
            v = v.reshape(v.shape[0], -1)
            x_v = v.astype("float32")
            data_x.append(x_v)
    
    args.n_views = len(data_x) 
    args.n_samples = Y.size
        
    return data_x, Y

class MultiviewDataset(Dataset):
    def __init__(self, data_x, label_y):
        self.n_views = len(data_x)
        self.data_x = []
        self.input_dims = []
        for v in data_x:
            # 删除方差为0的列，避免分母为0
            selector = VarianceThreshold(threshold=0)
            x_filtered = selector.fit_transform(v)
            # MinMaxScaler归一化到[0,1]，与解码器 sigmoid 输出一致
            scaler = MinMaxScaler()
            x_scaled = scaler.fit_transform(x_filtered).astype("float32")
            self.data_x.append(x_scaled)
            self.input_dims.append(int(x_scaled.shape[1]))
                        
        self.num_samples = int(label_y.size)
        self.targets = label_y

    def __len__(self):
        return self.num_samples
    def __getitem__(self, idx):
        # 切片操作 1D向量
        x_list = [torch.tensor(self.data_x[v][idx], dtype=torch.float32) for v in range(self.n_views)] 
        y = torch.tensor(self.targets[idx], dtype=torch.long)
        
        return idx, x_list, y
def load_dataset(args):
    data_x, targets = load_mat(args)
    dataset = MultiviewDataset(data_x=data_x, label_y=targets)
    args.input_dims = dataset.input_dims
    return args.input_dims, dataset