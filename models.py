import torch
import torch.nn as nn
import torch.nn.functional as F

class SemanticEncoder(nn.Module):
    """语义编码器 (MLP)"""
    def __init__(self, input_dim, hidden_dim, z_dim):
        super(SemanticEncoder, self).__init__()
        layers = []
        in_d = int(input_dim)
        for h in hidden_dim:
            layers.append(nn.Linear(in_d, h))
            layers.append(nn.ReLU(inplace=True))
            in_d = int(h)
        layers.append(nn.Linear(in_d, int(z_dim)))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class SemanticDecoder(nn.Module):
    """语义解码器 (MLP)"""
    def __init__(self, z_dim, hidden_dim, original_dim):
        super(SemanticDecoder, self).__init__()
        layers = []
        in_d = z_dim
        for h in hidden_dim:
            layers.append(nn.Linear(in_d, h))
            layers.append(nn.ReLU(inplace=True))
            in_d = int(h)
        layers.append(nn.Linear(in_d, original_dim))
        layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        return self.net(z)

class GCNLayer(nn.Module):
    """图卷积基础层"""
    def __init__(self, in_features, out_features):
        super(GCNLayer, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x, adj):
        support = torch.mm(x, self.weight)
        if adj.is_sparse:
            output = torch.spmm(adj, support)
        else:
            output = torch.mm(adj, support)
        return output

class StructuralEncoder(nn.Module):
    """结构编码器 (GCN)"""
    def __init__(self, input_dim, hidden_dim, h_dim):
        super(StructuralEncoder, self).__init__()
        dim_list = [input_dim] + [x for x in hidden_dim] + [h_dim]
        self.layers = nn.ModuleList()
        for i in range(len(dim_list) - 1):
            self.layers.append(GCNLayer(dim_list[i], dim_list[i + 1]))

    def forward(self, x, adj):
        h = x
        for i, layer in enumerate(self.layers):
            h = layer(h, adj)
            if i < len(self.layers) - 1:
                h = F.relu(h)
        return h

class MaskGenerator(nn.Module):
    """擦除掩码生成器"""
    def __init__(self, z_dim, h_dim, mask_dim):
        super(MaskGenerator, self).__init__()
        self.W_gate = nn.Linear(z_dim + h_dim, mask_dim)

    def forward(self, Z, h):
        # 对Z、h做L2行归一化再concat，避免某一侧尺度过大主导gate
        Z = F.normalize(Z, p=2, dim=1)
        h = F.normalize(h, p=2, dim=1)
        Z_h = torch.cat([Z, h], dim=-1)
        logits = self.W_gate(Z_h)
        M = torch.sigmoid(logits)
        return M
class ProjectionHead(nn.Module):
    """投影头"""
    def __init__(self, input_dim, hidden_dim, shared_dim):
        super(ProjectionHead, self).__init__()
        layers = []
        in_d = input_dim
        for h in hidden_dim:
            layers.append(nn.Linear(in_d, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU(inplace=True))
            in_d = h
        layers.append(nn.Linear(in_d, shared_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
class DSEG(nn.Module):
    def __init__(self, num_views, input_dims, hidden_dim, z_dim, h_dim, shared_dim):
        super(DSEG, self).__init__()
        self.num_views = num_views
        self.sem_encoders = nn.ModuleList([SemanticEncoder(input_dims[v], hidden_dim, z_dim) for v in range(self.num_views)])
        self.str_encoders = nn.ModuleList([StructuralEncoder(input_dims[v], hidden_dim, h_dim) for v in range(self.num_views)])
        self.sem_decoders = nn.ModuleList([SemanticDecoder(z_dim, hidden_dim, input_dims[v]) for v in range(self.num_views)])
        self.mask_generators = nn.ModuleList([MaskGenerator(z_dim, h_dim, mask_dim=z_dim) for v in range(self.num_views)])
        self.projectors = nn.ModuleList([ProjectionHead(z_dim, hidden_dim, shared_dim) for v in range(self.num_views)])

    def forward(self, X_list, A_norm_list):
        Z_list, h_list, A_hat_list = [], [], []
        M_list, E_list, S_list, X_hat_list, ProjE_list = [], [], [], [], []
        
        for v in range(self.num_views):
            Z = self.sem_encoders[v](X_list[v])
            
            h = self.str_encoders[v](X_list[v], A_norm_list[v])
            # h 行归一化后再算相似度，避免 h 范数过大导致 sigmoid 饱和、梯度消失
            h_norm = F.normalize(h, p=2, dim=1)
            A_hat = torch.sigmoid(torch.matmul(h_norm, h_norm.t()))            
            M = self.mask_generators[v](Z, h)
            E = M * Z         
            S = (1 - M) * Z     
            X_hat = self.sem_decoders[v](Z)
            ProjE = self.projectors[v](E)
            ProjE_list.append(ProjE)
            Z_list.append(Z)
            E_list.append(E)
            S_list.append(S)
            M_list.append(M)
            X_hat_list.append(X_hat)
            A_hat_list.append(A_hat)

        return X_hat_list, A_hat_list, E_list, S_list, M_list, ProjE_list