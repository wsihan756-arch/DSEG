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
        """Save initial weights"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        """Update momentum weights: shadow = decay * shadow + (1 - decay) * current"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        """Replace the model weights with EMA weights"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self):
        """Restore original weights"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.backup[name])
        self.backup = {}

def set_seed(seed):
    """Fixed random seed"""
    
    # Python
    random.seed(seed)
    # Hash
    os.environ['PYTHONHASHSEED'] = str(seed)
    # NumPy
    np.random.seed(seed)
    # PyTorch CPU
    torch.manual_seed(seed)
    # mps
    torch.mps.manual_seed(seed)

def knn_graph(X, k):
    """Construct the true adjacency matrix A of KNN based on the original feature X"""
    
    # The input is a tensor, which is first converted to a NumPy array for use with sklearn.
    X_np = X.detach().cpu().numpy()

    n_samples = X_np.shape[0]
    k = min(k, n_samples) 
    
    # Using sklearn to build KNN graphs
    # include_self=True:Add 1 on the diagonal to add a self-loop
    A_sparse = kneighbors_graph(X_np, n_neighbors=k, mode='connectivity', include_self=True)
    
    # Symmetry processing
    # A = max(A, A^T) This ensures that as long as one party considers the other a neighbor, the two parties are connected.
    A_sparse = A_sparse.maximum(A_sparse.T)
        
    # Convert scipy sparse matrices to dense format, then convert them to PyTorch Tensors.
    return torch.FloatTensor(A_sparse.toarray())
    
def normalize(A):
    """The normalized Laplacian matrix required for computing graph convolutional networks: D^{-1/2} A D^{-1/2}"""

    A = A.numpy() 
    # Calculate the degree matrix D 
    d = np.array(A.sum(1))

    # Calculate D^{-1/2}
    D1 = np.power(d, -0.5).flatten()
    
    # To prevent division by zero from resulting in infinity (in the case of isolated nodes).
    D1[np.isinf(D1)] = 0.
    
    # Construct a diagonal matrix D^{-1/2}
    D2 = sp.diags(D1)
    
    # A_norm = D^{-1/2} * A * D^{-1/2}
    A_sparse = sp.csr_matrix(A)
    A_norm = A_sparse.dot(D2).transpose().dot(D2)
    
    return torch.FloatTensor(A_norm.toarray())

