import torch
import torch.nn as nn
import torch.nn.functional as F

def rec(X_list, X_hat_list, A_list, A_hat_list):
    """Calculate l_rec (l_recon + l_str)"""
    V = len(X_list)
    l_recon, l_str = 0.0, 0.0
    
    for v in range(V):
        X_v, X_hat_v = X_list[v], X_hat_list[v]
        A_v, A_hat_v = A_list[v], A_hat_list[v]
        N = X_v.size(0)      
        
        # Semantic reconstruction loss (normalized 1/N)
        l_recon += torch.sum((X_hat_v - X_v) ** 2) / N
        
        # Structural reconstruction loss (normalized 1/N^2)
        l_str += torch.sum((A_hat_v - A_v) ** 2) / (N * N)
        
    return l_recon + l_str

def decorr(E_list, S_list, M_list):
    """Calculate l_decorr"""
    V = len(E_list)
    loss1, loss2 = 0.0, 0.0

    for v in range(V):
        e_v, s_v, m_v = E_list[v], S_list[v], M_list[v]
        
       # Perform L2 row normalization before calculating E^T S to ensure the values ​​of the correlation matrix are bounded and avoid L_decorr explosion.
        e_norm = F.normalize(e_v, p=2, dim=1) 
        s_norm = F.normalize(s_v, p=2, dim=1)
        
        es = torch.matmul(e_norm.t(), s_norm)
        
        loss1 += torch.sum(es ** 2)
        loss2 += torch.mean(m_v) 

    return (loss1 / V) + (loss2 / V)

def nce(ProjE_list, tau=0.5, tau_g=0.5):
    """Calculate l_nce"""
    V = len(ProjE_list)
    
    # Calculate the global guidance distribution Q_global (similarity S is detached using Z.detach, without backpropagation).
    Z = torch.cat(ProjE_list, dim=1)
    Z_norm = F.normalize(Z.detach(), p=2, dim=1)
    S = torch.matmul(Z_norm, Z_norm.t())
    Q_global = F.softmax(S / tau_g, dim=1)

    # Traverse the view pairs, calculate the cross probability P_cross, and align them.
    loss = 0.0
    for v in range(V):
        for u in range(V):
            if v == u:
                continue
    
            e_v_norm = F.normalize(ProjE_list[v], p=2, dim=1)
            e_u_norm = F.normalize(ProjE_list[u], p=2, dim=1)
            
            sim_vu = torch.matmul(e_v_norm, e_u_norm.t())  
            P_cross = F.softmax(sim_vu / tau, dim=1)

            log_P_cross = torch.log(P_cross + 1e-8)
            loss_vu = F.kl_div(input=log_P_cross, target=Q_global.detach(), reduction='batchmean')
            loss += loss_vu

    return loss / (V * (V - 1))

class MultiViewLossTracker(nn.Module):
    """A wrapper class that manages all loss functions"""
    def __init__(self, lambda_1, lambda_2, tau, tau_g):
        super(MultiViewLossTracker, self).__init__()
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.tau = tau
        self.tau_g = tau_g

    def forward(self, X_list, A_list, X_hat_list, A_hat_list, E_list, S_list, M_list, ProjE_list, warmup, epoch):
        l_rec = rec(X_list, X_hat_list, A_list, A_hat_list)
        l_decorr = decorr(E_list, S_list, M_list)
        l_nce = nce(ProjE_list, self.tau, self.tau_g)
        if epoch >= warmup:
            l_total = l_rec + self.lambda_1 * l_decorr + self.lambda_2 * l_nce
        else:
            l_total = l_rec + self.lambda_1 * l_decorr
        
        loss_details = {
            'L_total': l_total.item(),
            'L_rec': l_rec.item(),
            'L_decorr': l_decorr.item(),
            'L_nce': l_nce.item()
        }
        return l_total, loss_details
