from math import inf
import torch
import torch.optim as optim
import argparse
import configs
import json
import numpy as np
import random
import os
from models import DSEG
from losses import MultiViewLossTracker
from dataloader import load_dataset
from evaluation import evaluate_clustering
from utils import knn_graph, normalize, set_seed, EMA
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize as sk_normalize

# ----------------------- Argument Parser Setup -----------------------------------
parser = argparse.ArgumentParser(description='DSEG Super Parameters')

parser.add_argument('--dataset', type=str, default='synthetic3d', help='Dataset Name')
parser.add_argument('--lambda_1', type=float, default=0.1, help='Weight of l_decorr')
parser.add_argument('--lambda_2', type=float, default=1.0, help='Weight of l_nce')
parser.add_argument('--tau', type=float, default=0.5, help='Temperature parameter of cross-prediction distribution')
parser.add_argument('--tau_g', type=float, default=0.5, help='Temperature parameter of globally guided distribution')
parser.add_argument('--batch_size', type=int, default=1024, help='Training batch size')
parser.add_argument('--hidden_dim', type=str, default='256,512,1024', help='Hidden layer dimensions')
parser.add_argument('--z_dim', type=int, default=32, help='Semantic feature dimension')
parser.add_argument('--h_dim', type=int, default=32, help='Structural feature dimension')
parser.add_argument('--shared_dim', type=int, default=512, help='Projector head dimension')
parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
parser.add_argument('--weight_decay', type=float, default=5e-6, help='Weight decay')
parser.add_argument('--warmup', type=int, default=0, help='Preheating')
parser.add_argument('--k_nn', type=int, default=10, help='Number of neighboring nodes')
parser.add_argument('--n_init', type=int, default=10, help='KMeans initialization count')
parser.add_argument('--gate_lr', type=float, default=0.001, help='Learning rate of gate')
parser.add_argument('--w_ema', type=float, default=0.999, help='Weight decay of EMA')

args = parser.parse_args()

if __name__ == "__main__":
    
    device = torch.device("mps")
    args = configs.get_config(args)
    set_seed(args.seed)
    # ----------------------- Data Loading & Preparation -----------------------
    print('='*40)
    print(args)  
    print('='*40)

    input_dims, dataset = load_dataset(args)
    num_views = len(input_dims)
    n_clusters = len(np.unique(dataset.targets))
    hidden_dim = tuple(int(x.strip()) for x in args.hidden_dim.strip().split(','))
    # shuffle = True: Shuffle the order of all samples in the dataset
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # ----------------------- Model & Optimizer Setup --------------------------
    model = DSEG(num_views, input_dims, hidden_dim, args.z_dim, args.h_dim, args.shared_dim).to(device)

    loss_tracker = MultiViewLossTracker(args.lambda_1, args.lambda_2, args.tau, args.tau_g)

    ema = EMA(model, decay=args.w_ema)
    ema.register()

    # Adam optimizer group learning rate
    gate_params = []
    other_params = []
    for name, param in model.named_parameters():
        if "gate" in name or "mask" in name:
            gate_params.append(param)
        else:
            other_params.append(param)
    optimizer = torch.optim.Adam([
        {'params': other_params, 'lr': args.lr},
        {'params': gate_params, 'lr': args.lr*args.gate_lr }],
        weight_decay=args.weight_decay)
    
    # Cosine annealing learning rate scheduling
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Create results directory
    os.makedirs('train_results', exist_ok=True)
    from datetime import datetime
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # ------------------------------- Training ------------------------------------
    epoch_results = []
    for epoch in range(args.epochs):
        model.train()
        l_total = 0.0
        l_rec = 0.0
        l_decorr = 0.0
        l_nce = 0.0
        for idx, x_list, y in loader:
            x_list = [x.to(device) for x in x_list]
            # Each view is constructed using KNN and normalized
            A_real_list = []
            A_norm_list = []
            for v in range(num_views):
                A_real = knn_graph(x_list[v], args.k_nn)
                A_norm = normalize(A_real)
                A_real_list.append(A_real.to(device))
                A_norm_list.append(A_norm.to(device))

            optimizer.zero_grad()
            X_hat_list, A_hat_list, E_list, S_list, M_list, ProjE_list = model(x_list, A_norm_list)

            total_loss, loss_details = loss_tracker(x_list, A_real_list, X_hat_list, A_hat_list, E_list,
                                                S_list, M_list, ProjE_list, args.warmup, epoch)
            # Backpropagation
            total_loss.backward()
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            # Update parameters
            optimizer.step()
            # Update EMA parameters
            ema.update()

            l_total += total_loss.item()
            l_rec += loss_details['L_rec']
            l_decorr += loss_details['L_decorr']
            l_nce += loss_details['L_nce']
            
        # This training session is over. update lr
        scheduler.step()
        # ----------------------------- Evaluation ----------------------------------
        if (epoch + 1) % 10 == 0 and (args.dataset == 'BDGP' or args.dataset == 'MNIST_USPS' or args.dataset == 'Hdigit'):
            print(f"Epoch [{epoch+1}/{args.epochs}] | Total Loss: {total_loss:.4f}| L_rec: {l_rec:.4f}| L_decorr: {l_decorr:.4f}| L_NCE: {l_nce:.4f}")
            model.eval()
            with torch.no_grad():
                ema.apply_shadow()
                all_E = []
                all_y = []
                for idx, x_list, y in loader:
                    x_list = [x.to(device) for x in x_list]
                    A_norm_list = [normalize(knn_graph(x_list[v], args.k_nn)).to(device) for v in range(num_views)]
                    _, _, E_list, _, _, _ = model(x_list, A_norm_list)
                    all_E.append(torch.cat(E_list, dim=1).cpu().numpy())
                    all_y.append(y.numpy())
                Z_global = np.vstack(all_E)
                y_true = np.concatenate(all_y)
                
                # L2 normalization clusters on the hypersphere, mitigating scale differences and facilitating stable decision boundaries.
                Z_global = sk_normalize(Z_global, norm="l2", axis=1)
                
                kmeans = KMeans(n_clusters=n_clusters, n_init=args.n_init, random_state=args.seed)
                y_pred = kmeans.fit_predict(Z_global)
                acc, nmi, ari = evaluate_clustering(y_true, y_pred)
                print(f"   [Evaluation] ACC: {acc:.4f} | NMI: {nmi:.4f} | ARI: {ari:.4f}")

                # save results
                epoch_result = {
                    'epoch': epoch + 1,
                    'total_loss': l_total,
                    'l_rec': l_rec,
                    'l_decorr': l_decorr,
                    'l_nce': l_nce,
                    'acc': acc,
                    'nmi': nmi,
                    'ari': ari
                }
                epoch_results.append(epoch_result)

            ema.restore()
        elif args.dataset == 'synthetic3d' or args.dataset == 'Fashion' or args.dataset == 'Mfeat':
            print(f"Epoch [{epoch+1}/{args.epochs}] | Total Loss: {total_loss:.4f}| L_rec: {l_rec:.4f}| L_decorr: {l_decorr:.4f}| L_NCE: {l_nce:.4f}")
            model.eval()
            with torch.no_grad():
                ema.apply_shadow()
                all_E = []
                all_y = []
                for idx, x_list, y in loader:
                    x_list = [x.to(device) for x in x_list]
                    A_norm_list = [normalize(knn_graph(x_list[v], args.k_nn)).to(device) for v in range(num_views)]
                    _, _, E_list, _, _, _ = model(x_list, A_norm_list)
                    all_E.append(torch.cat(E_list, dim=1).cpu().numpy())
                    all_y.append(y.numpy())
                Z_global = np.vstack(all_E)
                y_true = np.concatenate(all_y)
                Z_global = sk_normalize(Z_global, norm="l2", axis=1)
                kmeans = KMeans(n_clusters=n_clusters, n_init=args.n_init, random_state=args.seed)
                y_pred = kmeans.fit_predict(Z_global)
                acc, nmi, ari = evaluate_clustering(y_true, y_pred)
                print(f"   [Evaluation] ACC: {acc:.4f} | NMI: {nmi:.4f} | ARI: {ari:.4f}")

                epoch_result = {
                    'epoch': epoch + 1,
                    'total_loss': l_total,
                    'l_rec': l_rec,
                    'l_decorr': l_decorr,
                    'l_nce': l_nce,
                    'acc': acc,
                    'nmi': nmi,
                    'ari': ari
                }
                epoch_results.append(epoch_result)

            ema.restore()

    filename = f"train_results/{args.dataset}_{timestamp}.json"
    save_data = {
        'dataset': args.dataset,
        'epoch_results': epoch_results,
    }
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(save_data, f, indent=4, ensure_ascii=False)    
    
