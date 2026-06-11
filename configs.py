def get_config(args):
    """Parameter configuration for datasets"""
    dataset_configs = {
        'synthetic3d': {
            "dataset": "synthetic3d",
            "lambda_1":0.1,
            "lambda_2":1.0,
            "epochs": 200,
            "lr": 3e-4,
            "batch_size": 1024,
            "hidden_dim": "256,512,1024",
            "z_dim": 32,
            "h_dim": 32,
            "shared_dim": 512,
            "tau": 0.5,
            "tau_g": 0.5,
            "weight_decay": 5e-6,
            "k_nn": 10,
            "n_init": 10,
            "gate_lr": 0.001,
            "seed": 42,
            "w_ema":0.999,
            "warmup":0,
        }
        # Add new dataset templates here
        # 'BDGP': {...}
    }
    # Update args with dataset-specific config
    config = dataset_configs[args.dataset]
    for key, value in config.items():
        setattr(args, key, value)
            
    return args
