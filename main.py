import argparse
from simulation import run_simulation

if __name__ == "__main__":
    # Add CPU optimization at the start
    import os
    os.environ['OMP_NUM_THREADS'] = '4'
    os.environ['MKL_NUM_THREADS'] = '4'
    os.environ['TORCH_NUM_THREADS'] = '4'
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    
    parser = argparse.ArgumentParser(description='Decentralized GitFL Simulation')
    parser.add_argument('--nodes', type=int, default=5, help='Number of nodes')
    parser.add_argument('--runtime', type=int, default=900, help='Simulation runtime in seconds')  # 15 minutes
    parser.add_argument('--epochs', type=int, default=2, help='Number of local epochs per training round')  # Match original
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')  # CPU optimized
    parser.add_argument('--iid', type=int, default=1, help='1 for IID, 0 for non-IID')
    parser.add_argument('--alpha', type=float, default=0.5, help='Dirichlet alpha parameter for non-IID data')
    parser.add_argument('--base_port', type=int, default=8000, help='Base port number')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')  # Fixed seed
    
    args = parser.parse_args()
    
    run_simulation(
        num_nodes=args.nodes,
        iid=args.iid == 1,
        alpha=args.alpha,
        runtime=args.runtime,
        base_port=args.base_port,
        local_epochs=args.epochs
    )
    
    
    # python main.py --nodes 5 --runtime 900 --epochs 2