import argparse
from simulation import run_simulation

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Decentralized GitFL Simulation')
    parser.add_argument('--nodes', type=int, default=5, help='Number of nodes')
    parser.add_argument('--iid', type=int, default=1, help='1 for IID, 0 for non-IID')
    parser.add_argument('--alpha', type=float, default=0.5, help='Dirichlet alpha parameter for non-IID data')
    parser.add_argument('--runtime', type=int, default=300, help='Simulation runtime in seconds')
    parser.add_argument('--base_port', type=int, default=8000, help='Base port number')
    parser.add_argument('--epochs', type=int, default=5, help='Number of local epochs per training round')
    
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