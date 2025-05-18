# Decentralized_GitFL

Decentralized GitFL enables privacy-preserving collaborative AI training by removing the central server bottleneck found in traditional federated learning systems. By applying Git's version control principles to a fully distributed architecture, devices can exchange model updates directly, manage model staleness effectively, and optimize communication patterns through reinforcement learning. Our experiments show 5% accuracy improvement and 7.2× faster convergence compared to centralized approaches, making the system ideal for resource-constrained IoT environments where privacy and resilience are critical.

# Decentralized GitFL

A peer-to-peer federated learning framework that brings Git-inspired version control to distributed machine learning without central coordination.

## Overview

Decentralized GitFL is an innovative approach to federated learning that enables collaborative AI training across distributed devices while preserving data privacy and eliminating central server dependencies. By applying Git's version control principles in a fully decentralized architecture, the system effectively manages model staleness and optimizes peer-to-peer communication patterns.

## Key Features

- **Fully Decentralized Architecture**: Eliminates single points of failure with direct peer-to-peer model exchanges
- **Distributed Version Control**: Tracks model versions across the network to manage staleness effectively
- **RL-Based Peer Selection**: Uses reinforcement learning to optimize communication patterns adaptively
- **Resilient Communication**: Maintains functionality even when portions of the network become unavailable
- **Efficient Knowledge Sharing**: Achieves faster convergence and higher accuracy than centralized approaches

## Results

Compared to traditional centralized federated learning approaches:

- 38.56% model accuracy (5% improvement)
- 7.2× faster convergence
- Enhanced resilience to network failures
- Better scalability with distributed computation

## System Architecture

The system consists of five key components:

1. **Node Controller**: Manages local training, peer discovery, and model sharing
2. **Repository**: Implements Git-inspired operations (push, pull, merge) with version tracking
3. **P2P Network**: Facilitates direct communication between nodes
4. **RL Selector**: Makes intelligent decisions about which peers to exchange models with
5. **Neural Network**: Represents the model being collaboratively trained

## Getting Started

### Prerequisites

- Python 3.8+
- PyTorch 1.9+
- NumPy
- Socket libraries

### Installation

```bash
git clone https://github.com/username/decentralized-gitfl.git
cd decentralized-gitfl
pip install -r requirements.txt
```

### Running a Simulation

To run a basic simulation with 5 nodes:

```bash
python main.py --nodes 5 --runtime 900 --epochs 2
```

Additional parameters:

- `--iid`: Set to 1 for IID data distribution, 0 for non-IID (default: 1)
- `--alpha`: Dirichlet alpha parameter for non-IID settings (default: 0.5)
- `--base_port`: Starting port number for P2P communication (default: 8000)

## Project Structure

```bash
DECENTRALIZED_GITFL/
│
├── data/cifar10/           # Dataset files
│
├── models/                 # Neural network models
│   ├── Nets.py
│   ├── resnetcifar.py
│   └── test.py
│
├── utils/                  # Core components
│   ├── get_dataset.py
│   ├── set_seed.py
│   ├── decentralized_repository.py
│   ├── DecentralizedGitFLNode.py
│   ├── distributed_rl_selector.py
│   ├── p2p_network.py
│   └── simulation.py
│
├── main.py                 # Entry point
├── requirements.txt
└── README.md
```

## Core Components

- **decentralized_repository.py**: Manages the distributed model storage and version tracking
- **DecentralizedGitFLNode.py**: Integrates all components representing a participating device
- **distributed_rl_selector.py**: Implements reinforcement learning for peer selection
- **p2p_network.py**: Handles all peer-to-peer communication between nodes
- **simulation.py**: Orchestrates multiple nodes in a controlled environment

## Future Directions

- Energy-aware peer selection for battery-powered devices
- Hierarchical peer organization for improved scalability
- Compressed model exchange to reduce bandwidth requirements
- Privacy-preserving mechanisms like differential privacy integration
- Real-world deployment studies on heterogeneous IoT devices

## Technology Stack

- Python
- PyTorch
- Socket Programming
- Reinforcement Learning
- Distributed Systems

## Citation

If you use this code for your research, please cite our work:

```bash
@article{bhattacharya2025decentralizedgitfl,
  title={Exploring Git-Inspired Version Control in Federated Learning: Decentralized GitFL Implementation},
  author={Bhattacharya, Tirthoraj},
  journal={},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Dr. Anshu S. Anand for research guidance
- Hu et al. for the original GitFL framework that inspired this work
