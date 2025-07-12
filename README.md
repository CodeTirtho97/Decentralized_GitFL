# Decentralized GitFL: Git-Inspired Version Control for Federated Learning

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Research](https://img.shields.io/badge/Research-IIIT%20Allahabad-green.svg)](https://iiita.ac.in/)

> **A novel peer-to-peer federated learning framework that eliminates central coordination while preserving Git-inspired version control for effective model staleness management.**

## 📋 Abstract

Federated Learning (FL) has emerged as a transformative distributed machine learning paradigm enabling privacy-preserving model training across multiple devices without sharing raw data. However, existing FL frameworks face critical challenges: device heterogeneity introduces "stragglers" that delay training, non-IID data distributions cause model divergence, and central server architectures create single points of failure.

This research addresses these limitations by developing **Decentralized GitFL**, a fully distributed federated learning system that applies Git-inspired version control principles to eliminate central coordination while effectively managing model staleness. Our implementation demonstrates **38.56% accuracy** (5% improvement) with **7.2× faster convergence** compared to centralized approaches.

## 🎯 Key Contributions

### 1. **Novel Decentralized Architecture**
- Eliminates central server dependencies and single points of failure
- Enables direct peer-to-peer model exchange without coordination overhead
- Maintains system functionality even with partial network failures

### 2. **Distributed Version Control System**
- Implements Git-inspired operations (push, pull, merge) for model management
- Version-weighted aggregation to mitigate staleness effects
- Distributed model repositories with limited history tracking

### 3. **Reinforcement Learning-Based Peer Selection**
- Multi-factor reward system considering version, curiosity, and recency
- Adaptive communication patterns that optimize knowledge dissemination
- Dynamic topology management for efficient network utilization

### 4. **Comprehensive Experimental Validation**
- Performance evaluation on CIFAR-10 with both IID and non-IID distributions
- Comparative analysis against centralized GitFL implementation
- Reproducible results with standardized evaluation metrics

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Decentralized GitFL Network             │
├─────────────────────────────────────────────────────────────┤
│  Node 0        Node 1        Node 2        Node 3        │
│ ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐     │
│ │Branch v3│◄──┤Branch v4│◄──┤Branch v2│◄──┤Branch v3│     │
│ └─────────┘   └─────────┘   └─────────┘   └─────────┘     │
│      ▲             ▲             ▲             ▲         │
│      │             │             │             │         │
│ ┌────▼────┐   ┌────▼────┐   ┌────▼────┐   ┌────▼────┐     │
│ │Repository│   │Repository│   │Repository│   │Repository│   │
│ │P2P Network│ │P2P Network│ │P2P Network│ │P2P Network│   │
│ │RL Selector│ │RL Selector│ │RL Selector│ │RL Selector│   │
│ │Controller │ │Controller │ │Controller │ │Controller │   │
│ └─────────┘   └─────────┘   └─────────┘   └─────────┘     │
└─────────────────────────────────────────────────────────────┘
```

### Core Components

| Component | Description | Implementation |
|-----------|-------------|----------------|
| **Node Controller** | Manages concurrent threads for training, discovery, and sharing | `DecentralizedGitFLNode.py` |
| **Repository** | Implements Git-inspired version control with distributed tracking | `decentralized_repository.py` |
| **P2P Network** | Facilitates reliable TCP-based communication between peers | `p2p_network.py` |
| **RL Selector** | Optimizes peer selection using multi-factor reward system | `distributed_rl_selector.py` |
| **Neural Network** | CNN architecture for collaborative model training | `models/Nets.py` |

## 🔬 Technical Innovation

### Version Control Mechanism

Our distributed version control system implements three key Git-inspired operations:

```python
# Version-weighted model merging
def compute_master_model(self):
    weights = {node_id: version / total_versions 
              for node_id, version in self.branch_versions.items()}
    
    # Weighted averaging of branch models
    for key in master_dict:
        master_dict[key] = sum(weights[node_id] * branch_models[node_id][key] 
                              for node_id in self.branch_models)
```

### Reinforcement Learning Peer Selection

The adaptive peer selection mechanism uses a composite reward function:

**R_peer = max(0.00001, R_version + R_curiosity + R_recency)**

Where:
- **R_version**: Balances version disparities across the network
- **R_curiosity**: Encourages exploration of less-frequently selected peers  
- **R_recency**: Promotes periodic interaction with all network participants

## 📊 Experimental Results

### Performance Comparison

| Metric | Centralized GitFL | Decentralized GitFL | Improvement |
|--------|-------------------|---------------------|-------------|
| **Final Accuracy** | 33.47% | 38.56% | **+5.09%** |
| **Convergence Time** | 6,438s | 900s | **7.2× faster** |
| **Network Resilience** | Single point of failure | Fault tolerant | **Eliminates bottleneck** |
| **Scalability** | Server bottleneck | Distributed load | **Enhanced** |

### Training Progress Analysis

```
Time (s)    Centralized    Decentralized    Performance Gap
────────    ────────────   ─────────────    ───────────────
0           10.09%         10.15%           -0.06%
900         ~15%*          38.56%           +23.56%
6,438       33.47%         N/A              N/A

* Extrapolated based on convergence rate
```

### Accuracy Evolution

The decentralized approach demonstrates superior learning dynamics:

- **Initial Phase (0-184s)**: Rapid local learning with 18.24% accuracy
- **Collaboration Phase (184-563s)**: Effective knowledge sharing reaching 32%
- **Convergence Phase (563-900s)**: Stable improvement to 38.56%

## 🚀 Getting Started

### Prerequisites

```bash
# Core Dependencies
Python >= 3.8
PyTorch >= 1.9
NumPy >= 1.21
torchvision >= 0.10
```

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/decentralized-gitfl.git
cd decentralized-gitfl

# Install dependencies
pip install torch torchvision numpy

# Verify installation
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
```

### Quick Start

```bash
# Run basic simulation with 5 nodes for 15 minutes
python main.py --nodes 5 --runtime 900 --epochs 2

# Advanced configuration with non-IID data
python main.py --nodes 10 --iid 0 --alpha 0.1 --runtime 1800

# Custom network topology
python main.py --nodes 7 --base_port 9000 --epochs 3
```

### Command Line Arguments

| Parameter | Description | Default | Range |
|-----------|-------------|---------|-------|
| `--nodes` | Number of participating nodes | 5 | 3-50 |
| `--runtime` | Simulation duration (seconds) | 300 | 60-3600 |
| `--epochs` | Local training epochs per round | 5 | 1-10 |
| `--iid` | Data distribution (1=IID, 0=non-IID) | 1 | 0,1 |
| `--alpha` | Dirichlet alpha for non-IID | 0.5 | 0.1-2.0 |
| `--base_port` | Starting port for P2P communication | 8000 | 1024-65535 |

## 📁 Project Structure

```
DECENTRALIZED_GITFL/
├── 📊 Core Implementation
│   ├── DecentralizedGitFLNode.py      # Main node integration
│   ├── decentralized_repository.py    # Distributed version control
│   ├── distributed_rl_selector.py     # RL-based peer selection
│   ├── p2p_network.py                 # Peer-to-peer communication
│   └── simulation.py                  # Orchestration and evaluation
│
├── 🧠 Neural Networks
│   ├── models/
│   │   ├── Nets.py                    # CNN architectures
│   │   ├── resnetcifar.py            # ResNet implementations
│   │   └── test.py                   # Model evaluation utilities
│
├── 🛠️ Utilities
│   ├── utils/
│   │   ├── get_dataset.py            # Dataset management
│   │   └── set_seed.py               # Reproducibility utilities
│
├── 🎯 Execution
│   ├── main.py                       # Entry point
│   └── requirements.txt              # Dependencies
│
└── 📚 Documentation
    ├── README.md                     # This file
    ├── LICENSE                       # MIT License
    └── .gitignore                    # Git exclusions
```

## 🔬 Research Methodology

### Experimental Design

Our evaluation follows rigorous research standards:

1. **Controlled Environment**: Fixed hardware specifications (Intel Core Ultra 5 125H, 16GB RAM)
2. **Reproducible Setup**: Standardized random seeds (42) across all experiments
3. **Comparative Analysis**: Direct comparison with centralized GitFL baseline
4. **Statistical Validation**: Multiple runs with confidence intervals

### Dataset Configuration

- **Primary Dataset**: CIFAR-10 (50,000 training, 10,000 test images)
- **Model Architecture**: Convolutional Neural Network with 6-16 filter progression
- **Data Distribution**: Both IID and non-IID (Dirichlet α=0.5) scenarios
- **Evaluation Metrics**: Test accuracy, convergence time, communication overhead

### Network Topology

- **Initial Configuration**: Ring topology for guaranteed connectivity
- **Dynamic Expansion**: Peer discovery enables mesh-like connections
- **Fault Tolerance**: System maintains functionality with node failures

## 🎯 Performance Analysis

### Convergence Characteristics

The decentralized approach exhibits superior convergence properties:

```python
# Typical accuracy progression
Time_points = [0, 184, 364, 563, 739, 900]
Accuracies = [10.15, 18.24, 31.59, 32.03, 37.73, 38.56]

# Convergence rate: ~0.031% per second
Convergence_rate = (38.56 - 10.15) / 900  # 0.0316% per second
```

### Communication Efficiency

| Pattern | Centralized | Decentralized | Advantage |
|---------|-------------|---------------|-----------|
| **Message Flow** | Hub-and-spoke | Mesh topology | Distributed load |
| **Bottlenecks** | Server capacity | Network bandwidth | Eliminated |
| **Scalability** | O(n) server load | O(1) per node | Linear improvement |

### Resource Utilization

- **Memory Efficiency**: Limited model history (max 5 versions per node)
- **Computational Load**: Distributed across all participants
- **Network Bandwidth**: Optimized through selective peer communication

## 🔮 Future Research Directions

### Immediate Extensions

1. **Energy-Aware Selection**: Incorporate battery state and power consumption
2. **Compressed Communication**: Implement model compression for bandwidth efficiency
3. **Hierarchical Organization**: Multi-level peer structures for enhanced scalability

### Advanced Research Opportunities

1. **Cross-Domain Applications**: Natural language processing, reinforcement learning tasks
2. **Privacy Enhancement**: Differential privacy integration with version control
3. **Theoretical Analysis**: Convergence guarantees under various network conditions
4. **Real-World Deployment**: Heterogeneous IoT device networks

### Open Problems

- **Byzantine Fault Tolerance**: Robustness against malicious participants
- **Dynamic Topology Optimization**: Adaptive network structure based on performance
- **Multi-Modal Learning**: Support for heterogeneous model architectures

## 📝 Publications and Citations

This work extends the original GitFL framework:

```bibtex
@article{bhattacharya2025decentralized,
  title={Exploring Git-Inspired Version Control in Federated Learning: Decentralized GitFL Implementation},
  author={Bhattacharya, Tirthoraj},
  journal={Master's Thesis, IIIT Allahabad},
  year={2025},
  institution={Indian Institute of Information Technology, Allahabad}
}

@inproceedings{hu2023gitfl,
  title={GitFL: Uncertainty-aware real-time asynchronous federated learning using version control},
  author={Hu, Ming and Xia, Zeke and Yan, Dengke and others},
  booktitle={2023 IEEE Real-Time Systems Symposium (RTSS)},
  pages={145--157},
  year={2023}
}
```

## 🤝 Contributing

We welcome contributions from the research community:

1. **Bug Reports**: Submit detailed issue descriptions with reproduction steps
2. **Feature Requests**: Propose enhancements with technical justification
3. **Code Contributions**: Follow PEP 8 standards with comprehensive documentation
4. **Research Collaborations**: Contact for joint research opportunities

### Development Guidelines

```bash
# Code style
python -m flake8 --max-line-length=88 *.py

# Type checking
python -m mypy --ignore-missing-imports *.py

# Testing
python -m pytest tests/ -v --cov=.
```

## 📞 Contact

**Tirthoraj Bhattacharya**  
Master of Technology (Information Technology)  
Indian Institute of Information Technology, Allahabad  

- 📧 Email: [mse2024008@iiita.ac.in](mailto:mse2024008@iiita.ac.in)
- 🏛️ Institution: [IIIT Allahabad](https://iiita.ac.in/)
- 👨‍🏫 Supervisor: Dr. Anshu S. Anand

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Dr. Anshu S. Anand** for research guidance and supervision
- **IIIT Allahabad** for providing research infrastructure
- **Original GitFL Authors** (Hu et al.) for foundational framework inspiration
- **PyTorch Community** for robust deep learning framework

---

<div align="center">

**Advancing Privacy-Preserving Collaborative Intelligence through Distributed Systems Innovation**

*Developed at the Indian Institute of Information Technology, Allahabad*

</div>