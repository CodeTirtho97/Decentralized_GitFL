import torch
import numpy as np
import random
import time
from torchvision import datasets, transforms
from models.Nets import CNNCifar
from decentralized_repository import DecentralizedRepository
from p2p_network import P2PNode
from distributed_rl_selector import PeerSelector
from DecentralizedGitFLNode import DecentralizedGitFLNode

def distribute_data(dataset, num_nodes, iid=True, alpha=0.5):
    """Distribute dataset across nodes"""
    if iid:
        # IID distribution: randomly shuffle and divide equally
        indices = list(range(len(dataset)))
        random.shuffle(indices)
        chunk_size = len(indices) // num_nodes
        return [indices[i*chunk_size:(i+1)*chunk_size] for i in range(num_nodes)]
    else:
        # Non-IID distribution using Dirichlet distribution
        labels = np.array(dataset.targets)
        num_classes = 10  # CIFAR-10 has 10 classes
        
        node_indices = [[] for _ in range(num_nodes)]
        
        # For each class, distribute samples according to Dirichlet distribution
        for class_idx in range(num_classes):
            class_indices = np.where(labels == class_idx)[0]
            random.shuffle(class_indices)
            
            # Generate Dirichlet distribution for this class
            proportions = np.random.dirichlet(np.repeat(alpha, num_nodes))
            
            # Distribute indices according to proportions
            class_size = len(class_indices)
            start_idx = 0
            for node_idx in range(num_nodes):
                end_idx = start_idx + int(proportions[node_idx] * class_size)
                end_idx = min(end_idx, class_size)  # Ensure we don't exceed bounds
                node_indices[node_idx].extend(class_indices[start_idx:end_idx])
                start_idx = end_idx
                
        # Shuffle indices for each node
        for node_idx in range(num_nodes):
            random.shuffle(node_indices[node_idx])
            
        return node_indices

def run_simulation(num_nodes=5, iid=True, alpha=0.5, runtime=300, base_port=8000, local_epochs=5):
    """Run a simulation of the decentralized GitFL system"""
    # Set random seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Prepare dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    dataset_train = datasets.CIFAR10('./data/cifar10', train=True, download=True, transform=transform)
    dataset_test = datasets.CIFAR10('./data/cifar10', train=False, download=True, transform=transform)
    
    # Distribute data across nodes
    node_data_indices = distribute_data(dataset_train, num_nodes, iid, alpha)
    
    # Create model architecture (used by all nodes)
    args = type('Args', (), {
        'num_channels': 3,
        'num_classes': 10,
        'device': torch.device("cuda" if torch.cuda.is_available() else "cpu")
    })
    model = CNNCifar(args)
    
    # Create nodes
    nodes = []
    for i in range(num_nodes):
        # Use localhost with different ports
        host = "127.0.0.1"
        port = base_port + i
        
        node = DecentralizedGitFLNode(
            node_id=i,
            host=host,
            port=port,
            model=model,
            dataset=dataset_train,
            local_data_indices=node_data_indices[i],
            learning_rate=0.01,
            local_epochs=local_epochs,
            batch_size=50
        )
        
        nodes.append(node)
    
    # Create initial connections (ring topology)
    for i in range(num_nodes):
        next_node = (i + 1) % num_nodes
        nodes[i].add_peer(next_node, "127.0.0.1", base_port + next_node)
        nodes[next_node].add_peer(i, "127.0.0.1", base_port + i)
    
    # Start all nodes
    for node in nodes:
        node.start()
        
    # Monitor and report progress
    start_time = time.time()
    try:
        while time.time() - start_time < runtime:
            # Evaluate models periodically
            if int((time.time() - start_time) / 30) % 2 == 0:  # Every 60 seconds
                print("\n" + "="*50)
                print(f"Simulation running for {int(time.time() - start_time)}s")
                
                # Evaluate each node's model
                accuracies = []
                for i, node in enumerate(nodes):
                    try:
                        accuracy = node.evaluate(dataset_test)
                        accuracies.append(accuracy)
                        print(f"Node {i} accuracy: {accuracy:.2f}%")
                    except Exception as e:
                        print(f"Error evaluating node {i}: {e}")
                
                if accuracies:
                    print(f"Average accuracy: {sum(accuracies)/len(accuracies):.2f}%")
                    print(f"Max accuracy: {max(accuracies):.2f}%")
            
            # Sleep to avoid hogging CPU
            time.sleep(5)
            
    except KeyboardInterrupt:
        print("\nSimulation interrupted by user")
    finally:
        # Stop all nodes
        for node in nodes:
            node.stop()
        
        print("Simulation complete")
        
        # Final evaluation
        final_accuracies = []
        for i, node in enumerate(nodes):
            try:
                accuracy = node.evaluate(dataset_test)
                final_accuracies.append(accuracy)
                print(f"Final Node {i} accuracy: {accuracy:.2f}%")
            except:
                pass
                
        if final_accuracies:
            print(f"Final Average accuracy: {sum(final_accuracies)/len(final_accuracies):.2f}%")
            print(f"Final Max accuracy: {max(final_accuracies):.2f}%")