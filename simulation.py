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

def run_simulation(num_nodes=5, iid=True, alpha=0.5, runtime=900, base_port=8000, local_epochs=2):
    """Run a simulation of the decentralized GitFL system with CPU optimizations"""
    # Set random seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Force CPU usage and disable CUDA
    torch.backends.cudnn.enabled = False
    
    print(f"🔧 Starting Decentralized GitFL Simulation")
    print(f"   Nodes: {num_nodes}")
    print(f"   Runtime: {runtime}s")
    print(f"   Local epochs: {local_epochs}")
    print(f"   IID: {iid}")
    print(f"   Device: CPU (forced)")
    
    # Prepare dataset with CPU-optimized settings
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    dataset_train = datasets.CIFAR10('./data/cifar10', train=True, download=True, transform=transform)
    dataset_test = datasets.CIFAR10('./data/cifar10', train=False, download=True, transform=transform)
    
    # Distribute data across nodes
    node_data_indices = distribute_data(dataset_train, num_nodes, iid, alpha)
    
    # Create model architecture (FORCE CPU DEVICE)
    args = type('Args', (), {
        'num_channels': 3,
        'num_classes': 10,
        'device': torch.device("cpu")  # Force CPU only
    })
    model = CNNCifar(args)
    
    # Create nodes with CPU-optimized parameters
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
            local_epochs=local_epochs,  # Use the passed parameter (default 2)
            batch_size=32  # Reduced from 50 for CPU optimization
        )
        
        nodes.append(node)
    
    # Create initial connections (ring topology)
    for i in range(num_nodes):
        next_node = (i + 1) % num_nodes
        nodes[i].add_peer(next_node, "127.0.0.1", base_port + next_node)
        nodes[next_node].add_peer(i, "127.0.0.1", base_port + i)
    
    # Start all nodes
    print("🚀 Starting all nodes...")
    for node in nodes:
        node.start()
    
    # Give nodes time to initialize
    time.sleep(5)
    
    # Monitor and report progress
    start_time = time.time()
    last_report_time = 0
    
    try:
        while time.time() - start_time < runtime:
            current_time = time.time() - start_time
            
            # Report progress every 60 seconds or at key intervals
            if current_time - last_report_time >= 60 or current_time in [180, 360, 540, 720]:
                print("\n" + "="*60)
                print(f"⏱️  Simulation Time: {int(current_time)}s / {runtime}s")
                
                # Evaluate each node's model
                accuracies = []
                max_accuracy = 0
                for i, node in enumerate(nodes):
                    try:
                        accuracy = node.evaluate(dataset_test)
                        accuracies.append(accuracy)
                        max_accuracy = max(max_accuracy, accuracy)
                        print(f"   Node {i}: {accuracy:.2f}% accuracy")
                    except Exception as e:
                        print(f"   Node {i}: Error - {str(e)[:50]}...")
                        accuracies.append(0)
                
                if accuracies:
                    avg_accuracy = sum(accuracies) / len(accuracies)
                    print(f"📊 Average Accuracy: {avg_accuracy:.2f}%")
                    print(f"📈 Maximum Accuracy: {max_accuracy:.2f}%")
                    print(f"📊 Min Accuracy: {min(accuracies):.2f}%")
                    
                    # Log key milestones for comparison
                    if int(current_time) in [184, 364, 563, 739]:
                        print(f"🎯 MILESTONE: Time {int(current_time)}s - Max Accuracy: {max_accuracy:.2f}%")
                
                last_report_time = current_time
            
            # Sleep to avoid excessive CPU usage
            time.sleep(10)
            
    except KeyboardInterrupt:
        print("\n⚠️  Simulation interrupted by user")
    finally:
        print("\n🛑 Stopping all nodes...")
        # Stop all nodes
        for node in nodes:
            try:
                node.stop()
            except:
                pass
        
        # Final evaluation
        print("\n" + "="*60)
        print("📋 FINAL RESULTS")
        print("="*60)
        
        final_accuracies = []
        final_time = time.time() - start_time
        
        for i, node in enumerate(nodes):
            try:
                accuracy = node.evaluate(dataset_test)
                final_accuracies.append(accuracy)
                print(f"   Node {i} Final Accuracy: {accuracy:.2f}%")
            except Exception as e:
                print(f"   Node {i} Final Accuracy: Error - {e}")
                final_accuracies.append(0)
        
        if final_accuracies:
            avg_final = sum(final_accuracies) / len(final_accuracies)
            max_final = max(final_accuracies)
            min_final = min(final_accuracies)
            
            print(f"\n🎯 SUMMARY STATISTICS:")
            print(f"   Total Runtime: {final_time:.1f}s")
            print(f"   Average Final Accuracy: {avg_final:.2f}%")
            print(f"   Maximum Final Accuracy: {max_final:.2f}%")
            print(f"   Minimum Final Accuracy: {min_final:.2f}%")
            print(f"   Accuracy Range: {max_final - min_final:.2f}%")
            
            # Comparison metrics for thesis
            print(f"\n📊 FOR THESIS COMPARISON:")
            print(f"   Decentralized GitFL Final Max Accuracy: {max_final:.2f}%")
            print(f"   Decentralized GitFL Convergence Time: {final_time:.1f}s")
            print(f"   Decentralized GitFL Average Accuracy: {avg_final:.2f}%")
        
        print("\n✅ Simulation completed successfully!")
        
        # Save results for later analysis
        results = {
            'final_max_accuracy': max_final if final_accuracies else 0,
            'final_avg_accuracy': avg_final if final_accuracies else 0,
            'total_runtime': final_time,
            'num_nodes': num_nodes,
            'local_epochs': local_epochs,
            'iid': iid,
            'alpha': alpha
        }
        
        import json
        with open(f'decentralized_gitfl_results_{int(time.time())}.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"📁 Results saved to: decentralized_gitfl_results_{int(time.time())}.json")