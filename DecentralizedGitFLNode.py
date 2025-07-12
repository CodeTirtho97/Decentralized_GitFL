import torch
import threading
import time
import random
import copy
import numpy as np
from torch.utils.data import DataLoader
from decentralized_repository import DecentralizedRepository
from p2p_network import P2PNode
from distributed_rl_selector import PeerSelector

class DecentralizedGitFLNode:
    """A complete node in the decentralized GitFL system"""
    
    def __init__(self, node_id, host, port, model, dataset, local_data_indices, 
                 learning_rate=0.01, local_epochs=2, batch_size=32):
        # Basic configuration
        self.node_id = node_id
        
        # CRITICAL FIX: Define device FIRST before using it
        self.device = torch.device("cpu")  # Force CPU usage
        
        # Now we can safely use self.device
        self.model = copy.deepcopy(model).to(self.device)
        self.dataset = dataset
        self.local_data_indices = local_data_indices
        self.lr = learning_rate
        
        # CPU-optimized parameters
        self.batch_size = batch_size
        self.local_epochs = local_epochs
        
        # Initialize components
        self.repository = DecentralizedRepository(self.model, node_id)  # Pass self.model instead of model
        self.network = P2PNode(node_id, host, port)
        self.peer_selector = PeerSelector(node_id)
        
        # Register message handlers
        self.network.register_message_handler('MODEL_SHARE', self._handle_model_share)
        self.network.register_message_handler('MODEL_REQUEST', self._handle_model_request)
        self.network.register_message_handler('PEER_DISCOVERY', self._handle_peer_discovery)
        self.network.register_message_handler('TRAINING_METRICS', self._handle_training_metrics)
        
        # State tracking
        self.running = False
        self.training_thread = None
        self.discovery_thread = None
        self.sharing_thread = None
        self.local_training_time = 0
        self.total_iterations = 0
        
        print(f"📱 Node {self.node_id} initialized on {host}:{port} (CPU mode)")
        
    def start(self):
        """Start all node operations"""
        self.running = True
        self.network.start()
        
        # Start background processes
        self.training_thread = threading.Thread(target=self._training_loop)
        self.discovery_thread = threading.Thread(target=self._discovery_loop)
        self.sharing_thread = threading.Thread(target=self._sharing_loop)
        
        self.training_thread.daemon = True
        self.discovery_thread.daemon = True
        self.sharing_thread.daemon = True
        
        self.training_thread.start()
        self.discovery_thread.start()
        self.sharing_thread.start()
        
        print(f"Node {self.node_id} started on {self.network.host}:{self.network.port}")
        
    def stop(self):
        """Stop all node operations"""
        self.running = False
        if self.training_thread:
            self.training_thread.join(timeout=2.0)
        if self.discovery_thread:
            self.discovery_thread.join(timeout=2.0)
        if self.sharing_thread:
            self.sharing_thread.join(timeout=2.0)
        self.network.stop()
        print(f"Node {self.node_id} stopped")
        
    def add_peer(self, peer_id, host, port):
        """Add a new peer to the network"""
        return self.network.add_neighbor(peer_id, host, port)
        
    def _training_loop(self):
        """Main loop for local model training"""
        while self.running:
            try:
                # Pull from master to local branch
                training_model = self.repository.pull_master_to_branch()
                training_model.to(self.device)
                
                # Prepare data loader with CPU optimization
                indices = torch.tensor(self.local_data_indices, dtype=torch.long)
                dataset = torch.utils.data.Subset(self.dataset, indices)
                data_loader = DataLoader(
                    dataset,
                    batch_size=self.batch_size,
                    shuffle=True,
                    num_workers=0,  # CPU optimization
                    pin_memory=False  # CPU optimization
                )
                
                # Setup training
                criterion = torch.nn.CrossEntropyLoss()
                optimizer = torch.optim.SGD(
                    training_model.parameters(),
                    lr=self.lr,
                    momentum=0.5
                )
                
                # Train for multiple local epochs
                start_time = time.time()
                training_model.train()  # Set model to training mode
                for epoch in range(self.local_epochs):
                    epoch_loss = 0
                    for batch_idx, (data, target) in enumerate(data_loader):
                        data, target = data.to(self.device), target.to(self.device)
                        optimizer.zero_grad()
                        
                        # Handle different model output formats
                        output = training_model(data)
                        if isinstance(output, dict):
                            output = output['output']
                        
                        loss = criterion(output, target)
                        loss.backward()
                        optimizer.step()
                        epoch_loss += loss.item()
                    
                    print(f"Node {self.node_id}, Epoch {epoch}: Loss {epoch_loss/len(data_loader):.4f}")
                
                self.local_training_time = time.time() - start_time
                self.total_iterations += 1
                
                # Update local branch with trained model
                training_model.to('cpu')
                self.repository.update_local_branch(training_model)
                
                # Log training progress
                print(f"Node {self.node_id} completed training iteration {self.total_iterations}, "
                    f"Version: {self.repository.branch_versions[self.node_id]}")
                
                # Random sleep to simulate varying device speeds
                time.sleep(random.uniform(0.5, 2.0))
                
            except Exception as e:
                print(f"Training error on node {self.node_id}: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(1)  # Avoid tight loop on persistent errors
                
    def _discovery_loop(self):
        """Periodically discover and update peer information"""
        while self.running:
            try:
                # Broadcast discovery message to find new peers
                if self.network.neighbors:
                    self.network.broadcast(
                        'PEER_DISCOVERY',
                        peers=[(p['node_id'], p['host'], p['port']) for p in self.network.neighbors]
                    )
                
                # Also broadcast training metrics for peer selection
                local_version = self.repository.branch_versions[self.node_id]
                self.network.broadcast(
                    'TRAINING_METRICS',
                    training_time=self.local_training_time,
                    model_version=local_version,
                    iterations=self.total_iterations
                )
                
                # Sleep between discovery cycles
                time.sleep(random.uniform(10, 20))
                
            except Exception as e:
                print(f"Discovery error on node {self.node_id}: {e}")
                time.sleep(5)
                
    def _sharing_loop(self):
        """Periodically share and request models with selected peers"""
        while self.running:
            try:
                # Only start sharing after some initial training
                if self.total_iterations < 2:
                    time.sleep(5)
                    continue
                    
                # Select peers to interact with
                all_peer_ids = [n['node_id'] for n in self.network.neighbors]
                local_version = self.repository.branch_versions[self.node_id]
                
                selected_peers = self.peer_selector.select_peers(local_version, all_peer_ids)
                
                # Request models from selected peers
                for peer_id in selected_peers:
                    peer = next((p for p in self.network.neighbors if p['node_id'] == peer_id), None)
                    if peer:
                        start_time = time.time()
                        success = self.network.send_message(
                            peer['host'],
                            peer['port'],
                            'MODEL_REQUEST',
                            requester_version=local_version
                        )
                        if success:
                            response_time = time.time() - start_time
                            peer_version = self.peer_selector.peer_versions.get(peer_id, 0)
                            self.peer_selector.update_peer_stats(peer_id, response_time, peer_version)
                
                # Sleep between sharing cycles
                time.sleep(random.uniform(5, 15))
                
            except Exception as e:
                print(f"Sharing error on node {self.node_id}: {e}")
                time.sleep(5)
                
    def _handle_model_share(self, message):
        """Handle receiving a model from another peer"""
        sender_id = message['sender_id']
        model_state = message['model_state']
        version = message['version']
        
        # Process the received model
        updated = self.repository.receive_branch_update(sender_id, model_state, version)
        
        if updated:
            print(f"Node {self.node_id} received model v{version} from node {sender_id}")
            
            # Update peer selector with actual version
            if sender_id in self.peer_selector.peer_versions:
                self.peer_selector.peer_versions[sender_id] = version
                
    def _handle_model_request(self, message):
        """Handle a request for our model from another peer"""
        requester_id = message['sender_id']
        requester_version = message.get('requester_version', 0)
        
        # Find the peer in our neighbors
        peer = next((p for p in self.network.neighbors if p['node_id'] == requester_id), None)
        if peer:
            # Send our current model
            self.network.send_message(
                peer['host'],
                peer['port'],
                'MODEL_SHARE',
                model_state=self.repository.local_branch,
                version=self.repository.branch_versions[self.node_id]
            )
            
            # Update peer selector (assuming zero response time since they requested from us)
            self.peer_selector.update_peer_stats(requester_id, 0.1, requester_version)
            
    def _handle_peer_discovery(self, message):
        """Handle peer discovery messages to expand our network"""
        sender_id = message['sender_id']
        new_peers = message.get('peers', [])
        
        # Add the sender if not already in our network
        sender_info = next((p for p in self.network.neighbors if p['node_id'] == sender_id), None)
        if not sender_info:
            # Extract sender info from message metadata
            for peer in new_peers:
                if peer[0] == sender_id:  # Found the sender in their peer list
                    self.network.add_neighbor(sender_id, peer[1], peer[2])
                    break
        
        # Process other peers
        for peer_id, host, port in new_peers:
            # Don't add ourselves
            if peer_id != self.node_id:
                self.network.add_neighbor(peer_id, host, port)
                
    def _handle_training_metrics(self, message):
        """Handle training metrics from other peers for peer selection"""
        sender_id = message['sender_id']
        training_time = message.get('training_time', 1.0)
        model_version = message.get('model_version', 0)
        
        # Update peer selector
        self.peer_selector.update_peer_stats(sender_id, training_time, model_version)
        
    def evaluate(self, test_dataset):
        """Evaluate the model on test data"""
        model = self.repository.compute_master_model()
        model.to(self.device)
        model.eval()
        
        # CPU-optimized test loader
        test_loader = DataLoader(
            test_dataset,
            batch_size=64,  # Smaller batch size for CPU
            shuffle=False,
            num_workers=0,  # CPU optimization
            pin_memory=False  # CPU optimization
        )
        
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                if isinstance(output, dict):
                    output = output['output']
                
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
                
        accuracy = 100 * correct / total
        print(f"Node {self.node_id} - Model version {self.repository.branch_versions[self.node_id]}, "
            f"Evaluation accuracy: {accuracy:.2f}%")
        return accuracy