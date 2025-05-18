import copy
import torch

class DecentralizedRepository:
    """Manages distributed storage of models across the network"""
    
    def __init__(self, initial_model, node_id, max_history=5):
        self.node_id = node_id
        self.local_branch = copy.deepcopy(initial_model.state_dict())
        self.branch_versions = {node_id: 0}  # Track versions of each node's branch
        self.branch_models = {node_id: self.local_branch}
        self.master_model = copy.deepcopy(initial_model)
        self.max_history = max_history
        self.model_history = {node_id: []}  # Limited history of previous models
        
    def update_local_branch(self, updated_model):
        """Update the local branch model after training"""
        self.local_branch = copy.deepcopy(updated_model.state_dict())
        self.branch_models[self.node_id] = self.local_branch
        prev_version = self.branch_versions[self.node_id]
        self.branch_versions[self.node_id] += 1
        
        print(f"Repository {self.node_id}: Updated local branch from version {prev_version} to {self.branch_versions[self.node_id]}")
        
        # Maintain limited history
        if len(self.model_history[self.node_id]) >= self.max_history:
            self.model_history[self.node_id].pop(0)
        self.model_history[self.node_id].append(copy.deepcopy(self.local_branch))
        
    def receive_branch_update(self, node_id, model_state_dict, version):
        """Process received branch model from another node"""
        # Add node to tracking if new
        if node_id not in self.branch_versions:
            self.branch_versions[node_id] = 0
            self.branch_models[node_id] = None
            self.model_history[node_id] = []
            
        # Only accept newer versions
        if version > self.branch_versions.get(node_id, -1):
            self.branch_models[node_id] = copy.deepcopy(model_state_dict)
            self.branch_versions[node_id] = version
            
            # Maintain limited history
            if len(self.model_history[node_id]) >= self.max_history:
                self.model_history[node_id].pop(0)
            self.model_history[node_id].append(copy.deepcopy(model_state_dict))
            
            # After receiving an update, compute new master model
            self.compute_master_model()
            return True
        return False
    
    def compute_master_model(self):
        """Generate master model by weighted averaging of branch models"""
        weights = {}
        total_versions = sum(self.branch_versions.values())
        
        # Compute weights based on version numbers
        for node_id, version in self.branch_versions.items():
            if total_versions > 0:
                weights[node_id] = version / total_versions
            else:
                weights[node_id] = 1.0 / len(self.branch_versions)
        
        # Apply weighted averaging
        master_dict = copy.deepcopy(self.master_model.state_dict())
        for key in master_dict:
            master_dict[key] = 0
            for node_id, branch_model in self.branch_models.items():
                if branch_model is not None:
                    master_dict[key] += weights[node_id] * branch_model[key]
                    
        self.master_model.load_state_dict(master_dict)
        return self.master_model
    
    def get_local_branch_model(self):
        """Return model loaded with local branch for training"""
        model = copy.deepcopy(self.master_model)
        model.load_state_dict(self.local_branch)
        return model
    
    def pull_master_to_branch(self, control_factor=10.0):
        """Pull updates from master to local branch with version-based weighting"""
        version_diff = self.branch_versions[self.node_id] - sum(self.branch_versions.values()) / len(self.branch_versions)
        
        # Calculate merging weight based on version difference
        branch_weight = max(control_factor + version_diff, 2.0)
        master_weight = 1.0
        
        # Merge master into branch
        merged_dict = copy.deepcopy(self.local_branch)
        master_dict = self.master_model.state_dict()
        
        for key in merged_dict:
            merged_dict[key] = (branch_weight * merged_dict[key] + master_weight * master_dict[key]) / (branch_weight + master_weight)
            
        self.local_branch = merged_dict
        self.branch_models[self.node_id] = merged_dict
        
        model = copy.deepcopy(self.master_model)
        model.load_state_dict(merged_dict)
        return model