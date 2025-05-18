import numpy as np
import random
import time

class PeerSelector:
    """Uses RL to select peers for model sharing based on performance history"""
    
    def __init__(self, node_id, max_peers=3):
        self.node_id = node_id
        self.max_peers = max_peers
        self.peer_performance = {}  # Average response time for each peer
        self.peer_selections = {}   # Number of times each peer was selected
        self.peer_versions = {}     # Last known version of each peer's model
        self.last_interaction = {}  # Timestamp of last interaction
        
    def update_peer_stats(self, peer_id, response_time, model_version):
        """Update statistics for a peer after interaction"""
        # Initialize if this is a new peer
        if peer_id not in self.peer_performance:
            self.peer_performance[peer_id] = response_time
            self.peer_selections[peer_id] = 1
            self.peer_versions[peer_id] = model_version
        else:
            # Update running average of response time
            n = self.peer_selections[peer_id]
            self.peer_performance[peer_id] = (self.peer_performance[peer_id] * n + response_time) / (n + 1)
            self.peer_selections[peer_id] += 1
            self.peer_versions[peer_id] = model_version
            
        self.last_interaction[peer_id] = time.time()
        
    def calculate_rewards(self, local_version, all_peers):
        """Calculate selection reward for each peer based on version and performance"""
        rewards = {}
        avg_version = sum(self.peer_versions.values()) / max(len(self.peer_versions), 1)
        avg_time = sum(self.peer_performance.values()) / max(len(self.peer_performance), 1)
        max_time = max(self.peer_performance.values()) if self.peer_performance else 1.0
        
        version_diff = local_version - avg_version
        
        for peer_id in all_peers:
            # Skip self
            if peer_id == self.node_id:
                continue
                
            # For new peers, assign default values
            if peer_id not in self.peer_performance:
                self.peer_performance[peer_id] = avg_time
                self.peer_selections[peer_id] = 0
                self.peer_versions[peer_id] = avg_version
                self.last_interaction[peer_id] = 0
                
            # Version-based component (match slow peers with high-version models)
            if peer_id in self.peer_versions:
                time_factor = (self.peer_performance[peer_id] - avg_time) / (max_time * 10)
                version_reward = version_diff * time_factor
            else:
                version_reward = 0
                
            # Curiosity-based component (encourage exploring less-selected peers)
            curiosity_reward = 1.0 / np.sqrt(self.peer_selections[peer_id] + 1)
            
            # Time since last interaction (encourage periodic check-ins)
            recency_factor = np.exp(-(time.time() - self.last_interaction.get(peer_id, 0)) / 3600)  # Decay over an hour
            recency_reward = 0.5 * (1 - recency_factor)  # Higher reward for peers not contacted recently
            
            # Combine rewards
            rewards[peer_id] = max(0.00001, version_reward + curiosity_reward + recency_reward)
            
        return rewards
        
    def select_peers(self, local_version, all_peers, exploration_factor=0.2):
        """Select peers to interact with using the reward-based approach"""
        rewards = self.calculate_rewards(local_version, all_peers)
        
        # Normalize rewards to create a probability distribution
        total_reward = sum(rewards.values())
        if total_reward == 0:
            # If all rewards are zero, use uniform distribution
            probs = {peer_id: 1.0/len(rewards) for peer_id in rewards}
        else:
            probs = {peer_id: reward/total_reward for peer_id, reward in rewards.items()}
            
        # Select peers based on probabilities, with some exploration
        selected_peers = []
        candidates = list(probs.keys())
        
        # Ensure we don't try to select more peers than available
        num_to_select = min(self.max_peers, len(candidates))
        
        for _ in range(num_to_select):
            if not candidates:
                break
                
            # Exploration: sometimes pick randomly
            if random.random() < exploration_factor:
                selected = random.choice(candidates)
            else:
                # Exploitation: weighted random choice based on rewards
                selected = random.choices(
                    candidates,
                    weights=[probs[pid] for pid in candidates],
                    k=1
                )[0]
                
            selected_peers.append(selected)
            candidates.remove(selected)
            
        return selected_peers