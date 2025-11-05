"""
GNN Encoder for Heterogeneous Graph State

Encodes heterogeneous graphs into fixed-length vectors for RL agent input.
Uses graph neural networks to process task and edge node relationships.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional
import src.constants as c


class HeteroGraphEncoder(nn.Module):
    """
    Encodes heterogeneous graphs into fixed-length state vectors.
    
    Architecture:
    - Task node encoder: Processes task features
    - Edge node encoder: Processes edge node features
    - Graph-level aggregation: Aggregates node embeddings
    - Output: Fixed-length state vector
    """
    
    def __init__(self, 
                 task_feature_dim: int = 6,
                 edge_feature_dim: int = 6,
                 hidden_dim: int = 64,
                 output_dim: int = 64,
                 num_layers: int = 2):
        """
        Initialize GNN encoder.
        
        Args:
            task_feature_dim: Dimension of task node features
            edge_feature_dim: Dimension of edge node features
            hidden_dim: Hidden dimension for GNN layers
            output_dim: Output state vector dimension
            num_layers: Number of GNN layers
        """
        super(HeteroGraphEncoder, self).__init__()
        
        self.task_feature_dim = task_feature_dim
        self.edge_feature_dim = edge_feature_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        
        # Task node encoder
        self.task_encoder = nn.Sequential(
            nn.Linear(task_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Edge node encoder
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Graph convolution layers (simple message passing)
        self.gnn_layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])
        
        # Aggregation layers
        self.task_aggregator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        self.edge_aggregator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Final output layer
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # task + edge aggregated
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, graph_data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Encode graph into fixed-length vector.
        
        Args:
            graph_data: Dictionary with graph structure
                - node_features: {'task': tensor, 'edge': tensor}
                - edge_index: {'queue': tensor, 'type': tensor, 'affinity': tensor, 'topology': tensor}
                - num_nodes: {'task': int, 'edge': int}
        
        Returns:
            State vector: [batch_size, output_dim]
        """
        node_features = graph_data['node_features']
        edge_index = graph_data['edge_index']
        num_nodes = graph_data['num_nodes']
        
        # Encode node features
        task_features = node_features['task']  # [num_tasks, task_feature_dim]
        edge_features = node_features['edge']   # [num_edges, edge_feature_dim]
        
        # Initial node embeddings
        if task_features.shape[0] > 0:
            task_embeddings = self.task_encoder(task_features)  # [num_tasks, hidden_dim]
        else:
            task_embeddings = torch.zeros((0, self.hidden_dim), device=task_features.device if task_features.numel() > 0 else torch.device('cpu'))
        
        if edge_features.shape[0] > 0:
            edge_embeddings = self.edge_encoder(edge_features)  # [num_edges, hidden_dim]
        else:
            edge_embeddings = torch.zeros((0, self.hidden_dim), device=edge_features.device if edge_features.numel() > 0 else torch.device('cpu'))
        
        # Graph convolution (message passing)
        # Process queue edges: propagate information along task queue
        if task_embeddings.shape[0] > 0:
            task_embeddings = self._apply_queue_convolution(
                task_embeddings, edge_index.get('queue', None)
            )
            
            # Process type edges: aggregate same-type tasks
            task_embeddings = self._apply_type_convolution(
                task_embeddings, edge_index.get('type', None)
            )
        
        # Process affinity edges: task-edge interactions
        if task_embeddings.shape[0] > 0 and edge_embeddings.shape[0] > 0:
            task_embeddings, edge_embeddings = self._apply_affinity_convolution(
                task_embeddings, edge_embeddings, edge_index.get('affinity', None)
            )
        
        # Process topology edges: edge-edge interactions
        if edge_embeddings.shape[0] > 1:
            edge_embeddings = self._apply_topology_convolution(
                edge_embeddings, edge_index.get('topology', None)
            )
        
        # Aggregate node embeddings to graph-level representation
        # Task aggregation: mean pooling over all tasks
        if task_embeddings.shape[0] > 0:
            task_aggregated = torch.mean(task_embeddings, dim=0)  # [hidden_dim]
            task_aggregated = self.task_aggregator(task_aggregated)  # [hidden_dim]
        else:
            task_aggregated = torch.zeros(self.hidden_dim, device=task_embeddings.device if task_embeddings.numel() > 0 else torch.device('cpu'))
        
        # Edge aggregation: mean pooling over all edges
        if edge_embeddings.shape[0] > 0:
            edge_aggregated = torch.mean(edge_embeddings, dim=0)  # [hidden_dim]
            edge_aggregated = self.edge_aggregator(edge_aggregated)  # [hidden_dim]
        else:
            edge_aggregated = torch.zeros(self.hidden_dim, device=edge_embeddings.device if edge_embeddings.numel() > 0 else torch.device('cpu'))
        
        # Concatenate and output
        combined = torch.cat([task_aggregated, edge_aggregated], dim=0)  # [hidden_dim * 2]
        output = self.output_layer(combined)  # [output_dim]
        
        return output
    
    def _apply_queue_convolution(self, task_embeddings: torch.Tensor, 
                                 queue_edges: Optional[torch.Tensor]) -> torch.Tensor:
        """Apply queue edge convolution (sequential message passing)."""
        if queue_edges is None or queue_edges.shape[1] == 0:
            return task_embeddings
        
        # Simple message passing: each task receives information from previous task
        updated_embeddings = task_embeddings.clone()
        
        for i in range(self.num_layers):
            # Aggregate messages from neighbors
            messages = torch.zeros_like(task_embeddings)
            
            # Queue edges: Task i → Task i+1
            if queue_edges.shape[1] > 0:
                source_indices = queue_edges[0]  # Source nodes
                target_indices = queue_edges[1]   # Target nodes
                
                # Aggregate messages from source to target
                for src, tgt in zip(source_indices, target_indices):
                    if src < task_embeddings.shape[0] and tgt < task_embeddings.shape[0]:
                        messages[tgt] += task_embeddings[src]
            
            # Update embeddings
            updated_embeddings = self.gnn_layers[i](updated_embeddings + messages * 0.5)
            updated_embeddings = F.relu(updated_embeddings)
        
        return updated_embeddings
    
    def _apply_type_convolution(self, task_embeddings: torch.Tensor,
                                type_edges: Optional[torch.Tensor]) -> torch.Tensor:
        """Apply type edge convolution (same-type task aggregation)."""
        if type_edges is None or type_edges.shape[1] == 0:
            return task_embeddings
        
        # Aggregate messages from same-type tasks
        messages = torch.zeros_like(task_embeddings)
        
        if type_edges.shape[1] > 0:
            source_indices = type_edges[0]
            target_indices = type_edges[1]
            
            # Count neighbors for normalization
            neighbor_count = torch.zeros(task_embeddings.shape[0], device=task_embeddings.device)
            
            for src, tgt in zip(source_indices, target_indices):
                if src < task_embeddings.shape[0] and tgt < task_embeddings.shape[0]:
                    messages[tgt] += task_embeddings[src]
                    neighbor_count[tgt] += 1
            
            # Normalize by number of neighbors
            neighbor_count = torch.clamp(neighbor_count, min=1.0)
            messages = messages / neighbor_count.unsqueeze(1)
        
        # Update embeddings
        updated_embeddings = task_embeddings + messages * 0.3  # Small contribution
        return updated_embeddings
    
    def _apply_affinity_convolution(self, task_embeddings: torch.Tensor,
                                    edge_embeddings: torch.Tensor,
                                    affinity_edges: Optional[torch.Tensor]) -> tuple:
        """Apply affinity edge convolution (task-edge interactions)."""
        if affinity_edges is None or affinity_edges.shape[1] == 0:
            return task_embeddings, edge_embeddings
        
        num_tasks = task_embeddings.shape[0]
        num_edges = edge_embeddings.shape[0]
        
        # Task nodes receive information from edge nodes
        task_messages = torch.zeros_like(task_embeddings)
        edge_messages = torch.zeros_like(edge_embeddings)
        
        if affinity_edges.shape[1] > 0:
            source_indices = affinity_edges[0]
            target_indices = affinity_edges[1]
            
            # Count neighbors for normalization
            task_neighbor_count = torch.zeros(num_tasks, device=task_embeddings.device)
            edge_neighbor_count = torch.zeros(num_edges, device=edge_embeddings.device)
            
            for src, tgt in zip(source_indices, target_indices):
                if src < num_tasks and tgt < num_edges:
                    # Task -> Edge
                    task_messages[src] += edge_embeddings[tgt]
                    task_neighbor_count[src] += 1
                    edge_messages[tgt] += task_embeddings[src]
                    edge_neighbor_count[tgt] += 1
                elif src < num_edges and tgt < num_tasks:
                    # Edge -> Task
                    task_messages[tgt] += edge_embeddings[src]
                    task_neighbor_count[tgt] += 1
                    edge_messages[src] += task_embeddings[tgt]
                    edge_neighbor_count[src] += 1
        
        # Normalize by number of neighbors
        if task_embeddings.shape[0] > 0:
            task_neighbor_count = torch.clamp(task_neighbor_count, min=1.0)
            task_messages = task_messages / task_neighbor_count.unsqueeze(1)
            task_embeddings = task_embeddings + task_messages * 0.3
        
        if edge_embeddings.shape[0] > 0:
            edge_neighbor_count = torch.clamp(edge_neighbor_count, min=1.0)
            edge_messages = edge_messages / edge_neighbor_count.unsqueeze(1)
            edge_embeddings = edge_embeddings + edge_messages * 0.3
        
        return task_embeddings, edge_embeddings
    
    def _apply_topology_convolution(self, edge_embeddings: torch.Tensor,
                                    topology_edges: Optional[torch.Tensor]) -> torch.Tensor:
        """Apply topology edge convolution (edge-edge interactions)."""
        if topology_edges is None or topology_edges.shape[1] == 0:
            return edge_embeddings
        
        # Aggregate messages from connected edge nodes
        messages = torch.zeros_like(edge_embeddings)
        
        if topology_edges.shape[1] > 0:
            source_indices = topology_edges[0]
            target_indices = topology_edges[1]
            
            neighbor_count = torch.zeros(edge_embeddings.shape[0], device=edge_embeddings.device)
            
            for src, tgt in zip(source_indices, target_indices):
                if src < edge_embeddings.shape[0] and tgt < edge_embeddings.shape[0]:
                    messages[tgt] += edge_embeddings[src]
                    neighbor_count[tgt] += 1
            
            # Normalize
            neighbor_count = torch.clamp(neighbor_count, min=1.0)
            messages = messages / neighbor_count.unsqueeze(1)
        
        # Update embeddings
        updated_embeddings = edge_embeddings + messages * 0.3
        return updated_embeddings


class GraphStateWrapper:
    """
    Wrapper to convert graph state environment to vector state for stable-baselines3.
    
    This wrapper converts the graph state returned by the environment into
    a fixed-length vector using the GNN encoder.
    """
    
    def __init__(self, env, encoder: Optional[HeteroGraphEncoder] = None, 
                 output_dim: int = 64, device: str = 'cpu'):
        """
        Initialize wrapper.
        
        Args:
            env: Environment that returns graph state
            encoder: Pre-trained GNN encoder (optional, will create if None)
            output_dim: Output dimension of state vector
            device: Device for encoder computation
        """
        self.env = env
        self.device = torch.device(device)
        
        # Create or use provided encoder
        if encoder is None:
            self.encoder = HeteroGraphEncoder(
                task_feature_dim=6,
                edge_feature_dim=6,
                hidden_dim=64,
                output_dim=output_dim,
                num_layers=2
            ).to(self.device)
            self.encoder.eval()  # Set to evaluation mode
        else:
            self.encoder = encoder.to(self.device)
        
        # Store original environment properties
        self.action_space = env.action_space
        self.output_dim = output_dim
        
        # Create observation space for encoded state
        from gymnasium import spaces
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(output_dim,),
            dtype=np.float32
        )
    
    def reset(self, seed=None, options=None):
        """Reset environment and encode graph state."""
        state, info = self.env.reset(seed=seed, options=options)
        
        # Convert graph to vector if needed
        if isinstance(state, dict):
            state_vector = self._encode_graph(state)
        else:
            state_vector = state
        
        return state_vector, info
    
    def step(self, action):
        """Step environment and encode graph state."""
        state, reward, terminated, truncated, info = self.env.step(action)
        
        # Convert graph to vector if needed
        if isinstance(state, dict):
            state_vector = self._encode_graph(state)
        else:
            state_vector = state
        
        return state_vector, reward, terminated, truncated, info
    
    def _encode_graph(self, graph_data: Dict) -> np.ndarray:
        """
        Encode graph data to vector using GNN encoder.
        
        Args:
            graph_data: Graph dictionary from environment
        
        Returns:
            State vector as numpy array
        """
        # Move graph data to device
        graph_data_device = self._move_graph_to_device(graph_data, self.device)
        
        # Encode graph
        with torch.no_grad():
            state_vector = self.encoder(graph_data_device)
        
        # Convert to numpy and return
        return state_vector.cpu().numpy().flatten()
    
    def _move_graph_to_device(self, graph_data: Dict, device: torch.device) -> Dict:
        """Move graph tensors to specified device."""
        graph_data_device = {}
        
        # Move node features
        graph_data_device['node_features'] = {}
        for node_type, features in graph_data['node_features'].items():
            if isinstance(features, torch.Tensor):
                graph_data_device['node_features'][node_type] = features.to(device)
            else:
                graph_data_device['node_features'][node_type] = torch.tensor(features, device=device)
        
        # Move edge indices
        graph_data_device['edge_index'] = {}
        for edge_type, indices in graph_data['edge_index'].items():
            if isinstance(indices, torch.Tensor):
                graph_data_device['edge_index'][edge_type] = indices.to(device)
            else:
                graph_data_device['edge_index'][edge_type] = torch.tensor(indices, device=device)
        
        # Copy num_nodes (not tensors)
        graph_data_device['num_nodes'] = graph_data['num_nodes']
        
        return graph_data_device
    
    def close(self):
        """Close wrapped environment."""
        return self.env.close()
    
    def render(self, mode='human'):
        """Render wrapped environment."""
        return self.env.render(mode)
    

