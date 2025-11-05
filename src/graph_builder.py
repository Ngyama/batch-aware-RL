"""
Graph Builder for Multi-Node Batch Scheduling

Builds heterogeneous graphs for GNN-based state representation.
Supports multiple edge nodes and task queues with various relationship types.

Node Types:
- Task: Tasks in the queue
- Edge: Computing nodes

Edge Types:
- queue: Task sequential order
- type: Same type tasks
- affinity: Task-Edge matching
- topology: Edge-Edge connections
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import deque
import src.constants as c


class HeteroGraphBuilder:
    """
    Builds heterogeneous graphs from environment state for GNN processing.
    
    Supports multi-node scenarios with task-queue and edge-node relationships.
    """
    
    def __init__(self, max_tasks: int = 100, num_edge_nodes: int = 1):
        """
        Initialize graph builder.
        
        Args:
            max_tasks: Maximum number of tasks to include in graph (for fixed-size graph)
            num_edge_nodes: Number of edge computing nodes
        """
        self.max_tasks = max_tasks
        self.num_edge_nodes = num_edge_nodes
        
        # Task node feature dimensions
        self.task_feature_dim = 6  # [type_id, time_to_deadline, wait_time, arrival_time, deadline, priority]
        
        # Edge node feature dimensions
        self.edge_feature_dim = 6  # [node_id, busy_status, time_until_free, utilization_rate, avg_processing_time, capability]
    
    def build_graph(self, 
                    task_queue: deque,
                    edge_nodes: List[Dict],
                    current_time: float,
                    task_types: List[Dict],
                    recent_stats: Optional[Dict] = None) -> Dict[str, torch.Tensor]:
        """
        Build heterogeneous graph from environment state.
        
        Args:
            task_queue: Queue of tasks waiting to be processed
            edge_nodes: List of edge node state dictionaries
            current_time: Current simulation time
            task_types: List of task type definitions
            recent_stats: Optional recent statistics for node features
            
        Returns:
            Dictionary with graph data:
            - node_features: Dict with 'task' and 'edge' node features
            - edge_index: Dict with edge indices for each edge type
            - edge_attr: Dict with edge attributes (optional)
            - num_nodes: Dict with number of nodes per type
        """
        # Extract tasks from queue (limit to max_tasks for fixed-size graph)
        tasks = list(task_queue)[:self.max_tasks]
        num_tasks = len(tasks)
        num_edges = len(edge_nodes)
        
        # ============================================================
        # Build Task Nodes
        # ============================================================
        task_features = self._build_task_nodes(tasks, current_time, task_types)
        
        # ============================================================
        # Build Edge Nodes
        # ============================================================
        edge_features = self._build_edge_nodes(edge_nodes, recent_stats)
        
        # ============================================================
        # Build Edges
        # ============================================================
        edge_data = self._build_edges(tasks, edge_nodes, num_tasks, num_edges)
        
        return {
            'node_features': {
                'task': task_features,
                'edge': edge_features
            },
            'edge_index': edge_data['edge_index'],
            'edge_attr': edge_data.get('edge_attr', {}),
            'num_nodes': {
                'task': num_tasks,
                'edge': num_edges
            }
        }
    
    def _build_task_nodes(self, 
                          tasks: List[Dict], 
                          current_time: float,
                          task_types: List[Dict]) -> torch.Tensor:
        """
        Build task node features.
        
        Task node features:
        [0] task_type_id: Task type ID (normalized)
        [1] time_to_deadline: Time until deadline (seconds)
        [2] wait_time: Time waited in queue (seconds)
        [3] arrival_time: Arrival time (normalized)
        [4] deadline: Absolute deadline (normalized)
        [5] priority: Priority score (based on deadline urgency)
        """
        if len(tasks) == 0:
            return torch.zeros((0, self.task_feature_dim), dtype=torch.float32)
        
        features = []
        max_deadline = max(t['deadline'] for t in tasks) if tasks else 1.0
        max_time = max(current_time, max_deadline) if tasks else 1.0
        
        for task in tasks:
            task_type_id = task.get('task_type_id', 0)
            deadline = task.get('deadline', current_time + 1.0)
            arrival_time = task.get('arrival_time', current_time)
            
            time_to_deadline = max(0.0, deadline - current_time)
            wait_time = current_time - arrival_time
            
            # Priority: higher priority for urgent tasks
            priority = 1.0 / (time_to_deadline + 0.001)  # Avoid division by zero
            
            # Normalize features
            normalized_type_id = task_type_id / max(len(task_types), 1)
            normalized_time_to_deadline = time_to_deadline / max(max_deadline, 0.05)
            normalized_wait_time = wait_time / max(max_time, 1.0)
            normalized_arrival_time = arrival_time / max(max_time, 1.0)
            normalized_deadline = deadline / max(max_deadline, 1.0)
            normalized_priority = priority / (1.0 / 0.001)  # Normalize priority
            
            features.append([
                normalized_type_id,
                normalized_time_to_deadline,
                normalized_wait_time,
                normalized_arrival_time,
                normalized_deadline,
                normalized_priority
            ])
        
        return torch.tensor(features, dtype=torch.float32)
    
    def _build_edge_nodes(self, 
                          edge_nodes: List[Dict],
                          recent_stats: Optional[Dict] = None) -> torch.Tensor:
        """
        Build edge node features.
        
        Edge node features:
        [0] node_id: Node ID (normalized)
        [1] busy_status: Whether node is busy (0.0 or 1.0)
        [2] time_until_free: Time until node is free (seconds, normalized)
        [3] utilization_rate: Node utilization rate (0.0-1.0)
        [4] avg_processing_time: Average processing time (normalized)
        [5] capability: Node capability score (normalized)
        """
        if len(edge_nodes) == 0:
            return torch.zeros((0, self.edge_feature_dim), dtype=torch.float32)
        
        features = []
        max_processing_time = 0.012  # Based on PERFORMANCE_PROFILE max
        
        for i, node in enumerate(edge_nodes):
            node_id = i
            busy_status = 1.0 if node.get('busy', False) else 0.0
            free_at_time = node.get('free_at_time', 0.0)
            current_time = node.get('current_time', 0.0)
            
            time_until_free = max(0.0, free_at_time - current_time)
            
            # Get utilization and processing time from recent stats if available
            if recent_stats and i < len(recent_stats.get('node_utilization', [])):
                utilization_rate = recent_stats['node_utilization'][i]
            else:
                utilization_rate = busy_status  # Fallback to current busy status
            
            if recent_stats and i < len(recent_stats.get('node_avg_processing_time', [])):
                avg_processing_time = recent_stats['node_avg_processing_time'][i]
            else:
                avg_processing_time = 0.0016  # Default from PERFORMANCE_PROFILE
            
            # Capability: simple score (can be extended with GPU type, memory, etc.)
            capability = 1.0  # All nodes same capability for now
            
            # Normalize features
            normalized_node_id = node_id / max(len(edge_nodes), 1)
            normalized_time_until_free = time_until_free / max(max_processing_time, 0.05)
            normalized_avg_processing_time = avg_processing_time / max_processing_time
            
            features.append([
                normalized_node_id,
                busy_status,
                normalized_time_until_free,
                utilization_rate,
                normalized_avg_processing_time,
                capability
            ])
        
        return torch.tensor(features, dtype=torch.float32)
    
    def _build_edges(self,
                     tasks: List[Dict],
                     edge_nodes: List[Dict],
                     num_tasks: int,
                     num_edges: int) -> Dict:
        """
        Build edges between nodes.
        
        Edge types:
        - queue: Task → Task (sequential order in queue)
        - type: Task ↔ Task (same task type)
        - affinity: Task ↔ Edge (task-node matching)
        - topology: Edge ↔ Edge (node connections)
        """
        edge_index = {}
        edge_attr = {}
        
        # ============================================================
        # Queue edges: Sequential order in queue
        # ============================================================
        queue_edges = []
        for i in range(num_tasks - 1):
            queue_edges.append([i, i + 1])  # Task i → Task i+1
        
        if queue_edges:
            edge_index['queue'] = torch.tensor(queue_edges, dtype=torch.long).t().contiguous()
        else:
            edge_index['queue'] = torch.zeros((2, 0), dtype=torch.long)
        
        # ============================================================
        # Type edges: Same type tasks (bidirectional)
        # ============================================================
        type_edges = []
        for i in range(num_tasks):
            for j in range(i + 1, num_tasks):
                if tasks[i].get('task_type_id') == tasks[j].get('task_type_id'):
                    type_edges.append([i, j])
                    type_edges.append([j, i])  # Bidirectional
        
        if type_edges:
            edge_index['type'] = torch.tensor(type_edges, dtype=torch.long).t().contiguous()
        else:
            edge_index['type'] = torch.zeros((2, 0), dtype=torch.long)
        
        # ============================================================
        # Affinity edges: Task-Edge matching (bidirectional)
        # Note: In the graph, task indices are 0..num_tasks-1, edge indices are 0..num_edges-1
        # ============================================================
        affinity_edges = []
        for task_idx in range(num_tasks):
            for edge_idx in range(num_edges):
                # Task -> Edge connection
                affinity_edges.append([task_idx, edge_idx])
                # Edge -> Task connection (bidirectional)
                affinity_edges.append([edge_idx, task_idx])
        
        if affinity_edges:
            edge_index['affinity'] = torch.tensor(affinity_edges, dtype=torch.long).t().contiguous()
        else:
            edge_index['affinity'] = torch.zeros((2, 0), dtype=torch.long)
        
        # ============================================================
        # Topology edges: Edge-Edge connections (for multi-node)
        # ============================================================
        topology_edges = []
        if num_edges > 1:
            # Fully connected topology (all nodes can communicate)
            for i in range(num_edges):
                for j in range(i + 1, num_edges):
                    topology_edges.append([i, j])
                    topology_edges.append([j, i])  # Bidirectional
        
        if topology_edges:
            edge_index['topology'] = torch.tensor(topology_edges, dtype=torch.long).t().contiguous()
        else:
            edge_index['topology'] = torch.zeros((2, 0), dtype=torch.long)
        
        return {
            'edge_index': edge_index,
            'edge_attr': edge_attr  # Can add edge weights/attributes later
        }
    
    def get_graph_info(self, graph_data: Dict) -> Dict:
        """
        Get information about the built graph.
        
        Returns:
            Dictionary with graph statistics
        """
        num_nodes = graph_data['num_nodes']
        edge_index = graph_data['edge_index']
        
        info = {
            'num_task_nodes': num_nodes['task'],
            'num_edge_nodes': num_nodes['edge'],
            'num_edges': {}
        }
        
        for edge_type, edges in edge_index.items():
            info['num_edges'][edge_type] = edges.shape[1] if edges.shape[0] > 0 else 0
        
        return info

