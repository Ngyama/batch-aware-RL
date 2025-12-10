"""
State-aware node selection utility.
"""

from typing import List, Optional

import numpy as np


class StateAwareNodeSelector:
    """Selects the best edge node based on availability and state embedding."""

    def __init__(self, descriptor_dim: int, history_window: int):
        self.descriptor_dim = descriptor_dim
        self.history_window = history_window

    def select_node(
        self,
        edge_nodes: List[dict],
        current_time: float,
        recent_busy_history: List,
        recent_processing_times: List,
        state_vector: Optional[np.ndarray],
        node_embeddings: Optional[np.ndarray] = None,
    ):
        """Select an available edge node."""
        available_nodes = [node for node in edge_nodes if not node["busy"]]
        if not available_nodes:
            return None

        if (
            node_embeddings is not None
            and len(node_embeddings) >= len(edge_nodes)
            and state_vector is not None
            and state_vector.size > 0
        ):
            selected = self._select_with_embeddings(
                available_nodes,
                node_embeddings,
                state_vector,
            )
            if selected is not None:
                return selected

        return self._select_with_descriptors(
            available_nodes,
            current_time,
            recent_busy_history,
            recent_processing_times,
            state_vector,
        )

    def _select_with_embeddings(
        self,
        available_nodes: List[dict],
        node_embeddings: np.ndarray,
        state_vector: np.ndarray,
    ) -> Optional[dict]:
        state_vec = np.asarray(state_vector, dtype=np.float32).flatten()
        state_slice = state_vec[: node_embeddings.shape[1]]
        best_node = None
        best_score = -np.inf

        for node in available_nodes:
            node_id = node.get("node_id", 0)
            if node_id >= node_embeddings.shape[0]:
                continue
            node_vec = node_embeddings[node_id]
            score = float(np.dot(node_vec, state_slice))
            if score > best_score:
                best_score = score
                best_node = node

        return best_node

    def _select_with_descriptors(
        self,
        available_nodes: List[dict],
        current_time: float,
        recent_busy_history: List,
        recent_processing_times: List,
        state_vector: Optional[np.ndarray],
    ) -> Optional[dict]:
        if state_vector is None or state_vector.size == 0:
            return available_nodes[0]

        state_vec = np.asarray(state_vector, dtype=np.float32).flatten()

        best_node = None
        best_score = -np.inf

        for node in available_nodes:
            descriptor = self._build_node_descriptor(
                node, current_time, recent_busy_history, recent_processing_times
            )
            score = float(np.dot(state_vec[: self.descriptor_dim], descriptor))
            if score > best_score:
                best_score = score
                best_node = node

        return best_node or available_nodes[0]

    def _build_node_descriptor(
        self,
        node: dict,
        current_time: float,
        recent_busy_history: List,
        recent_processing_times: List,
    ) -> np.ndarray:
        node_id = node.get("node_id", 0)
        time_until_free = max(0.0, node["free_at_time"] - current_time)
        utilization = self._get_node_utilization(node_id, recent_busy_history, node["busy"])
        avg_processing_time = self._get_node_avg_processing_time(node_id, recent_processing_times)
        history_ratio = (
            len(recent_processing_times[node_id]) / max(1, self.history_window)
            if node_id < len(recent_processing_times)
            else 0.0
        )

        availability = 1.0
        free_time_score = 1.0 / (1.0 + time_until_free)
        utilization_score = 1.0 - utilization
        speed_score = 1.0 / (1.0 + avg_processing_time * 1000.0)
        history_confidence = history_ratio
        bias = 1.0

        base_features = np.array(
            [
                availability,
                free_time_score,
                utilization_score,
                speed_score,
                history_confidence,
                bias,
            ],
            dtype=np.float32,
        )

        if self.descriptor_dim <= base_features.size:
            return base_features[: self.descriptor_dim]

        repeat_times = int(np.ceil(self.descriptor_dim / base_features.size))
        tiled = np.tile(base_features, repeat_times)
        return tiled[: self.descriptor_dim]

    def _get_node_utilization(self, node_id: int, busy_history: List, busy_flag: bool) -> float:
        if node_id < len(busy_history) and len(busy_history[node_id]) > 0:
            return float(np.mean(list(busy_history[node_id])))
        return 1.0 if busy_flag else 0.0

    def _get_node_avg_processing_time(self, node_id: int, processing_history: List) -> float:
        if node_id < len(processing_history) and len(processing_history[node_id]) > 0:
            return float(np.mean(list(processing_history[node_id])))
        return 0.0016

