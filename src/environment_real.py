"""
Real Data Environment for Batch-Aware RL Scheduler
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import collections
import random
import time
import os

import torch
import torchaudio
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

import src.constants as c
from src.graph_builder import HeteroGraphBuilder
from src.node_selector import StateAwareNodeSelector


class SchedulingEnvReal(gym.Env):

    def __init__(self):
        # Initialize the parent class
        super().__init__()

        # Define action and observation spaces
        self.action_space = spaces.Discrete(c.NUM_ACTIONS)
        
        # Enhanced 12-dimensional state space (between 0 and infinity)
        low = np.array([0] * c.NUM_STATE_FEATURES, dtype=np.float32)
        high = np.array([np.inf] * c.NUM_STATE_FEATURES, dtype=np.float32)
        self.observation_space = spaces.Box(low, high, dtype=np.float32)
        
        # Setup neural network model and dataset
        self.device = torch.device(c.INFERENCE_DEVICE if torch.cuda.is_available() else "cpu")
        
        # Load ResNet-18 model for image tasks
        self.image_model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.image_model.eval()
        self.image_model.to(self.device)
        
        # Define image preprocessing pipeline
        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Load Imagenette dataset for image tasks
        train_path = os.path.join(c.IMAGENETTE_PATH, 'train')
        self.image_dataset = ImageFolder(root=train_path, transform=self.preprocess)
        self.image_data_loader = DataLoader(self.image_dataset, batch_size=1, shuffle=True)
        self.image_data_iterator = iter(self.image_data_loader)
        
        # Load audio model (simple CNN for speech commands)
        self.audio_model = self._create_audio_model()
        self.audio_model.eval()
        self.audio_model.to(self.device)
        
        # Load Google Speech Commands Dataset for audio tasks
        audio_path = c.SPEECH_COMMANDS_PATH
        try:
            from torchaudio.datasets import SPEECHCOMMANDS
            if not os.path.exists(audio_path) or len(os.listdir(audio_path)) == 0:
                self.audio_dataset = None
            else:
                self.audio_dataset = SPEECHCOMMANDS(root=audio_path, download=False, subset='training')
        except Exception:
            self.audio_dataset = None
        
        if self.audio_dataset is not None:
            self.audio_data_loader = DataLoader(self.audio_dataset, batch_size=1, shuffle=True)
            self.audio_data_iterator = iter(self.audio_data_loader)
        else:
            self.audio_data_loader = None
            self.audio_data_iterator = None
        
        self.model = self.image_model
        
        # Warmup GPU for consistent timing
        if self.device.type == 'cuda':
            dummy_image = torch.randn(1, 3, 224, 224).to(self.device)
            dummy_audio = torch.randn(1, 1, 16000).to(self.device)
            with torch.no_grad():
                for _ in range(10):
                    _ = self.image_model(dummy_image)
                for _ in range(10):
                    _ = self.audio_model(dummy_audio)
            torch.cuda.synchronize()
        
        # Initialize simulation state
        self.current_time = 0.0
        self.task_queue = collections.deque()
        self.episode_steps = 0  # Track number of steps in current episode
        
        # List of edge nodes with their states(Only one node for now)
        num_nodes = getattr(c, 'NUM_EDGE_NODES', 1)
        self.edge_nodes = [
            {'busy': False, 'free_at_time': 0.0, 'node_id': i, 'current_time': 0.0}
            for i in range(num_nodes)
        ]

        self.edge_node = self.edge_nodes[0] if len(self.edge_nodes) > 0 else None
        
        # Initialize graph builder for GNN state representation
        self.graph_builder = HeteroGraphBuilder(
            max_tasks=getattr(c, 'MAX_TASKS_IN_GRAPH', 100),
            num_edge_nodes=len(self.edge_nodes)
        )
        self.use_graph_state = getattr(c, 'USE_GRAPH_STATE', True)
        self.node_descriptor_dim = getattr(c, 'GNN_OUTPUT_DIM', c.NUM_STATE_FEATURES)
        self.last_state_vector = None
        self.last_node_embeddings = None
        self.node_selector = StateAwareNodeSelector(
            descriptor_dim=self.node_descriptor_dim,
            history_window=c.HISTORY_WINDOW_SIZE
        )
        
        # Statistics tracking
        self.stats = {
            'total_tasks_arrived': 0,
            'total_tasks_completed': 0,
            'total_tasks_failed': 0,
            'total_latency': 0.0,
            'total_inference_time': 0.0
        }
        
        # History tracking for enhanced state features
        self.recent_queue_lengths = collections.deque(maxlen=c.HISTORY_WINDOW_SIZE)
        self.recent_dispatches = collections.deque(maxlen=c.HISTORY_WINDOW_SIZE)  # (num_success, num_total)
        
        # Node state tracking for enhanced state features (per node)
        self.recent_node_busy_status = [collections.deque(maxlen=c.HISTORY_WINDOW_SIZE) for _ in self.edge_nodes]
        self.recent_processing_times = [collections.deque(maxlen=c.HISTORY_WINDOW_SIZE) for _ in self.edge_nodes]
        
        # Independent arrival process: each task type has its own arrival schedule
        if c.USE_SINGLE_TASK_TYPE:
            # Single type mode: only use first task type (camera)
            self.task_types = [c.TASK_TYPES[0]]
        else:
            # Multi-type mode: use all task types
            self.task_types = c.TASK_TYPES
        
        # Initialize next arrival time for each task type (independent arrival processes)
        self.next_arrival_times = {}
        for i, task_type in enumerate(self.task_types):
            # For audio tasks, use random arrival interval
            if task_type['name'] == 'audio' and 'random_arrival_range' in task_type:
                min_interval, max_interval = task_type['random_arrival_range']
                arrival_delay = random.uniform(min_interval, max_interval)
            else:
                # For ADAS tasks, use fixed interval with small random offset
                arrival_delay = task_type['arrival_interval'] + random.uniform(0, task_type['arrival_interval'] * 0.1)
            self.next_arrival_times[i] = self.current_time + arrival_delay

    def reset(self, seed=None, options=None):
        """Reset environment to initial state for a new episode."""
        super().reset(seed=seed)
        
        self.current_time = 0.0
        self.task_queue.clear()
        self.episode_steps = 0  # Reset episode step counter
        
        # Reset edge nodes
        for node in self.edge_nodes:
            node['busy'] = False
            node['free_at_time'] = 0.0
            node['current_time'] = 0.0
        self.edge_node = self.edge_nodes[0]
        
        self.stats = {
            'total_tasks_arrived': 0,
            'total_tasks_completed': 0,
            'total_tasks_failed': 0,
            'total_latency': 0.0,
            'total_inference_time': 0.0
        }
        
        # Reset history tracking
        self.recent_queue_lengths.clear()
        self.recent_dispatches.clear()
        for busy_status in self.recent_node_busy_status:
            busy_status.clear()
        for processing_times in self.recent_processing_times:
            processing_times.clear()
        
        # Reset independent arrival times for each task type
        self.next_arrival_times = {}
        for i, task_type in enumerate(self.task_types):
            # For audio tasks, use random arrival interval
            if task_type['name'] == 'audio' and 'random_arrival_range' in task_type:
                min_interval, max_interval = task_type['random_arrival_range']
                arrival_delay = random.uniform(min_interval, max_interval)
            else:
                # For ADAS tasks, use fixed interval with small random offset
                arrival_delay = task_type['arrival_interval'] + random.uniform(0, task_type['arrival_interval'] * 0.1)
            self.next_arrival_times[i] = self.current_time + arrival_delay
        
        self.last_state_vector = None
        self.last_node_embeddings = None
        
        # Reset data iterators
        self.image_data_iterator = iter(self.image_data_loader)
        if self.audio_data_loader is not None:
            self.audio_data_iterator = iter(self.audio_data_loader)
        
        # Return state based on mode (graph or vector)
        if self.use_graph_state:
            state = self._get_graph_state()
        else:
            state = self._get_state()
        
        return state, {}

    def step(self, action):
        """
        Execute one environment step with real inference.
        
        Args:
            action: Agent's action (0=WAIT, 1-7=dispatch batch)
        
        Returns:
            state, reward, terminated, truncated, info
        """
        reward = 0.0
        
        # Process agent's action
        if action == c.ACTION_WAIT:
            reward -= c.PENALTY_WAIT * len(self.task_queue)
        else:
            desired_batch_size = c.BATCH_SIZE_OPTIONS[action - 1]
            
            if len(self.task_queue) == 0:
                reward -= c.PENALTY_EMPTY_QUEUE
            else:
                # Find available edge node (multi-node support)
                state_vector = self.last_state_vector if self.use_graph_state else None
                node_embeddings = self.last_node_embeddings if self.use_graph_state else None
                available_node = self.node_selector.select_node(
                    edge_nodes=self.edge_nodes,
                    current_time=self.current_time,
                    recent_busy_history=self.recent_node_busy_status,
                    recent_processing_times=self.recent_processing_times,
                    state_vector=state_vector,
                    node_embeddings=node_embeddings
                )
                
                if available_node is None:
                    reward -= c.PENALTY_NODE_BUSY
                else:
                    # Execute real batch processing with actual inference on selected node
                    reward += self._execute_real_batch_dispatch(desired_batch_size, available_node)
        
        # Advance simulation time and increment episode step counter
        self.current_time += c.SIM_STEP_SECONDS
        self.episode_steps += 1
        
        # Update world state
        for node in self.edge_nodes:
            node['current_time'] = self.current_time
            if node['busy'] and self.current_time >= node['free_at_time']:
                node['busy'] = False
        self.edge_node = self.edge_nodes[0]
        
        # Check for independent task arrivals from each sensor type
        for type_id, task_type in enumerate(self.task_types):
            if self.current_time >= self.next_arrival_times[type_id]:
                self._add_new_task_with_image(type_id, task_type)
                # Schedule next arrival for this specific type
                self._schedule_next_arrival_for_type(type_id, task_type)
        
        # Check for expired tasks
        expired_penalty = self._remove_expired_tasks()
        reward -= expired_penalty
        
        # Update history tracking
        self.recent_queue_lengths.append(len(self.task_queue))
        for i, node in enumerate(self.edge_nodes):
            if i < len(self.recent_node_busy_status):
                self.recent_node_busy_status[i].append(1.0 if node['busy'] else 0.0)
        
        # Prepare return values (support both graph and vector state)
        if self.use_graph_state:
            next_state = self._get_graph_state()
        else:
            next_state = self._get_state()
        # Check episode termination conditions
        terminated = expired_penalty > 0  # Episode ends if tasks expired
        max_episode_steps = getattr(c, 'MAX_EPISODE_STEPS', 10000)
        truncated = (self.episode_steps >= max_episode_steps)  # Episode truncated if too long
        
        info = {
            'stats': self.stats.copy(),
            'current_time': self.current_time,
            'queue_length': len(self.task_queue)
        }
        
        return next_state, reward, terminated, truncated, info

    def _create_audio_model(self):
        """
        Create a simple CNN model for audio classification (speech commands).
        This is a lightweight model for inference timing.
        """
        class SimpleAudioCNN(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # Input: 1 channel, 16000 samples (1 second at 16kHz)
                # Use 1D convolutions for audio
                self.conv1 = torch.nn.Conv1d(1, 32, kernel_size=80, stride=16)
                self.conv2 = torch.nn.Conv1d(32, 64, kernel_size=3)
                self.conv3 = torch.nn.Conv1d(64, 64, kernel_size=3)
                self.pool = torch.nn.AdaptiveAvgPool1d(1)
                self.fc = torch.nn.Linear(64, 10)  # 10 classes for speech commands
                
            def forward(self, x):
                # x: [batch, 1, 16000]
                x = self.conv1(x)
                x = torch.relu(x)
                x = self.conv2(x)
                x = torch.relu(x)
                x = self.conv3(x)
                x = torch.relu(x)
                x = self.pool(x)
                x = x.view(x.size(0), -1)
                x = self.fc(x)
                return x
        
        return SimpleAudioCNN()

    def _execute_real_batch_dispatch(self, desired_batch_size, edge_node):
        """
        Execute batch dispatch with REAL ResNet-18 inference on specified edge node.
        
        This is the key difference from simulation:
        - Actually runs neural network inference on real images
        - Measures actual processing time
        - Considers GPU state, memory, etc.
        
        Args:
            desired_batch_size: Requested batch size
            edge_node: Edge node dictionary to process the batch
        
        Returns:
            reward: Calculated reward for this dispatch
        """
        # Actual batch size limited by queue length
        actual_batch_size = min(desired_batch_size, len(self.task_queue))
        batch_tasks = [self.task_queue.popleft() for _ in range(actual_batch_size)]
        
        # Separate tasks by type (image vs audio)
        image_tasks = [t for t in batch_tasks if 'image' in t]
        audio_tasks = [t for t in batch_tasks if 'audio' in t]
        
        # Process image tasks if any
        if image_tasks:
            batch_images = [task['image'] for task in image_tasks]
            batch_tensor = torch.cat(batch_images, dim=0).to(self.device)
            
            with torch.no_grad():
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                
                start_time = time.perf_counter()
                outputs = self.image_model(batch_tensor)  # Real image inference!
                
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                
                end_time = time.perf_counter()
                processing_time = end_time - start_time
        else:
            processing_time = 0.0
        
        # Process audio tasks if any
        if audio_tasks:
            batch_audios = [task['audio'] for task in audio_tasks]
            batch_tensor = torch.cat(batch_audios, dim=0).to(self.device)
            
            with torch.no_grad():
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                
                start_time = time.perf_counter()
                outputs = self.audio_model(batch_tensor)  # Real audio inference!
                
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                
                end_time = time.perf_counter()
                audio_processing_time = end_time - start_time
                processing_time += audio_processing_time
        
        # Measure actual processing time
        self.stats['total_inference_time'] += processing_time
        
        # Track processing time for the specific node
        node_id = edge_node.get('node_id', 0)
        if node_id < len(self.recent_processing_times):
            self.recent_processing_times[node_id].append(processing_time)
        
        # Update edge node state
        edge_node['busy'] = True
        completion_time = self.current_time + processing_time
        edge_node['free_at_time'] = completion_time
        
        return self._calculate_batch_reward(batch_tasks, completion_time, actual_batch_size)

    def _calculate_batch_reward(self, batch_tasks, completion_time, batch_size):
        """
        Calculate reward for a dispatched batch.
        
        Reward components:
        1. Success/failure for each task (deadline met or not)
        2. Latency penalty (minimize overall latency)
        3. Batch efficiency bonus (encourage appropriate batching)
        """
        reward = 0.0
        num_success = 0
        
        for task in batch_tasks:
            task_latency = completion_time - task['arrival_time']
            
            if completion_time <= task['deadline']:
                reward += c.REWARD_TASK_SUCCESS
                self.stats['total_tasks_completed'] += 1
                reward -= task_latency * c.LATENCY_PENALTY_COEFF
                self.stats['total_latency'] += task_latency
                num_success += 1
            else:
                reward -= c.PENALTY_TASK_MISS
                self.stats['total_tasks_failed'] += 1
        
        # Track dispatch success for history
        self.recent_dispatches.append((num_success, batch_size))
        
        # Batch efficiency bonus
        if batch_size > 1:
            reward += np.log(batch_size) * c.BATCH_BONUS_COEFF
        
        return reward

    def _get_state(self):
        """
        Get current state observation.
        
        Returns state vector organized by categories:
        
        Category 1: Queue State (Features 0-2)
        Category 2: Task Type Distribution (Features 3-5)
        Category 3: Node State (Features 6, 9-11)
        Category 4: Historical Statistics (Features 7-8)
        Category 5: Future Extensions (Reserved)
        """
        queue_length = len(self.task_queue)
        
        # ============================================================
        # Category 1: Queue State (Instant queue information)
        # ============================================================
        if queue_length == 0:
            # Use maximum deadline from task types as default
            max_deadline = max(t['deadline'] for t in self.task_types) if self.task_types else 0.05
            time_to_nearest_deadline = max_deadline
            time_since_oldest_task = 0.0
        else:
            min_deadline = min(task['deadline'] for task in self.task_queue)
            time_to_nearest_deadline = max(0.0, min_deadline - self.current_time)
            
            oldest_task = self.task_queue[0]
            time_since_oldest_task = self.current_time - oldest_task['arrival_time']
        
        # [0] queue_length: Current number of tasks in queue
        # [1] time_to_nearest_deadline: Time until most urgent task's deadline (seconds)
        # [2] time_since_oldest_task: How long the oldest task has been waiting (seconds)
        
        # ============================================================
        # Category 2: Task Type Distribution (Task type counts)
        # ============================================================
        # Count tasks by type in the queue
        count_by_type = [0] * c.NUM_TASK_TYPES
        for task in self.task_queue:
            if 'task_type_id' in task:
                type_id = task['task_type_id']
                if 0 <= type_id < c.NUM_TASK_TYPES:
                    count_by_type[type_id] += 1
        
        # Ensure we have exactly 3 features for task types (pad with zeros if needed)
        count_type_0 = count_by_type[0] if len(count_by_type) > 0 else 0
        count_type_1 = count_by_type[1] if len(count_by_type) > 1 else 0
        count_type_2 = count_by_type[2] if len(count_by_type) > 2 else 0
        
        # [3] count_type_0: Number of type 0 tasks in queue (e.g., camera)
        # [4] count_type_1: Number of type 1 tasks in queue (e.g., radar)
        # [5] count_type_2: Number of type 2 tasks in queue (e.g., lidar)
        
        # Future extensions for Category 2:
        # [RESERVED] avg_wait_time_by_type: Average waiting time per task type
        # [RESERVED] urgent_tasks_by_type: Count of urgent tasks (deadline < threshold) per type
        # [RESERVED] deadline_variance: Variance of deadlines in queue (urgency spread)
        
        # ============================================================
        # Category 3: Node State (Computing node information)
        # ============================================================
        first_node = self.edge_nodes[0]
        time_until_node_free = max(0.0, first_node['free_at_time'] - self.current_time)
        node_busy_status = 1.0 if first_node['busy'] else 0.0
        
        # Get utilization rate from first node
        if len(self.recent_node_busy_status) > 0 and len(self.recent_node_busy_status[0]) > 0:
            node_utilization_rate = np.mean(list(self.recent_node_busy_status[0]))
        else:
            node_utilization_rate = 0.0  # No history, assume idle
        
        # Get average processing time from first node
        if len(self.recent_processing_times) > 0 and len(self.recent_processing_times[0]) > 0:
            node_avg_processing_time = np.mean(list(self.recent_processing_times[0]))
        else:
            # No processing history yet, use a default value based on typical batch size
            # Estimate based on PERFORMANCE_PROFILE for batch_size=1
            node_avg_processing_time = 0.0016  # ~1.6ms from PERFORMANCE_PROFILE
        
        # [6] time_until_node_free: How long until edge node finishes current batch (seconds)
        # [9] node_busy_status: Current node busy status (1.0 if busy, 0.0 if free)
        # [10] node_utilization_rate: Fraction of time node was busy over last N steps
        # [11] node_avg_processing_time: Average processing time of last N batches (seconds)
        
        # Future extensions for Category 3:
        # [RESERVED] node_throughput: Tasks processed per second (recent)
        # [RESERVED] node_pending_batch_size: Size of current batch being processed
        # [RESERVED] node_memory_usage: Memory utilization (if multiple nodes)
        
        # ============================================================
        # Category 4: Historical Statistics (Trend information)
        # ============================================================
        if len(self.recent_queue_lengths) == 0:
            avg_queue_length_recent = queue_length
        else:
            avg_queue_length_recent = np.mean(self.recent_queue_lengths)
        
        if len(self.recent_dispatches) == 0:
            recent_success_rate = 1.0  # Assume 100% at start
        else:
            total_success = sum(success for success, _ in self.recent_dispatches)
            total_tasks = sum(total for _, total in self.recent_dispatches)
            recent_success_rate = total_success / total_tasks if total_tasks > 0 else 1.0
        
        # [7] avg_queue_length_recent: Average queue length over last N steps
        # [8] recent_success_rate: Success rate of last N dispatches
        
        # Future extensions for Category 4:
        # [RESERVED] queue_length_trend: Trend of queue length (increasing/decreasing rate)
        # [RESERVED] avg_latency_recent: Average task latency in recent dispatches
        # [RESERVED] deadline_miss_rate: Rate of deadline misses in recent history
        
        # ============================================================
        # Category 5: Future Extensions (Reserved for dependencies, etc.)
        # ============================================================
        # [RESERVED] task_dependency_count: Number of tasks waiting for dependencies
        # [RESERVED] dependent_task_ratio: Ratio of tasks with dependencies
        # [RESERVED] system_load_prediction: Predicted load in next time window
        # [RESERVED] multi_node_state: State information if multiple nodes (future)
        
        # ============================================================
        # Assemble state vector (12 dimensions)
        # ============================================================
        return np.array([
            # Category 1: Queue State (Features 0-2)
            queue_length,                    # [0]
            time_to_nearest_deadline,        # [1]
            time_since_oldest_task,          # [2]
            
            # Category 2: Task Type Distribution (Features 3-5)
            count_type_0,                    # [3]
            count_type_1,                    # [4]
            count_type_2,                    # [5]
            
            # Category 3: Node State (Features 6, 9-11)
            time_until_node_free,            # [6]
            avg_queue_length_recent,         # [7]
            recent_success_rate,             # [8]
            node_busy_status,                # [9]
            node_utilization_rate,           # [10]
            node_avg_processing_time,        # [11]
        ], dtype=np.float32)
    
    def _get_graph_state(self):
        """
        Get current state as heterogeneous graph for GNN processing.
        
        Returns:
            Dictionary with graph data structure (compatible with PyTorch Geometric or DGL)
        """
        # Prepare recent statistics for graph builder
        recent_stats = {
            'node_utilization': [],
            'node_avg_processing_time': []
        }
        
        for i in range(len(self.edge_nodes)):
            # Get utilization rate for this node
            if i < len(self.recent_node_busy_status) and len(self.recent_node_busy_status[i]) > 0:
                utilization = np.mean(list(self.recent_node_busy_status[i]))
            else:
                utilization = 1.0 if self.edge_nodes[i]['busy'] else 0.0
            
            # Get average processing time for this node
            if i < len(self.recent_processing_times) and len(self.recent_processing_times[i]) > 0:
                avg_processing_time = np.mean(list(self.recent_processing_times[i]))
            else:
                avg_processing_time = 0.0016  # Default
            
            recent_stats['node_utilization'].append(utilization)
            recent_stats['node_avg_processing_time'].append(avg_processing_time)
        
        # Build graph
        graph_data = self.graph_builder.build_graph(
            task_queue=self.task_queue,
            edge_nodes=self.edge_nodes,
            current_time=self.current_time,
            task_types=self.task_types,
            recent_stats=recent_stats
        )
        
        return graph_data

    def _schedule_next_arrival_for_type(self, type_id, task_type):
        """
        Schedule next arrival time for a specific task type (independent arrival process).
        
        Args:
            type_id: Task type ID
            task_type: Task type dictionary with arrival_interval or random_arrival_range
        """
        # For audio tasks, use random arrival interval
        if task_type['name'] == 'audio' and 'random_arrival_range' in task_type:
            min_interval, max_interval = task_type['random_arrival_range']
            arrival_delay = random.uniform(min_interval, max_interval)
        else:
            # Use fixed interval for ADAS sensor types
            arrival_delay = task_type['arrival_interval']
        self.next_arrival_times[type_id] = self.current_time + arrival_delay

    def _add_new_task_with_image(self, type_id, task_type):
        """
        Add a new task to the queue WITH REAL DATA (image or audio).
        
        Loads actual data from dataset and assigns the specified task type.
        This method is called for each independent sensor arrival.
        
        Args:
            type_id: Task type ID (determined by which sensor is arriving)
            task_type: Task type dictionary with deadline and name
        """
        # Handle audio tasks differently
        if task_type['name'] == 'audio':
            # Load audio data
            if self.audio_data_iterator is None:
                # Fallback: create dummy audio data if dataset not available
                audio_data = torch.randn(1, 16000).to(self.device)  # 1 second at 16kHz
                label = torch.tensor([0])
            else:
                try:
                    waveform, sample_rate, label, speaker_id, utterance_number = next(self.audio_data_iterator)
                    # Resample to 16kHz if needed and convert to mono
                    if sample_rate != 16000:
                        resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                        waveform = resampler(waveform)
                    if waveform.shape[0] > 1:
                        waveform = waveform[0:1]  # Take first channel
                    # Pad or truncate to 1 second (16000 samples)
                    if waveform.shape[1] < 16000:
                        waveform = torch.nn.functional.pad(waveform, (0, 16000 - waveform.shape[1]))
                    elif waveform.shape[1] > 16000:
                        waveform = waveform[:, :16000]
                    audio_data = waveform.to(self.device)
                except StopIteration:
                    # Dataset exhausted, restart iterator
                    try:
                        self.audio_data_iterator = iter(self.audio_data_loader)
                        waveform, sample_rate, label, speaker_id, utterance_number = next(self.audio_data_iterator)
                        if sample_rate != 16000:
                            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                            waveform = resampler(waveform)
                        if waveform.shape[0] > 1:
                            waveform = waveform[0:1]
                        if waveform.shape[1] < 16000:
                            waveform = torch.nn.functional.pad(waveform, (0, 16000 - waveform.shape[1]))
                        elif waveform.shape[1] > 16000:
                            waveform = waveform[:, :16000]
                        audio_data = waveform.to(self.device)
                    except (StopIteration, Exception) as e:
                        # If still fails (e.g., audio backend issue), use placeholder
                        audio_data = torch.randn(1, 16000).to(self.device)
                        label = torch.tensor([0])
                except Exception as e:
                    # Handle audio loading errors (e.g., missing backend, corrupted files)
                    # Use placeholder audio data
                    audio_data = torch.randn(1, 16000).to(self.device)
                    label = torch.tensor([0])
            
            # Random deadline for audio tasks
            if 'random_deadline_range' in task_type:
                min_deadline, max_deadline = task_type['random_deadline_range']
                deadline_offset = random.uniform(min_deadline, max_deadline)
            else:
                deadline_offset = 0.3  # Default
            
            new_task = {
                'arrival_time': self.current_time,
                'deadline': self.current_time + deadline_offset,
                'task_type': task_type['name'],
                'task_type_id': type_id,
                'task_id': self.stats['total_tasks_arrived'],
                'audio': audio_data,  # Audio waveform tensor
                'label': label
            }
        else:
            # Handle image tasks (camera, radar, lidar)
            try:
                image, label = next(self.image_data_iterator)
            except StopIteration:
                # Dataset exhausted, restart iterator
                self.image_data_iterator = iter(self.image_data_loader)
                image, label = next(self.image_data_iterator)
            
            new_task = {
                'arrival_time': self.current_time,
                'deadline': self.current_time + task_type['deadline'],
                'task_type': task_type['name'],
                'task_type_id': type_id,
                'task_id': self.stats['total_tasks_arrived'],
                'image': image,  # Actual image tensor
                'label': label
            }
        
        self.task_queue.append(new_task)
        self.stats['total_tasks_arrived'] += 1

    def _remove_expired_tasks(self):
        """
        Remove tasks that expired while waiting in queue.
        
        Returns:
            penalty: Total penalty for expired tasks
        """
        penalty = 0.0
        tasks_to_remove = []
        
        for task in self.task_queue:
            if self.current_time > task['deadline']:
                penalty += c.PENALTY_QUEUE_EXPIRY
                tasks_to_remove.append(task)
                self.stats['total_tasks_failed'] += 1
        
        for task in tasks_to_remove:
            self.task_queue.remove(task)
        
        return penalty

    def render(self, mode='human'):
        """Display current environment state with inference statistics."""
        state = self._get_state()
        avg_inference_time = (self.stats['total_inference_time'] / 
                             max(1, self.stats['total_tasks_completed'] + self.stats['total_tasks_failed']))
        
        print(
            f"[REAL] Time: {self.current_time:.2f}s | "
            f"Queue: {int(state[0])} | "
            f"Nearest Deadline: {state[1]:.3f}s | "
            f"Oldest Wait: {state[2]:.3f}s | "
            f"Node: {'BUSY' if self.edge_node['busy'] else 'FREE'} | "
            f"Success/Fail: {self.stats['total_tasks_completed']}/{self.stats['total_tasks_failed']} | "
            f"Avg Inference: {avg_inference_time*1000:.2f}ms"
        )

    def close(self):
        """Clean up resources."""
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
