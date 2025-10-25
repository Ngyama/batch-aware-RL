"""
Real Data Environment for Batch-Aware RL Scheduler

Processes actual images from Imagenette dataset using real ResNet-18 inference,
providing realistic timing measurements at the cost of slower training.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import collections
import random
import time
import os

import torch
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

import src.constants as c


class SchedulingEnvReal(gym.Env):
    """
    Real-data environment for batch-aware scheduling.
    
    Loads actual images and runs real ResNet-18 inference to measure
    processing times, providing realistic environment behavior.
    """

    def __init__(self):
        super(SchedulingEnvReal, self).__init__()

        # Define action and observation spaces
        self.action_space = spaces.Discrete(c.NUM_ACTIONS)
        
        # Enhanced 9-dimensional state space
        low = np.array([0] * c.NUM_STATE_FEATURES, dtype=np.float32)
        high = np.array([np.inf] * c.NUM_STATE_FEATURES, dtype=np.float32)
        self.observation_space = spaces.Box(low, high, dtype=np.float32)
        
        # Setup neural network model and dataset
        print("[INIT] Initializing Real Environment...")
        
        self.device = torch.device(c.INFERENCE_DEVICE if torch.cuda.is_available() else "cpu")
        print(f"  - Using device: {self.device}")
        
        # Load ResNet-18 model
        print("  - Loading ResNet-18 model...")
        self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.model.eval()
        self.model.to(self.device)
        
        # Define image preprocessing pipeline
        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Load Imagenette dataset
        train_path = os.path.join(c.IMAGENETTE_PATH, 'train')
        if not os.path.exists(train_path):
            raise FileNotFoundError(
                f"Imagenette dataset not found at {train_path}. "
                f"Please ensure the dataset is downloaded."
            )
        
        print(f"  - Loading Imagenette dataset from {train_path}...")
        self.dataset = ImageFolder(root=train_path, transform=self.preprocess)
        print(f"  - Dataset loaded: {len(self.dataset)} images")
        
        self.data_loader = DataLoader(self.dataset, batch_size=1, shuffle=True)
        self.data_iterator = iter(self.data_loader)
        
        # Warmup GPU for consistent timing
        if self.device.type == 'cuda':
            print("  - Warming up GPU...")
            dummy_input = torch.randn(1, 3, 224, 224).to(self.device)
            with torch.no_grad():
                for _ in range(10):
                    _ = self.model(dummy_input)
            torch.cuda.synchronize()
        
        print("[√] Real Environment initialized successfully!")
        
        # Initialize simulation state
        self.current_time = 0.0
        self.task_queue = collections.deque()
        self.edge_node = {'busy': False, 'free_at_time': 0.0}
        
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
        
        self._schedule_next_task_arrival()

    def reset(self, seed=None, options=None):
        """Reset environment to initial state for a new episode."""
        super().reset(seed=seed)
        
        self.current_time = 0.0
        self.task_queue.clear()
        self.edge_node = {'busy': False, 'free_at_time': 0.0}
        
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
        
        self.data_iterator = iter(self.data_loader)
        self._schedule_next_task_arrival()
        
        return self._get_state(), {}

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
            elif self.edge_node['busy']:
                reward -= c.PENALTY_NODE_BUSY
            else:
                # Execute real batch processing with actual inference
                reward += self._execute_real_batch_dispatch(desired_batch_size)
        
        # Advance simulation time
        self.current_time += c.SIM_STEP_SECONDS
        
        # Update world state
        if self.edge_node['busy'] and self.current_time >= self.edge_node['free_at_time']:
            self.edge_node['busy'] = False
        
        if self.current_time >= self.next_task_arrival_time:
            self._add_new_task_with_image()
            self._schedule_next_task_arrival()
        
        # Check for expired tasks
        expired_penalty = self._remove_expired_tasks()
        reward -= expired_penalty
        
        # Update history tracking for enhanced state features
        self.recent_queue_lengths.append(len(self.task_queue))
        
        # Prepare return values
        next_state = self._get_state()
        terminated = expired_penalty > 0
        truncated = False
        
        info = {
            'stats': self.stats.copy(),
            'current_time': self.current_time,
            'queue_length': len(self.task_queue)
        }
        
        return next_state, reward, terminated, truncated, info

    def _execute_real_batch_dispatch(self, desired_batch_size):
        """
        Execute batch dispatch with REAL ResNet-18 inference.
        
        This is the key difference from simulation:
        - Actually runs neural network inference on real images
        - Measures actual processing time
        - Considers GPU state, memory, etc.
        
        Args:
            desired_batch_size: Requested batch size
        
        Returns:
            reward: Calculated reward for this dispatch
        """
        # Actual batch size limited by queue length
        actual_batch_size = min(desired_batch_size, len(self.task_queue))
        batch_tasks = [self.task_queue.popleft() for _ in range(actual_batch_size)]
        
        # Collect all images in the batch
        batch_images = [task['image'] for task in batch_tasks]
        batch_tensor = torch.cat(batch_images, dim=0).to(self.device)
        
        # Perform actual neural network inference and measure time
        with torch.no_grad():
            if self.device.type == 'cuda':
                torch.cuda.synchronize()
            
            start_time = time.perf_counter()
            outputs = self.model(batch_tensor)  # Real inference!
            
            if self.device.type == 'cuda':
                torch.cuda.synchronize()
            
            end_time = time.perf_counter()
        
        # Measure actual processing time
        processing_time = end_time - start_time
        self.stats['total_inference_time'] += processing_time
        
        # Update edge node state
        self.edge_node['busy'] = True
        completion_time = self.current_time + processing_time
        self.edge_node['free_at_time'] = completion_time
        
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
        
        Returns enhanced 9-dimensional state vector:
            [0] queue_length: Current number of tasks in queue
            [1] time_to_nearest_deadline: Time until most urgent task's deadline (seconds)
            [2] time_since_oldest_task: How long the oldest task has been waiting (seconds)
            [3] ratio_urgent_tasks: Fraction of tasks with deadline < 10ms
            [4] ratio_medium_tasks: Fraction of tasks with deadline 10-30ms
            [5] ratio_relaxed_tasks: Fraction of tasks with deadline > 30ms
            [6] time_until_node_free: How long until edge node finishes current batch (seconds)
            [7] avg_queue_length_recent: Average queue length over last N steps
            [8] recent_success_rate: Success rate of last N dispatches
        """
        queue_length = len(self.task_queue)
        
        # Features 0-2: Basic queue state (original features)
        if queue_length == 0:
            time_to_nearest_deadline = c.TASK_DEADLINE_SECONDS
            time_since_oldest_task = 0.0
        else:
            min_deadline = min(task['deadline'] for task in self.task_queue)
            time_to_nearest_deadline = max(0.0, min_deadline - self.current_time)
            
            oldest_task = self.task_queue[0]
            time_since_oldest_task = self.current_time - oldest_task['arrival_time']
        
        # Features 3-5: Task urgency distribution
        if queue_length == 0:
            ratio_urgent = 0.0
            ratio_medium = 0.0
            ratio_relaxed = 0.0
        else:
            num_urgent = sum(1 for task in self.task_queue 
                           if (task['deadline'] - self.current_time) < c.URGENT_THRESHOLD)
            num_medium = sum(1 for task in self.task_queue 
                           if c.MEDIUM_THRESHOLD_LOW <= (task['deadline'] - self.current_time) < c.MEDIUM_THRESHOLD_HIGH)
            num_relaxed = queue_length - num_urgent - num_medium
            
            ratio_urgent = num_urgent / queue_length
            ratio_medium = num_medium / queue_length
            ratio_relaxed = num_relaxed / queue_length
        
        # Feature 6: Node availability
        time_until_node_free = max(0.0, self.edge_node['free_at_time'] - self.current_time)
        
        # Feature 7: Recent average queue length
        if len(self.recent_queue_lengths) == 0:
            avg_queue_length_recent = queue_length
        else:
            avg_queue_length_recent = np.mean(self.recent_queue_lengths)
        
        # Feature 8: Recent success rate
        if len(self.recent_dispatches) == 0:
            recent_success_rate = 1.0  # Assume 100% at start
        else:
            total_success = sum(success for success, _ in self.recent_dispatches)
            total_tasks = sum(total for _, total in self.recent_dispatches)
            recent_success_rate = total_success / total_tasks if total_tasks > 0 else 1.0
        
        return np.array([
            queue_length,
            time_to_nearest_deadline,
            time_since_oldest_task,
            ratio_urgent,
            ratio_medium,
            ratio_relaxed,
            time_until_node_free,
            avg_queue_length_recent,
            recent_success_rate
        ], dtype=np.float32)

    def _schedule_next_task_arrival(self):
        """Schedule next task arrival time (Poisson process or fixed rate)."""
        if c.TASK_ARRIVAL_MODE == "fixed_rate":
            arrival_delay = 1.0 / c.FIXED_FRAME_RATE
        else:
            arrival_delay = random.expovariate(1.0 / c.TASK_ARRIVAL_INTERVAL_SECONDS)
        
        self.next_task_arrival_time = self.current_time + arrival_delay

    def _add_new_task_with_image(self):
        """
        Add a new task to the queue WITH A REAL IMAGE.
        
        Loads an actual image from Imagenette dataset.
        """
        try:
            image, label = next(self.data_iterator)
        except StopIteration:
            # Dataset exhausted, restart iterator
            self.data_iterator = iter(self.data_loader)
            image, label = next(self.data_iterator)
        
        # Calculate deadline based on mode
        if hasattr(c, 'TASK_DEADLINE_MODE') and c.TASK_DEADLINE_MODE == "random":
            # Random deadline: simulates heterogeneous task urgency
            deadline_duration = random.uniform(c.TASK_DEADLINE_MIN, c.TASK_DEADLINE_MAX)
        else:
            # Fixed deadline
            deadline_duration = c.TASK_DEADLINE_SECONDS
        
        # Create task with real image data
        new_task = {
            'arrival_time': self.current_time,
            'deadline': self.current_time + deadline_duration,
            'task_id': self.stats['total_tasks_arrived'],
            'image': image,  # Actual image tensor
            'label': label   # Ground truth label
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
