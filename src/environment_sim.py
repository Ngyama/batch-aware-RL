"""
Simulation Environment for Batch-Aware RL Scheduler

Fast simulation environment using pre-profiled performance data for rapid training.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import collections
import random
import src.constants as c


class SchedulingEnvSim(gym.Env):
    """
    Simulation-based environment for batch-aware scheduling.
    
    Uses lookup tables for processing times instead of actual inference,
    enabling fast training iterations.
    """

    def __init__(self):
        super(SchedulingEnvSim, self).__init__()

        # Define action and observation spaces
        self.action_space = spaces.Discrete(c.NUM_ACTIONS)
        
        # Enhanced 9-dimensional state space
        low = np.array([0] * c.NUM_STATE_FEATURES, dtype=np.float32)
        high = np.array([np.inf] * c.NUM_STATE_FEATURES, dtype=np.float32)
        self.observation_space = spaces.Box(low, high, dtype=np.float32)
        
        # Initialize simulation state
        self.current_time = 0.0
        self.task_queue = collections.deque()
        self.edge_node = {'busy': False, 'free_at_time': 0.0}
        
        # Statistics tracking
        self.stats = {
            'total_tasks_arrived': 0,
            'total_tasks_completed': 0,
            'total_tasks_failed': 0,
            'total_latency': 0.0
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
            'total_latency': 0.0
        }
        
        # Reset history tracking
        self.recent_queue_lengths.clear()
        self.recent_dispatches.clear()
        
        self._schedule_next_task_arrival()
        
        return self._get_state(), {}

    def step(self, action):
        """
        Execute one environment step.
        
        Args:
            action: Agent's action (0=WAIT, 1-7=dispatch batch)
        
        Returns:
            state, reward, terminated, truncated, info
        """
        reward = 0.0
        
        # Process agent's action
        if action == c.ACTION_WAIT:
            # Small penalty for waiting (proportional to queue size)
            reward -= c.PENALTY_WAIT * len(self.task_queue)
        else:
            # Dispatch batch with chosen size
            desired_batch_size = c.BATCH_SIZE_OPTIONS[action - 1]
            
            if len(self.task_queue) == 0:
                reward -= c.PENALTY_EMPTY_QUEUE
            elif self.edge_node['busy']:
                reward -= c.PENALTY_NODE_BUSY
            else:
                reward += self._execute_batch_dispatch(desired_batch_size)
        
        # Advance simulation time
        self.current_time += c.SIM_STEP_SECONDS
        
        # Update world state
        if self.edge_node['busy'] and self.current_time >= self.edge_node['free_at_time']:
            self.edge_node['busy'] = False
        
        if self.current_time >= self.next_task_arrival_time:
            self._add_new_task()
            self._schedule_next_task_arrival()
        
        # Check for expired tasks (heavy penalty)
        expired_penalty = self._remove_expired_tasks()
        reward -= expired_penalty
        
        # Update history tracking for enhanced state features
        self.recent_queue_lengths.append(len(self.task_queue))
        
        # Prepare return values
        next_state = self._get_state()
        terminated = expired_penalty > 0  # Episode ends if task expires
        truncated = False
        
        info = {
            'stats': self.stats.copy(),
            'current_time': self.current_time,
            'queue_length': len(self.task_queue)
        }
        
        return next_state, reward, terminated, truncated, info

    def _execute_batch_dispatch(self, desired_batch_size):
        """
        Dispatch a batch for processing.
        
        Args:
            desired_batch_size: Requested batch size
        
        Returns:
            reward: Calculated reward for this dispatch
        """
        # Actual batch size limited by queue length
        actual_batch_size = min(desired_batch_size, len(self.task_queue))
        batch_tasks = [self.task_queue.popleft() for _ in range(actual_batch_size)]
        
        # Lookup processing time from performance profile
        processing_time = self._get_processing_time(actual_batch_size)
        
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
                # Task completed successfully
                reward += c.REWARD_TASK_SUCCESS
                self.stats['total_tasks_completed'] += 1
                reward -= task_latency * c.LATENCY_PENALTY_COEFF
                self.stats['total_latency'] += task_latency
                num_success += 1
            else:
                # Task missed deadline
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

    def _add_new_task(self):
        """Add a new task to the queue."""
        # Calculate deadline based on mode
        if hasattr(c, 'TASK_DEADLINE_MODE') and c.TASK_DEADLINE_MODE == "random":
            # Random deadline: simulates heterogeneous task urgency
            deadline_duration = random.uniform(c.TASK_DEADLINE_MIN, c.TASK_DEADLINE_MAX)
        else:
            # Fixed deadline
            deadline_duration = c.TASK_DEADLINE_SECONDS
        
        new_task = {
            'arrival_time': self.current_time,
            'deadline': self.current_time + deadline_duration,
            'task_id': self.stats['total_tasks_arrived']
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

    def _get_processing_time(self, batch_size):
        """
        Get processing time for a given batch size from performance profile.
        
        Args:
            batch_size: Number of tasks in the batch
        
        Returns:
            processing_time: Time in seconds to process this batch
        """
        if batch_size <= 0:
            return 0.0
        
        profiled_sizes = sorted(c.PERFORMANCE_PROFILE.keys())
        
        # Use exact match if available
        if batch_size in c.PERFORMANCE_PROFILE:
            return c.PERFORMANCE_PROFILE[batch_size]
        
        # Use next larger profiled size (conservative estimate)
        for profiled_size in profiled_sizes:
            if batch_size <= profiled_size:
                return c.PERFORMANCE_PROFILE[profiled_size]
        
        # Extrapolate linearly for larger batches
        max_profiled_size = max(profiled_sizes)
        max_latency = c.PERFORMANCE_PROFILE[max_profiled_size]
        return max_latency * (batch_size / max_profiled_size)

    def render(self, mode='human'):
        """Display current environment state."""
        state = self._get_state()
        print(
            f"[SIM] Time: {self.current_time:.2f}s | "
            f"Queue: {int(state[0])} | "
            f"Nearest Deadline: {state[1]:.3f}s | "
            f"Oldest Wait: {state[2]:.3f}s | "
            f"Node: {'BUSY' if self.edge_node['busy'] else 'FREE'} | "
            f"Success/Fail: {self.stats['total_tasks_completed']}/{self.stats['total_tasks_failed']}"
        )

    def close(self):
        """Clean up resources."""
        pass
