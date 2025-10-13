import gymnasium as gym
from gymnasium import spaces
import numpy as np
import collections
import random

# Import all our constants from the control panel
import src.constants as c

class SchedulingEnv(gym.Env):
    
    """
    A custom Gymnasium environment for the Batch-Aware RL Scheduler.

    This environment simulates the arrival of tasks into a queue and the
    processing of these tasks by a single Edge Node. The RL agent's goal
    is to learn a policy for when to dispatch batches of tasks to maximize
    the number of tasks completed before their deadlines.
    """

    def __init__(self):
        super(SchedulingEnv, self).__init__()

        # ================================================================
        # 1. DEFINE ACTION & OBSERVATION SPACES (The Agent's Interface)
        # ================================================================
        # Action space: {0: WAIT, 1: DISPATCH}
        self.action_space = spaces.Discrete(2)

        # Observation space (State): A vector of 4 continuous values
        # [queue_length, time_since_oldest, time_to_deadline, node_status]
        # We define the theoretical min/max values for each feature.
        low = np.array([0, 0, 0, 0], dtype=np.float32)
        high = np.array([np.inf, np.inf, np.inf, 1], dtype=np.float32)
        self.observation_space = spaces.Box(low, high, dtype=np.float32)
        
        # ================================================================
        # 2. INITIALIZE THE SIMULATION WORLD
        # ================================================================
        self.current_time = 0.0  # Simulation clock
        self.task_queue = collections.deque() # Use a deque for efficient appends and pops
        
        # The Edge Node is represented as a dictionary holding its state
        self.edge_node = {'busy': False, 'free_at_time': 0.0}
        
        # Schedule the very first task arrival
        self._schedule_next_task_arrival()

    def reset(self, seed=None, options=None):
        """
        Resets the environment to its initial state for a new episode.
        """
        super().reset(seed=seed)
        
        self.current_time = 0.0
        self.task_queue.clear()
        self.edge_node = {'busy': False, 'free_at_time': 0.0}
        self._schedule_next_task_arrival()
        
        # Return the initial state and an empty info dict
        initial_state = self._get_state()
        return initial_state, {}

    def step(self, action):
        """
        Executes one time step within the environment.
        This is the core of the simulation.
        """
        reward = 0.0
        
        # --- 1. APPLY AGENT'S ACTION ---
        if action == c.ACTION_DISPATCH:
            if not self.task_queue:
                # Penalty for trying to dispatch an empty queue
                reward -= 0.5
            elif self.edge_node['busy']:
                # Penalty for trying to dispatch to a busy node
                reward -= 1.0
            else:
                # This is a valid dispatch action
                batch_size = len(self.task_queue)
                processing_time = self._get_processing_time(batch_size)
                
                # Set the node to busy
                self.edge_node['busy'] = True
                self.edge_node['free_at_time'] = self.current_time + processing_time
                
                # Check which tasks in the batch will meet their deadline
                dispatched_tasks = list(self.task_queue)
                self.task_queue.clear() # Empty the queue
                
                for task in dispatched_tasks:
                    if self.edge_node['free_at_time'] <= task['deadline']:
                        reward += 1.0 # Reward for each successful task
                    else:
                        reward -= 1.0 # Penalty for dispatching a task that will fail
        
        elif action == c.ACTION_WAIT:
            # A small cost for waiting, to encourage action
            reward -= 0.01

        # --- 2. ADVANCE SIMULATION TIME ---
        self.current_time += c.SIM_STEP_SECONDS

        # --- 3. UPDATE WORLD STATE BASED ON TIME PASSING ---
        # A. Check if the edge node has finished its work
        if self.edge_node['busy'] and self.current_time >= self.edge_node['free_at_time']:
            self.edge_node['busy'] = False

        # B. Check for new task arrivals
        if self.current_time >= self.next_task_arrival_time:
            new_task = {
                'arrival_time': self.current_time,
                'deadline': self.current_time + c.TASK_DEADLINE_SECONDS
            }
            self.task_queue.append(new_task)
            self._schedule_next_task_arrival()

        # C. Check for tasks that failed due to waiting too long
        tasks_to_remove = []
        for task in self.task_queue:
            if self.current_time > task['deadline']:
                reward -= 10.0  # Heavy penalty for letting a task expire in the queue
                tasks_to_remove.append(task)
        
        # Remove expired tasks from the queue
        for task in tasks_to_remove:
            self.task_queue.remove(task)

        # --- 4. PREPARE RETURN VALUES ---
        next_state = self._get_state()
        
        # The episode terminates if a task expires in the queue
        terminated = len(tasks_to_remove) > 0
        truncated = False # We don't have a fixed episode length
        info = {}

        return next_state, reward, terminated, truncated, info

    def _get_state(self):
        """
        "Measures" the current state of the world and returns it as a vector.
        This is the "instrument panel" for our RL agent.
        """
        queue_length = len(self.task_queue)
        
        if queue_length == 0:
            time_since_oldest = 0.0
            time_to_nearest_deadline = c.TASK_DEADLINE_SECONDS
        else:
            oldest_task = self.task_queue[0]
            time_since_oldest = self.current_time - oldest_task['arrival_time']
            
            min_deadline = min(t['deadline'] for t in self.task_queue)
            time_to_nearest_deadline = max(0, min_deadline - self.current_time)
            
        edge_node_status = 1.0 if self.edge_node['busy'] else 0.0
        
        return np.array([
            queue_length,
            time_since_oldest,
            time_to_nearest_deadline,
            edge_node_status
        ], dtype=np.float32)

    def _schedule_next_task_arrival(self):
        """
        Schedules the arrival time of the next task using an exponential distribution
        to simulate random arrivals (Poisson process).
        """
        # The rate (lambda) of the Poisson process is 1 / average_interval
        arrival_delay = random.expovariate(1.0 / c.TASK_ARRIVAL_INTERVAL_SECONDS)
        self.next_task_arrival_time = self.current_time + arrival_delay

    def _get_processing_time(self, batch_size):
        """
        Looks up the processing time from our performance profile.
        Handles cases where the exact batch size is not in the profile.
        """
        if batch_size <= 0:
            return 0.0
        
        # Find the smallest key in the profile that is >= our batch_size
        # This is a conservative estimate: we assume the latency for a batch of 5
        # is the same as for a batch of 8 if only {4: ..., 8: ...} is profiled.
        for profiled_size in sorted(c.PERFORMANCE_PROFILE.keys()):
            if batch_size <= profiled_size:
                return c.PERFORMANCE_PROFILE[profiled_size]
        
        # If batch_size is larger than any profiled size, extrapolate linearly (simple approach)
        max_size = max(c.PERFORMANCE_PROFILE.keys())
        return c.PERFORMANCE_PROFILE[max_size] * (batch_size / max_size)

    def render(self, mode='human'):
        """
        (Optional) Renders the environment state for human observation.
        """
        state = self._get_state()
        print(
            f"Time: {self.current_time:.2f}s | "
            f"Queue: {int(state[0])} | "
            f"Oldest Wait: {state[1]:.2f}s | "
            f"Nearest Deadline: {state[2]:.2f}s | "
            f"Node Busy: {bool(state[3])}"
        )