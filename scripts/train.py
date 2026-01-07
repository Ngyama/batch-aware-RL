"""
Training Script for Batch-Aware RL Scheduler
"""

import sys
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np

from src.environment_real import SchedulingEnvReal
from src.gnn_encoder import GraphStateWrapper
import src.constants as c


class TrainingMonitorCallback(BaseCallback):
    """Monitor training progress with additional real environment metrics."""
    
    def __init__(self, check_freq=5, progress_freq=10000, verbose=1):
        super(TrainingMonitorCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.progress_freq = progress_freq
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_stats = []
        self.current_episode_reward = 0
        self.current_episode_length = 0
        self.last_progress_timestep = 0
    
    def _on_step(self):
        self.current_episode_reward += self.locals['rewards'][0]
        self.current_episode_length += 1
        
        current_timestep = self.num_timesteps
        if current_timestep - self.last_progress_timestep >= self.progress_freq:
            total_timesteps = None
            if hasattr(self, 'model') and self.model is not None:
                total_timesteps = getattr(self.model, 'total_timesteps', None)
            
            if total_timesteps:
                progress_pct = (current_timestep / total_timesteps * 100)
                print(f"[PROGRESS] {current_timestep:,}/{total_timesteps:,} ({progress_pct:.1f}%), episodes: {len(self.episode_rewards)}")
            else:
                print(f"[PROGRESS] {current_timestep:,} steps, episodes: {len(self.episode_rewards)}")
            self.last_progress_timestep = current_timestep
        
        if self.locals['dones'][0]:
            info = self.locals['infos'][0]
            
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_lengths.append(self.current_episode_length)
            self.episode_stats.append(info.get('stats', {}))
            
            self.current_episode_reward = 0
            self.current_episode_length = 0
            
            if len(self.episode_rewards) % self.check_freq == 0:
                recent_rewards = self.episode_rewards[-self.check_freq:]
                recent_stats = self.episode_stats[-self.check_freq:]
                total_completed = sum(s.get('total_tasks_completed', 0) for s in recent_stats)
                total_failed = sum(s.get('total_tasks_failed', 0) for s in recent_stats)
                success_rate = total_completed / max(1, total_completed + total_failed) * 100
                
                print(f"\nEpisode {len(self.episode_rewards)}: "
                      f"mean reward={np.mean(recent_rewards):.2f}±{np.std(recent_rewards):.2f}, "
                      f"success rate={success_rate:.1f}%")
        
        return True


def main():
    print(f"Training config: {c.TOTAL_TIMESTEPS:,} steps, device: {c.INFERENCE_DEVICE}")
    
    if c.INFERENCE_DEVICE == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA not available, using CPU")
    
    results_path = "results/real_gnn"
    os.makedirs(results_path, exist_ok=True)
    
    try:
        env = SchedulingEnvReal()
    except Exception as e:
        print(f"[ERROR] Failed to create environment: {e}")
        return
    
    if hasattr(env, 'use_graph_state') and not getattr(env, 'use_graph_state'):
        env.use_graph_state = True
    
    training_device = 'cuda' if c.INFERENCE_DEVICE == 'cuda' and torch.cuda.is_available() else 'cpu'
    gnn_output_dim = getattr(c, 'GNN_OUTPUT_DIM', 64)
    env = GraphStateWrapper(
        env,
        encoder=None,
        output_dim=gnn_output_dim,
        device=training_device
    )
    
    model = DQN(
        'MlpPolicy',
        env,
        learning_rate=c.LEARNING_RATE,
        buffer_size=c.BUFFER_SIZE,
        learning_starts=c.LEARNING_STARTS,
        gamma=c.GAMMA,
        verbose=1,
        tensorboard_log=os.path.join(results_path, "tensorboard"),
        device=training_device
    )
    
    callback = TrainingMonitorCallback(check_freq=5, progress_freq=10000, verbose=1)
    
    try:
        model.learn(
            total_timesteps=c.TOTAL_TIMESTEPS,
            callback=callback,
            progress_bar=False
        )
        actual_timesteps = model.num_timesteps
        if actual_timesteps < c.TOTAL_TIMESTEPS:
            print(f"[WARN] Training stopped early: {actual_timesteps:,}/{c.TOTAL_TIMESTEPS:,} ({100*actual_timesteps/c.TOTAL_TIMESTEPS:.1f}%)")
    except KeyboardInterrupt:
        print("\n[WARN] Training interrupted")
    except Exception as e:
        print(f"\n[ERROR] Training error: {e}")
    
    model_name = f"dqn_real_gnn_{c.TOTAL_TIMESTEPS}_steps.zip"
    model.save(os.path.join(results_path, model_name))
    print(f"[SAVED] Model saved: {os.path.join(results_path, model_name)}")
    
    if hasattr(env, 'encoder'):
        encoder_path = os.path.join(results_path, "gnn_encoder.pt")
        torch.save(env.encoder.state_dict(), encoder_path)
    
    stats_path = os.path.join(results_path, "training_stats.npz")
    np.savez(
        stats_path,
        episode_rewards=callback.episode_rewards,
        episode_lengths=callback.episode_lengths
    )
    
    if len(callback.episode_rewards) > 0:
        total_completed = sum(s.get('total_tasks_completed', 0) for s in callback.episode_stats)
        total_failed = sum(s.get('total_tasks_failed', 0) for s in callback.episode_stats)
        overall_success_rate = total_completed / max(1, total_completed + total_failed) * 100
        
        print(f"\nTraining summary: {len(callback.episode_rewards)} episodes, "
              f"mean reward: {np.mean(callback.episode_rewards):.2f}, "
              f"success rate: {overall_success_rate:.1f}%")
    
    env.close()


if __name__ == "__main__":
    main()
