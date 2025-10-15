"""
Training Script for Simulation Environment

Fast training using pure simulation environment.

Usage:
    python scripts/train_sim.py
"""

import os
import torch
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np

from src.environment_sim import SchedulingEnvSim
import src.constants as c

torch.set_default_device('cpu')
os.environ['CUDA_VISIBLE_DEVICES'] = ''


class TrainingMonitorCallback(BaseCallback):
    """Monitor training progress."""
    
    def __init__(self, check_freq=10, verbose=1):
        super(TrainingMonitorCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.episode_rewards = []
        self.episode_lengths = []
        self.current_episode_reward = 0
        self.current_episode_length = 0
    
    def _on_step(self):
        self.current_episode_reward += self.locals['rewards'][0]
        self.current_episode_length += 1
        
        if self.locals['dones'][0]:
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_lengths.append(self.current_episode_length)
            self.current_episode_reward = 0
            self.current_episode_length = 0
            
            if len(self.episode_rewards) % self.check_freq == 0:
                recent_rewards = self.episode_rewards[-self.check_freq:]
                recent_lengths = self.episode_lengths[-self.check_freq:]
                print(f"\n--- Episode {len(self.episode_rewards)} ---")
                print(f"  Mean Reward (last {self.check_freq}): {np.mean(recent_rewards):.2f}")
                print(f"  Mean Length (last {self.check_freq}): {np.mean(recent_lengths):.2f}")
                print(f"  Reward Std: {np.std(recent_rewards):.2f}")
        
        return True


def main():
    print("="*70)
    print("🚀 BATCH-AWARE RL SCHEDULER - SIMULATION TRAINING")
    print("="*70)
    print(f"\nTotal Timesteps: {c.TOTAL_TIMESTEPS:,}")
    print(f"Learning Rate: {c.LEARNING_RATE}")
    print(f"Batch Sizes: {c.BATCH_SIZE_OPTIONS}")
    print("="*70 + "\n")
    
    results_path = "results/simulation"
    os.makedirs(results_path, exist_ok=True)
    
    env = SchedulingEnvSim()
    print("✅ Environment created\n")
    
    model = DQN(
        'MlpPolicy',
        env,
        learning_rate=c.LEARNING_RATE,
        buffer_size=c.BUFFER_SIZE,
        learning_starts=c.LEARNING_STARTS,
        gamma=c.GAMMA,
        verbose=1,
        tensorboard_log=os.path.join(results_path, "tensorboard"),
        device='cpu'
    )
    print("✅ DQN agent created\n")
    
    print("="*70)
    print("🎯 STARTING TRAINING")
    print("="*70 + "\n")
    
    callback = TrainingMonitorCallback(check_freq=10, verbose=1)
    
    try:
        model.learn(
            total_timesteps=c.TOTAL_TIMESTEPS,
            callback=callback,
            progress_bar=False
        )
        print("\n" + "="*70)
        print("✅ TRAINING COMPLETED")
        print("="*70 + "\n")
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted\n")
    
    model_save_path = os.path.join(results_path, f"dqn_sim_{c.TOTAL_TIMESTEPS}_steps.zip")
    model.save(model_save_path)
    print(f"💾 Model saved to: {model_save_path}")
    
    stats_path = os.path.join(results_path, "training_stats.npz")
    np.savez(
        stats_path,
        episode_rewards=callback.episode_rewards,
        episode_lengths=callback.episode_lengths
    )
    print(f"📊 Statistics saved to: {stats_path}")
    
    if len(callback.episode_rewards) > 0:
        print("\n" + "="*70)
        print("📈 TRAINING SUMMARY")
        print("="*70)
        print(f"  Total Episodes: {len(callback.episode_rewards)}")
        print(f"  Mean Reward: {np.mean(callback.episode_rewards):.2f}")
        print(f"  Best Reward: {np.max(callback.episode_rewards):.2f}")
        print("="*70 + "\n")
    
    env.close()
    print("🏁 Training complete!")


if __name__ == "__main__":
    main()
