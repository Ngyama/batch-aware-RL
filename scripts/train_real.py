"""
Training Script for Real Data Environment

Trains RL agent using real environment with actual image processing.
Training is slower but provides realistic performance evaluation.

Usage:
    python scripts/train_real.py
"""

import os
import torch
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np

from src.environment_real import SchedulingEnvReal
import src.constants as c

torch.set_default_device('cpu')
os.environ['CUDA_VISIBLE_DEVICES'] = ''


class TrainingMonitorCallback(BaseCallback):
    """Monitor training progress with additional real environment metrics."""
    
    def __init__(self, check_freq=5, verbose=1):
        super(TrainingMonitorCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_stats = []
        self.current_episode_reward = 0
        self.current_episode_length = 0
    
    def _on_step(self):
        self.current_episode_reward += self.locals['rewards'][0]
        self.current_episode_length += 1
        
        if self.locals['dones'][0]:
            info = self.locals['infos'][0]
            
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_lengths.append(self.current_episode_length)
            self.episode_stats.append(info.get('stats', {}))
            
            self.current_episode_reward = 0
            self.current_episode_length = 0
            
            # Print statistics periodically
            if len(self.episode_rewards) % self.check_freq == 0:
                recent_rewards = self.episode_rewards[-self.check_freq:]
                recent_lengths = self.episode_lengths[-self.check_freq:]
                
                # Calculate success rate
                recent_stats = self.episode_stats[-self.check_freq:]
                total_completed = sum(s.get('total_tasks_completed', 0) for s in recent_stats)
                total_failed = sum(s.get('total_tasks_failed', 0) for s in recent_stats)
                success_rate = total_completed / max(1, total_completed + total_failed) * 100
                
                # Calculate average inference time
                total_inference = sum(s.get('total_inference_time', 0) for s in recent_stats)
                avg_inference_time = total_inference / max(1, total_completed + total_failed)
                
                print(f"\n--- Episode {len(self.episode_rewards)} ---")
                print(f"  Mean Reward: {np.mean(recent_rewards):.2f} ± {np.std(recent_rewards):.2f}")
                print(f"  Mean Length: {np.mean(recent_lengths):.2f}")
                print(f"  Success Rate: {success_rate:.1f}%")
                print(f"  Avg Inference Time: {avg_inference_time*1000:.2f}ms")
        
        return True


def main():
    print("="*70)
    print("🚀 BATCH-AWARE RL SCHEDULER - REAL DATA TRAINING")
    print("="*70)
    print("\n⚠️  Real data training is computationally intensive!")
    print("   - Requires GPU for reasonable speed")
    print("   - Training will be much slower than simulation")
    print(f"\nTotal Timesteps: {c.TOTAL_TIMESTEPS:,}")
    print(f"Inference Device: {c.INFERENCE_DEVICE}")
    print("="*70 + "\n")
    
    # Check CUDA availability
    if c.INFERENCE_DEVICE == "cuda" and not torch.cuda.is_available():
        print("⚠️  WARNING: CUDA requested but not available!")
        response = input("   Continue with CPU? (y/n): ")
        if response.lower() != 'y':
            print("Exiting...")
            return
    
    # Create results directory
    results_path = "results/real"
    os.makedirs(results_path, exist_ok=True)
    
    # Create environment
    print("🔧 Creating real data environment...")
    try:
        env = SchedulingEnvReal()
    except Exception as e:
        print(f"\n❌ Error creating environment: {e}")
        print("\nPossible issues:")
        print("  - Imagenette dataset not found")
        print("  - CUDA/PyTorch installation issues")
        return
    
    print("\n✅ Environment created\n")
    
    # Create DQN agent
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
    print("🎯 STARTING TRAINING (This will take a while...)")
    print("="*70 + "\n")
    
    callback = TrainingMonitorCallback(check_freq=5, verbose=1)
    
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
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
    
    # Save model
    model_save_path = os.path.join(results_path, f"dqn_real_{c.TOTAL_TIMESTEPS}_steps.zip")
    model.save(model_save_path)
    print(f"💾 Model saved to: {model_save_path}")
    
    # Save statistics
    stats_path = os.path.join(results_path, "training_stats.npz")
    np.savez(
        stats_path,
        episode_rewards=callback.episode_rewards,
        episode_lengths=callback.episode_lengths
    )
    print(f"📊 Statistics saved to: {stats_path}")
    
    # Print summary
    if len(callback.episode_rewards) > 0:
        print("\n" + "="*70)
        print("📈 TRAINING SUMMARY")
        print("="*70)
        print(f"  Total Episodes: {len(callback.episode_rewards)}")
        print(f"  Mean Reward: {np.mean(callback.episode_rewards):.2f}")
        print(f"  Best Reward: {np.max(callback.episode_rewards):.2f}")
        
        total_completed = sum(s.get('total_tasks_completed', 0) for s in callback.episode_stats)
        total_failed = sum(s.get('total_tasks_failed', 0) for s in callback.episode_stats)
        overall_success_rate = total_completed / max(1, total_completed + total_failed) * 100
        
        print(f"  Overall Success Rate: {overall_success_rate:.1f}%")
        print(f"  Total Tasks Completed: {total_completed}")
        print("="*70 + "\n")
    
    env.close()
    print("🏁 Training complete!")


if __name__ == "__main__":
    main()
