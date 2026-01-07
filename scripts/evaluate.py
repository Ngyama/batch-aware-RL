"""
Evaluation Script for Batch-Aware RL Scheduler

This script evaluates trained models on the real environment,
providing detailed performance metrics and comparisons with baseline strategies.

Usage:
    # Evaluate real model
    python scripts/evaluate.py --model results/real/dqn_real_100000_steps.zip
    
    # Compare with baselines
    python scripts/evaluate.py --model results/real/dqn_real_100000_steps.zip --compare-baselines
"""

import sys
import os

# Fix OpenMP library conflict
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
from stable_baselines3 import DQN

from src.environment_real import SchedulingEnvReal
from src.gnn_encoder import GraphStateWrapper, HeteroGraphEncoder
import src.constants as c


class BaselinePolicy:
    """
    Baseline scheduling policies for comparison.
    """
    
    @staticmethod
    def greedy_fill(state):
        """
        Greedy Fill: Always dispatch when queue is non-empty and node is free.
        Dispatches the entire queue as one batch.
        """
        queue_length = int(state[0])
        
        if queue_length == 0:
            return c.ACTION_WAIT
        
        # Find the closest batch size option
        for i, batch_size in enumerate(c.BATCH_SIZE_OPTIONS):
            if queue_length <= batch_size:
                return i + 1  # +1 because action 0 is WAIT
        
        # If queue is larger than max batch size, use max
        return len(c.BATCH_SIZE_OPTIONS)
    
    @staticmethod
    def fixed_batch(state, target_batch_size=8):
        """
        Fixed Batch: Wait until queue reaches target batch size, then dispatch.
        """
        queue_length = int(state[0])
        
        if queue_length >= target_batch_size:
            # Find the action corresponding to target batch size
            for i, batch_size in enumerate(c.BATCH_SIZE_OPTIONS):
                if batch_size >= target_batch_size:
                    return i + 1
        
        return c.ACTION_WAIT
    
    @staticmethod
    def deadline_aware(state):
        """
        Deadline Aware: Dispatch when nearest deadline is approaching,
        using appropriate batch size based on urgency.
        """
        queue_length = int(state[0])
        time_to_deadline = state[1]
        
        if queue_length == 0:
            return c.ACTION_WAIT
        
        # Urgent: deadline < 0.1s
        if time_to_deadline < 0.1:
            # Dispatch immediately with small batch
            return 1  # Batch size 1
        
        # Moderate urgency: 0.1s < deadline < 0.5s
        elif time_to_deadline < 0.5:
            # Dispatch with medium batch
            for i, batch_size in enumerate(c.BATCH_SIZE_OPTIONS):
                if queue_length <= batch_size and batch_size <= 8:
                    return i + 1
            return 3  # Default to batch size 4
        
        # Not urgent: wait for more tasks
        else:
            if queue_length >= 16:
                # Good batch size accumulated, dispatch
                return 5  # Batch size 16
            return c.ACTION_WAIT


def evaluate_policy(env, policy, num_episodes=10, max_steps_per_episode=1000, policy_name="Policy"):
    """
    Evaluate a policy (RL agent or baseline) on the environment.
    
    Args:
        env: The environment instance
        policy: Either a trained model or a baseline function
        num_episodes: Number of episodes to evaluate
        max_steps_per_episode: Maximum steps per episode
        policy_name: Name for logging
    
    Returns:
        dict: Evaluation metrics
    """
    episode_rewards = []
    episode_lengths = []
    episode_stats = []
    
    print(f"\n[EVAL] Evaluating {policy_name} ({num_episodes} episodes)...")
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        steps = 0
        
        while not done and steps < max_steps_per_episode:
            if isinstance(policy, DQN):
                action, _ = policy.predict(state, deterministic=False)
            else:
                action = policy(state)
            
            state, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            steps += 1
            done = terminated or truncated
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(steps)
        episode_stats.append(info.get('stats', {}))
        
        if (episode + 1) % 5 == 0 or (episode + 1) == num_episodes:
            stats = info.get('stats', {})
            success_rate = (stats.get('total_tasks_completed', 0) / 
                           max(1, stats.get('total_tasks_completed', 0) + stats.get('total_tasks_failed', 0))) * 100
            print(f"  Episode {episode+1}/{num_episodes}: reward={episode_reward:.2f}, success rate={success_rate:.1f}%")
    
    # Calculate aggregate metrics
    metrics = {
        'policy_name': policy_name,
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'mean_length': np.mean(episode_lengths),
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths
    }
    
    # Calculate task completion metrics
    total_completed = sum(s.get('total_tasks_completed', 0) for s in episode_stats)
    total_failed = sum(s.get('total_tasks_failed', 0) for s in episode_stats)
    total_arrived = sum(s.get('total_tasks_arrived', 0) for s in episode_stats)
    total_latency = sum(s.get('total_latency', 0) for s in episode_stats)
    
    metrics['total_tasks_completed'] = total_completed
    metrics['total_tasks_failed'] = total_failed
    metrics['total_tasks_arrived'] = total_arrived
    metrics['success_rate'] = total_completed / max(1, total_completed + total_failed) * 100
    metrics['avg_latency'] = total_latency / max(1, total_completed)
    
    print(f"\n[SUMMARY] {policy_name}: "
          f"mean reward={metrics['mean_reward']:.2f}±{metrics['std_reward']:.2f}, "
          f"success rate={metrics['success_rate']:.2f}%, "
          f"avg latency={metrics['avg_latency']*1000:.2f}ms")
    
    return metrics


def plot_comparison(all_metrics, save_path=None):
    """
    Plot comparison of different policies.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Batch-Aware RL Scheduler - Policy Comparison', fontsize=16)
    
    policy_names = [m['policy_name'] for m in all_metrics]
    
    # 1. Mean Reward Comparison
    ax = axes[0, 0]
    mean_rewards = [m['mean_reward'] for m in all_metrics]
    std_rewards = [m['std_reward'] for m in all_metrics]
    ax.bar(policy_names, mean_rewards, yerr=std_rewards, capsize=5, alpha=0.7)
    ax.set_ylabel('Mean Episode Reward')
    ax.set_title('Episode Reward Comparison')
    ax.grid(axis='y', alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # 2. Success Rate Comparison
    ax = axes[0, 1]
    success_rates = [m['success_rate'] for m in all_metrics]
    ax.bar(policy_names, success_rates, alpha=0.7, color='green')
    ax.set_ylabel('Success Rate (%)')
    ax.set_title('Task Completion Success Rate')
    ax.set_ylim([0, 105])
    ax.grid(axis='y', alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # 3. Average Latency Comparison
    ax = axes[1, 0]
    avg_latencies = [m['avg_latency'] * 1000 for m in all_metrics]  # Convert to ms
    ax.bar(policy_names, avg_latencies, alpha=0.7, color='orange')
    ax.set_ylabel('Average Latency (ms)')
    ax.set_title('Average Task Latency')
    ax.grid(axis='y', alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # 4. Episode Length Comparison
    ax = axes[1, 1]
    mean_lengths = [m['mean_length'] for m in all_metrics]
    ax.bar(policy_names, mean_lengths, alpha=0.7, color='purple')
    ax.set_ylabel('Mean Episode Length')
    ax.set_title('Episode Duration (steps)')
    ax.grid(axis='y', alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')


def main():
    parser = argparse.ArgumentParser(description='Evaluate Batch-Aware RL Scheduler')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model (.zip file)')
    parser.add_argument('--episodes', type=int, default=10,
                       help='Number of evaluation episodes')
    parser.add_argument('--compare-baselines', action='store_true',
                       help='Compare with baseline policies')
    parser.add_argument('--output-dir', type=str, default='results/evaluation',
                       help='Directory to save evaluation results')
    
    args = parser.parse_args()
    
    print(f"Evaluation config: model={args.model}, episodes={args.episodes}, compare_baselines={args.compare_baselines}\n")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    env = SchedulingEnvReal()
    if hasattr(env, 'use_graph_state') and not getattr(env, 'use_graph_state'):
        env.use_graph_state = True
    
    eval_device = 'cuda' if c.INFERENCE_DEVICE == 'cuda' and torch.cuda.is_available() else 'cpu'
    gnn_output_dim = getattr(c, 'GNN_OUTPUT_DIM', 64)
    gnn_hidden_dim = getattr(c, 'GNN_HIDDEN_DIM', 64)
    gnn_layers = getattr(c, 'GNN_NUM_LAYERS', 2)
    
    encoder = HeteroGraphEncoder(
        task_feature_dim=6,
        edge_feature_dim=6,
        hidden_dim=gnn_hidden_dim,
        output_dim=gnn_output_dim,
        num_layers=gnn_layers
    )
    
    encoder_path = os.path.join(os.path.dirname(args.model), "gnn_encoder.pt")
    if os.path.exists(encoder_path):
        encoder.load_state_dict(torch.load(encoder_path, map_location=eval_device))
    encoder.eval()
    
    env = GraphStateWrapper(
        env,
        encoder=encoder,
        output_dim=gnn_output_dim,
        device=eval_device
    )
    
    try:
        model = DQN.load(args.model, device='auto')
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        return
    
    # Evaluate trained model
    all_metrics = []
    
    rl_metrics = evaluate_policy(
        env, model, 
        num_episodes=args.episodes,
        policy_name="RL Agent (DQN)"
    )
    all_metrics.append(rl_metrics)
    
    if args.compare_baselines:
        print("\n[EVAL] Evaluating baseline policies...")
        baselines = [
            (BaselinePolicy.greedy_fill, "Baseline: Greedy Fill"),
            (lambda s: BaselinePolicy.fixed_batch(s, 8), "Baseline: Fixed Batch-8"),
            (lambda s: BaselinePolicy.fixed_batch(s, 16), "Baseline: Fixed Batch-16"),
            (BaselinePolicy.deadline_aware, "Baseline: Deadline Aware")
        ]
        
        for baseline_fn, baseline_name in baselines:
            baseline_metrics = evaluate_policy(
                env, baseline_fn,
                num_episodes=args.episodes,
                policy_name=baseline_name
            )
            all_metrics.append(baseline_metrics)
    
    print("\n[COMPARISON] Final Results:")
    print(f"{'Policy':<30} {'Mean Reward':<18} {'Success Rate':<15} {'Avg Latency (ms)':<18}")
    print("-"*80)
    for m in all_metrics:
        print(f"{m['policy_name']:<30} "
              f"{m['mean_reward']:>8.2f}±{m['std_reward']:<7.2f} "
              f"{m['success_rate']:>12.2f}% "
              f"{m['avg_latency']*1000:>15.2f}")
    
    metrics_path = os.path.join(args.output_dir, "evaluation_metrics_real.npz")
    np.savez(metrics_path, metrics=all_metrics)
    
    if args.compare_baselines:
        plot_path = os.path.join(args.output_dir, "comparison_real.png")
        plot_comparison(all_metrics, save_path=plot_path)
    
    env.close()
    print("\n[DONE] Evaluation complete")


if __name__ == "__main__":
    main()

