# Batch-Aware Edge Computing with Reinforcement Learning

**This is a research project** investigating intelligent batch scheduling for edge computing systems using deep reinforcement learning. The project focuses on optimizing task scheduling in ADAS (Advanced Driver Assistance Systems) scenarios where real-time processing requirements must be balanced with computational efficiency.

## Overview

This research explores how reinforcement learning can learn optimal batch scheduling strategies to minimize system latency while meeting task deadlines. The system learns when and how to batch tasks for processing on edge computing nodes.

### Key Features

- **Dual Environment Support**: Simulation environment for fast iteration and real data environment for realistic evaluation
- **Intelligent Batch Selection**: RL agent learns to choose optimal batch sizes (1, 2, 4, 8, 16, 32, 64)
- **Real-time Constraints**: Handles task deadlines and queue management
- **Comprehensive Evaluation**: Includes baseline strategy comparisons

## Project Structure

```
batch-aware/
├── src/
│   ├── environment_sim.py      # Simulation environment
│   ├── environment_real.py     # Real data environment  
│   ├── constants.py            # Configuration parameters
│   └── __init__.py
├── scripts/
│   ├── train_sim.py            # Simulation training
│   ├── train_real.py           # Real data training
│   ├── evaluate.py             # Model evaluation
│   └── utils/                   # Utility scripts
├── data/
│   └── imagenette2/            # Dataset directory
└── results/                    # Training results
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download Dataset

```bash
cd scripts
python utils/download_dataset.py
cd ..
```

### 3. Training Options

**Option A: Simulation Training (Recommended for initial experiments)**
- Fast training (~5-10 minutes for 100K steps)
- Good for debugging and hyperparameter tuning

```bash
python scripts/train_sim.py
```

**Option B: Real Data Training**
- Slower training (~1-2 hours for 100K steps)
- Uses actual image processing with GPU inference
- More realistic performance evaluation

```bash
python scripts/train_real.py
```

### 4. Evaluate Models

```bash
# Evaluate simulation model
python scripts/evaluate.py --env sim --model results/simulation/dqn_sim_100000_steps.zip --episodes 20

# Evaluate real model  
python scripts/evaluate.py --env real --model results/real/dqn_real_100000_steps.zip --episodes 20

# Compare with baselines
python scripts/evaluate.py --env sim --model results/simulation/dqn_sim_100000_steps.zip --episodes 20 --compare-baselines
```

## Configuration

Key parameters can be modified in `src/constants.py`:

- `BATCH_SIZE_OPTIONS`: Available batch sizes for scheduling
- `TASK_ARRIVAL_INTERVAL_SECONDS`: Task arrival rate
- `TASK_DEADLINE_SECONDS`: Task deadline constraints
- `TOTAL_TIMESTEPS`: Training duration
- `LEARNING_RATE`: RL agent learning rate
- `REWARD_TASK_SUCCESS`: Reward for successful task completion
- `PENALTY_TASK_MISS`: Penalty for missing deadlines

## Research Directions

### Hyperparameter Optimization
- Experiment with different reward function weights
- Test various task arrival patterns and deadlines
- Compare different RL algorithms (DQN, PPO, A2C)

### Environment Enhancements
- Add more sophisticated state features
- Implement dynamic deadline assignment
- Test with different datasets and models

### Performance Analysis
- Compare simulation vs real environment results
- Analyze batch size selection patterns
- Evaluate robustness under different load conditions

## Results

Training generates:
- Model checkpoints in `results/` directory
- TensorBoard logs for visualization
- Training statistics and evaluation metrics

View training progress:
```bash
tensorboard --logdir results/simulation/tensorboard
# or
tensorboard --logdir results/real/tensorboard
```

## Research Context

This project investigates the application of reinforcement learning to edge computing scheduling problems, specifically focusing on:

1. **Batch Processing Optimization**: Learning optimal batch sizes for different workload conditions
2. **Real-time Constraint Handling**: Managing task deadlines while maximizing throughput
3. **Edge Computing Efficiency**: Balancing latency and computational resource utilization

The research contributes to understanding how RL can improve resource management in time-critical edge computing applications.

## Requirements

- Python 3.8+
- PyTorch (CPU or GPU)
- Stable-Baselines3
- Gymnasium
- TensorBoard for visualization

See `requirements.txt` for complete dependency list.

## License

This is a research project. Please cite appropriately if used in academic work.
