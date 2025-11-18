# Batch-Aware Edge Computing with Reinforcement Learning

**This is a research project** investigating intelligent batch scheduling for edge computing systems using deep reinforcement learning. The project focuses on optimizing task scheduling in ADAS (Advanced Driver Assistance Systems) scenarios where real-time processing requirements must be balanced with computational efficiency.

## Overview

This research explores how reinforcement learning can learn optimal batch scheduling strategies to minimize system latency while meeting task deadlines. The system learns when and how to batch tasks for processing on edge computing nodes.

### Key Features

- **Real Data Environment**: Uses actual ResNet-18 inference with Imagenette dataset for realistic evaluation
- **Intelligent Batch Selection**: RL agent learns to choose optimal batch sizes (1, 2, 4, 8, 16, 32, 64)
- **Real-time Constraints**: Handles task deadlines and queue management
- **Comprehensive Evaluation**: Includes baseline strategy comparisons

## Project Structure

```
batch-aware/
├── src/
│   ├── environment_real.py     # Real-data Gymnasium env
│   ├── graph_builder.py        # Heterogeneous graph construction
│   ├── gnn_encoder.py          # GNN encoder + wrapper
│   ├── node_selector.py        # State-aware node scoring
│   └── constants.py            # Centralized configuration
├── scripts/
│   ├── train.py                # Training pipeline (vector or graph state)
│   ├── evaluate.py             # Evaluation + metrics export
│   └── utils/                  # Dataset download & profiling tools
├── docs/                       # Design notes (GNN usage, graph spec, flow)
├── tests/                      # Smoke tests (`test_environments.py`)
├── data/imagenette2/           # Imagenette dataset (downloaded locally)
└── results/                    # Checkpoints, metrics, TensorBoard logs
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

### 3. Train Model

```bash
python scripts/train.py
```

Training uses actual ResNet-18 inference with GPU acceleration:
- Training time: ~1-2 hours for 100K steps (depends on GPU)
- Uses real image processing for realistic performance evaluation

### 4. Evaluate Models

```bash
# Evaluate model
python scripts/evaluate.py --model results/real/dqn_real_100000_steps.zip --episodes 20

# Compare with baselines
python scripts/evaluate.py --model results/real/dqn_real_100000_steps.zip --episodes 20 --compare-baselines
```

## Configuration

Key parameters live in `src/constants.py`:

- `TASK_TYPES`: Sensor-specific deadline + arrival interval definitions
- `USE_SINGLE_TASK_TYPE`: Force single-modality workloads for ablation
- `BATCH_SIZE_OPTIONS`: Discrete batch sizes the agent can dispatch
- `NUM_EDGE_NODES`: Number of edge devices simulated
- `USE_GRAPH_STATE` / `GNN_OUTPUT_DIM`: Enable GNN encoder and pick output size
- `TOTAL_TIMESTEPS`, `LEARNING_RATE`, `BUFFER_SIZE`, `GAMMA`: RL hyperparameters
- `REWARD_TASK_SUCCESS`, `PENALTY_TASK_MISS`, `LATENCY_PENALTY_COEFF`: reward shaping

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
- Analyze batch size selection patterns
- Evaluate robustness under different load conditions
- Compare with baseline scheduling strategies

## Results

Training generates:
- Model checkpoints in `results/` directory
- TensorBoard logs for visualization
- Training statistics and evaluation metrics

View training progress:
```bash
tensorboard --logdir results/real/tensorboard
```

## Documentation

The most up-to-date design notes live under `docs/`:

- `GNN_USAGE.md`: how to enable graph state + training tips
- `GRAPH_STATE_DESIGN.md`: node/edge schemas and feature scaling
- `SYSTEM_FLOW_CN.md`: 中文版数据流向说明（含节点打分链路）

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
