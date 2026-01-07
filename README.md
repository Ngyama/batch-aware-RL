# Batch-Aware RL

**研究项目** - 基于深度强化学习的边缘计算智能批处理调度系统

本项目研究如何利用强化学习优化边缘计算系统中的任务调度，特别是在 ADAS（高级驾驶辅助系统）场景中平衡实时处理需求和计算效率。

## 概述

系统学习最优的批处理调度策略，在满足任务截止时间的同时最小化系统延迟。智能体学会何时以及如何将任务批处理以在边缘计算节点上处理。

### 核心特性

- **真实数据环境**: 使用 ResNet-18 推理和 Imagenette 数据集进行真实评估
- **智能批次选择**: RL 智能体学习选择最优批次大小（1, 2, 4, 8, 16, 32, 64）
- **实时约束**: 处理任务截止时间和队列管理
- **综合评估**: 包含基线策略对比

## 项目结构

```
batch-aware/
├── src/                    # 核心代码
│   ├── environment_real.py    # 真实数据环境
│   ├── graph_builder.py       # 异构图构建
│   ├── gnn_encoder.py         # GNN 编码器
│   ├── node_selector.py       # 节点选择器
│   └── constants.py           # 配置参数
├── scripts/                # 脚本
│   ├── train.py              # 训练脚本
│   ├── evaluate.py           # 评估脚本
│   └── utils/                # 工具脚本
├── docs/                  # 文档
├── tests/                 # 测试
├── data/                  # 数据集
└── results/               # 结果（模型、日志等）
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 下载数据集

```bash
cd scripts
python utils/download_dataset.py
cd ..
```

### 3. 训练模型

```bash
python scripts/train.py
```

训练使用真实的 ResNet-18 推理和 GPU 加速：
- 训练时间：100K 步约需 1-2 小时（取决于 GPU）
- 使用真实图像处理进行性能评估

### 4. 评估模型

```bash
# 评估模型
python scripts/evaluate.py --model results/real_gnn/dqn_real_gnn_100000_steps.zip --episodes 20

# 与基线策略对比
python scripts/evaluate.py --model results/real_gnn/dqn_real_gnn_100000_steps.zip --episodes 20 --compare-baselines
```

## 配置

主要参数在 `src/constants.py` 中配置：

- `TASK_TYPES`: 传感器任务类型定义（截止时间、到达间隔）
- `BATCH_SIZE_OPTIONS`: 可选择的批次大小
- `NUM_EDGE_NODES`: 边缘节点数量
- `USE_GRAPH_STATE`: 是否使用图状态表示
- `TOTAL_TIMESTEPS`: 训练总步数
- `LEARNING_RATE`, `GAMMA`: RL 超参数
- `REWARD_TASK_SUCCESS`, `PENALTY_TASK_MISS`: 奖励函数参数

## 结果

训练生成：
- 模型检查点保存在 `results/` 目录
- TensorBoard 日志用于可视化
- 训练统计和评估指标

查看训练进度：
```bash
tensorboard --logdir results/real_gnn/tensorboard
```

## 文档

详细设计文档位于 `docs/` 目录：
- `GNN_USAGE.md`: GNN 使用指南
- `GRAPH_STATE_DESIGN.md`: 图状态设计
- `SYSTEM_FLOW_CN.md`: 系统流程说明（中文）

## 系统要求

- Python 3.8+
- PyTorch（支持 CPU 或 GPU）
- Stable-Baselines3
- Gymnasium
- TensorBoard（用于可视化）

完整依赖列表见 `requirements.txt`。

## 研究内容

本项目研究强化学习在边缘计算调度问题中的应用，重点关注：

1. **批处理优化**: 学习不同负载条件下的最优批次大小
2. **实时约束处理**: 在最大化吞吐量的同时管理任务截止时间
3. **边缘计算效率**: 平衡延迟和计算资源利用率

## 许可证

本项目为研究用途。如在学术工作中使用，请适当引用。