# Batch-Aware RL

**研究项目** - 基于深度强化学习的边缘计算智能批处理调度系统

本项目研究如何利用强化学习优化边缘计算系统中的任务调度，特别是在 ADAS（高级驾驶辅助系统）场景中平衡实时处理需求和计算效率。

## 概述

系统学习最优的批处理调度策略，在满足任务截止时间的同时最小化系统延迟。智能体学会何时以及如何将任务批处理以在边缘计算节点上处理。

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

## 用法

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

### 4. 评估模型

```bash
# 评估模型
python scripts/evaluate.py --model results/real_gnn/dqn_real_gnn_100000_steps.zip --episodes 20
```