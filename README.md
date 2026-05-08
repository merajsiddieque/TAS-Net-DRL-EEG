# TAS-Net-DRL-EEG

### Unsupervised Time-Aware Sampling Network with Deep Reinforcement Learning for EEG-Based Emotion Recognition

---

## Overview

This project explores advanced Deep Reinforcement Learning (DRL) techniques for improving EEG-based emotion recognition using the TAS-Net framework.

The primary objective is to enhance temporal segment selection from EEG signals by introducing multiple reinforcement learning methodologies and comparing their performance against the original TAS-Net baseline.

The project investigates how intelligent sampling strategies can improve emotion recognition accuracy from noisy and non-stationary EEG brain signals.

---

# Team Details

| Role | Name | ID |
|------|------|------|
| Team Leader | Anshul Reddy | U23AI107 |
| Member | Mihir Hajare | U23AI092 |
| Member | Meraj Alam | U23AI094 |

**Contact (Team Leader):**  
+91 891 905 9046

---

# Project Objectives

- Improve EEG-based emotion recognition using DRL
- Enhance temporal segment selection from EEG signals
- Reduce instability in reinforcement learning training
- Compare multiple RL methodologies against TAS-Net baseline
- Analyze reward-learning and recall performance across methods

---

# Baseline Framework

The project is built upon the original **TAS-Net** architecture which uses:

- REINFORCE Policy Gradient
- Bernoulli Temporal Sampling
- EEG Feature Extraction
- Unsupervised Temporal Segment Detection

### Limitations of Baseline

- High variance training
- Unstable convergence
- Weak temporal continuity
- Poor reward alignment
- Fragmented sampling behavior

---

# Proposed Reinforcement Learning Methodologies

## Method 1 — PPO + Top-K + Multi-Reward
- Stabilized policy optimization
- Structured temporal selection
- Multi-objective reward learning

---

## Method 2 — Actor-Critic + Gumbel + Entropy Regularization
- Reduced variance learning
- Differentiable sampling
- Improved exploration strategy

---

## Method 3 — Hierarchical RL + TD Learning
- Temporal credit assignment
- Hierarchical decision making
- Timestep-based rewards

---

## Method 4 — RL + Contrastive Learning + Transformer
- Transformer-based temporal modeling
- Enhanced feature representation
- Contrastive representation learning

---

## Method 5 — Multi-Agent Reinforcement Learning
- Multi-agent temporal exploration
- Diverse sampling strategies
- Coverage optimization

---

# Features

- EEG-based Emotion Recognition
- Deep Reinforcement Learning Framework
- Multiple RL Strategy Comparisons
- Transformer + Contrastive Learning
- PPO and Actor-Critic Optimization
- Multi-Agent Reinforcement Learning
- Temporal EEG Segment Sampling
- Performance and Recall Analysis

---

# Dataset

This project utilizes EEG emotion recognition datasets such as:

- **SEED Dataset**

Ensure dataset paths are configured correctly before execution.

---

# Environment Setup

## Option 1 — Conda Environment

### Create Environment

```bash
conda create -n eeg_rl python=3.8
conda activate eeg_rl
```

### Install Dependencies

```bash
pip install torch torchvision torchaudio
pip install numpy pandas matplotlib scipy h5py
```

---

## Option 2 — Python Virtual Environment

### Create Environment

```bash
python -m venv eeg_rl_env
```

### Activate Environment

#### Linux / macOS

```bash
source eeg_rl_env/bin/activate
```

#### Windows

```bash
eeg_rl_env\\Scripts\\activate
```

### Install Dependencies

```bash
pip install torch numpy pandas matplotlib scipy h5py
```

---

## Option 3 — Google Colab

### Mount Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')
```

### Install Required Libraries

```python
!pip install h5py scipy
```

---

# Hardware Requirements

| Requirement | Specification |
|---|---|
| GPU | NVIDIA CUDA Recommended |
| RAM | Minimum 8GB |
| Storage | ~5GB Dataset Space |

---

# Project Structure

```text
baselines/
method1/
method2/
method3/
method4/
method5/
features/
TAS-output/
```

---

# Notes

- Ensure correct dataset paths before training
- GPU selection can be configured using the `--gpu` argument
- CUDA-enabled GPUs are highly recommended
- Large EEG datasets may require additional storage and memory

---

# Research Focus

This work aims to analyze how different reinforcement learning strategies affect:

- Training Stability
- Reward Optimization
- Recall Performance
- Temporal Segment Selection
- Feature Representation Quality

---

# Authors

- Anshul Reddy
- Meraj Alam
- Mihir Hajare
