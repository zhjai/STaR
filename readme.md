<div align="center">

<img src="STaR.png" alt="STaR Logo" width="420"/>

# ⭐ STaR: Slow-Thinking for Table Reasoning

[![arXiv](https://img.shields.io/badge/arXiv-2511.11233-b31b1b.svg?style=flat-square&logo=arxiv)](https://arxiv.org/abs/2511.11233)
[![Hugging Face Datasets](https://img.shields.io/badge/🤗%20Datasets-STaR--Datasets-ff8c00?style=flat-square)](https://huggingface.co/datasets/zhjai/STaR-Datasets)
[![License](https://img.shields.io/badge/License-MIT-green.svg?style=flat-square)](LICENSE)

**A Cognitive Framework for Table Reasoning with LLMs**

[📄 Paper](https://arxiv.org/abs/2511.11233) • [🤗 Datasets](https://huggingface.co/datasets/zhjai/STaR-Datasets) • [🏠 GitHub](https://github.com/zhjai/STaR)

</div>

---

## 📌 Overview

<div align="center">
<img src="star-framework.png" alt="STaR Framework" width="90%"/>
</div>

**STaR** (Slow-Thinking for Table Reasoning) is a novel framework that equips LLMs with slow-thinking capabilities for cognitive table reasoning by explicitly modeling step-by-step thinking and uncertainty-aware inference.

### ✨ Key Features

- 🧠 **Cognitive Reasoning**: Mimics human-like iterative and reflective thought processes
- 📈 **Two-Stage DRL**: Difficulty-aware reinforcement learning from simple to complex queries
- 🎯 **Uncertainty Quantification**: Trajectory-level confidence for reliable reasoning paths
- 🚀 **Strong Generalization**: Excellent out-of-domain performance

---

## 📋 Abstract

Table reasoning with large language models (LLMs) is a fundamental path toward building intelligent systems that can understand and analyze structured data. While recent progress has shown promising results, they still suffer from two key limitations: (i) the reasoning processes lack the depth and iterative refinement characteristic of human cognition; and (ii) the reasoning processes exhibit instability, which compromises their reliability in downstream applications. 

In this work, we present **STaR**, a new framework achieving cognitive table reasoning, in which LLMs are equipped with slow-thinking capabilities by explicitly modeling step-by-step thinking and uncertainty-aware inference. During training, STaR employs **two-stage difficulty-aware reinforcement learning (DRL)**, progressively learning from simple to complex queries under a composite reward. During inference, STaR performs **trajectory-level uncertainty quantification** by integrating token-level confidence and answer consistency, enabling selection of more credible reasoning paths.

---

## 📁 Project Structure

```
STaR/
├── 📂 data/                    # Datasets
│   ├── STaR-sft.parquet        # SFT training data
│   ├── STaR-train-easy.parquet # Easy training samples
│   ├── STaR-train-hard.parquet # Hard training samples
│   ├── STaR-train-all.parquet  # All training samples
│   ├── STaR-eval.parquet       # Evaluation data
│   └── STaR-test.parquet       # Test data
├── 📂 model/                   # Pre-trained models
├── 📂 sh/                      # Training & evaluation scripts
├── 📂 verl/                    # VERL framework
├── 📂 checkpoints/             # Model checkpoints
├── 📄 reward.py                # Reward function
├── 📄 eval-by-trajectory.py    # Evaluation script
└── 📄 requirements.txt         # Dependencies
```

---

## 🛠️ Installation

> **Requirements**: Python 3.10+ and CUDA GPUs

```bash
# 1️⃣ Clone the repository
git clone https://github.com/zhjai/STaR.git
cd STaR

# 2️⃣ Install Python dependencies
pip install -r requirements.txt

# 3️⃣ Install verl in editable mode
cd verl
pip install -e .
cd ..
```

---

## 📦 Data & Models

### 🤗 Datasets

Download the datasets from Hugging Face and place them in the `data/` folder:

| Dataset | Description | Link |
|---------|-------------|------|
| STaR-Datasets | Full training & evaluation data | [![Hugging Face](https://img.shields.io/badge/🤗-Datasets-ff8c00)](https://huggingface.co/datasets/zhjai/STaR-Datasets) |

### 🤖 Base Models

Download the base models and place them in the `model/` folder:

| Model | Parameters | Link |
|-------|------------|------|
| Qwen3-0.6B | 0.6B | [![Hugging Face](https://img.shields.io/badge/🤗-Qwen3--0.6B-ff8c00)](https://huggingface.co/Qwen/Qwen3-0.6B) |
| Qwen3-8B | 8B | [![Hugging Face](https://img.shields.io/badge/🤗-Qwen3--8B-ff8c00)](https://huggingface.co/Qwen/Qwen3-8B) |

### 🏆 Trained Checkpoints

Our trained model weights are available on Hugging Face:

| Model | Parameters | Link |
|-------|------------|------|
| STaR-0.6B | 0.6B | [![Hugging Face](https://img.shields.io/badge/🤗-STaR--0.6B-ff8c00)](https://huggingface.co/zhjai/STaR-0.6B) |
| STaR-8B | 8B | [![Hugging Face](https://img.shields.io/badge/🤗-STaR--8B-ff8c00)](https://huggingface.co/zhjai/STaR-8B) |

---

## 🚀 Training

Training scripts are located in the `sh/` directory. Adjust paths and hyperparameters as needed.

### 📚 Stage 0: Supervised Fine-Tuning (SFT)

```bash
# Qwen3-0.6B
bash sh/STaR-sft-qwen3-0.6b.sh

# Qwen3-8B
bash sh/STaR-sft-qwen3-8b.sh
```

### 🎯 Stage 1: GRPO (Easy Samples)

```bash
# Qwen3-0.6B
bash sh/STaR-sft-stage1-qwen3-0.6b.sh

# Qwen3-8B
bash sh/STaR-sft-stage1-qwen3-8b.sh
```

### 🔥 Stage 2: GRPO (Hard Samples)

```bash
# Qwen3-0.6B
bash sh/STaR-sft-stage1-stage2-qwen3-0.6b.sh

# Qwen3-8B
bash sh/STaR-sft-stage1-stage2-qwen3-8b.sh
```

---

## 📊 Evaluation

### 1️⃣ Generate Trajectories

```bash
bash sh/STaR-eval.sh
```

### 2️⃣ Compute Metrics

```bash
python eval-by-trajectory.py
```

---

## 📖 Citation

If you find this work useful, please cite our paper:

```bibtex
@article{zhang2025star,
  title={STaR: Towards Cognitive Table Reasoning via Slow-Thinking Large Language Models},
  author={Zhang, Huajian and Cheng, Mingyue and Luo, Yucong and Tao, Xiaoyu},
  journal={arXiv preprint arXiv:2511.11233},
  year={2025}
}
```

---

## 🙏 Acknowledgements

- This work builds on the excellent [**VERL**](https://github.com/volcengine/verl) framework
- Base models from [**Qwen**](https://github.com/QwenLM/Qwen) team at Alibaba
- Thanks to the open-source community for tools and datasets

---

<div align="center">

**⭐ Star this repo if you find it helpful! ⭐**

Made with ❤️ for the research community

</div>
