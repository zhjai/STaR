[English](readme.md) | [中文](readme_cn.md)

<div align="center">

<img src="STaR.png" alt="STaR Logo" width="420"/>

# ⭐ STaR: Towards Effective and Stable Table Reasoning via Slow-Thinking Large Language Models

[![arXiv](https://img.shields.io/badge/arXiv-2511.11233-b31b1b.svg?style=flat-square&logo=arxiv)](https://arxiv.org/abs/2511.11233)
[![Hugging Face Datasets](https://img.shields.io/badge/🤗%20Datasets-STaR--Datasets-yellow?style=flat-square)](https://huggingface.co/datasets/zhjai/STaR-Datasets)
[![License](https://img.shields.io/badge/License-MIT-green.svg?style=flat-square)](LICENSE)

**A Novel Slow-Thinking Model for Effective and Stable Table Reasoning**

[📄 Paper](https://arxiv.org/abs/2511.11233) • [🤗 Datasets](https://huggingface.co/datasets/zhjai/STaR-Datasets) • [🏠 GitHub](https://github.com/zhjai/STaR)

</div>

---

## 📌 Overview

<div align="center">
<img src="star-framework.png" alt="STaR Framework" width="90%"/>
</div>

**STaR** (Slow-Thinking Table Reasoning) is a novel slow-thinking model that can achieve effective and stable table reasoning. It enables effective multi-step reasoning through a two-stage training framework (SFT + RFT) and improves reasoning stability via trajectory-level uncertainty quantification.

### ✨ Key Features

- 🧠 **Effective Multi-Step Reasoning**: Two-stage training framework with SFT warm-up and reinforced fine-tuning (RFT)
- 📈 **Difficulty-Aware RL**: Reinforcement learning mechanism that progressively handles complex reasoning
- 🎯 **Stable Reasoning**: Trajectory-level uncertainty quantification fusing token-level confidence with answer-level consistency
- 🚀 **Strong Generalization**: State-of-the-art in-domain performance and excellent out-of-domain generalization

---

## 📋 Abstract

Table reasoning with large language models (LLMs) plays a critical role in building intelligent systems capable of understanding and analyzing tabular data. Despite recent progress, existing methods still face key limitations: their reasoning processes lacks depth and explicit multi-step reasoning, often relying solely on implicit language model understanding. In addition, their reasoning processes suffer from instability, primarily caused by model uncertainty.

In this work, we propose **STaR**, a novel slow-thinking model that can achieve effective and stable table reasoning. To enable effective multi-step reasoning, we design a **two-stage training framework** consisting of supervised fine-tuning (SFT) warm-up followed by reinforced fine-tuning (RFT). Specifically, in the SFT stage, we construct a high-quality dataset through automatic self-verification. In the RFT stage, we introduce a **difficulty-aware reinforcement learning mechanism** to further enhance reasoning capabilities. Furthermore, to improve reasoning stability, we introduce **trajectory-level uncertainty quantification**, which fuses token-level confidence with answer-level consistency, enabling the selection of better reasoning trajectories. Extensive experiments demonstrate that STaR-8B achieves state-of-the-art performance on in-domain benchmarks and exhibits strong generalization to out-of-domain datasets, highlighting its potential for enhancing both effectiveness and stability in table reasoning.

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
| STaR-Datasets | Full training & evaluation data | [![Hugging Face](https://img.shields.io/badge/🤗-Datasets-yellow)](https://huggingface.co/datasets/zhjai/STaR-Datasets) |

### 🤖 Base Models

Download the base models and place them in the `model/` folder:

| Model | Parameters | Link |
|-------|------------|------|
| Qwen3-0.6B | 0.6B | [![Hugging Face](https://img.shields.io/badge/🤗-Qwen3--0.6B-yellow)](https://huggingface.co/Qwen/Qwen3-0.6B) |
| Qwen3-8B | 8B | [![Hugging Face](https://img.shields.io/badge/🤗-Qwen3--8B-yellow)](https://huggingface.co/Qwen/Qwen3-8B) |

### 🏆 Trained Checkpoints

Our trained model weights are available on Hugging Face:

| Model | Parameters | Link |
|-------|------------|------|
| STaR-0.6B | 0.6B | [![Hugging Face](https://img.shields.io/badge/🤗-STaR--0.6B-yellow)](https://huggingface.co/zhjai/STaR-0.6B) |
| STaR-8B | 8B | [![Hugging Face](https://img.shields.io/badge/🤗-STaR--8B-yellow)](https://huggingface.co/zhjai/STaR-8B) |

---

## 🚀 Training

Training scripts are located in the `sh/` directory. Adjust paths and hyperparameters as needed.

### 📚 Stage 1: Supervised Fine-Tuning (SFT)

```bash
# Qwen3-0.6B
bash sh/STaR-sft-qwen3-0.6b.sh

# Qwen3-8B
bash sh/STaR-sft-qwen3-8b.sh
```

### 🎯 Stage 2: Reinforced Fine-Tuning (RFT) - Foundational Training

```bash
# Qwen3-0.6B
bash sh/STaR-sft-stage1-qwen3-0.6b.sh

# Qwen3-8B
bash sh/STaR-sft-stage1-qwen3-8b.sh
```

### 🔥 Stage 2: Reinforced Fine-Tuning (RFT) - Progressive Training

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

> **Note:** The citation on Google Scholar may still display the old title. The correct title is: *STaR: Towards Effective and Stable Table Reasoning via Slow-Thinking Large Language Models*

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
