<div align="center">

<img src="STaR.jpg" alt="STaR" width="420"/>

# STaR: Slow-Thinking for Table Reasoning (Anonymous)

</div>

## Abstract

Table reasoning with the large language models (LLMs) is a fundamental path toward building intelligent systems that can understand and analyze over structured data. While recent progress has shown promising results, they still suffer from two key limitations: (i) the reasoning processes lack the depth and iterative refinement characteristic of human cognition; and (ii) the reasoning processes exhibit instability, which compromises their reliability in downstream applications. In this work, we present STaR (slow-thinking for table reasoning), a new framework achieving cognitive table reasoning, in which LLMs are equipped with slow-thinking capabilities by explicitly modeling step-by-step thinking and uncertainty-aware inference. During training, STaR employs two-stage difficulty-aware reinforcement learning (DRL), progressively learning from simple to complex queries under a composite reward. During inference, STaR performs trajectory-level uncertainty quantification by integrating token-level confidence and answer consistency, enabling selection of more credible reasoning paths. Extensive experiments on benchmarks demonstrate that STaR achieves superior performance and enhanced reasoning stability. Moreover, strong generalization over out-of-domain datasets further demonstrates STaR's potential as a reliable and cognitively inspired solution for table reasoning with LLMs.

This repository provides the anonymous implementation of STaR based on the `verl` framework.

## Installation

Tested with Python 3.10 and CUDA GPUs.

```bash
# 1) Clone (anonymous placeholder URL)
git clone [REPO_URL]
cd ano-repo

# 2) Install Python dependencies
pip install -r requirements.txt

# 3) Install verl in editable mode
cd verl
pip install -e .
cd -
```

## Training

Shell scripts are under `sh/`. Adjust paths and hyperparameters inside the scripts as needed.

- SFT
  - `bash sh/STaR-sft-qwen3-0.6b.sh`
  - `bash sh/STaR-sft-qwen3-8b.sh`

- GRPO — Stage 1
  - `bash sh/STaR-sft-stage1-qwen3-0.6b.sh`
  - `bash sh/STaR-sft-stage1-qwen3-8b.sh`

- GRPO — Stage 2
  - `bash sh/STaR-sft-stage1-stage2-qwen3-0.6b.sh`
  - `bash sh/STaR-sft-stage1-stage2-qwen3-8b.sh`

## Evaluation

1) Run rollout to generate trajectories:

```bash
bash sh/STaR-eval.sh
```

2) Compute EM metric by trajectory:

```bash
python eval-by-trajectory.py
```

## Acknowledgements

This work builds on the excellent `verl` framework. We thank the community for open-source tools and datasets used in our experiments.

