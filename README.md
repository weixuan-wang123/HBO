# Hierarchical Balancing Optimization (HBO)

This repository contains the data and codes for our paper " [Hierarchical Balancing Optimization for Fine-Tuning Large Language Models](https://arxiv.org/pdf/2505.12300)". HBO is a framework for adaptive dataset sampling to improve fine-tuning efficiency of large language models. It jointly optimizes inter-category and intra-category (difficulty-level) sampling via learned actors that receive rewards derived from model signals. 




## Quick setup

1. Create and activate a Python environment (recommended: conda or venv).
2. Install dependencies:
   pip install -r requirements.txt


## Usage

Two experimental scripts:
- jobscripts/run_hbo.sh — run the model with HBO experiments
- jobscripts/run_heuristic.sh — run using heuristic sampling or baselines



## Citation
If you find this work is useful or use the data in your work, please consider cite our paper:

```
@article{wang2025hbo,
  title={HBO: Hierarchical Balancing Optimization for Fine-Tuning Large Language Models},
  author={Wang, Weixuan and Wu, Minghao and Haddow, Barry and Birch, Alexandra},
  journal={arXiv preprint arXiv:2505.12300},
  year={2025}
}
```
