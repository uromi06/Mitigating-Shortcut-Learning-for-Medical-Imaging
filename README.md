# Pneumothorax-chest drain shortcut learning

Project @ [UBRA AI Toolbox Hackathon](https://www.bremen-research.de/en/ai-toolbox-hackathon)
## Overview
This project studies **shortcut learning** in chest X-ray pneumothorax prediction: models may over-rely on the presence of a **chest drain** (a treatment artifact) instead of learning radiographic evidence of pneumothorax. This can cause failures when the shortcut distribution shifts (e.g., pneumothorax without a drain).
## Code entrypoints
- **Baseline (original scaffold):** `cxp_pneu.py`
- **My extended version (SupCon + multi-run):** `cxp_pneu_v2.py`
- **SupCon loss module:** `supcon.py`


## Getting set up
- [ ] Create a GitHub account (if you do not have one yet)
- [ ] Fork this repository and give all group members access to the fork; I suggest using this as the primary code synchronization method
- [ ] Create a [Weights & Biases](https://wandb.ai/) account (if you do not have one yet - one per group is sufficient)
- [ ] Take notice of the CheXpert dataset research use agreement and create an individual account [here](https://stanfordaimi.azurewebsites.net/datasets/8cbd9ed4-2eb9-4565-affc-111cf4f7ebe2)
- [ ] [Get onto the VM](VM.md) and verify you can run cxp_pneu.py
- [ ] Go through [the baseline code](cxp_pneu.py); make sure you understand it
- [ ] Implement [the Serna et al. approach](https://www.sciencedirect.com/science/article/pii/S0004370222000224?via%3Dihub) as a potential shortcut learning mitigation method
- [ ] Go through the list of open issues; address what you find interesting or explore your own ideas freely



## Dataset
We use the **CheXpert** dataset (research use agreement required). We evaluate performance across clinically relevant slices defined by:
- Pneumothorax label (pos/neg)
- Chest drain presence (present/absent)

Key robustness slice: **Pneumothorax+ & Drain−** (where shortcut reliance is most harmful).

## Training details
- Backbone: DenseNet121 (ImageNet weights)
- Loss: BCEWithLogitsLoss; optional `BCE + λ·SupCon` (λ=0.50)
- SupCon: projection head 1024→256→128; temperature τ=0.10
- EMA: exponential moving average model (decay 0.9) used for eval
- Repeats: NUM_RUNS=10; final metrics reported as mean ± std

## Supervised Contrastive Loss (SupCon) — implementation note
This repo includes a **single-GPU** supervised contrastive loss module used to encourage label-consistent clustering in representation space.

**Code:** `SupervisedContrastiveLoss`  
**Inputs:**
- `feats`: embedding tensor of shape `[B, D]` (expected from a projection head)
- `labels`: tensor of shape `[B]` (binary or multi-class)

## Reading materials (optional)
General shortcut learning:
- [Shortcut learning in deep neural networks](https://www.nature.com/articles/s42256-020-00257-z)
- [The risk of shortcutting in deep learning algorithms for medical imaging research](https://www.nature.com/articles/s41598-024-79838-6)

Papers that cover pneumothorax/chest drain shortcut learning:
- [Hidden stratification causes clinically meaningful failures in machine learning for medical imaging](https://dl.acm.org/doi/10.1145/3368555.3384468)
- [DETECTING SHORTCUTS IN MEDICAL IMAGES - A CASE STUDY IN CHEST X-RAYS](https://arxiv.org/pdf/2211.04279)
- [Slicing Through Bias: Explaining Performance Gaps in Medical Image Analysis using Slice Discovery Methods](https://arxiv.org/html/2406.12142v2)

Drawbacks of DANN / CDANN (baseline method for 'domain invariance' which can also be used to address shortcut learning):
- [Fundamental Limits and Tradeoffs in Invariant Representation Learning](https://www.jmlr.org/papers/v23/21-1078.html)
- [10 Years of Fair Representations: Challenges and Opportunities](https://arxiv.org/abs/2407.03834)
- [Are demographically invariant models and representations in medical imaging fair?](https://arxiv.org/html/2305.01397v3)
- [The limits of fair medical imaging AI in real-world generalization](https://www.nature.com/articles/s41591-024-03113-4)
- [MEDFAIR: Benchmarking Fairness for Medical Imaging](https://arxiv.org/abs/2210.01725)

Alternative approach pursued here: [Sensitive loss: Improving accuracy and fairness of face representations with discrimination-aware deep learning](https://www.sciencedirect.com/science/article/pii/S0004370222000224)

## Credits & Contributions
This repository was developed as part of the **UBRA AI Toolbox Hackathon**.

- **Baseline code and project framework:** Eike Petersen  
- **SupCon loss:** Based on the Supervised Contrastive Learning objective (Khosla et al., NeurIPS 2020)
- **Modifications in this fork (Uromi):**
  - Multi-run evaluation (mean ± std across runs)
  - Balanced validation option + drain-aware train balancing (WeightedRandomSampler)
  - SupCon training option (projection head + combined BCE + λ·SupCon objective)
