---
title: MedAlloc-RL Results
tags:
- reinforcement-learning
- openenv
- hospital-allocation
- grpo
- meta-pytorch
---

# MedAlloc-RL: Hospital Resource Allocation RL Environment

**Top 800 Finalists | Meta PyTorch OpenEnv Hackathon x Scaler SST 2026**
**Out of 31,000+ registered teams**

## Live Demo
- HF Space: https://huggingface.co/spaces/MSathish/hospital-env
- API Docs: https://msathish-hospital-env.hf.space/docs
- GitHub: https://github.com/MSathish01/MedAlloc-RL-Intelligent-Hospital-Resource-Allocation-Environment

## Quick Start
```python
import requests
ENV = "https://msathish-hospital-env.hf.space"

# Reset environment
obs = requests.post(f"{ENV}/reset", params={"task": "hard"}).json()

# Take action
result = requests.post(f"{ENV}/step", json={"allocate": 4}).json()
print(f"Reward: {result['reward']}, Score: {result['score']}")
```

## Training Results

| Metric | Before Training | After Training |
|--------|----------------|----------------|
| Average Reward | 13.84 | 14.31 |
| Peak Reward | 21.80 | 27.20 |
| Hard Task | Baseline | +10% improvement |

## Reward Function

| Event | Reward |
|-------|--------|
| High severity treated | +3.0 |
| Emergency treated | +1.0 bonus |
| Medium severity treated | +2.0 |
| Low severity treated | +1.0 |
| High severity untreated | -2.0 |
| Emergency untreated | -1.5 |
| Bed wasted | -0.3 |
| Patient deterioration | -0.5 |

## Environment Features
- Priority-based triage (low/medium/high severity)
- Emergency patient arrivals (20% chance per step)
- Patient deterioration over time
- 3 difficulty levels: easy/medium/hard
- Normalized scoring 0.001 to 0.999

## Built With
- OpenEnv + FastAPI
- HuggingFace Spaces (Docker)
- GRPO Algorithm
- Unsloth + TRL

*Team Sathish | SVCET Tiruchirappalli | April 2026*
