"""
Upload blog/results to HuggingFace as a free dataset.
No Pro account needed.
Run: python hf_upload_blog.py
"""
from huggingface_hub import HfApi, create_repo
import os

# Your HF credentials
HF_TOKEN = os.environ.get("HF_TOKEN", "")  # or paste your token here
HF_USERNAME = os.environ.get("HF_USERNAME", "MSathish")
REPO_NAME = os.environ.get("HF_REPO_NAME", "medalloc-rl-results")
HF_REPO_ID = os.environ.get("HF_REPO_ID", "")  # optional: "username-or-org/repo"

# Blog content
BLOG_CONTENT = """---
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
"""


def upload_to_hf():
    repo_id = HF_REPO_ID.strip() or f"{HF_USERNAME}/{REPO_NAME}"
    if not HF_TOKEN:
        print("ERROR: Set HF_TOKEN first!")
        print("Get your token from: https://huggingface.co/settings/tokens")
        token = input("Paste your HF token here: ").strip()
    else:
        token = HF_TOKEN

    api = HfApi(token=token)

    # Helpful diagnostic: verify which HF account the token belongs to.
    try:
        me = api.whoami()
        token_user = me.get("name") or me.get("fullname") or "(unknown)"
        print(f"Authenticated as: {token_user}")
    except Exception as exc:
        print(f"Token authentication failed: {exc}")
        print("Make sure your token is valid and has Write permissions.")
        return

    if "/" in repo_id:
        target_namespace = repo_id.split("/", 1)[0]
        if token_user != "(unknown)" and target_namespace != token_user:
            print(
                "WARNING: Your token user does not match the target namespace.\n"
                f"  token user : {token_user}\n"
                f"  namespace  : {target_namespace}\n"
                "If this is an org repo, ensure you're a member and your token has access."
            )

    # If the repo already exists, don't call create_repo (it can 403 even when you have write access
    # but lack permission to create new repos in that namespace).
    try:
        api.dataset_info(repo_id=repo_id)
        print(f"Dataset repo exists: {repo_id}")
    except Exception:
        print(f"Dataset repo not found; attempting to create: {repo_id}")
        try:
            create_repo(
                repo_id=repo_id,
                repo_type="dataset",
                private=False,
                token=token,
                exist_ok=True,
            )
            print("Repo created/exists.")
        except Exception as exc:
            print(f"Repo create error: {exc}")
            print("Next steps:")
            print("  - Create the dataset repo manually in the browser, then rerun this script")
            print("  - Or set HF_REPO_ID to a namespace your token can write to")
            return

    # Upload README
    print("Uploading README...")
    api.upload_file(
        path_or_fileobj=BLOG_CONTENT.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
        token=token,
    )
    print("README uploaded.")

    # Upload reward curve if it exists
    if os.path.exists("medalloc_reward_curves.png"):
        print("Uploading reward curve chart...")
        api.upload_file(
            path_or_fileobj="medalloc_reward_curves.png",
            path_in_repo="medalloc_reward_curves.png",
            repo_id=repo_id,
            repo_type="dataset",
            token=token,
        )
        print("Chart uploaded.")

    print("\nDONE! Your blog is live at:")
    print(f"https://huggingface.co/datasets/{repo_id}")


if __name__ == "__main__":
    upload_to_hf()
