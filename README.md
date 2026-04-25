---
title: hospital-env
emoji: 🏥
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
---

# MedAlloc-RL: Hospital Resource Allocation Environment

A real-world reinforcement learning environment where an AI agent 
allocates limited hospital beds to patients with varying severity levels.

## Problem Statement
Hospitals face critical resource allocation decisions daily. This environment
simulates that challenge — the agent must decide how many beds to allocate
each step, prioritizing high-severity and emergency patients.

## Features
- Priority-based triage: low / medium / high severity
- Emergency patient arrivals (20% chance per step)
- Patient deterioration over time (waiting patients get worse)
- Dynamic new patient arrivals each step
- 3 difficulty levels: easy / medium / hard

## Reward Function
| Event | Reward |
|-------|--------|
| High severity treated | +3.0 |
| Medium severity treated | +2.0 |
| Low severity treated | +1.0 |
| Emergency treated bonus | +1.0 |
| High severity untreated | -2.0 |
| Emergency untreated | -1.5 |
| Wasted bed | -0.3 |
| Patient deterioration | -0.5 |

## Tasks
| Task | Beds | Patients | Challenge |
|------|------|----------|-----------|
| Easy | 10 | 5 | Learn basic priority allocation |
| Medium | 8 | 8 | Balanced pressure, emergencies begin |
| Hard | 5 | 10 | Crisis mode, scarce resources |

## API
- `POST /reset?task=easy|medium|hard` — Start new episode
- `POST /step` — Take action `{"allocate": int}`
- `GET /state` — Current environment state
- `GET /grade` — Current episode score (0.001 to 0.999)
- `GET /health` — Health check

## Action Space
`allocate`: integer — number of beds to allocate this step

## Observation Space
- `beds`: available beds this step
- `patients`: list of waiting patients with severity and emergency flag
- `step`: current step number
- `max_steps`: episode length

## Local Scripts

### Interactive Demo (Pitch Demo)

Run a manual, terminal-based episode against the deployed HF Space API:

```powershell
python interactive_demo.py
```

If `requests` is missing:

```powershell
python -m pip install requests
```

### HF Blog Upload (Dataset README)

This uploads a `README.md` to a dataset repo on the Hugging Face Hub (free).

1) Create a token with **Write** permissions: https://huggingface.co/settings/tokens
2) (If needed) create the dataset repo manually in the browser under your namespace, e.g.
	`https://huggingface.co/new-dataset`
3) Run:

```powershell
python hf_upload_blog.py
```

Notes:
- The Hugging Face namespace is case-sensitive (use `MSathish`, not `msathish`).
- If you're in a hurry, you can upload via the website by creating `README.md` and pasting the content from [hf_blog.md](hf_blog.md).

Optional environment variables:

```powershell
$env:HF_TOKEN = "<your_token>"
$env:HF_REPO_ID = "MSathish/medalloc-rl-results"
```