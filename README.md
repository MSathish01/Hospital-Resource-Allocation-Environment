# 🏥 Hospital Resource Allocation Environment (OpenEnv)

## 📌 Overview

This project simulates a **real-world hospital resource allocation problem** using the OpenEnv framework.

An AI agent must decide how to allocate limited hospital beds to incoming patients efficiently, maximizing rewards while minimizing waste.

---

## 🎯 Problem Statement

Efficient allocation of hospital resources is critical in real-world healthcare systems.
This environment models:

* Limited beds
* Varying patient load
* Decision-based allocation

---

## ⚙️ Environment Design

### 🔹 State

* `beds` → Available hospital beds
* `patients` → Patients waiting
* `step` → Current step count
* `difficulty` → easy / medium / hard

---

### 🔹 Actions

```json
{
  "allocate": <number_of_beds_to_assign>
}
```

---

### 🔹 Tasks (Difficulty Levels)

| Level  | Beds | Patients | Description         |
| ------ | ---- | -------- | ------------------- |
| Easy   | 10   | 5        | Plenty of resources |
| Medium | 8    | 8        | Balanced            |
| Hard   | 5    | 10       | Resource scarcity   |

---

## 🧠 Reward Function

| Condition            | Reward         |
| -------------------- | -------------- |
| Correct allocation   | +1 per patient |
| No allocation        | -0.5           |
| Over allocation      | -1             |
| All patients treated | +5 bonus       |

---

## 📊 Grading System

Score is normalized between **0.0 → 1.0**:

```python
score = max(0.0, min(1.0, reward / 10.0))
```

---

## 🔌 API Endpoints

| Endpoint  | Method | Description       |
| --------- | ------ | ----------------- |
| `/reset`  | POST   | Start new episode |
| `/step`   | POST   | Perform action    |
| `/state`  | GET    | Get current state |
| `/health` | GET    | Health check      |
| `/web`    | GET    | UI page           |
| `/docs`   | GET    | API documentation |

---

## 🚀 Deployment

Live API:
👉 https://msathish-hospital-env.hf.space

Swagger Docs:
👉 https://msathish-hospital-env.hf.space/docs

---

## 🧪 Run Inference (Baseline Agent)

```bash
python inference.py
```

---

## 📦 Tech Stack

* FastAPI
* OpenEnv Framework
* Hugging Face Spaces
* Python

---

## 🏆 Key Features

* Real-world simulation
* Multiple difficulty levels
* Reward shaping
* Normalized grading system
* Deployable API

---

## 📌 Future Improvements

* Smarter RL agent
* Dynamic patient arrival
* Multi-hospital system
* Priority-based allocation

---

## 👤 Author

Sathish
