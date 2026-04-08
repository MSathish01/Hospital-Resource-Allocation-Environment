from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List
import random

app = FastAPI(title="MedAlloc-RL", description="Hospital Resource Allocation RL Environment")

state_data = {}

TASK_CONFIG = {
    "easy":   {"beds": 10, "patients": 5,  "max_steps": 5},
    "medium": {"beds": 8,  "patients": 8,  "max_steps": 5},
    "hard":   {"beds": 5,  "patients": 10, "max_steps": 5},
}

def make_patients(n: int, start_id: int = 0) -> List[dict]:
    return [
        {
            "id": start_id + i,
            "severity": random.choice(["low", "medium", "high"]),
            "emergency": False,
            "waiting_steps": 0,
        }
        for i in range(n)
    ]

def safe_score(raw: float) -> float:
    """Always return strictly between 0 and 1."""
    score = round(max(0.001, min(0.999, raw)), 3)
    return score

@app.post("/reset")
def reset(task: str = Query(default="easy")):
    global state_data
    if task not in TASK_CONFIG:
        task = "easy"
    cfg = TASK_CONFIG[task]
    state_data = {
        "beds":               cfg["beds"],
        "total_beds":         cfg["beds"],
        "patients":           make_patients(cfg["patients"]),
        "step":               0,
        "max_steps":          cfg["max_steps"],
        "difficulty":         task,
        "total_reward":       0.0,
        "patient_id_counter": cfg["patients"],
        "treated_count":      0,
        "emergency_count":    0,
    }
    return {
        "observation": _clean_obs(state_data),
        "reward": 0.0,
        "done": False,
        "task": task,
    }

class Action(BaseModel):
    allocate: int

@app.post("/step")
def step(action: Action):
    global state_data
    if not state_data:
        return {"error": "Call /reset first"}

    beds     = state_data["beds"]
    patients = state_data["patients"]

    def priority_key(p):
        sev = {"high": 0, "medium": 1, "low": 2}[p["severity"]]
        emergency_bonus = -10 if p.get("emergency") else 0
        return emergency_bonus + sev - p.get("waiting_steps", 0) * 0.1

    patients_sorted = sorted(patients, key=priority_key)
    allocate  = max(0, min(action.allocate, beds, len(patients_sorted)))
    treated   = patients_sorted[:allocate]
    remaining = patients_sorted[allocate:]

    reward = 0.0
    for p in treated:
        if p["severity"] == "high":
            reward += 3.0
            if p.get("emergency"):
                reward += 1.0
        elif p["severity"] == "medium":
            reward += 2.0
        else:
            reward += 1.0
        state_data["treated_count"] += 1

    for p in remaining:
        if p["severity"] == "high":
            reward -= 2.0
        elif p["severity"] == "medium":
            reward -= 0.5
        if p.get("emergency"):
            reward -= 1.5

    unused = beds - allocate
    if len(patients_sorted) > 0:
        reward -= unused * 0.3

    for p in remaining:
        p["waiting_steps"] = p.get("waiting_steps", 0) + 1

    for p in remaining:
        if p["waiting_steps"] >= 2 and p["severity"] == "medium":
            p["severity"] = "high"
            reward -= 0.5
        elif p["waiting_steps"] >= 3 and p["severity"] == "low":
            p["severity"] = "medium"

    state_data["beds"] = state_data["total_beds"]
    state_data["step"] += 1
    state_data["total_reward"] += reward

    counter     = state_data["patient_id_counter"]
    new_arrivals = make_patients(random.randint(0, 2), start_id=counter)
    state_data["patient_id_counter"] = counter + len(new_arrivals)

    if random.random() < 0.2:
        state_data["patients"].append({
            "id":            state_data["patient_id_counter"],
            "severity":      "high",
            "emergency":     True,
            "waiting_steps": 0,
        })
        state_data["patient_id_counter"] += 1
        state_data["emergency_count"]    += 1

    state_data["patients"] = remaining + new_arrivals

    done = (
        state_data["step"] >= state_data["max_steps"]
        or len(state_data["patients"]) == 0
    )

    max_possible = max(1.0, state_data["total_beds"] * 3.0)
    score = safe_score(state_data["total_reward"] / max_possible)

    return {
        "observation": _clean_obs(state_data),
        "reward":      round(reward, 2),
        "score":       score,
        "done":        done,
        "info": {
            "step":             state_data["step"],
            "treated_total":    state_data["treated_count"],
            "emergencies_seen": state_data["emergency_count"],
            "total_reward":     round(state_data["total_reward"], 2),
        }
    }

@app.get("/grade")
def grade():
    if not state_data:
        return {"score": 0.5}
    max_possible = max(1.0, state_data["total_beds"] * 3.0)
    score = safe_score(state_data["total_reward"] / max_possible)
    return {
        "score":         score,
        "total_reward":  round(state_data["total_reward"], 2),
        "treated_count": state_data.get("treated_count", 0),
        "steps_taken":   state_data["step"],
        "difficulty":    state_data["difficulty"],
    }

@app.get("/state")
def get_state():
    if not state_data:
        return {"error": "No active episode. Call /reset first."}
    return _clean_obs(state_data)

@app.get("/health")
def health():
    return {"status": "healthy", "environment": "MedAlloc-RL", "version": "2.0.0"}

@app.get("/")
def home():
    return {
        "message": "MedAlloc-RL Hospital Resource Allocation Environment",
        "version": "2.0.0",
        "endpoints": {
            "reset":  "POST /reset?task=easy|medium|hard",
            "step":   "POST /step  {allocate: int}",
            "state":  "GET  /state",
            "grade":  "GET  /grade",
            "health": "GET  /health",
        }
    }

@app.get("/web", response_class=HTMLResponse)
def web_ui():
    return """
    <html>
      <head><title>MedAlloc-RL</title></head>
      <body style="font-family:Arial;max-width:700px;margin:40px auto;padding:20px">
        <h1>MedAlloc-RL Hospital Environment</h1>
        <p>Hospital Resource Allocation - Reinforcement Learning Environment</p>
        <h3>Features</h3>
        <ul>
          <li>Priority-based patient triage (low/medium/high)</li>
          <li>Emergency patient arrivals (20 percent per step)</li>
          <li>Patient deterioration over time</li>
          <li>Time pressure with step limits</li>
          <li>Normalized scoring 0.001 to 0.999</li>
          <li>3 difficulty levels: easy / medium / hard</li>
        </ul>
        <h3>Quick Links</h3>
        <ul>
          <li><a href="/docs">API Docs</a></li>
          <li><a href="/health">Health Check</a></li>
          <li><a href="/grade">Current Grade</a></li>
        </ul>
      </body>
    </html>
    """

def _clean_obs(s: dict) -> dict:
    return {
        "beds":       s["beds"],
        "total_beds": s["total_beds"],
        "patients":   s["patients"],
        "step":       s["step"],
        "max_steps":  s["max_steps"],
        "difficulty": s["difficulty"],
    }

def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860, reload=False)

if __name__ == "__main__":
    main()