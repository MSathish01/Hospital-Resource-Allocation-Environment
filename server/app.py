from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import random

app = FastAPI()

state_data = {}

TASK_CONFIG = {
    "easy":   {"beds": 10, "patients": 5},
    "medium": {"beds": 8,  "patients": 8},
    "hard":   {"beds": 5,  "patients": 10},
}

def make_patients(n):
    return [
        {"id": i, "severity": random.choice(["low", "medium", "high"])}
        for i in range(n)
    ]

@app.post("/reset")
def reset(task: str = Query(default="easy")):
    global state_data
    if task not in TASK_CONFIG:
        task = "easy"
    cfg = TASK_CONFIG[task]
    state_data = {
        "beds":        cfg["beds"],
        "total_beds":  cfg["beds"],   # beds RESTORE each step
        "patients":    make_patients(cfg["patients"]),
        "step":        0,
        "max_steps":   5,
        "difficulty":  task,
        "total_reward": 0.0,
    }
    return {
        "observation": state_data,
        "reward": 0.0,
        "done": False,
        "task": task
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

    # Sort by severity so agent always treats worst first
    sev_order = {"high": 0, "medium": 1, "low": 2}
    patients_sorted = sorted(patients, key=lambda p: sev_order[p["severity"]])

    allocate = max(0, min(action.allocate, beds, len(patients_sorted)))
    treated  = patients_sorted[:allocate]
    remaining = patients_sorted[allocate:]

    reward = 0.0
    for p in treated:
        if p["severity"] == "high":
            reward += 3.0
        elif p["severity"] == "medium":
            reward += 2.0
        else:
            reward += 1.0

    # Penalty: untreated high-severity patients
    for p in remaining:
        if p["severity"] == "high":
            reward -= 2.0
        elif p["severity"] == "medium":
            reward -= 0.5

    # Penalty: wasting beds when patients are waiting
    unused = beds - allocate
    if len(patients_sorted) > 0:
        reward -= unused * 0.3

    # Beds restore each step (treated patients leave)
    state_data["beds"] = state_data["total_beds"]
    state_data["step"] += 1

    # Dynamic new arrivals
    new_arrivals = make_patients(random.randint(0, 2))
    state_data["patients"] = remaining + new_arrivals
    state_data["total_reward"] += reward

    done = (
        state_data["step"] >= state_data["max_steps"]
        or len(state_data["patients"]) == 0
    )

    # Score: 0.0 to 1.0 — based on max possible reward
    max_possible = state_data["total_beds"] * 3.0
    score = round(max(0.0, min(1.0, state_data["total_reward"] / (max_possible * state_data["max_steps"]))), 3)

    return {
        "observation": state_data,
        "reward":      round(reward, 2),
        "score":       score,
        "done":        done
    }

@app.get("/state")
def state():
    return state_data

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.get("/")
def home():
    return {"message": "MedAlloc-RL Running"}

@app.get("/web", response_class=HTMLResponse)
def web_ui():
    return """
    <html>
      <head><title>MedAlloc-RL</title></head>
      <body>
        <h1>🏥 MedAlloc-RL</h1>
        <p>Priority-based hospital allocation environment</p>
        <ul>
          <li><a href="/docs">API Docs</a></li>
          <li><a href="/health">Health</a></li>
        </ul>
      </body>
    </html>
    """

def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
