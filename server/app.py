from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import random

app = FastAPI()

# -------------------------------
# GLOBAL STATE
# -------------------------------
state_data = {}

# -------------------------------
# HOME
# -------------------------------
@app.get("/")
def home():
    return {"message": "Hospital Environment Running"}

# -------------------------------
# WEB UI (HF Spaces REQUIRED)
# -------------------------------
@app.get("/web", response_class=HTMLResponse)
def web_ui():
    return """
    <html>
        <head><title>Hospital Env</title></head>
        <body>
            <h1>🏥 Hospital Resource Allocation</h1>
            <p>🎯 Allocate beds efficiently to treat patients</p>
            <ul>
                <li><a href="/docs">API Docs</a></li>
                <li><a href="/health">Health Check</a></li>
            </ul>
        </body>
    </html>
    """

# -------------------------------
# HEALTH
# -------------------------------
@app.get("/health")
def health():
    return {"status": "healthy"}

# -------------------------------
# RESET
# -------------------------------
@app.post("/reset")
def reset():
    global state_data

    difficulty = random.choice(["easy", "medium", "hard"])

    if difficulty == "easy":
        beds = 10
        patients = 5
    elif difficulty == "medium":
        beds = 8
        patients = 8
    else:
        beds = 5
        patients = 10

    state_data = {
        "beds": beds,
        "patients": patients,
        "step": 0,
        "difficulty": difficulty
    }

    return {
        "observation": state_data,
        "reward": 0.0,
        "score": 0.0,
        "done": False,
        "task": difficulty
    }

# -------------------------------
# ACTION MODEL
# -------------------------------
class Action(BaseModel):
    allocate: int

# -------------------------------
# STEP
# -------------------------------
@app.post("/step")
def step(action: Action):
    global state_data

    allocate = action.allocate
    beds = state_data["beds"]
    patients = state_data["patients"]

    # Prevent invalid allocation
    allocate = max(0, min(allocate, beds))

    state_data["step"] += 1
    state_data["beds"] -= allocate
    state_data["patients"] -= allocate

    # -------------------------------
    # REWARD LOGIC
    # -------------------------------
    if allocate == 0:
        reward = -0.5
    elif allocate > patients:
        reward = -1.0
    else:
        reward = allocate * 1.0

    # Bonus if completed
    if state_data["patients"] <= 0:
        reward += 5.0

    # -------------------------------
    # GRADER (0 → 1 score)
    # -------------------------------
    max_reward = 10.0
    score = max(0.0, min(1.0, reward / max_reward))

    # -------------------------------
    # DONE
    # -------------------------------
    done = (
        state_data["step"] >= 5
        or state_data["patients"] <= 0
        or state_data["beds"] <= 0
    )

    return {
        "observation": state_data,
        "reward": reward,
        "score": score,
        "done": done
    }

# -------------------------------
# STATE
# -------------------------------
@app.get("/state")
def state():
    return state_data