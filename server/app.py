from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, RedirectResponse
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
    return {"status": "healthy", "environment": "MedAlloc-RL", "version": "2.0.3"}

@app.get("/api")
def api_index():
    return {
        "message": "MedAlloc-RL Hospital Resource Allocation Environment",
        "version": "2.0.3",
        "endpoints": {
            "reset": "POST /reset?task=easy|medium|hard",
            "step": "POST /step  {allocate: int}",
            "state": "GET  /state",
            "grade": "GET  /grade",
            "health": "GET  /health",
            "docs": "GET /docs",
        },
    }


@app.get("/doc", include_in_schema=False)
def doc_redirect():
    return RedirectResponse(url="/docs")

def _interactive_html() -> str:
        return """
        <!doctype html>
        <html>
            <head>
                <meta charset=\"utf-8\" />
                <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
                <title>MedAlloc-RL Interactive</title>
            </head>
            <body style=\"font-family:Arial, sans-serif; max-width:980px; margin:28px auto; padding:0 16px;\">
                <h1 style=\"margin:0 0 6px;\">MedAlloc-RL — Interactive Hospital Environment</h1>
                <div style=\"margin-bottom:14px; color:#444;\">
                    Reset an episode and step through actions. API docs: <a href=\"/docs\">/docs</a> · JSON index: <a href=\"/api\">/api</a>
                </div>

                <div id=\"banner\" style=\"display:none; padding:10px 12px; border-radius:8px; margin-bottom:14px; background:#f6f6f6;\"></div>

                <div style=\"display:flex; gap:14px; flex-wrap:wrap; align-items:flex-end; margin-bottom:14px;\">
                    <div>
                        <label for=\"task\" style=\"display:block; font-weight:600; margin-bottom:6px;\">Difficulty</label>
                        <select id=\"task\" style=\"padding:8px; min-width:180px;\">
                            <option value=\"easy\">easy</option>
                            <option value=\"medium\" selected>medium</option>
                            <option value=\"hard\">hard</option>
                        </select>
                    </div>

                    <div>
                        <button id=\"resetBtn\" style=\"padding:9px 14px;\">Reset</button>
                    </div>

                    <div>
                        <label for=\"allocate\" style=\"display:block; font-weight:600; margin-bottom:6px;\">Allocate beds</label>
                        <input id=\"allocate\" type=\"number\" min=\"0\" value=\"0\" style=\"padding:8px; width:120px;\" />
                    </div>

                    <div>
                        <div style=\"font-weight:600; margin-bottom:6px;\">Recommendation</div>
                        <div style=\"display:flex; gap:8px; align-items:center;\">
                            <span id=\"recText\" style=\"color:#444;\">—</span>
                            <button id=\"useRecBtn\" style=\"padding:8px 10px;\">Use</button>
                        </div>
                    </div>

                    <div>
                        <button id=\"stepBtn\" style=\"padding:9px 14px;\">Step</button>
                    </div>
                </div>

                <div style=\"display:flex; gap:18px; flex-wrap:wrap;\">
                    <div style=\"flex: 1 1 560px;\">
                        <div style=\"font-weight:700; margin-bottom:6px;\">Overview</div>
                        <div id=\"summary\" style=\"background:#f6f6f6; padding:12px; border-radius:8px; margin-bottom:12px;\"></div>

                        <div style=\"font-weight:700; margin-bottom:6px;\">Patients Waiting</div>
                        <div style=\"background:#f6f6f6; padding:12px; border-radius:8px; overflow:auto;\">
                            <table style=\"width:100%; border-collapse:collapse;\">
                                <thead>
                                    <tr>
                                        <th style=\"text-align:left; padding:6px 8px;\">ID</th>
                                        <th style=\"text-align:left; padding:6px 8px;\">Severity</th>
                                        <th style=\"text-align:left; padding:6px 8px;\">Emergency</th>
                                        <th style=\"text-align:left; padding:6px 8px;\">Waiting</th>
                                    </tr>
                                </thead>
                                <tbody id=\"patients\"></tbody>
                            </table>
                        </div>

                        <details style=\"margin-top:12px;\">
                            <summary style=\"cursor:pointer; font-weight:700;\">Raw observation JSON</summary>
                            <pre id=\"stateRaw\" style=\"background:#f6f6f6; padding:12px; border-radius:8px; overflow:auto; margin-top:8px;\"></pre>
                        </details>
                    </div>

                    <div style=\"flex: 1 1 320px;\">
                        <div style=\"font-weight:700; margin-bottom:6px;\">Last Step</div>
                        <pre id=\"last\" style=\"background:#f6f6f6; padding:12px; border-radius:8px; overflow:auto; min-height:360px;\"></pre>
                    </div>
                </div>

                <script>
                    const bannerEl = document.getElementById('banner');
                    const summaryEl = document.getElementById('summary');
                    const patientsEl = document.getElementById('patients');
                    const stateRawEl = document.getElementById('stateRaw');
                    const lastEl = document.getElementById('last');
                    const allocateEl = document.getElementById('allocate');
                    const taskEl = document.getElementById('task');
                    const stepBtn = document.getElementById('stepBtn');
                    const resetBtn = document.getElementById('resetBtn');
                    const recTextEl = document.getElementById('recText');
                    const useRecBtn = document.getElementById('useRecBtn');

                    let currentObs = null;
                    let currentDone = false;
                    let lastRecommendation = 0;

                    function pretty(obj) { return JSON.stringify(obj, null, 2); }

                    function setBanner(text) {
                        if (!text) {
                            bannerEl.style.display = 'none';
                            bannerEl.textContent = '';
                            return;
                        }
                        bannerEl.style.display = 'block';
                        bannerEl.textContent = text;
                    }

                    function recommendAllocate(obs) {
                        if (!obs) return 0;
                        const beds = Number(obs.beds || 0);
                        const patients = Array.isArray(obs.patients) ? obs.patients : [];
                        const highOrEmerg = patients.filter(p => p.severity === 'high' || p.emergency).length;
                        const medium = patients.filter(p => p.severity === 'medium').length;
                        const base = highOrEmerg > 0 ? highOrEmerg : (medium > 0 ? medium : 1);
                        return Math.max(0, Math.min(base, beds));
                    }

                    function renderRecommendation(obs) {
                        const rec = recommendAllocate(obs);
                        lastRecommendation = rec;
                        recTextEl.textContent = String(rec);
                    }

                    function renderSummary(obs) {
                        if (!obs) {
                            summaryEl.textContent = 'No state yet. Click Reset.';
                            return;
                        }
                        const patients = Array.isArray(obs.patients) ? obs.patients : [];
                        const emerg = patients.filter(p => !!p.emergency).length;
                        const high = patients.filter(p => p.severity === 'high').length;
                        const medium = patients.filter(p => p.severity === 'medium').length;
                        const low = patients.filter(p => p.severity === 'low').length;
                        summaryEl.innerHTML = `
                            <div style=\"display:flex; gap:16px; flex-wrap:wrap;\">
                                <div><div style=\"font-weight:700;\">Beds</div><div>${obs.beds} / ${obs.total_beds}</div></div>
                                <div><div style=\"font-weight:700;\">Step</div><div>${obs.step} / ${obs.max_steps}</div></div>
                                <div><div style=\"font-weight:700;\">Difficulty</div><div>${String(obs.difficulty || '').toUpperCase()}</div></div>
                                <div><div style=\"font-weight:700;\">Waiting</div><div>${patients.length}</div></div>
                                <div><div style=\"font-weight:700;\">High / Med / Low</div><div>${high} / ${medium} / ${low}</div></div>
                                <div><div style=\"font-weight:700;\">Emergencies</div><div>${emerg}</div></div>
                            </div>
                        `;
                    }

                    function renderPatients(obs) {
                        const patients = obs && Array.isArray(obs.patients) ? obs.patients : [];
                        if (patients.length === 0) {
                            patientsEl.innerHTML = `<tr><td colspan=\"4\" style=\"padding:8px; color:#444;\">No waiting patients.</td></tr>`;
                            return;
                        }
                        const rows = patients.map(p => {
                            const sev = String(p.severity || 'unknown');
                            const emerg = p.emergency ? 'YES' : '';
                            const waiting = (p.waiting_steps ?? 0);
                            return `
                                <tr>
                                    <td style=\"padding:6px 8px;\">${p.id ?? ''}</td>
                                    <td style=\"padding:6px 8px;\">${sev.toUpperCase()}</td>
                                    <td style=\"padding:6px 8px;\">${emerg}</td>
                                    <td style=\"padding:6px 8px;\">${waiting}</td>
                                </tr>
                            `;
                        }).join('');
                        patientsEl.innerHTML = rows;
                    }

                    function renderRaw(obs) {
                        stateRawEl.textContent = pretty(obs || {});
                    }

                    function setObs(obs) {
                        currentObs = obs;
                        if (obs && typeof obs.beds === 'number') {
                            allocateEl.max = String(obs.beds);
                            if (Number(allocateEl.value) > obs.beds) allocateEl.value = String(obs.beds);
                        }
                        renderSummary(obs);
                        renderPatients(obs);
                        renderRaw(obs);
                        renderRecommendation(obs);
                    }

                    function setDone(done) {
                        currentDone = !!done;
                        stepBtn.disabled = currentDone;
                        if (currentDone) setBanner('Episode complete. Click Reset to start a new one.');
                    }

                    async function doReset() {
                        setBanner('');
                        lastEl.textContent = '';
                        stepBtn.disabled = true;
                        const task = taskEl.value;
                        const res = await fetch(`/reset?task=${encodeURIComponent(task)}`, { method: 'POST' });
                        const data = await res.json();
                        if (data.observation) {
                            setDone(false);
                            setObs(data.observation);
                            allocateEl.value = String(lastRecommendation);
                            stepBtn.disabled = false;
                        } else {
                            lastEl.textContent = pretty(data);
                            setBanner('Reset failed. See Last Step.');
                        }
                    }

                    async function doStep() {
                        if (currentDone) return;
                        setBanner('');
                        stepBtn.disabled = true;
                        const allocate = Number(allocateEl.value || 0);
                        const res = await fetch('/step', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({ allocate })
                        });
                        const data = await res.json();
                        lastEl.textContent = pretty(data);

                        if (data.observation) setObs(data.observation);

                        if (data.error) {
                            setBanner(String(data.error));
                            stepBtn.disabled = false;
                            return;
                        }

                        if (data.done) {
                            setDone(true);
                            try {
                                const gradeRes = await fetch('/grade');
                                const grade = await gradeRes.json();
                                lastEl.textContent = pretty({ ...data, final_grade: grade });
                            } catch (e) {}
                            return;
                        }

                        stepBtn.disabled = false;
                    }

                    resetBtn.addEventListener('click', () => doReset());
                    stepBtn.addEventListener('click', () => doStep());
                    useRecBtn.addEventListener('click', () => { allocateEl.value = String(lastRecommendation); });
                    doReset();
                </script>
            </body>
        </html>
        """


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def home_ui():
        return _interactive_html()


@app.get("/web", response_class=HTMLResponse, include_in_schema=False)
def web_ui():
        return _interactive_html()

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