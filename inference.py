import os
import re
import time
import requests
from openai import OpenAI

API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN     = os.environ.get("HF_TOKEN")

ENV_URL = "https://msathish-hospital-env.hf.space"

client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN or "sk-placeholder",
)

def wake_up_space():
    """Ping health endpoint until space is awake."""
    for attempt in range(10):
        try:
            r = requests.get(f"{ENV_URL}/health", timeout=15)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(6)
    return False


def api_call_with_retry(method: str, url: str, **kwargs) -> requests.Response:
    """Make API call with retry on 503."""
    for attempt in range(5):
        try:
            if method == "post":
                res = requests.post(url, timeout=30, **kwargs)
            else:
                res = requests.get(url, timeout=30, **kwargs)
            if res.status_code == 503:
                time.sleep(8)
                continue
            res.raise_for_status()
            return res
        except requests.exceptions.HTTPError as e:
            if "503" in str(e):
                time.sleep(8)
                continue
            raise
        except Exception:
            time.sleep(4)
            continue
    raise Exception(f"Failed after 5 retries: {url}")


def get_action(observation: dict) -> int:
    patients = observation.get("patients", [])
    beds = observation.get("beds", 0)
    if not patients or beds == 0:
        return 0

    high = sum(1 for p in patients if p["severity"] == "high")
    medium = sum(1 for p in patients if p["severity"] == "medium")
    emergency = sum(1 for p in patients if p.get("emergency", False))

    prompt = f"""You are a hospital resource allocation agent.
Available beds: {beds}
Patients waiting: {len(patients)} (high={high}, medium={medium}, emergency={emergency})
Reply with ONLY a single integer - number of beds to allocate. Prioritize high severity and emergencies."""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=5,
            temperature=0,
        )
        raw = response.choices[0].message.content.strip()
        numbers = re.findall(r'\d+', raw)
        if numbers:
            return max(0, min(int(numbers[0]), beds))
    except Exception:
        pass

    # Greedy fallback
    high_p = [p for p in patients if p["severity"] == "high" or p.get("emergency")]
    if high_p:
        return min(len(high_p), beds)
    medium_p = [p for p in patients if p["severity"] == "medium"]
    if medium_p:
        return min(len(medium_p), beds)
    return min(len(patients), beds)


def run_task(task: str) -> float:
    res = api_call_with_retry("post", f"{ENV_URL}/reset", params={"task": task})
    data = res.json()
    obs = data["observation"]

    print(f"[START] task={task}", flush=True)

    step_num = 0
    final_score = 0.0
    max_steps = obs.get("max_steps", 5)

    while True:
        allocate = get_action(obs)

        step_res = api_call_with_retry("post", f"{ENV_URL}/step", json={"allocate": allocate})
        step_data = step_res.json()

        obs        = step_data.get("observation", obs)
        reward     = float(step_data.get("reward", 0.0))
        score      = float(step_data.get("score", 0.0))
        done       = bool(step_data.get("done", False))
        step_num  += 1
        final_score = score

        print(f"[STEP] step={step_num} reward={reward:.2f} score={score:.3f} action=allocate({allocate}) done={str(done).lower()}", flush=True)

        if done or step_num >= max_steps:
            break

    print(f"[END] task={task} score={final_score:.3f} steps={step_num}", flush=True)
    return final_score


if __name__ == "__main__":
    # Wake up the space first
    wake_up_space()

    all_scores = {}
    for task in ["easy", "medium", "hard"]:
        try:
            score = run_task(task)
            all_scores[task] = score
        except Exception as e:
            print(f"[END] task={task} score=0.0 steps=0 error={str(e)}", flush=True)
            all_scores[task] = 0.0

    avg = sum(all_scores.values()) / len(all_scores)
    print(f"[SUMMARY] easy={all_scores.get('easy',0):.3f} medium={all_scores.get('medium',0):.3f} hard={all_scores.get('hard',0):.3f} avg={avg:.3f}", flush=True)