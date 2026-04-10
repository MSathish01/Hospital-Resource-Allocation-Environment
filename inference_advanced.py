import os
import re
import time
import requests
from openai import OpenAI

# Required env vars — exact format from official guidelines
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.getenv("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN     = os.getenv("HF_TOKEN")

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN,
)

ENV_URL   = "https://msathish-hospital-env.hf.space"
ENV_NAME  = "medalloc"


def wake_up():
    for _ in range(10):
        try:
            r = requests.get(f"{ENV_URL}/health", timeout=15)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(6)
    return False


def api_post(url, **kwargs):
    for _ in range(5):
        try:
            r = requests.post(url, timeout=30, **kwargs)
            if r.status_code == 503:
                time.sleep(8)
                continue
            r.raise_for_status()
            return r
        except Exception:
            time.sleep(4)
    raise Exception(f"Failed after 5 retries: {url}")


def get_action(observation: dict) -> int:
    patients = observation.get("patients", [])
    beds     = observation.get("beds", 0)
    if not patients or beds == 0:
        return 0

    high      = sum(1 for p in patients if p["severity"] == "high")
    medium    = sum(1 for p in patients if p["severity"] == "medium")
    emergency = sum(1 for p in patients if p.get("emergency", False))

    prompt = f"""You are a hospital resource allocation agent.
Available beds: {beds}
Patients waiting: {len(patients)} (high={high}, medium={medium}, emergency={emergency})
Prioritize: emergency patients first, then high severity, then medium, then low.
Reply with ONLY a single integer — number of beds to allocate this step."""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=5,
            temperature=0,
        )
        raw     = response.choices[0].message.content.strip()
        numbers = re.findall(r'\d+', raw)
        if numbers:
            return max(0, min(int(numbers[0]), beds))
    except Exception:
        pass

    # Greedy fallback
    priority = [p for p in patients if p.get("emergency") or p["severity"] == "high"]
    if priority:
        return min(len(priority), beds)
    medium_p = [p for p in patients if p["severity"] == "medium"]
    if medium_p:
        return min(len(medium_p), beds)
    return min(len(patients), beds)


def run_task(task: str):
    res  = api_post(f"{ENV_URL}/reset", params={"task": task})
    data = res.json()
    obs  = data["observation"]

    # [START] — official format
    print(f"[START] task={task} env={ENV_NAME} model={MODEL_NAME}", flush=True)

    step_num   = 0
    rewards    = []
    last_error = None
    success    = False
    max_steps  = obs.get("max_steps", 5)

    while True:
        allocate    = get_action(obs)
        action_str  = f"allocate({allocate})"
        last_error  = None

        try:
            step_res  = api_post(f"{ENV_URL}/step", json={"allocate": allocate})
            step_data = step_res.json()
            obs       = step_data.get("observation", obs)
            reward    = float(step_data.get("reward", 0.0))
            done      = bool(step_data.get("done", False))
        except Exception as e:
            reward     = 0.0
            done       = True
            last_error = str(e)

        step_num += 1
        rewards.append(reward)

        error_str = last_error if last_error else "null"

        # [STEP] — official format
        print(
            f"[STEP] step={step_num} action={action_str} "
            f"reward={reward:.2f} done={str(done).lower()} error={error_str}",
            flush=True
        )

        if done or step_num >= max_steps:
            success = done and last_error is None
            break

    rewards_str = ",".join(f"{r:.2f}" for r in rewards)

    # [END] — official format
    print(
        f"[END] success={str(success).lower()} steps={step_num} rewards={rewards_str}",
        flush=True
    )


if __name__ == "__main__":
    wake_up()

    for task in ["easy", "medium", "hard"]:
        try:
            run_task(task)
        except Exception as e:
            print(f"[END] success=false steps=0 rewards=0.00 error={str(e)}", flush=True)
