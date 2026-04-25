import requests
import time

BASE_URL = "https://msathish-hospital-env.hf.space"
ENV_NAME = "medalloc"
MODEL_NAME = "gpt-4o-mini"


def check_server():
    """Check if server is running"""
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False


def run_task(task: str):
    """Run a single task with official output format"""
    try:
        # Reset environment
        res = requests.post(f"{BASE_URL}/reset", params={"task": task}, timeout=10)
        data = res.json()
        obs = data.get("observation", {})
    except Exception as e:
        print(f"[END] success=false steps=0 rewards=0.00 error={str(e)}", flush=True)
        return

    # [START] - official format
    print(f"[START] task={task} env={ENV_NAME} model={MODEL_NAME}", flush=True)

    step_num = 0
    rewards = []
    success = False
    max_steps = obs.get("max_steps", 5)

    while True:
        # Simple greedy allocation strategy
        beds = obs.get("beds", 0)
        patients = obs.get("patients", [])
        
        # Allocate beds based on patient severity
        high_severity = sum(1 for p in patients if p.get("severity") == "high")
        allocate = min(beds, max(high_severity, len(patients)))
        
        action_str = f"allocate({allocate})"
        last_error = None

        try:
            # Take step
            step_res = requests.post(
                f"{BASE_URL}/step",
                json={"allocate": allocate},
                timeout=10
            )
            step_data = step_res.json()
            
            obs = step_data.get("observation", obs)
            reward = float(step_data.get("reward", 0.0))
            done = bool(step_data.get("done", False))
            
        except Exception as e:
            reward = 0.0
            done = True
            last_error = str(e)

        step_num += 1
        rewards.append(reward)

        error_str = last_error if last_error else "null"

        # [STEP] - official format
        print(
            f"[STEP] step={step_num} action={action_str} "
            f"reward={reward:.2f} done={str(done).lower()} error={error_str}",
            flush=True
        )

        if done or step_num >= max_steps:
            success = done and last_error is None
            break

    rewards_str = ",".join(f"{r:.2f}" for r in rewards)

    # [END] - official format
    print(
        f"[END] success={str(success).lower()} steps={step_num} rewards={rewards_str}",
        flush=True
    )


if __name__ == "__main__":
    # Check if server is running
    if not check_server():
        print("Error: Server is not running at https://msathish-hospital-env.hf.space")
        print("Please start the server first: uvicorn server.app:app --reload")
        exit(1)

    # Run all tasks
    for task in ["easy", "medium", "hard"]:
        try:
            run_task(task)
        except Exception as e:
            print(f"[END] success=false steps=0 rewards=0.00 error={str(e)}", flush=True)
        
        # Small delay between tasks
        time.sleep(0.5)
