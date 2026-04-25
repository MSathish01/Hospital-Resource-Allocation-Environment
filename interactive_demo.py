"""
MedAlloc-RL Interactive Demo
Run this to manually play the hospital environment.
"""
import requests

ENV_URL = "https://msathish-hospital-env.hf.space"


def print_state(obs):
    print("\n" + "=" * 50)
    print("  HOSPITAL STATE")
    print("=" * 50)
    print(f"  Beds Available : {obs['beds']} / {obs['total_beds']}")
    print(f"  Step           : {obs['step']} / {obs['max_steps']}")
    print(f"  Difficulty     : {obs['difficulty'].upper()}")
    print(f"  Patients Waiting: {len(obs['patients'])}")
    print("-" * 50)
    for p in obs["patients"]:
        emergency = " EMERGENCY" if p.get("emergency") else ""
        waiting = (
            f" (waiting {p['waiting_steps']} steps)"
            if p.get("waiting_steps", 0) > 0
            else ""
        )
        sev_color = {"high": "[HIGH]", "medium": "[MED]", "low": "[LOW]"}
        print(
            f"  Patient {p['id']:3d} | "
            f"{sev_color.get(p['severity'], '[UNK]')} "
            f"{p['severity'].upper():<6}{emergency}{waiting}"
        )
    print("=" * 50)


def interactive_demo():
    print("\nMedAlloc-RL Interactive Demo")
    print("=" * 32)

    # Choose task
    print("\nChoose difficulty:")
    print("  1. easy   (10 beds, 5 patients)")
    print("  2. medium (8 beds, 8 patients)")
    print("  3. hard   (5 beds, 10 patients)")

    choice = input("\nEnter 1/2/3 (default=2): ").strip()
    task = {"1": "easy", "2": "medium", "3": "hard"}.get(choice, "medium")

    # Reset
    print(f"\nStarting {task.upper()} episode...")
    res = requests.post(f"{ENV_URL}/reset", params={"task": task})
    data = res.json()
    obs = data["observation"]

    total_reward = 0.0

    while True:
        print_state(obs)

        # Show recommendation
        patients = obs["patients"]
        beds = obs["beds"]
        high = sum(1 for p in patients if p["severity"] == "high" or p.get("emergency"))
        medium = sum(1 for p in patients if p["severity"] == "medium")
        recommended = min(max(high, 1), beds) if high else min(max(medium, 1), beds)

        print(f"\n  Recommendation: allocate {recommended} beds")
        print(f"  (High/Emergency: {high}, Medium: {medium})")

        try:
            user_input = input(f"\n  How many beds to allocate? (0-{beds}): ").strip()
            allocate = int(user_input) if user_input else recommended
            allocate = max(0, min(allocate, beds))
        except ValueError:
            allocate = recommended

        # Step
        res = requests.post(f"{ENV_URL}/step", json={"allocate": allocate})
        step_data = res.json()

        obs = step_data.get("observation", obs)
        reward = float(step_data.get("reward", 0))
        score = float(step_data.get("score", 0))
        done = step_data.get("done", False)
        total_reward += reward

        print(f"\n  Allocated {allocate} beds")
        print(f"  Reward this step : {reward:+.2f}")
        print(f"  Total reward     : {total_reward:.2f}")
        print(f"  Current score    : {score:.3f}")

        if done:
            # Get final grade
            grade = requests.get(f"{ENV_URL}/grade").json()
            print("\n" + "=" * 50)
            print("  EPISODE COMPLETE!")
            print("=" * 50)
            print(f"  Final Score    : {grade.get('score', score):.3f}")
            print(f"  Total Reward   : {grade.get('total_reward', total_reward):.2f}")
            print(f"  Patients Treated: {grade.get('treated_count', 0)}")
            print("=" * 50)

            again = input("\n  Play again? (y/n): ").strip().lower()
            if again == "y":
                interactive_demo()
            break


if __name__ == "__main__":
    # First check if space is awake
    print("Checking HF Space status...")
    try:
        r = requests.get(f"{ENV_URL}/health", timeout=10)
        if r.status_code == 200:
            print("Space is running.")
            interactive_demo()
        else:
            print("Space returned error.")
    except Exception as exc:
        print(f"Cannot reach space: {exc}")
        print(f"Check: {ENV_URL}/health")
