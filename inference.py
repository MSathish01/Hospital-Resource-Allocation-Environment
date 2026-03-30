import requests

BASE_URL = "https://msathish-hospital-env.hf.space"

# -------------------------------
# RESET ENVIRONMENT
# -------------------------------
response = requests.post(f"{BASE_URL}/reset")
data = response.json()

print("🔄 RESET ENVIRONMENT")
print("Initial State:", data)

done = False
step_count = 0

# -------------------------------
# SIMPLE AGENT LOOP
# -------------------------------
while not done:
    step_count += 1

    # Simple strategy
    action = {
        "allocate": 2
    }

    response = requests.post(f"{BASE_URL}/step", json=action)
    data = response.json()

    print(f"\n➡ Step {step_count}")
    print("Observation:", data["observation"])
    print("Reward:", data["reward"])
    print("Score:", data.get("score", "N/A"))  # Important
    print("Done:", data["done"])

    done = data["done"]

print("\n🏁 EPISODE FINISHED")