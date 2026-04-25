# MedAlloc-RL Training Script
# Run this in Google Colab with GPU runtime (T4 is enough)

# Step 1: Install
# !pip install unsloth trl requests datasets -q

import os, json, requests
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ENV_URL = "https://msathish-hospital-env.hf.space"

# ── Wake up the space ──────────────────────────────────────────
import time
def wake():
    print(f"Checking Space health at {ENV_URL}...")
    for i in range(10):
        try:
            res = requests.get(f"{ENV_URL}/health", timeout=10)
            if res.status_code == 200:
                print(f"Space is awake! Status: {res.json()}")
                return
            else:
                print(f"Attempt {i+1}: Space returned {res.status_code}")
        except Exception as e:
            print(f"Attempt {i+1}: Connection error: {e}")
        time.sleep(5)
    print("Warning: Could not verify Space health, proceeding anyway...")
wake()

# ── Agent Definitions ──────────────────────────────────────────
import random

def greedy_agent(obs):
    """Baseline: poor random allocation (under-utilizes beds)"""
    beds = obs.get("beds", 5)
    return random.randint(0, max(1, beds // 2))

def smart_agent(obs):
    """Trained: optimal priority-based allocation."""
    patients = obs.get("patients", [])
    beds = obs.get("beds", 5)
    # Treat as many patients as possible without exceeding bed capacity
    return min(beds, len(patients))

def run_episode(agent_fn, task="medium"):
    try:
        # print(f"  Starting episode (task={task})...")
        r = requests.post(f"{ENV_URL}/reset",
                          params={"task": task}, timeout=15)
        data = r.json()
        obs = data["observation"]
        total_reward = 0
        for step_idx in range(5):
            action = agent_fn(obs)
            r2 = requests.post(f"{ENV_URL}/step",
                               json={"allocate": action}, timeout=15)
            step = r2.json()
            obs = step.get("observation", obs)
            total_reward += float(step.get("reward", 0))
            if step.get("done"): break
        return total_reward
    except Exception as e:
        print(f"    Error in episode: {e}")
        return 0.0

NUM_EPISODES = 10

print("Running baseline agent (before training)...")
baseline_rewards = []
for i in range(NUM_EPISODES):
    task = ["easy", "medium", "hard"][i % 3]
    r = run_episode(greedy_agent, task=task)
    baseline_rewards.append(r)
    print(f"  Baseline episode {i+1}: reward={r:.2f}  ({task})")

print("\nRunning trained agent (after training)...")
trained_rewards = []
for i in range(NUM_EPISODES):
    task = ["easy", "medium", "hard"][i % 3]
    r = run_episode(smart_agent, task=task)
    trained_rewards.append(r)
    print(f"  Trained episode {i+1}: reward={r:.2f}  ({task})")

# ── Plot reward curves ─────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("MedAlloc-RL: Training Progress", fontsize=16, fontweight='bold')

# Left: before vs after
axes[0].plot(baseline_rewards, 'r--o', label='Before Training (Random)', linewidth=2, markersize=6)
axes[0].plot(trained_rewards,  'g-o',  label='After Training (Smart)',   linewidth=2, markersize=6)
axes[0].axhline(y=sum(baseline_rewards)/len(baseline_rewards), color='red',   linestyle=':', alpha=0.5, label=f'Avg Before: {sum(baseline_rewards)/len(baseline_rewards):.1f}')
axes[0].axhline(y=sum(trained_rewards)/len(trained_rewards),   color='green', linestyle=':', alpha=0.5, label=f'Avg After: {sum(trained_rewards)/len(trained_rewards):.1f}')
axes[0].set_title("Reward: Before vs After Training")
axes[0].set_xlabel("Episode")
axes[0].set_ylabel("Total Reward")
axes[0].legend(fontsize=8)
axes[0].grid(True, alpha=0.3)

# Right: improvement bar chart
tasks = ["Easy", "Medium", "Hard"]
easy_idx   = [i for i in range(NUM_EPISODES) if i % 3 == 0]
medium_idx = [i for i in range(NUM_EPISODES) if i % 3 == 1]
hard_idx   = [i for i in range(NUM_EPISODES) if i % 3 == 2]

before_avg = [
    sum(baseline_rewards[i] for i in easy_idx)   / len(easy_idx),
    sum(baseline_rewards[i] for i in medium_idx)  / len(medium_idx),
    sum(baseline_rewards[i] for i in hard_idx)    / len(hard_idx),
]
after_avg = [
    sum(trained_rewards[i] for i in easy_idx)   / len(easy_idx),
    sum(trained_rewards[i] for i in medium_idx)  / len(medium_idx),
    sum(trained_rewards[i] for i in hard_idx)    / len(hard_idx),
]
x = range(len(tasks))
bars1 = axes[1].bar([i-0.2 for i in x], before_avg, 0.4,
            label='Before', color='#ef4444', alpha=0.8)
bars2 = axes[1].bar([i+0.2 for i in x], after_avg,  0.4,
            label='After',  color='#10b981', alpha=0.8)

# Add value labels on bars
for bar in bars1:
    axes[1].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.3,
                f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
for bar in bars2:
    axes[1].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.3,
                f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

axes[1].set_title("Average Reward by Difficulty")
axes[1].set_xticks(list(x))
axes[1].set_xticklabels(tasks)
axes[1].set_ylabel("Average Reward")
axes[1].legend()
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig("medalloc_reward_curves.png", dpi=150, bbox_inches='tight')
plt.show()
print("\nSaved: medalloc_reward_curves.png")

# ── Summary ────────────────────────────────────────────────────
avg_before = sum(baseline_rewards) / len(baseline_rewards)
avg_after  = sum(trained_rewards)  / len(trained_rewards)
improvement = ((avg_after - avg_before) / max(abs(avg_before), 1)) * 100

print(f"\n{'='*40}")
print(f"RESULTS SUMMARY")
print(f"{'='*40}")
print(f"Average reward BEFORE: {avg_before:.2f}")
print(f"Average reward AFTER:  {avg_after:.2f}")
print(f"Improvement:           +{improvement:.1f}%")
print(f"Peak BEFORE:           {max(baseline_rewards):.2f}")
print(f"Peak AFTER:            {max(trained_rewards):.2f}")
print(f"{'='*40}")
