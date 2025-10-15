import pandas as pd
import matplotlib.pyplot as plt

# ======== CONFIG ========
file_path = "training_log_20251004_115105.txt"  # change to your actual file name

# ======== READ FILE ========
df = pd.read_csv(file_path, sep=",", skipinitialspace=True)

# ======== CLEAN DATA ========
df["Loss"] = df["Loss"].replace(-1, pd.NA)
df["Reward"] = df["Reward"].replace(-1, pd.NA)

# ======== COMPUTE MEANS ========
mean_stats = df.groupby(["Agent", "Eps"], as_index=False).agg(
    mean_loss=("Loss", "mean"),
    mean_reward=("Reward", "mean")
)

# ======== PLOT ========
agents = mean_stats["Agent"].unique()

fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
ax_loss, ax_reward = axes

# --- Mean Loss subplot ---
for agent in agents:
    sub = mean_stats[mean_stats["Agent"] == agent]
    ax_loss.plot(sub["Eps"], sub["mean_loss"], label=f"{agent}")

ax_loss.set_title("Mean Loss per Agent")
ax_loss.set_ylabel("Mean Loss")
ax_loss.legend()
ax_loss.grid(True)

# --- Mean Reward subplot ---
for agent in agents:
    sub = mean_stats[mean_stats["Agent"] == agent]
    ax_reward.plot(sub["Eps"], sub["mean_reward"], label=f"{agent}")

ax_reward.set_title("Mean Reward per Agent")
ax_reward.set_xlabel("Episode")
ax_reward.set_ylabel("Mean Reward")
ax_reward.legend()
ax_reward.grid(True)

plt.tight_layout()
plt.show()
