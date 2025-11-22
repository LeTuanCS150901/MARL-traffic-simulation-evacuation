import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ======== CONFIG ========
file_path = "training_log_20251023_180815.txt"  # change to your actual file name

# ======== READ FILE ========
df = pd.read_csv(file_path, sep=",", skipinitialspace=True)

# ======== CLEAN DATA ========
df["Loss"] = df["Loss"].replace(-1, np.nan)
df["Reward"] = df["Reward"].replace(-1, np.nan)
df = df.dropna(subset=["Loss", "Reward"], how="all")

# ======== COMPUTE MEAN AND CONFIDENCE INTERVAL ========
agg_stats = df.groupby(["Agent", "Eps"], as_index=False).agg(
    mean_loss=("Loss", "mean"),
    std_loss=("Loss", "std"),
    n_loss=("Loss", "count"),
    mean_reward=("Reward", "mean"),
    std_reward=("Reward", "std"),
    n_reward=("Reward", "count")
)

agg_stats = agg_stats.fillna(0)

# 75% confidence interval → z = 1.15
z = 1.15
agg_stats["ci_loss"] = np.where(
    agg_stats["n_loss"] > 1,
    z * agg_stats["std_loss"] / np.sqrt(agg_stats["n_loss"]),
    0
)
agg_stats["ci_reward"] = np.where(
    agg_stats["n_reward"] > 1,
    z * agg_stats["std_reward"] / np.sqrt(agg_stats["n_reward"]),
    0
)

# ======== PLOT STYLE SETTINGS ========
plt.style.use("seaborn-v0_8-whitegrid")
color = "#1f77b4"  # blue like your example
line_width = 1.5
fill_alpha = 0.35  # more opaque to highlight variance

agents = agg_stats["Agent"].unique()

# ===== FIGURE 1: Mean Loss per Agent =====
fig_loss, axes_loss = plt.subplots(nrows=len(agents), ncols=1, figsize=(8, 3 * len(agents)))

if len(agents) == 1:
    axes_loss = [axes_loss]

for i, agent in enumerate(agents):
    ax = axes_loss[i]
    sub = agg_stats[agg_stats["Agent"] == agent]

    ax.fill_between(sub["Eps"],
                    sub["mean_loss"] - sub["ci_loss"],
                    sub["mean_loss"] + sub["ci_loss"],
                    color=color,
                    alpha=fill_alpha,
                    linewidth=0)
    ax.plot(sub["Eps"], sub["mean_loss"], color=color, linewidth=line_width)
    ax.set_title(f"Mean Loss - Agent {agent}")
    ax.set_ylabel("Mean Loss")
    ax.grid(True, alpha=0.3)

fig_loss.suptitle("Cooperative Mean Loss per Agent (75% CI)", fontsize=14)
fig_loss.tight_layout(rect=[0, 0, 1, 0.97])

# ===== FIGURE 2: Mean Reward per Agent =====
fig_reward, axes_reward = plt.subplots(nrows=len(agents), ncols=1, figsize=(8, 3 * len(agents)))

if len(agents) == 1:
    axes_reward = [axes_reward]

for i, agent in enumerate(agents):
    ax = axes_reward[i]
    sub = agg_stats[agg_stats["Agent"] == agent]

    ax.fill_between(sub["Eps"],
                    sub["mean_reward"] - sub["ci_reward"],
                    sub["mean_reward"] + sub["ci_reward"],
                    color=color,
                    alpha=fill_alpha,
                    linewidth=0)
    ax.plot(sub["Eps"], sub["mean_reward"], color=color, linewidth=line_width)
    ax.set_title(f"Mean Reward - Agent {agent}")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Mean Reward")
    ax.grid(True, alpha=0.3)

fig_reward.suptitle("Cooperative Mean Reward per Agent (75% CI)", fontsize=14)
fig_reward.tight_layout(rect=[0, 0, 1, 0.97])

plt.show()
