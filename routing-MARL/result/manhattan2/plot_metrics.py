import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ======== CONFIG ========
file_path = "training_log_20251120_235832.txt"  # change to your actual file name

# ======== READ FILE ========
df = pd.read_csv(file_path, sep=",", skipinitialspace=True)

# ======== CLEAN DATA ========
df["Loss"] = df["Loss"].replace(-1, pd.NA)
df["Reward"] = df["Reward"].replace(-1, pd.NA)

df["Loss"] = pd.to_numeric(df["Loss"], errors="coerce")
df["Reward"] = pd.to_numeric(df["Reward"], errors="coerce")
# ======== COMPUTE MEANS ========
mean_stats = df.groupby(["Agent", "Eps"], as_index=False).agg(
    mean_loss=("Loss", "mean"),
    mean_reward=("Reward", "mean")
)

# ======== PLOT ========
agents = mean_stats["Agent"].unique()

# fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
# ax_loss, ax_reward = axes

# print(agents)
# ===== FIGURE 1: Mean Loss per Agent =====
fig_loss, axes_loss = plt.subplots(nrows=len(agents), ncols=1, figsize=(8, 3 * len(agents)))

if len(agents) == 1:
    axes_loss = [axes_loss]  # make iterable if only one agent

for i, agent in enumerate(agents):
    ax = axes_loss[i]
    sub = mean_stats[mean_stats["Agent"] == agent]
    # value_loss = np.array(sub["mean_loss"].values, dtype=float)
    value_loss = np.log10(np.array(sub["mean_loss"].values,dtype=float))

    # Plot
    ax.plot(sub["Eps"].values, value_loss, label=f"{agent}")
    ax.set_title(f"Mean Loss - Agent {agent}")
    ax.set_ylabel("Mean Loss")
    ax.legend()
    ax.grid(True)

fig_loss.suptitle("Mean Loss per Agent", fontsize=14)
fig_loss.tight_layout(rect=[0, 0, 1, 0.97])

# ===== FIGURE 2: Mean Reward per Agent =====
fig_reward, axes_reward = plt.subplots(nrows=len(agents), ncols=1, figsize=(8, 3 * len(agents)))

if len(agents) == 1:
    axes_reward = [axes_reward]

for i, agent in enumerate(agents):
    ax = axes_reward[i]
    sub = mean_stats[mean_stats["Agent"] == agent]
    value_reward = np.array(sub["mean_reward"].values, dtype=float)

    # Plot
    ax.plot(sub["Eps"].values, value_reward, label=f"{agent}")
    ax.set_title(f"Mean Reward - Agent {agent}")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Mean Reward")
    ax.legend()
    ax.grid(True)

fig_reward.suptitle("Mean Reward per Agent", fontsize=14)
fig_reward.tight_layout(rect=[0, 0, 1, 0.97])


for i, agent in enumerate(agents):
    sub = mean_stats[mean_stats["Agent"] == agent]
    value_loss = np.array(sub["mean_loss"].values, dtype=float)
    print(f"Agent: {agent}, min loss {np.min(value_loss)}, max loss {np.max(value_loss)}")
    

plt.show()