import pandas as pd
import matplotlib.pyplot as plt
filename = 'survived_vehicles_20251120_235832.txt'

# --- 2. Read the data from the TXT file ---
# 'sep' handles the comma delimiter, 'skipinitialspace' cleans up spaces after the comma,
# and 'usecols' selects only the required columns.
df = pd.read_csv(filename, sep=',', skipinitialspace=True, usecols=['Eps', 'Survived_Vehicles'])

# --- 3. Calculate the Survival Rate (0-1) ---
TOTAL_VEHICLES = df['Survived_Vehicles'].max() # This is 8
df['Survival_Rate'] = df['Survived_Vehicles'] / TOTAL_VEHICLES

# --- 4. Plot the data ---
plt.figure(figsize=(10, 6))
plt.plot(df['Eps'], df['Survival_Rate'], marker='o', linestyle='-', color='b')

# Set titles and labels
plt.title('Rate of Survived Vehicles Over Episode', fontsize=14)
plt.xlabel('Episode (Eps)', fontsize=12)
plt.ylabel(f'Survival Rate (Survived Vehicles / {TOTAL_VEHICLES})', fontsize=12)
# plt.ylim(0, 1)
plt.grid(True, linestyle='--', alpha=0.6)
# plt.xticks(df['Eps'].astype(int))

plt.legend()
plt.tight_layout()
plt.show()
# # Save the plot to a file
# plot_filename = 'survival_rate_over_episode_from_file.png'
# plt.savefig(plot_filename)