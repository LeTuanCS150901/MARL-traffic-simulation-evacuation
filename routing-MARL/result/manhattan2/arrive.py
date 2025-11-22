import pandas as pd
import matplotlib.pyplot as plt

# Read the data file (update the path as needed)
file_path = "arrived_vehicles_20251119_181656.txt"
data = pd.read_csv(file_path, sep=",", skipinitialspace=True)

# Compute normalized rate (number / maximum)
data["Rate"] = data["Arrived_Vehicles"] / data["Arrived_Vehicles"].max()

# Plot normalized rate
plt.figure(figsize=(10, 6))
plt.plot(data["Eps"], data["Rate"], marker='o', linestyle='-', color='b', label="Arrived / Max")
plt.title("Arrival Rate over Time")
plt.xlabel("Eps")
plt.ylabel("Normalized Arrival Rate")
# plt.ylim(0,1)
plt.grid(True, linestyle='--', alpha=0.6)
# plt.xticks(data['Eps'].astype(int))

plt.legend()
plt.tight_layout()
plt.show()
