import numpy as np  
import matplotlib.pyplot as plt  

# Define the range for d and T  
d_values = np.linspace(0, 1, 1024)  
T_values = np.linspace(0, 1, 1024)  

# Create meshgrid for d and T  
D, T = np.meshgrid(d_values, T_values)  

# Compute R  
R = (1 - D) / (1 + T)  

# Plot heatmap  
plt.figure(figsize=(8, 6))  

heatmap = plt.imshow(R, origin='lower', extent=[0, 1, 0, 1], aspect='auto', cmap='jet')  

plt.colorbar(heatmap, label='Reward R')  
plt.xlabel(r'$d/d_{max}$')  
plt.ylabel(r'$t$')  
plt.title(r'Heatmap of $R = \alpha(1 - d/d_{max}) / (1 + t)$')  
plt.show()

# Parameters  
t_max = 10  
d_max = 5  
alpha = 0.7  

# Create grid of t and d values  
t_values = np.linspace(0, t_max, 100)  
d_values = np.linspace(0, d_max, 100)  
T, D = np.meshgrid(t_values, d_values)  

# Compute reward R  
R = (1 - T / t_max)**2 + alpha * (1 - D / d_max)**2 

# Plot heatmap  
plt.figure(figsize=(8, 6))  
heatmap = plt.contourf(D, T, R, levels=50, cmap='jet')  
plt.colorbar(heatmap, label='Reward R')  

plt.xlabel(r'$d/d_{max}$')  
plt.ylabel(r'$t$')  
plt.title(r'Reward Heatmap: $R = \left(1 - \frac{t}{t_{max}}\right)^2 + \alpha \left(1 - \frac{d}{d_{max}}\right)^2$')  
plt.show()