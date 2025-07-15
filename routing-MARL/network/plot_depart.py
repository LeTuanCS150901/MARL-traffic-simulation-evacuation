import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import lognorm
import math

def lognorm_params(mean, variance):
    sigma_sq = math.log(1 + variance / mean ** 2)
    mu = math.log(mean) - 0.5 * sigma_sq
    sigma = math.sqrt(sigma_sq)
    return mu, sigma

def plot_distribution(dep_mean, dep_variance, limit=1000):
    mu, sigma = lognorm_params(dep_mean, dep_variance)
    dist = lognorm(s=sigma, scale=np.exp(mu))
    x = np.linspace(0, limit, 1000)
    y = dist.pdf(x)

    plt.plot(x, y, label=f"mean={dep_mean}, var={dep_variance}")
    plt.title("Evacuation Departure Time Distribution")
    plt.xlabel("Time (s)")
    plt.ylabel("Density")
    plt.grid(True)
    plt.legend()

# Plot all behaviors
plt.figure(figsize=(10, 6))
plot_distribution(30, 100)
plot_distribution(300, 900)
plot_distribution(90, 900)
plot_distribution(180, 3600)
plt.show()
