import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Parse the routes.xml file
def parse_routes_xml(xml_file):
    """Extract departure times from routes.xml file"""
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    departure_times = []
    for trip in root.findall('trip'):
        depart = float(trip.get('depart'))
        departure_times.append(depart / 60.0)  # Convert to minutes
    
    return np.array(departure_times)

# Load data
xml_file = 'manhattan2/manhattan2.rou.xml'
departure_times = parse_routes_xml(xml_file)

# Fit log-normal distribution
shape, loc, scale = stats.lognorm.fit(departure_times, floc=0)

# Create figure
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

# ========== TOP PLOT: Histogram with scaled PDF ==========
# Create histogram (counts, not density)
n, bins, patches = ax1.hist(departure_times, bins=30, alpha=0.7, 
                            color='lightblue', edgecolor='black', 
                            linewidth=0.8, label='Histogram')

# Generate PDF and scale it to match histogram
x = np.linspace(departure_times.min(), departure_times.max(), 1000)
pdf = stats.lognorm.pdf(x, shape, loc, scale)

# Scale PDF to match histogram counts
bin_width = bins[1] - bins[0]
pdf_scaled = pdf * len(departure_times) * bin_width

# Plot scaled PDF
ax1.plot(x, pdf_scaled, 'r-', linewidth=2.5, label='Log-normal PDF')

# Formatting
ax1.set_xlabel('Departure Time (minutes)', fontsize=15)
ax1.set_ylabel('Departing Evacuees', fontsize=15)
ax1.set_title('Log-normal Distribution (PDF) of Departure Time of Evacuees', 
             fontsize=18)
ax1.grid(True)
ax1.legend(loc='upper right', fontsize=10)

# ========== BOTTOM PLOT: CDF ==========
cdf = stats.lognorm.cdf(x, shape, loc, scale)
ax2.plot(x, cdf, 'g-', linewidth=2.5, label='Log-normal CDF')

ax2.set_xlabel('Departure Time (minutes)', fontsize=15)
ax2.set_ylabel('Cumulative Probability', fontsize=15)
ax2.set_title('Cumulative Distribution (CDF) of Departure Time of Evacuees', 
             fontsize=18)
ax2.grid(True)
ax2.legend(loc='upper left', fontsize=10)
ax2.set_ylim([0, 1.05])

plt.tight_layout()
# Adjust spacing between subplots
plt.subplots_adjust(hspace=0.3)  # Increase this value for more space (default is ~0.2)
plt.show()



# Statistics
print(f"\nStatistics:")
print(f"Total vehicles: {len(departure_times)}")
print(f"Mean departure time: {departure_times.mean():.2f} minutes")
print(f"Median departure time: {np.median(departure_times):.2f} minutes")
print(f"Std deviation: {departure_times.std():.2f} minutes")
print(f"Min: {departure_times.min():.2f} minutes")
print(f"Max: {departure_times.max():.2f} minutes")