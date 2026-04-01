import xml.etree.ElementTree as ET
import numpy as np

input_file = "karl/karl.net.xml"

tree = ET.parse(input_file)
root = tree.getroot()

speeds_ms = []

for edge in root.findall('edge'):
    for lane in edge.findall('lane'):
        speed = lane.get('speed')
        if speed:
            try:
                speeds_ms.append(float(speed))
            except ValueError:
                continue

if speeds_ms:
    # Convert list to numpy array and transform to km/h
    speeds_kmh = np.array(speeds_ms) * 3.6
    
    # Statistical Calculations (km/h)
    mean_speed = np.mean(speeds_kmh)
    q1 = np.percentile(speeds_kmh, 25)
    q3 = np.percentile(speeds_kmh, 75)
    max_speed = np.max(speeds_kmh)
    min_speed = np.min(speeds_kmh)
    median_speed = np.median(speeds_kmh)

    print("--- Network Speed Statistics (km/h) ---")
    print(f"Median Speed: {median_speed:.2f} km/h")
    print(f"Mean Speed:    {mean_speed:.2f} km/h")
    print(f"Min Speed:     {min_speed:.2f} km/h")
    print(f"Q1 (25th%):    {q1:.2f} km/h")
    print(f"Q3 (75th%):    {q3:.2f} km/h")
    print(f"Max Speed:     {max_speed:.2f} km/h")
    print("-" * 39)
    # print(f"Unique values (km/h): {np.unique(speeds_kmh.round(2))}")
else:
    print("No speed data found.")