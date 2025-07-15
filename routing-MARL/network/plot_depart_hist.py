import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt

# Path to your XML file
# xml_file = r"c:\Tuan_code\Tuan\marl-traffic-simulation-evacuation-main\routing-MARL\network\luxemb\luxemb.rou.xml"
xml_file = "luxemb/luxemb.rou.xml"

# Parse XML and extract depart times
tree = ET.parse(xml_file)
root = tree.getroot()
depart_times = [float(trip.attrib['depart']) for trip in root.findall('trip')]

# Plot histogram
plt.figure(figsize=(10, 6))
plt.hist(depart_times, bins=20, color='skyblue', edgecolor='black')
plt.xlabel('Depart Time')
plt.ylabel('Number of Vehicles')
plt.title('Histogram of Depart Times')
plt.grid(True)
plt.show()