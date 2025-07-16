import xml.etree.ElementTree as ET

input_file = "manhattan2/manhattan2.net.xml"

tree = ET.parse(input_file)
root = tree.getroot()

max_speed = 0.0
for edge in root.findall('edge'):
    for lane in edge.findall('lane'):
        speed = lane.get('speed')
        if speed:
            try:
                speed_val = float(speed)
                if speed_val > max_speed:
                    max_speed = speed_val
            except ValueError:
                continue

print(f"Maximum travel speed in the network: {max_speed} m/s")