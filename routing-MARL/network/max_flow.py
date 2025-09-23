import xml.etree.ElementTree as ET

tree = ET.parse("manhattan2/manhattan2.net.xml")
root = tree.getroot()

vehicle_length = 5  # meters

for t in root.findall('type'):
    type_id = t.get('id')
    num_lanes = int(t.get('numLanes'))
    speed = float(t.get('speed'))  # m/s
    max_flow = num_lanes * speed * (3600 / vehicle_length)
    print(f"Type: {type_id}, Max Flow: {max_flow:.0f} vehicles/hour")