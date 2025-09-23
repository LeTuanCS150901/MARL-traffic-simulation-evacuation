import xml.etree.ElementTree as ET

# Path to your .net.xml file
file_path = "manhattan2/manhattan2.net.xml"

tree = ET.parse(file_path)
root = tree.getroot()

max_length = 0
max_edge_id = None

# Iterate over all edge elements
for edge in root.findall('edge'):
    for lane in edge.findall('lane'):
        length = lane.get('length')
        if length is not None:
            length = float(length)
            if length > max_length:
                max_length = length
                max_edge_id = edge.get('id')

print(f"Edge with highest length: {max_edge_id}, Length: {max_length}")