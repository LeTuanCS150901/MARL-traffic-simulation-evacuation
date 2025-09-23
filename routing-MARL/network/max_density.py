import xml.etree.ElementTree as ET

tree = ET.parse("manhattan2/manhattan2.net.xml")
root = tree.getroot()

car_keywords = {"private", "passenger", "car", "taxi", "bus", "coach", "emergency", "hov", "vip"}
vehicle_length = 5  # meters




max_vehicles_per_lane = 0
max_type = None
max_edge_id = None
min_vehicles_per_lane = float('inf')
min_type = None
min_edge_id = None


# First, collect all car-allowed types

# Only consider these types
allowed_types = {"highway.primary", "highway.secondary", "highway.primary_link", "highway.secondary_link"}
car_types = set()
type_lanes = {}
for t in root.findall('type'):
    type_id = t.get('id')
    num_lanes = int(t.get('numLanes'))
    if type_id in allowed_types:
        car_types.add(type_id)
        type_lanes[type_id] = num_lanes


# Now, scan edges for max and min vehicles per lane
for edge in root.findall('edge'):
    edge_type = edge.get('type')
    if edge_type in car_types:
        for lane in edge.findall('lane'):
            lane_length = lane.get('length')
            if lane_length is not None:
                lane_length = float(lane_length)
                vehicles_per_lane = lane_length // vehicle_length
                if vehicles_per_lane > max_vehicles_per_lane:
                    max_vehicles_per_lane = vehicles_per_lane
                    max_type = edge_type
                    max_edge_id = edge.get('id')
                if vehicles_per_lane < min_vehicles_per_lane:
                    min_vehicles_per_lane = vehicles_per_lane
                    min_type = edge_type
                    min_edge_id = edge.get('id')


print(f"Max vehicles per lane: {int(max_vehicles_per_lane)} vehicles/lane, Road type: {max_type}")
if max_edge_id:
    print(f"Edge with max vehicles per lane: {max_edge_id}")
else:
    print("No edge found for specified types.")

print(f"Min vehicles per lane: {int(min_vehicles_per_lane)} vehicles/lane, Road type: {min_type}")
if min_edge_id:
    print(f"Edge with min vehicles per lane: {min_edge_id}")
else:
    print("No edge found for specified types.")