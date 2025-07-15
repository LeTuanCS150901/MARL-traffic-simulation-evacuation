import math
import xml.etree.ElementTree as ET
import numpy as np
from scipy.stats import lognorm

def lognorm_params(mean, variance):
    # Calculate log-normal mu and sigma from mean and variance (both in seconds!)
    sigma_sq = math.log(1 + variance / mean ** 2)
    mu = math.log(mean) - 0.5 * sigma_sq
    sigma = math.sqrt(sigma_sq)
    print(f"Log-normal parameters, mu: {mu:.4f}, sigma: {sigma:.4f}")
    return mu, sigma

def generate_sorted_distributed_trips_xml(
    od_pairs,
    filename="routes.xml",
    num_list=None,
    vtype_list=None,
    color_list=None,
    dep_mean=30,           # Real mean of departure time in seconds
    dep_variance=20,       # Real variance in seconds^2
    evacuation_time_limit=None  # Optional max cap for departure times (e.g., 1800 seconds)
):
    assert len(od_pairs) == len(num_list) == len(vtype_list) == len(color_list), \
        "List lengths of OD-pairs, num_list, vtype_list and color_list must match."
    
    mu, sigma = lognorm_params(dep_mean, dep_variance)
    departure_dist = lognorm(s=sigma, scale=math.exp(mu))

    trips = []
    for ((from_edge, to_edge), veh_type, count, color) in zip(od_pairs, vtype_list, num_list, color_list):
        # Sample all departure times at once
        dep_times = departure_dist.rvs(size=count)

        # Optional: clip to evacuation time window
        if evacuation_time_limit is not None:
            dep_times = np.clip(dep_times, 0, evacuation_time_limit)

        for depart_time in dep_times:
            depart_time = round(depart_time, 2)
            trips.append((depart_time, {
                "type":  veh_type,
                "from":  from_edge,
                "to":    to_edge,
                "depart": str(depart_time),
            }))

    # Sort and assign IDs
    trips.sort(key=lambda x: x[0])
    for idx, (_, attrs) in enumerate(trips):
        attrs["id"] = f"trip_{idx}"

    # Build XML
    routes = ET.Element(
        "routes",
        {
            "xmlns:xsi": "http://www.w3.org/2001/XMLSchema-instance",
            "xsi:noNamespaceSchemaLocation": "http://sumo.dlr.de/xsd/routes_file.xsd"
        }
    )
    for veh_type, color in zip(vtype_list, color_list):
        ET.SubElement(routes, "vType", {
            "id": veh_type,
            "vClass": "passenger",
            "color": color,
        })
    for _, attrs in trips:
        ET.SubElement(routes, "trip", attrs)

    tree = ET.ElementTree(routes)
    ET.indent(tree, space="    ")
    tree.write(filename, encoding="UTF-8", xml_declaration=True)
    print(f"Generated sorted XML file: {filename}")

if __name__ == "__main__":
    od_pairs = [
        ('--30256#1', '-31622#16'),
        ('-32924#14', '--32766'),
        ('--30528#2', '-31448'),
        ('--32924#0', '--32970')
    ]
    num_list =   [20, 20, 20, 20]
    vtype_list = ['veh_blue', 'veh_red', 'veh_green', 'veh_yellow']
    color_list = ['#0000FF',  '#FF0000', '#00FF00', '#FFFF00']

    generate_sorted_distributed_trips_xml(
        od_pairs=od_pairs,
        filename="luxemb/luxemb.rou.xml",
        num_list=num_list,
        vtype_list=vtype_list,
        color_list=color_list,
        dep_mean=90,              # People mostly leave around 1,5 minute
        dep_variance=900,         # Large spread (sqrt(900)=30s std deviation)
        evacuation_time_limit=1000  # Optional: 30-minute evacuation window. In our case, depend on max simulation time
    )
