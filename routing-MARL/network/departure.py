import random
import math
import xml.etree.ElementTree as ET

def lognorm_params(mean, variance):
    # Calculate log-normal mu and sigma from mean and variance (both in seconds!)
    sigma_sq = math.log(1 + variance / mean ** 2)
    mu = math.log(mean) - 0.5 * sigma_sq
    sigma = math.sqrt(sigma_sq)
    print(f"Log-normal parameters, mu: {mu} sigma:{sigma}")
    return mu, sigma

def generate_sorted_distributed_trips_xml(
    od_pairs,
    filename="routes.xml",
    num_list=None,           # List of vehicle counts for each OD-pair
    vtype_list=None,         # List of vType names (e.g. ['veh_blue', ...])
    color_list=None,         # List of vehicle colors
    dep_mean=8,              # Mean of log-normal in seconds (real mean!)
    dep_variance=1           # Variance of log-normal in seconds^2 (real variance!)
):
    assert len(od_pairs) == len(num_list) == len(vtype_list) == len(color_list), \
        "List lengths of OD-pairs, num_list, vtype_list and color_list must match."
    
    mu, sigma = lognorm_params(dep_mean, dep_variance)

    # 1) Generate all trips
    trips = []
    for ((from_edge, to_edge), veh_type, count, color) in zip(od_pairs, vtype_list, num_list, color_list):
        for _ in range(count):
            # Generate log-normal distributed departure time (>= 0)
            while True:
                depart_time = random.lognormvariate(mu, sigma)
                if depart_time >= 0:
                    break
            depart_time = round(depart_time, 2)
            trips.append((depart_time, {
                "type":  veh_type,
                "from":  from_edge,
                "to":    to_edge,
                "depart": str(depart_time),
            }))

    # 2) Sort & reassign IDs
    trips.sort(key=lambda x: x[0])
    for idx, (_, attrs) in enumerate(trips):
        attrs["id"] = f"trip_{idx}"

    # 3) Build XML
    routes = ET.Element(
        "routes",
        {
            "xmlns:xsi": "http://www.w3.org/2001/XMLSchema-instance",
            "xsi:noNamespaceSchemaLocation": "http://sumo.dlr.de/xsd/routes_file.xsd"
        }
    )
    # vType definitions
    for veh_type, color in zip(vtype_list, color_list):
        ET.SubElement(routes, "vType", {
            "id": veh_type,
            "vClass": "passenger",
            "color": color,
        })

    # Trips
    for _, attrs in trips:
        ET.SubElement(routes, "trip", attrs)

    # Write
    tree = ET.ElementTree(routes)
    ET.indent(tree, space="    ")
    tree.write(filename, encoding="UTF-8", xml_declaration=True)
    print(f"Generated sorted XML file: {filename}")

if __name__ == "__main__":
    # od_pairs = [
    #     ("68674962#0", "1106469541#0"),
    #     ("353220115#0",  "68674962#47"),
    #     ("5671357#0",   "5671357#17")
    # ]
    od_pairs = [
    ('--30256#1', '-31622#16'),
    ('-32924#14', '--32766'),
    ('--30528#2', '-31448'),
    ('--32924#0', '--32970')
    ]
    num_list =   [200, 200, 200, 200]  # vehicles per OD pair
    vtype_list = ['veh_blue', 'veh_red', 'veh_green', 'veh_yellow']
    color_list = ['#0000FF',  '#FF0000', '#00FF00', '#FFFF00']  # colors for each vType

    # *** EXAMPLE: mean = 8s, variance = 1s^2 ***
    generate_sorted_distributed_trips_xml(
        od_pairs=od_pairs,
        filename="luxemb/luxemb.rou.xml",
        num_list=num_list,
        vtype_list=vtype_list,
        color_list=color_list,
        dep_mean=20,
        dep_variance=10
    )