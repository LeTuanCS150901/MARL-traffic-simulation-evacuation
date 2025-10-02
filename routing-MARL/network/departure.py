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
    
    print( len(od_pairs), len(num_list), len(vtype_list), len(color_list)  )
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
    # od_pairs = [
    #     ('384649121#0', '542258337#0'),
    #     ('986615541#0', '46139034#0'),
    #     ('464683329#0', '195745917#0'),
    #     ('542096279#0', '1030123275#0')
    # ]

    # od_pairs = [
    # ('46481747#0', '542258337#0'),
    # ('986615541#0', '-438901427#2'),
    # ('464683329#0', '195745917#0'),
    # ('542096279#0', '458166897#36')
    # ]

    # od_pairs = [
    # ('384649121#0', '542258337#0'),
    # ('986615541#0', '68325062#0'),
    # ('464683329#0', '195745917#0'),
    # ('542096279#0', '1030123275#0'),
    # ]
    # num_list =   [750, 750, 750, 750]
    # vtype_list = ['veh_blue', 'veh_red', 'veh_green', 'veh_yellow']
    # color_list = ['#0000FF',  '#FF0000', '#00FF00', '#FFFF00']

    # od_pairs = [
    # ('353220115#0', '68674962#47'),
    # ('68674962#0', '1106469541#0'),
    # ('5671357#0', '5671357#17'),
    # ('5670797#0', '56577925#0'),
    # ]

    # num_list =   [10, 10, 10, 10]
    # vtype_list = ['veh_blue', 'veh_red', 'veh_green', 'veh_yellow']
    # color_list = ['#0000FF',  '#FF0000', '#00FF00', '#FFFF00']

    # od_pairs = [
    # ('46481747#0', '542258337#0'),
    # ('542096279#0', '458166897#36')
    # ]

    # num_list =   [25,25]
    # vtype_list = ['veh_blue', 'veh_red']
    # color_list = ['#0000FF', '#FF0000']

    # od_pairs = [
    # ('538864401', '10295137'),
    # ('-10294329', '10290527')]
    # num_list =   [750, 750]
    # vtype_list = ['veh_blue', 'veh_red']
    # color_list = ['#0000FF',  '#FF0000']


    destinations_edges= ['-184839999#1', '10293408#2', '622089262'] # ensure that these destination edges are connected with shelter nodes in the route file 
    origin_edges= ['-10291544', '10298118', '-10296294']


    # Cartesian product: every origin can go to every destination
    # Pair origins and destinations 1-to-1
    od_pairs = list(zip(origin_edges, destinations_edges))

    num_list =   [1200, 1200, 1200]
    vtype_list = ['veh_blue', 'veh_red', 'veh_yellow']
    color_list = ['#0000FF',  '#FF0000', '#ffff00']

    # mean, var, time limit are in seconds
    generate_sorted_distributed_trips_xml(
        od_pairs=od_pairs,
        filename="campfire/campfire.rou.xml",
        num_list=num_list,
        vtype_list=vtype_list,
        color_list=color_list,
        dep_mean=1800,              # People mostly leave around 1,5 minute
        dep_variance=810000,         # Large spread (sqrt(var) = sqrt(900) = 30 std deviation)
        evacuation_time_limit=3600  # Optional: 30-minute evacuation window. In our case, depend on max simulation time
                                    # this mean that the last evacuee should left < evacuation_time_limit
    )
