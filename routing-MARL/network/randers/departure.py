import math
import xml.etree.ElementTree as ET
import numpy as np
from scipy.stats import lognorm
import matplotlib.pyplot as plt


time_jitter = 0.5   # small random offset (s)
min_gap = 0.3       # ensure no two depart too close
last_depart = -1


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
    temp = None
    for ((from_edge, to_edge), veh_type, count, color) in zip(od_pairs, vtype_list, num_list, color_list):
        # Sample all departure times at once
        dep_times = departure_dist.rvs(size=count) 
        

        # Optional: clip to evacuation time window
        if evacuation_time_limit is not None:
            dep_times = np.clip(dep_times, 0, evacuation_time_limit)

        # Sort the departure times for windowed processing
        dep_times.sort()
        temp =np.array(dep_times)
        # Apply windowed departure time calculation
        s = min(75, len(dep_times))  # Handle cases where count < window_size

        if s > 1:
            for k in range(len(dep_times) - 1):
                # Get the window endpoint (k+s), but don't exceed array bounds
                end_idx = min(k + s, len(dep_times) - 1)
                
                # Calculate: t_{k+1} = t_k + (t_{k+s} - t_k) / (s - 1)
                if end_idx > k:  # Ensure we have a valid window
                    dep_times[k + 1] = dep_times[k] + (dep_times[end_idx] - dep_times[k]) / (s - 1)

        for depart_time in dep_times:
            # depart_time = round(depart_time, 2)
            trips.append((depart_time, {
                "type":  veh_type,
                "from":  from_edge,
                "to":    to_edge,
                "depart": str(depart_time),
                "departLane": "best",
                "departPos": "last"
            }))

        # for depart_time in dep_times:
        #     depart_time = round(depart_time, 2)
        #     trips.append((depart_time, {
        #         "type":  veh_type,
        #         "from":  from_edge,
        #         "to":    to_edge,
        #         "depart": str(depart_time),
        #     }))
        # dep_times.sort()
        # for i in range(1, len(dep_times)):
        #     depart_time = max(dep_times[i], dep_times[i-1] + 2.0)
        #     trips.append((depart_time, {
        #         "type":  veh_type,
        #         "from":  from_edge,
        #         "to":    to_edge,
        #         "depart": str(depart_time),
        #     }))

    # Sort and assign IDs
    trips.sort(key=lambda x: x[0])
    for idx, (_, attrs) in enumerate(trips):
        attrs["id"] = f"trip_{idx}"


    derivative = np.diff(dep_times)
    derivative_tmp = np.diff(temp)
    # plt.plot( range(len(derivative)),  derivative, label="window")
    # plt.plot( range(len(derivative_tmp)),  derivative_tmp, label="temp")
    # #plt.plot( range(len(derivative)), derivative_tmp - derivative)
    # plt.legend()
    # plt.show()
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # First subplot
    ax1.plot(range(len(derivative)), derivative, label="window")
    ax1.set_title("Window Derivative")
    ax1.legend()
    ax1.grid(True)
    ax1.set_ylim([0,60])

    # Second subplot
    ax2.plot(range(len(derivative_tmp)), derivative_tmp, label="temp")
    ax2.set_title("Temp Derivative")
    ax2.legend()
    ax2.grid(True)
    ax2.set_ylim([0,60])


    plt.tight_layout()
    plt.show()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # First subplot
    ax1.scatter(range(len(dep_times)), dep_times, label="window")
    ax1.set_title("Window Derivative")
    ax1.legend()
    ax1.grid(True)
    #ax1.set_ylim([0,60])

    # Second subplot
    ax2.scatter( range(len(temp)), temp, label="temp")
    ax2.set_title("Temp Derivative")
    ax2.legend()
    ax2.grid(True)
    #ax2.set_ylim([0,60])


    plt.tight_layout()
    plt.show()
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
            # "accel": "3.0",
            # "decel": "3.5",
            # "lcCooperative": "1.0",
            # "lcStrategic": "1.0",
            # "sigma": "0",
            # "tau": "0.5"
        })
    for _, attrs in trips:
        ET.SubElement(routes, "trip", attrs)

    tree = ET.ElementTree(routes)
    ET.indent(tree, space="    ")
    tree.write(filename, encoding="UTF-8", xml_declaration=True)
    print(f"Generated sorted XML file: {filename}")

if __name__ == "__main__":
    # Pair origins and destinations 1-to-1
    destinations_edges= [
    "233955597#2",
    "-65372792#1",
    "52458578#0",
    "-60995980"
    ] 

    origin_edges = [
    "-302024489#0",
    "-301949382#1",
    "301810426#1",
    "301676368"
    ]
    od_pairs = list(zip(origin_edges, destinations_edges))

    num_list =   [30, 30, 30, 30]
    vtype_list = ['veh_blue', 'veh_red', 'veh_yellow', 'veh_green']
    color_list = ['#0000FF',  '#FF0000', '#ffff00', "#00FF00"]

    # mean, var, time limit are in seconds
    generate_sorted_distributed_trips_xml(
        od_pairs=od_pairs,
        # filename="../../test_teleport/manhattan2.rou.xml",
        filename="randers.rou.xml",
        num_list=num_list,
        vtype_list=vtype_list,
        color_list=color_list,
        dep_mean=70,              # People mostly leave around 1,5 minute
        dep_variance=50**2,         # Large spread (sqrt(900)=30s std deviation)
        # evacuation_time_limit=1800  # Optional: 30-minute evacuation window. In our case, depend on max simulation time
                                    # this mean that the last evacuee should left < evacuation_time_limit
    )



    
