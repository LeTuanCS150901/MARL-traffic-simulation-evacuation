import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt 

def calculate_average_mean_TT(summary_file, latest_step):
    tree = ET.parse(summary_file)
    root = tree.getroot()

    mtt_values = [
        float(step.attrib.get("meanTravelTime", "-1"))
        for step in root.findall("step")
    ]
    mtt_array = np.array(mtt_values)

    # filter until lastest step where our controlled vehicle reach dest
    mtt_array = mtt_array[0:latest_step]
    # Filter out -1
    valid_mtt = mtt_array[mtt_array > 0]

    if valid_mtt.size > 0:
        return np.mean(valid_mtt)
    else:
        return None
    
def get_duration_trips(tripinfo_file):

    # Parse tripinfo.xml and extract arrival times for trips starting with "trip_"
    tripinfo_tree = ET.parse(tripinfo_file)
    tripinfo_root = tripinfo_tree.getroot()

    durations_time = []
    arrival_times = []
    for t in tripinfo_root.findall('tripinfo'):
        trip_id = t.get('id')
        if trip_id.startswith("trip_"):
            duration = t.get('duration')
            arrival = t.get('arrival')
            if duration is not None:
                durations_time.append(float(duration))
                arrival_times.append(float(arrival))

    # print("duration times:\n", sorted(durations_time))
    print()
    print("MTT:", np.average(durations_time))
    print("network clearance time", list(sorted(arrival_times))[-1])
    print("max travel time", max(durations_time))
 

if __name__ == "__main__":
    ## trip info for A*
    tripinfo_file = r"D:\PhD_Life\Work\Code\marl-traffic-simulation-evacuation\routing-MARL\network\manhattan2\tripinfo.xml"
    get_duration_trips(tripinfo_file)