import xml.etree.ElementTree as ET
import numpy as np

def get_duration_trips(file_dict):
    for algo_name, tripinfo_file in file_dict.items():
        print(f"\n=== {algo_name} ===")
        try:
            tripinfo_tree = ET.parse(tripinfo_file)
            tripinfo_root = tripinfo_tree.getroot()

            durations_time = []
            arrival_times = []

            for t in tripinfo_root.findall('tripinfo'):
                trip_id = t.get('id')
                if trip_id and trip_id.startswith("trip_"):
                    duration = t.get('duration')
                    arrival = t.get('arrival')
                    if duration is not None and arrival is not None:
                        durations_time.append(float(duration))
                        arrival_times.append(float(arrival))

            if not durations_time:
                print("No trips found.")
                continue

            print("MTT:", np.average(durations_time))
            print("Network clearance time:", max(arrival_times))
            print("Max travel time:", max(durations_time))

        except Exception as e:
            print(f"Error reading {tripinfo_file}: {e}")

if __name__ == "__main__":
    file_dict = {
        'Dynamic A*': 'dynamic_a/tripinfo.xml',
        'SUMO DUE': 'due/tripinfo_001.xml',
        'C-logit': 'hassan/tripinfo_399.xml',
        'MARL no disaster': 'RL/no_disaster/tripinfo.xml'
    }

    get_duration_trips(file_dict)
