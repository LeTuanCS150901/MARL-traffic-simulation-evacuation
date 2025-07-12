import traci
import xml.etree.ElementTree as ET
import pandas as pd
import sys
import os

# Get the current script directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the path to folder A
folder_a_path = os.path.join(current_dir, '..', 'routing_MARL')

# Add folder A to the system path
sys.path.append(folder_a_path)

from graph_manager import GraphManager
from read_config import read_config

def extract_trip_destinations(xml_file, print_output=False):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    trip_destinations = {}
    for trip in root.findall('trip'):
        trip_id = trip.attrib['id']
        trip_to = trip.attrib['to']
        trip_destinations[trip_id] = trip_to

    if print_output:
        print("trip_destinations = {")
        for k, v in trip_destinations.items():
            print(f'    "{k}": "{v}",')
        print("}")
    return trip_destinations

def log_disaster_metric(demand_file, network_file, output_routes, trips_to_control, testcase,
                        saved_file):
    
    # READ GRAPH Set the config file
    CONFIG = read_config(config_filename=testcase)
    net_file = CONFIG['net_file']
    route_file = CONFIG['route_file']
    graph = GraphManager(route_file, net_file)

    # Calculate distances
    graph.calculate_distances(net_file)
    graph.calculate_distance_to_disaster_zones(net_file)
    graph.calculate_dist_OD(net_file)

    # Load the output routes XML file
    try:
        tree = ET.parse(output_routes)
        root = tree.getroot()
    except ET.ParseError as e:
        print(f"Error parsing XML file: {e}")
        return
    except FileNotFoundError:
        print(f"Route file not found: {output_routes}")
        return

    path_survival_probabilities = []

    # Log information on found vehicles
    vehicle_count = 0
    for vehicle in root.findall('vehicle'):
        vehicle_id = vehicle.get('id')
     

        if vehicle_id in trips_to_control:
            vehicle_count += 1
            # Get the route element
            route = vehicle.find('route')
            
            if route is not None:
                # Get edges attribute, split it into a list
                edges = route.get('edges').split()  # Split the space-separated string

                print(f"\nProcessing vehicle {vehicle_id} with route edges: {edges}")

                # Initialize accumulated length and survival probability sum
                total_length = 0
                # weighted_survival_sum = 0
                product_survival = 1

                
                # Loop through each edge in the current vehicle's route
                for edge in edges:
                    min_distance, min_disaster_edge, disaster_zone = compute_min_distance_edge(graph, edge)
                    print("-------------------------zone consider", disaster_zone)
                    survival_probability = calculate_survival_probability(graph, edge, disaster_zone)
                    
                    edge_length = graph.edges[edge]['length']
                    total_length += edge_length
                    survive_proba_power_length = survival_probability **edge_length

                    product_survival *= survive_proba_power_length
                    
                    # edge_length = graph.edges[edge]['length']
                    # weighted_survival_sum += edge_length * survival_probability
                    
                    print(f"Edge: {edge}, Min Disaster Edge: {min_disaster_edge}, Survival Probability: {survival_probability}")

                # Calculate the weighted average survival probability for the path
                if total_length > 0:
                    print("++++++++++++++++++++++++++++++++++++++++++")
                    print(f"Total length for vehicle {vehicle_id}: {total_length}")
                    print("product_survival ", product_survival)
                    path_survival_probability = product_survival **(1/total_length)
                    print(f"Path survival probability for vehicle {vehicle_id}: {path_survival_probability}")
                    # path_survival_probability = weighted_survival_sum / total_length
                    path_survival_probabilities.append({
                        'vehicle_id': vehicle_id,
                        'survival_probability': path_survival_probability
                    })

    # Save results to CSV
    save_to_csv(path_survival_probabilities, saved_file)
    print(f"Total vehicles controlled found in route file: {vehicle_count}")

def compute_min_distance_edge(graph, current_edge):
    min_distance = float('inf')
    min_disaster_edge = None

    # Loop through disaster edges
    disaster_zones = graph.CONFIG['disaster']['fire']['disaster_zones']
    selected_zone_id = None

    for zone_id in disaster_zones:
        for edge_disaster in disaster_zones[zone_id]:
            distance = graph.euclidean_distance_between_edges(current_edge, edge_disaster)

            if distance < min_distance:
                min_distance = distance
                min_disaster_edge = edge_disaster
                selected_zone_id = zone_id

    print("Minimum distance found:", min_distance, "for edge:", current_edge, "to disaster edge:", min_disaster_edge)
    return min_distance, min_disaster_edge, selected_zone_id

def calculate_survival_probability(graph, current_edge, disaster_zone):

    disaster_radius = graph.disaster_info[disaster_zone]['radius']
    
    if current_edge in graph.edges_dist_disaster:
        ds = graph.edges_dist_disaster[current_edge][disaster_zone]
        print(f"Edge {current_edge} has distance to disaster {disaster_zone}: {ds}; disaster radius: {disaster_radius}")

        # Calculate not survival probability
        not_survive_proba = 0 if ds > disaster_radius else 1 - (ds / disaster_radius)
        print("dead probability:", not_survive_proba)
        return 1 - not_survive_proba  # Survival probability
    else:
        return 0.0

def save_to_csv(data, saved_file):
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'result/disaster_metric') 
    os.makedirs(output_dir, exist_ok=True)

    filepath = os.path.join(output_dir, saved_file)

    df = pd.DataFrame(data)
    df.to_csv(filepath, index=False)
    print(f"Results saved to: {filepath}")

if __name__ == "__main__":
    # network_file = '../network/manhattan/manhattan.net.xml'
    # demand_file = '../network/manhattan/manhattan.rou.xml'
    # testcase = 'manhattan'
    # output_routes = '../network/manhattan/routes.xml'
    # saved_file = 'RL_dead_proba_R1R2.csv'

    # network_file = '../network/toynet2/net2.net.xml'
    # demand_file = '../network/toynet2/net2.rou.xml'
    # testcase = 'toynet2'
    # output_routes = '../network/toynet2/routes.xml'
    # saved_file = 'dynamic_A_dead_proba_R1R2.csv'

    # SUMO DUE small network
    # network_file = '../network/toynet2/net2.net.xml'
    # demand_file = '../network/toynet2/net2.rou.xml'
    # testcase = 'toynet2'
    # output_routes = '../analyse/400/net2_400.rou.xml'
    # saved_file = 'SUMO_dead_proba_R1R2.csv'

    # SUMO DUE Manhattan network
    network_file = '../network/manhattan/manhattan.net.xml'
    demand_file = '../network/manhattan/manhattan.rou.xml'
    testcase = 'manhattan'
    output_routes = '../analyse/999/manhattan_999.rou.xml'
    saved_file = 'SUMO_dead_proba_R1R2.csv'

    # Extract controlled trips
    trip_destinations = extract_trip_destinations(demand_file, print_output=True)
    trips_to_control = set(trip_destinations.keys())

    # LOG the disaster metric
    log_disaster_metric(demand_file, network_file, output_routes, trips_to_control, testcase,
                        saved_file)