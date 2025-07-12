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

    # Load the outout routes XML file
    try:
        tree = ET.parse(output_routes)
        root = tree.getroot()
    except ET.ParseError as e:
        print(f"Error parsing XML file: {e}")
        return
    except FileNotFoundError:
        print(f"Route file not found: {route_file}")
        return

    print("---------------------------------------------------------------------- log disaster metrics")
    
    # Log information on found vehicles
    vehicle_count = 0
    for vehicle in root.findall('vehicle'):
        vehicle_id = vehicle.get('id')
        vehicle_count += 1

        if vehicle_id in trips_to_control:
            # Get the route element
            route = vehicle.find('route')
            
            if route is not None:
                # Get edges attribute, split it into a list
                edges = route.get('edges').split()  # Split the space-separated string

                print(f"\nProcessing vehicle {vehicle_id} with route edges: {edges}")

                # Compute distance to disaster for each edge in the route
                for edge in edges:
                    compute_distance_static_disaster(graph, edge, saved_file)
                
                print(f"\nVehicle ID: {vehicle_id}, Route: {edges}")
            else:
                print(f"Warning: Route not found for vehicle ID: {vehicle_id}")

    print(f"Total vehicles found in route file: {vehicle_count}")

def compute_distance_static_disaster(graph, current_edge, saved_file):
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'result/disaster_metric') 
    os.makedirs(output_dir, exist_ok=True)

    print(f"Computing distance to disaster for edge: {current_edge}")
    
    ###### distance to disaster reward ######
    for disaster_id in graph.disaster_info:
        disaster = graph.disaster_info[disaster_id]
        disaster_radius = disaster['radius']
        
        # Get the distance from current edge to the disaster zone
        if current_edge in graph.edges_dist_disaster:
            ds = graph.edges_dist_disaster[current_edge][disaster_id]
            print(f"Edge {current_edge} has dist to disaster: {ds}; disaster radius: {disaster_radius}")

            # Calculate not survival probability
            not_survive_proba = 0 if ds > disaster_radius else 1 - (ds / disaster_radius)

            # Create new data row
            new_data = {
                'disaster_radius': disaster_radius,
                'distance': ds,
                'not_survive_probability': not_survive_proba
            }
            
            # Define file path with a valid filename
            filepath = os.path.join(output_dir, saved_file)  # Changed here
            
            # Append to existing CSV or create new one
            if os.path.exists(filepath):
                df_existing = pd.read_csv(filepath)
                df_new = pd.DataFrame([new_data])
                df_combined = pd.concat([df_existing, df_new], ignore_index=True)
                df_combined.to_csv(filepath, index=False)
            else:
                df_new = pd.DataFrame([new_data])
                df_new.to_csv(filepath, index=False)
            
            print(f"Result appended to: {filepath}")
        else:
            print(f"Warning: Edge {current_edge} not found in disaster distances.")

if __name__ == "__main__":
    # network_file = '../network/manhattan/manhattan.net.xml'
    # demand_file = '../network/manhattan/manhattan.rou.xml'
    # testcase = 'manhattan'
    # output_routes = '../network/manhattan/routes.xml'
    # saved_file = 'SUMO_dead_proba_R1R2.csv'

    # network_file = '../network/toynet2/net2.net.xml'
    # demand_file = '../network/toynet2/net2.rou.xml'
    # testcase = 'toynet2'
    # output_routes = '../network/toynet2/routes.xml'
    
    # survive proba for each edge
    proba_edge = []
    
    network_file = '../network/toynet2/net2.net.xml'
    demand_file = '../network/toynet2/net2.rou.xml'
    testcase = 'toynet2'
    output_routes = '../analyse/999/net2_999.rou.xml'
    saved_file = 'SUMO_dead_proba_R1R2.csv'

    # Extract controlled trips
    trip_destinations = extract_trip_destinations(demand_file, print_output=True)
    trips_to_control = set(trip_destinations.keys())

    # LOG the disaster metric
    log_disaster_metric(demand_file, network_file, output_routes, trips_to_control, testcase,
                        saved_file)