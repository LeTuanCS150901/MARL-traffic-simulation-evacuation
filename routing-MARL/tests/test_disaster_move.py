import os
import sys
import traci

# Get the current script directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the path to folder A
folder_a_path = os.path.join(current_dir, '..', 'routing_MARL')

# Add folder A to the system path
sys.path.append(folder_a_path)


from graph_manager import GraphManager
from disaster_infor.disaster_move import DisasterMove
from read_config import read_config


if __name__ == "__main__":

    testcase = "campfire"
    CONFIG = read_config(config_filename=testcase)
    net_file = CONFIG['net_file']
    route_file = CONFIG['route_file']
    sumo_config = CONFIG['sumo_config']

    graph = GraphManager(route_file, net_file)
    poi_file = "../network/campfire/generated_pois_large.add.xml"  # Replace with your POI XML file path
    disaster_move = DisasterMove(poi_file, graph)
    disaster_move.init_disaster_params()

    # Start SUMO simulation with TraCI
    traci.start(["sumo", "-c", sumo_config])


    # Example simulation loop using while
    step = 0
    while step <= disaster_move.max_time:
        traci.simulationStep()  # Advance the simulation by one step
        disaster_move.simulate_points(step)

        # Recalculate risks every 10 seconds
        if step % 10 == 0:
            # print(f"current step {step}")
            # disaster_move.compute_risks()
            # disaster_move.compute_risks_vectorized()
            disaster_move.compute_risks_numba()
            print(f"Recalculated risks at step {step}")
        
        step += 1  # Increment step counter

    # Print computed risks for validation
    print("Computed Risks:")
    for edge_id, risk in disaster_move.edge_risks.items():
        print(f"Edge {edge_id}: Risk {risk}")