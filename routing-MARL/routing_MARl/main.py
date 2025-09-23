import time
import argparse  
from read_config import read_config

from group_agents import MultiAgentManager
from graph_manager import GraphManager
from logs.logger import logger  # import our custom logger
from simulation import run_simulation, reset_simulation, run_simulation_inference
from disaster_infor.disaster_move import DisasterMove

def initialize(CONFIG):
    net_file = CONFIG['net_file']
    route_file = CONFIG['route_file']
    graph =  GraphManager(route_file, net_file)
    # calculate distance of edges to destination edges
    graph.calculate_distances(net_file)
    # calculate distance of edges to disaster zones
    graph.calculate_distance_to_disaster_zones(net_file)
    # calculate distance from O->d
    graph.calculate_dist_OD(net_file)

    logger.info(f"Graph loaded. Number of nodes: {graph.network.number_of_nodes()}")
    logger.info(f"Number of edges: {graph.network.number_of_edges()}")

    # Global dictionary to track each vehicle's last processed edge.
    # visited_edge = {vehicle_id: [] for vehicle_id in graph.trips.keys()}
    visited_edge = {}
    multi_agents = MultiAgentManager(graph)
    multi_agents.divide_trips_to_agent_groups()

    # Init disaster object
    poi_file = CONFIG['disaster']['POI_path'] # path to file contains disaster coordinates
    disaster_move = None
    if poi_file:
        logger.success("ACtivate offline training for data collection")
        disaster_move = DisasterMove(poi_file, graph)
        disaster_move.init_disaster_params()

    return graph.edges, graph.trips, visited_edge, multi_agents, disaster_move

def train(num_episodes, simulation_steps, save_freq=5, CONFIG=None):  
    """Training function with model saving"""  
    # Initialize environment and agents  
    edges, trips, last_edge, multi_agents, disaster_move = initialize(CONFIG)  
    
    # Initialize plotter  
    multi_agents.initialize_plotter(max_episodes=num_episodes)  
    
    for episode in range(num_episodes):  
        logger.info(f"Starting episode {episode + 1}/{num_episodes}")  
        
        # Reset the simulation  
        reset_simulation(episode, CONFIG)  
        last_edge.clear()  
        
        # random activate/deactiate disaster for a new episode
        # if no disaster provided. The function do nothing
        multi_agents.random_active_disaster()  
        
        # Run one episode  
        run_simulation(edges=edges, trips=trips, last_edge=last_edge,  
                      multi_agents=multi_agents, CONFIG=CONFIG, simulation_steps=simulation_steps,
                      disaster_move=disaster_move)  
        
        logger.info(f"Episode {episode + 1} completed.")  
        multi_agents.plotter.increase_episode()  
        
        # Save models periodically, but not on the final episode (we'll do that outside the loop)  
        if (episode + 1) % save_freq == 0 and episode < num_episodes - 1:  
            multi_agents.save_all_models(episode=episode+1)  
        
        time.sleep(1)  

    # Final save at the end of training (only once)  
    timestamp = multi_agents.save_all_models(episode=num_episodes)  
    multi_agents.plotter.write_to_file() 
    
    logger.info(f"Training completed. Models saved with timestamp: {timestamp}")  
    return timestamp  

def run_inference(index=None, episode=None, simulation_steps=None, CONFIG=None):  
    """Run inference using saved models"""  
    if simulation_steps is None:  
        simulation_steps = CONFIG['simulation_steps']  
    
    # Initialize environment and agents  
    edges, trips, last_edge, multi_agents = initialize(CONFIG) 
        
    # initialize plotter
    multi_agents.initialize_plotter(max_episodes=1)  
     
    # set mode as inference
    multi_agents.set_training(mode=False)

    # Load saved models  
    success = multi_agents.load_models(index=index, episode=episode, eval_mode=True)  
    if not success:  
        logger.error("Failed to load models. Cannot run inference.")  
        return  
    
    logger.info("Models loaded successfully. Running inference...")  

    # Reset simulation  
    reset_simulation(episode=0, CONFIG=CONFIG)  
    last_edge.clear()  
    
    # random activate/deactiate disaster for a new episode
    # if no disaster provided. The function do nothing
    multi_agents.random_active_disaster()  
        
    # Run simulation with loaded models (no learning) 
    # For now no disaster is added into the model
    run_simulation_inference(edges=edges, trips=trips, last_edge=last_edge,  
                  multi_agents=multi_agents, CONFIG=CONFIG, simulation_steps=simulation_steps)  
    
    logger.info("Inference completed")  

def main():  
    parser = argparse.ArgumentParser(description='RL for Traffic Routing')  
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'inference'],  
                       help='Run mode: train or inference')  
    parser.add_argument('--episodes', type=int, default=None,  
                       help='Number of episodes for training')  
    parser.add_argument('--testcase', type=str, default=None,  
                       help='The testcase config yaml file in config folder')  
    parser.add_argument('--steps', type=int, default=None,  
                       help='Number of simulation steps per episode')  
    parser.add_argument('--save_freq', type=int, default=1,  
                       help='Save frequency (episodes)')  
    parser.add_argument('--index', type=int, default=None,  
                       help='index is the run number of training. E.g, traning number 1,2,3,...')  
    parser.add_argument('--episode', type=int, default=None,  
                       help='Episode number of saved models to load for inference')  
    parser.add_argument('--gui',       action='store_true',
                        help='Launch SUMO with GUI (sumo-gui) instead of sumo') # boolean flag for gui
    
    args = parser.parse_args()  

    # Set the config file
    CONFIG = read_config(config_filename=args.testcase)
    
    # Override CONFIG values if specified  
    num_episodes = args.episodes if args.episodes is not None else CONFIG['num_episodes']  
    simulation_steps = args.steps if args.steps is not None else CONFIG['simulation_steps']  
    # Set which SUMO binary to call
    CONFIG['sumo_binary'] = 'sumo-gui' if args.gui else 'sumo'

    if args.mode == 'train':  
        logger.info(f"Starting training for {num_episodes} episodes, {simulation_steps} steps per episode")  
        train(num_episodes, simulation_steps, save_freq=args.save_freq, CONFIG=CONFIG)  
    else:  # inference mode  
        logger.info("Starting inference with saved models")  
        run_inference(index=args.index, episode=args.episode, simulation_steps=simulation_steps, CONFIG=CONFIG)  

if __name__ == "__main__":  
    main()  
    