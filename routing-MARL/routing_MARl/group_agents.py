import libsumo as traci
import torch
import os
import json
import glob
import numpy as np
import re
import time
from collections import defaultdict
from collections import deque

from agent import Agent
from logs.logger import logger  # import our custom logger
from plotter import MultiAgentPlotter
from read_config import get_config
from utils.reward import Reward
from groups_shared_infor import GroupsSharedInfo
from database.h5_database import OfflineRLDataset
from database.sumo_db import SumoDB

class MultiAgentManager:
    """
    This architectur:
    obs1 concat obs1^, ...
    => transformer
    => embedding
    => Q1, ...
    => Q total
    -> action selection from individual Q values

    obs1: current edge index, connectivity edges
    obs1^: density on current edge 

    
    """
    def __init__(self, graph, activation_prob=1):
        
        self.CONFIG = get_config()
        self.graph = graph
        self.edges = graph.edges
        self.trips = graph.trips  
        self.edges_dist_disaster = graph.edges_dist_disaster  # distance of all edges to mean of all disaster zones
        self.edges_distance = graph.edges_distance # distance of all edges to destination edges
        self.disaster_zones = graph.disaster_zones  # disaster zones in the network

        # Create reward instance, passing self as the agents manager  
        self.reward = Reward(self)  

        # Initialize the shared info manager to allow sharing information between agents
        self.shared_info = GroupsSharedInfo(graph) 

        # agents are grouped by their destination
        # agents share the same destination, so they are in the same group
        self.groups = {} 
        # configs for save and load model
        self.saved_folder = os.path.join(os.path.dirname(__file__), '..', 'new_models') 

        # the dimension of state is index of [current edge],[feasible next edges], [flows of feasible next edges]
        # and [mean x disaster, mean y disaster, radius disaster]
        self.obs_dim = 1 + len(self.edges) + 1 # current edge + next feasible edges + density on current edge)
        self.action_dim = len(self.edges)  # action dimension based on the number of edges in the network
        
        ### create offlinedataset for RL if required
        # Create dataset
        try:        
            offline_data_path = self.CONFIG['offline_data_path']
            self.offline_dataset = OfflineRLDataset(filename=offline_data_path, 
                                           state_dim=self.obs_dim, action_dim=self.action_dim)
        except Exception:
            logger.warning("No offline data path for storing offline data collection")

        # init SUMO db to keep track of edge connectivity to destination edges
        self.SumoDB_instance = SumoDB()

        # Convert edges dict into a list
        # the index of the edge in the list is the encoded-number of the edge
        self.edges_list = list(self.edges.keys())
        self.max_vehicles_edge = self.CONFIG['max_vehicles_edge'] # maximum flows on in each edge
        self.max_travel_time = graph.max_travel_time  # maximum travel time for each edge
        self.destination_edges = self.CONFIG['destinations_edges']  # destination edges for the vehicles

        self.reward_type = self.CONFIG['reward_type'] # R1, R1R2
        # self.d_max = graph.max_edge_to_dest_distance  # maximum distance from an edge to edge distance for reward calculation
        self.w1 = self.CONFIG['reward']['R1R2']['w1']  # weight for travel time reward
        self.w2 = self.CONFIG['reward']['R1R2']['w2']  # weight distance reward

        self.d_weight = self.CONFIG['reward']['MSE']['d_weight']  # weight for distance reward
        self.max_speed_vehs = self.CONFIG['max_speed']  # maximum speed of vehicles on an edge

        self.dead_edge =  self.CONFIG['reward']['MSE']['dead_edge'] # penalty for dead edges
        self.disaster_info = graph.disaster_info # dict to hold the radius of each disaster zone
        self.dist_OD = graph.dist_OD # euclid distance of origin => dest
        
        # mode to activate/deactive disaster
        self.is_disaster_active = False  
        self.activation_prob = activation_prob # probability of activating disaster
        # plotter for visualizing agent performance.
        self.plotter = None  
        # set mode for training/ inference
        self.training = True
        self.offline_infer = False

        # Track timestep transitions for each group
        self.group_timestep_buffers = {}  # {destination: list of (vehicle_id, obs, action, next_obs, done)}
         # Track rewards for each group in CURRENT timestep
        self.current_timestep_rewards = {}  # {destination: [reward1, reward2, ...]}


        # --- NEW: History Tracking ---
        self.seq_len = 12 # Trajectory length S_{t-4}, S_{t-3} ... S_t
        self.rolling_histories = {} # {vehicle_id: deque}


    # --- NEW HISTORY HELPER FUNCTIONS ---
    def get_current_history(self, vehicle_id, current_obs):
        """Updates and returns the trajectory up to S_t"""
        obs_flat = current_obs.squeeze(0).clone().cpu()
        if vehicle_id not in self.rolling_histories:
            # Fill with zeros if the vehicle just spawned
            self.rolling_histories[vehicle_id] = deque([torch.zeros_like(obs_flat)] * self.seq_len, maxlen=self.seq_len)
        
        self.rolling_histories[vehicle_id].append(obs_flat)
        return torch.stack(list(self.rolling_histories[vehicle_id])) # [seq_len, obs_dim]

    def get_next_history_preview(self, vehicle_id, next_obs):
        """Previews S_{t+1} without mutating the buffer (needed for the target network)"""
        obs_flat = next_obs.squeeze(0).clone().cpu()
        temp_list = list(self.rolling_histories[vehicle_id])[1:] # Drop the oldest
        temp_list.append(obs_flat) # Append the future state
        return torch.stack(temp_list)

    def cleanup_vehicle_history(self, vehicle_id):
        if vehicle_id in self.rolling_histories:
            del self.rolling_histories[vehicle_id]

    def set_training(self, mode: bool) -> None:
        """Turn on/off training mode."""
        self.training = mode

    def set_offline_infer(self, mode: bool) -> None:
        self.offline_infer = True

    def random_active_disaster(self) -> None:
        """
        Randomly activate/deactivate a disaster episode.

        If the config did not contain disaster information, the flag
        will stay False and a debug line is emitted.
        """
        if not self.disaster_info:                         # no disaster in cfg
            logger.warning("random_active_disaster(): "
                         "no disaster data configured - cannot activate.")
            self.is_disaster_active = False
            return

        self.is_disaster_active = np.random.random() < self.activation_prob
        if self.is_disaster_active:
            logger.warning("Disaster ACTIVATED for this episode.")
        else:
            logger.warning("No disaster for this episode.")

    def get_nearest_disaster(self, next_edge: str):
        """
        Return (zone_id, distance) of the nearest disaster zone to `edge_id`
        or (None, None) when nothing is active / configured.
        """
        # 1. Config check -------------------------------------------------
        if not self.disaster_info:
            logger.info("get_nearest_disaster(): no disaster configured.")
            return None, None

        # 2. Episode state check -----------------------------------------
        if not self.is_disaster_active:
            logger.info("get_nearest_disaster(): disaster not active.")
            return None, None

        # 3. Distance map check ------------------------------------------
        if not self.edges_dist_disaster:
            logger.warning("get_nearest_disaster(): "
                           "distance lookup table empty.")
            return None, None

        if next_edge not in self.edges_dist_disaster:
            logger.warning("get_nearest_disaster(): edge_id '%s' unknown.", next_edge)
            return None, None

        # 4. Retrieve nearest zone ---------------------------------------
        disaster_distances = self.edges_dist_disaster[next_edge]  
        # Find minimum distance and nearest zone
        min_distance = min(disaster_distances.values())  
        nearest_zone = [k for k, v in disaster_distances.items() if v == min_distance][0] 
        # return nearest disaster zone id and the distance to it 
        return nearest_zone, min_distance  
    
    def divide_trips_to_agent_groups(self):
        """Initialize groups"""
        for trip_id, trip_info in self.trips.items():
            destination = trip_info['to']
            
            if destination not in self.groups:
                self.groups[destination] = Agent(
                    self.obs_dim, 
                    self.action_dim, 
                    group_id=destination
                )
                self.current_timestep_rewards[destination] = []
                self.group_timestep_buffers[destination] = []
            
            # self.groups[destination].add_vehicle(trip_id)

    # Assuming this is in your multi_agents class (containing step, collect_offline_data, etc.)  
    def batch_step(self, vehicles_to_route, disaster_move=None):  
        """  
        Batch process routing decisions for multiple vehicles.  
        
        vehicles_to_route: List of (vehicle_id, curr_edge, connected_edges) tuples  
        disaster_move: disaster simulation object  
        
        Returns: List of (next_edge, append_route) tuples (one per vehicle)  
        """  
        if not vehicles_to_route:  
            return []  
        
        # ============================================================  
        # Collect per-vehicle data and group by destination (agent group)  
        # ============================================================  
        # Structure: {dest_edge: {'vehicles': [...], 'observations': [...], ...}}  
        group_batches = defaultdict(lambda: {  
            'vehicles': [],           # List of (vehicle_id, curr_edge, connected_edges, original_idx)  
            'observations': [],  
            'encoded_disasters': [],  
            'feasible_actions': [],  
        })  
        
        skipped_indices = set()  # Track which original indices were skipped  
        
        for original_idx, (vehicle_id, curr_edge, connected_edges) in enumerate(vehicles_to_route):  
            dest_edge = self.trips[vehicle_id]['to']  
            
            # Create observation (preserves logic from step)  
            try:  
                encoded_observation = self.encode_oberservation(curr_edge, connected_edges)  
                encoded_disaster_obs = self.get_disaster_obs(curr_edge)  
            except Exception as e:  
                logger.error(f"Error creating observation for vehicle {vehicle_id}: {e}")  
                skipped_indices.add(original_idx)  
                continue  # Skip if observation fails (preserves logic)  
            
            # Feasible next actions (preserves logic from step)  
            feasible_actions = [  
                1 if (edge in connected_edges and   
                    (not self.is_dead_edge(edge) or edge in self.destination_edges))  
                else 0   
                for edge in self.edges_list  
            ]  
            feasible_actions = self.filter_edges_not_connected_dest(feasible_actions, dest_edge)  
            
            if not any(feasible_actions):  
                logger.warning(f"All feasible actions from {curr_edge} is empty")  
                skipped_indices.add(original_idx)  
                continue  # Skip if no actions (preserves logic)  
            
            # Collect into group batch  
            group_batches[dest_edge]['vehicles'].append((vehicle_id, curr_edge, connected_edges, original_idx))  
            group_batches[dest_edge]['observations'].append(encoded_observation)  
            group_batches[dest_edge]['encoded_disasters'].append(encoded_disaster_obs)  
            group_batches[dest_edge]['feasible_actions'].append(feasible_actions)  
        
        # Initialize results array with None for all vehicles  
        full_results = [(None, None) for _ in vehicles_to_route]  
        
        if not group_batches:  
            return full_results  # All skipped  
        
        # ============================================================  
        # Process each agent group separately  
        # ============================================================  
        for dest_edge, batch_data in group_batches.items():  
            group_agent = self.groups[dest_edge]  
            
            observations_list = batch_data['observations']  
            encoded_disaster_list = batch_data['encoded_disasters']  
            feasible_actions_list = batch_data['feasible_actions']  
            vehicles = batch_data['vehicles']  
            
            batch_size = len(observations_list)  
            if batch_size == 0:  
                continue  
            
            # ============================================================  
            # Stack tensors for batch processing (per group)  
            # ============================================================  
            observations_batch = torch.cat(observations_list, dim=0)  # [batch_size, obs_dim]  
            
            if all(ed is None for ed in encoded_disaster_list):  
                encoded_disaster_batch = None  
            else:  
                encoded_disaster_batch = torch.cat(  
                    [ed if ed is not None else torch.zeros(1, self.dis_dim) for ed in encoded_disaster_list],   
                    dim=0  
                )  
            
            # ============================================================  
            # Batch action selection (SINGLE GPU CALL per group)  
            # ============================================================  
            actions = group_agent.batch_softmax_select_action(  
                observations_batch,  
                feasible_actions_list,  
                encoded_disaster_batch  
            )  
            
            # ============================================================  
            # Per-vehicle: Execute actions, store buffers/rewards  
            # Cache reward/append_route for same (group, curr_edge) to save computation  
            # ============================================================  
            # Cache structure: {curr_edge: (reward, append_route, next_edge)} for this group  
            reward_cache = {}  
            
            for i, (vehicle_id, curr_edge, connected_edges, original_idx) in enumerate(vehicles):  
                action = actions[i]  
                if action is None:  
                    full_results[original_idx] = (None, None)  
                    continue  
                
                try:  
                    if group_agent.training:  
                        next_edge = self.edges_list[action]  
                        
                        # Check cache: same group + same curr_edge + same action => same reward/append_route  
                        cache_key = (curr_edge, action)  
                        
                        if cache_key in reward_cache:  
                            # Reuse cached reward and append_route  
                            individual_reward, append_route = reward_cache[cache_key]  
                            logger.info(f"Vehicle {vehicle_id} (cached) took action: {next_edge}, received reward: {individual_reward}")  
                        else:  
                            # Compute reward and append_route, then cache  
                            individual_reward, append_route = self.reward.get_reward_5(  
                                curr_edge, next_edge, dest_edge, vehicle_id, feasible_actions_list[i]  
                            )  
                            reward_cache[cache_key] = (individual_reward, append_route)  
                            logger.info(f"Vehicle {vehicle_id} took action: {next_edge}, received reward: {individual_reward}")  
                        
                        encoded_next_observation = self.encode_next_oberservation(next_edge)  
                        done = (next_edge in self.destination_edges)  
                        
                        # Store in temporary buffer for this timestep  
                        self.group_timestep_buffers[dest_edge].append(  
                            (vehicle_id, observations_list[i], action, encoded_next_observation, done, encoded_disaster_list[i])  
                        )  
                        
                        # Accumulate reward for this timestep of current agent group  
                        self.current_timestep_rewards[dest_edge].append(individual_reward)  
                    else:  
                        # inference mode
                        next_edge = self.edges_list[action]
                        append_route = self.inference_get_append_route(next_edge, dest_edge)

                except Exception as e:  
                    logger.error(f"Error selecting/executing action for vehicle {vehicle_id}: {e}")  
                    next_edge = None  
                    append_route = None  
                
                full_results[original_idx] = (next_edge, append_route)  
        
        return full_results  


    def step(self, vehicle_id, curr_edge, connected_edges, disaster_move=None, episode_checker=None):
        """Step for individual vehicle"""
        dest_edge = self.trips[vehicle_id]['to']
        group_agent = self.groups[dest_edge]

        ############## Obsevation
        try:
            # Create observation for the agent (e.g., current node, travel time)
            encoded_observation = self.encode_oberservation(curr_edge, connected_edges) 
            encoded_disaster_obs = self.get_disaster_obs(curr_edge)
        except Exception as e:
            logger.error(f"Error creating observation for vehicle {vehicle_id}: {e}")

        ##############  feasbile next actions 
        feasible_actions = [
            1 if (edge in connected_edges and 
                  (not self.is_dead_edge(edge) or edge in self.destination_edges))
            else 0 
            for edge in self.edges_list
        ]
        
        # mask out next edge options: remove edges that are not connected to dest edge
        feasible_actions = self.filter_edges_not_connected_dest(feasible_actions, dest_edge)

        if not any(feasible_actions):
            logger.warning(f"All feasible actions from {curr_edge} is empty")
            return

        # NEW: Fetch history S_1...S_t
        # Get history to feed to the network
        hist_obs = self.get_current_history(vehicle_id, encoded_observation)

        try: 
            # Make sure this takes hist_obs now!
            action = group_agent.softmax_select_action(encoded_observation, feasible_actions, encoded_disaster_obs, hist_obs)

            # indices = np.where(np.array(feasible_actions) == 1)[0]
            # action = np.random.choice(indices)

        except Exception as e:
            logger.error(f"Error selecting action for vehicle {vehicle_id}: {e}")  
        

        ##############  Excecute action => receive reward and next obsevation 
        ############## Or do inference of action 
        if self.training:
            next_edge, individual_reward, append_route = self.execute_action(
                action, curr_edge, dest_edge, vehicle_id, feasible_actions, disaster_move, episode_checker
            )
            
            encoded_next_observation = self.encode_next_oberservation(next_edge)
            done = (next_edge in self.destination_edges)
            
            # --- NEW: Get the current and next histories ---
            hist_obs = self.get_current_history(vehicle_id, encoded_observation)
            next_hist_obs = self.get_next_history_preview(vehicle_id, encoded_next_observation)
            
            # --- MODIFIED: Store 8 items instead of 6 ---
            self.group_timestep_buffers[dest_edge].append(
                (vehicle_id, encoded_observation, action, encoded_next_observation, done, encoded_disaster_obs, hist_obs.numpy(), next_hist_obs.numpy())
            )

            # Accumulate reward for this timestep of current agent group
            self.current_timestep_rewards[dest_edge].append(individual_reward)
            return next_edge, append_route
        else:
            next_edge = self.edges_list[action]
            append_route = self.inference_get_append_route(next_edge, dest_edge)
            return next_edge, append_route
        
    def end_timestep(self, time_step, step_train=2):
        """
        Call this at the end of each simulation timestep.
        Stores joint transitions and triggers training.
        """
        if not self.training:
            return
        
        start_time = time.time()
        for destination, group_agent in self.groups.items():
            # Get all transitions for this group in current timestep

            try:
                transitions = self.group_timestep_buffers[destination]
            except Exception as e:
                logger.error(f"No transition for group {destination}: {e}. Skipping.")
                continue
            
            if len(transitions) > 0:
                # Calculate global reward for this group
                global_reward = self.calculate_group_reward(destination)
                logger.debug(f"Global reward for group {destination}: {global_reward}")
                
                # # Store joint transition
                # for vehicle_id, obs, action, next_obs, done, disaster_obs in transitions:
                #     group_agent.store_transition_timestep(vehicle_id, obs, action, next_obs, done, disaster_obs)

                # NEW: Unpack with hist_obs and next_hist_obs
                for vehicle_id, obs, action, next_obs, done, disaster_obs, h_o, nh_o in transitions:
                    group_agent.store_transition_timestep(vehicle_id, obs, action, next_obs, done, disaster_obs, h_o, nh_o)
                
                # Store joint transition with global reward
                group_agent.end_timestep(global_reward)
                
                # we train the network every 10 steps
                if time_step % step_train == 0:
                    # Update/training network with backpropagation
                    loss = group_agent.update(plotter=self.plotter)
                    logger.debug(f"Training loss for group {destination}: {loss}")
                
                if self.plotter:
                    self.plotter.record_reward(destination, global_reward)
            
            # Clear buffer for next timestep
            self.group_timestep_buffers[destination] = []
        end_time = time.time()
        duration = end_time-start_time
        # print(f"Training takes: {duration}s")


    # def end_timestep(self, time_step, step_train=4):
    #     """
    #     Call this at the end of each simulation timestep.
    #     Stores joint transitions (SHARED across all groups) and triggers training.
    #     """
    #     if not self.training:
    #         return
        
    #     start_time = time.time()

    #     # --- STEP 1: Aggregate data from ALL groups ---
    #     all_transitions = []
    #     for dest, transitions in self.group_timestep_buffers.items():
    #         all_transitions.extend(transitions)

    #     # If no data anywhere, skip
    #     if len(all_transitions) == 0:
    #         return

    #     # --- STEP 2: Distribute shared data to each agent and Train ---
    #     for destination, group_agent in self.groups.items():
            
    #         # Calculate global reward for this SPECIFIC group
    #         # (Even though data is shared, the reward usually remains specific to the agent's goal)
    #         global_reward = self.calculate_group_reward(destination)
    #         logger.debug(f"Global reward for group {destination}: {global_reward}")
            
    #         # Store ALL transitions (from every group) into this agent
    #         for vehicle_id, obs, action, next_obs, done, disaster_obs in all_transitions:
    #             group_agent.store_transition_timestep(vehicle_id, obs, action, next_obs, done, disaster_obs)
            
    #         # Commit the timestep with the reward
    #         group_agent.end_timestep(global_reward)
            
    #         # --- Training Logic ---
    #         loss = None
    #         if time_step % step_train == 0:
    #             # Update/training network with backpropagation
    #             loss = group_agent.update(plotter=self.plotter)
            
    #         if loss and self.plotter:
    #             self.plotter.record_reward(destination, global_reward)
        
    #     # --- STEP 3: Clear buffers (only after everyone has collected the data) ---
    #     for destination in self.group_timestep_buffers:
    #         self.group_timestep_buffers[destination] = []

    #     end_time = time.time()
    #     duration = end_time - start_time
    #     # print(f"Training takes: {duration}s")

    def reset_timestep_rewards(self):
        """Reset reward buffers (call at start of new episode or timestep)"""
        for destination in self.current_timestep_rewards:
            self.current_timestep_rewards[destination] = []

    def calculate_group_reward(self, destination):
        """
        Calculate global reward as average of rewards from all vehicles 
        in this group that acted in the current timestep.
        """
        if destination not in self.current_timestep_rewards:
            return 0.0
        
        rewards = self.current_timestep_rewards[destination]
        
        if len(rewards) == 0:
            return 0.0
        
        # Average reward across all vehicles in group for this timestep
        avg_reward = np.mean(rewards)
        
        return avg_reward
    
    def collect_offline_data(self, vehicle_id, curr_edge, connected_edges, disaster_move):
        """
        Collect a single offline transition (S, A, R, S', group_id done)
        where action is the next edge chosen by an A* route.
        The transition is stored in the dataset.
        """
         
        dest_edge = self.trips[vehicle_id]['to']  # destination edge
        feasible_actions = [
            1 if (edge in connected_edges and 
                (not self.is_dead_edge(edge) or edge in self.destination_edges))
            else 0
            for edge in self.edges_list
        ]
        feasible_actions = self.filter_edges_not_connected_dest(feasible_actions, dest_edge)

        if not any(feasible_actions):
            logger.error(f"No feasible actions for vehicle {vehicle_id} on edge {curr_edge}")
            return None

        try:
            # Create observation for the agent (e.g., current node, travel time)
            encoded_observation = self.encode_oberservation(curr_edge, connected_edges) 
        except Exception as e:
            logger.error(f"Offline training. Error creating observation for vehicle {vehicle_id}: {e}")

        # --- Action selection: from A* path instead of RL agent ---
        # this is human imitation learning. Because we collect offline data from expert A* path
        # we first apply new edges weights to all edges (new edge weight = edge travel time + edge risk)
        # self.apply_new_edges_weight(disaster_move)
        route = traci.simulation.findRoute(curr_edge, dest_edge, routingMode=traci.constants.ROUTING_MODE_AGGREGATED)
        try:
            if len(route.edges) != 0:  
                next_edge = route.edges[1]  # take the first edge in the route as the next edge. the zero edge is current edge
            else:
                logger.error(f"A* could not find a route from {curr_edge} to {dest_edge} for vehicle {vehicle_id}")
                return None
        except Exception as e:
            logger.exception(f"The error route is {route.edges}. Error {e}")   
        
        encoded_next_observation = self.encode_next_oberservation(next_edge)
        action = self.edges_list.index(next_edge)  # map edge → discrete index
        
        # reward = self.compute_reward(curr_edge, next_edge, dest_edge, route, disaster_move) 
        # reward is current edge cost (already included travel time + risk)
        reward = -traci.edge.getTraveltime(next_edge) # reward should be reverse of cost

        # reward = self.reward._get_reward_5(curr_edge, next_edge, dest_edge, vehicle_id, feasible_actions)
        logger.info(f"Vehicle {action} took action: {next_edge}, received reward: {reward}")

        # --- Done flag ---
        done = next_edge in self.destination_edges

        # --- Store transition in HDF5 dataset ---
        # vehicles are grouped based on their similar destination => same destination => same group id
        group_id = dest_edge
        self.offline_dataset.append(encoded_observation, action, reward, encoded_next_observation, group_id, done)

        return next_edge
    
    def apply_new_edges_weight(self, disaster_move):
            """
            Reads all edge data, calculates new weights using NumPy vectorization,
            and writes them back to the simulation in a batch.
            """
            edge_ids = self.edges

            # ===================================================================
            # 1. READ PHASE: Gather all data from the simulation first
            # ===================================================================
            # Get all travel times and risk values. These loops are for I/O,
            # not calculation. This minimizes back-and-forth with SUMO.
            
            # Note: traci.edge.getTraveltime doesn't have a batch version, so
            # a list comprehension is the most efficient way to call it.
            travel_times_list = [traci.edge.getTraveltime(edge_id) for edge_id in edge_ids]

            # Process risks, handling cases where a risk might be None or 0.
            # The list comprehension directly translates your original 'if not edge_risk' logic.
            risks_list = [disaster_move.get_risk_edge(edge_id) for edge_id in edge_ids]
            
            # Convert the Python lists into efficient NumPy arrays for calculation.
            travel_times = np.array(travel_times_list, dtype=np.float64)
            
            # This comprehension handles the `if not edge_risk: edge_risk = 0` logic.
            # It correctly processes values that are `None`, `0`, or `False`.
            risks = np.array([r if r else 0.0 for r in risks_list], dtype=np.float64)

            # ===================================================================
            # 2. CALCULATE PHASE: Use fast, vectorized NumPy operations
            # ===================================================================
            # This single line replaces the entire calculation loop. It operates on
            # all elements of the arrays at once, executed in highly optimized C code.
            new_weights = self.w1 * travel_times + self.w2 * risks

            # ===================================================================
            # 3. WRITE PHASE: Apply all changes back to the simulation
            # ===================================================================
            # Now, loop one last time just to send the results to SUMO.
            for i, edge_id in enumerate(edge_ids):
                # We use 'i' to get the corresponding weight from our calculated array.
                traci.edge.adaptTraveltime(edge_id, new_weights[i])
                # logger.debug(f"New cost for edge {edge_id}: {new_weights[i]}")
        
    def compute_reward(self, curr_edge, next_edge, dest_edge, route, disaster_move):

        # reward = self.reward.get_reward_5(curr_edge, next_edge, 
                                        #   dest_edge, vehicle_id, feasible_actions)

        reward = self.reward.get_reward_offline_training(curr_edge, next_edge, dest_edge, 
                                    route, disaster_move)
        return reward 


    def encode_oberservation(self, curr_edge, connected_edges):
        """
        Transform the observation into a tensor format
        """
        return self.encode(curr_edge, connected_edges)

    def encode_next_oberservation(self, next_edge):
        """
        Transform then next observation into a tensor format
        """
        return self.encode(next_edge, connected_edges=self.edges[next_edge]["connected_edges"])

    def encode(self, curr_edge, connected_edges):
        """
        Encode the state of the current agent (obs):
        - [0] : Index of current edge
        - [1 : 1+num_edges] : one-hot connectivity edges from current edge
        
        then concat with [density of current edge] (obs^)

        Returns:
            obs (np.ndarray): state vector of fixed length
        """

        # Initialize state vector
        # we take obs dim -1 because we will concat with obs^ later
        obs = torch.zeros(self.obs_dim-1, dtype=torch.float32)

        # Encode current edge index (normalized)
        if curr_edge in self.edges_list:
            edge_idx = self.edges_list.index(curr_edge)
            obs[0] = edge_idx 
        else:
            logger.error(f"{curr_edge} not in edge list for encoding")
            obs[0] = -1.0  # use -1 if not on a known edge

        # Encode feasible edges (one-hot)
        for edge_id in connected_edges:
            if edge_id in self.edges_list:
                idx = self.edges_list.index(edge_id)
                obs[1 + idx] = 1.0

        # Encode obs^ - density on current edge
        num_vehicles = len(traci.edge.getLastStepVehicleIDs(curr_edge))  # Note: using curr_edge, not edge_id
        if num_vehicles > self.max_vehicles_edge:
            norm_flow = 1.0
        else:
            norm_flow = num_vehicles / self.max_vehicles_edge if self.max_vehicles_edge > 0 else 0.0

        # Create obs^ as a tensor
        obs_hat = torch.tensor([norm_flow], dtype=torch.float32)
        
        # Concatenate obs and obs^
        full_obs = torch.cat([obs, obs_hat], dim=0)
        
        return full_obs.unsqueeze(0)  # convert from 1D to 2D tensor (1, obs_dim)
    
    def get_disaster_obs(self, curr_edge):
        """
        Return a tensor MxD, where M is the number of disaster zones, D is the number of disaster features
        we can add binary mask/channel or full -1 tensors to indicate if there is no disaster zones
        """
        disaster_ids = list(self.disaster_info.keys())
        # if no disasters
        if len(disaster_ids) == 0:
            logger.warning("get_disaster_state(): no disaster zones configured.")
            return None

        # 2d vector
        distances = np.array([
                [self.edges_dist_disaster[curr_edge][disaster_id] ]
                for disaster_id in disaster_ids
        ]) # (4,))

        return torch.tensor(distances, dtype=torch.float32).unsqueeze(0) # convert from 2D to 3D tensor 
        ########### Add disaster info to the state: (mean x, mean y) and radius of disaster 
        # mean_x, mean_y = 0.0, 0.0
        # radius = 0.0 
        # nearest_zone, _ = self.get_nearest_disaster(curr_edge)  
        # if nearest_zone:
        #     mean_x, mean_y = self.disaster_info[nearest_zone]['coordinate']   
        #     radius = self.disaster_info[nearest_zone]['radius']  dius of disaster 
        # mean_x, mean_y = 0.0, 0.0
        # radius = 0.0 
        # nearest_zone, _ = self.get_nearest_disaster(curr_edge)  
        # if nearest_zone:
        #     mean_x, mean_y = self.disaster_info[nearest_zone]['coordinate']   
        #     radius = self.disaster_info[nearest_zone]['radius']  

        # ########## TODO: Normalize the state disaster
        # # Fixed indices for disaster information
        # disaster_start_idx = 1 + 2 * len(self.edges_list)  
        # self.obs[disaster_start_idx] = mean_x  
        # self.obs[disaster_start_idx + 1] = mean_y  
        # self.obs[disaster_start_idx + 2] = radius  
        # logger.error(f"disaster info state {mean_x} {mean_y} {radius}")

        #TODO: Add Disaster info to the state (disaster radius/distance to disaster)
        # nearest_zone, dist_disaster = self.get_nearest_disaster(curr_edge)
        # if nearest_zone:
        #     radius = self.disaster_info[nearest_zone]['radius']
        # disaster_state_index = 1 + 2 * len(self.edges_list)
        # disaster_state = 1 / (1+ np.sqrt( (radius/dist_disaster)**2 ))
        # self.obs[disaster_state_index] = disaster_state
        # logger.error(f"disaster info state {disaster_state}")

        # return self.obs.unsqueeze(0)  # convert from 1D to 2D tensor
        
    def execute_action(self, action, curr_edge, dest_edge, vehicle_id, feasible_actions,
                       disaster_move=None, episode_checker=None):
        """
        The agent executes the action in the environment and receives the next state and reward.
        The action is the index of next edge to traverse.
        """
        next_edge = self.edges_list[action]
        # reward = self.get_reward_MSE(next_edge, dest_edge)
        # reward = self.get_reward_based_d_weight(next_edge, dest_edge)
        # reward = self.get_reward_risk_model(next_edge, dest_edge)
        # reward = self.reward.get_reward_risk_model(next_edge, dest_edge)
        # reward = self.reward.get_reward_1(curr_edge, next_edge, dest_edge, vehicle_id, feasible_actions)
        # reward = self.reward.get_reward_3(curr_edge, next_edge, dest_edge, vehicle_id, feasible_actions)
        # reward = self.reward.get_reward_4(curr_edge, next_edge, dest_edge, vehicle_id, feasible_actions)
        # reward = self.reward.get_reward_5(curr_edge, next_edge, dest_edge, vehicle_id, feasible_actions)
        # if self.CONFIG['offline_training']: 
        #     reward = self.reward.get_reward_offline_training(curr_edge, next_edge, dest_edge, disaster_move)
        #     logger.success("Take reward using offline training")
        # else:
        reward, append_route = self.reward.get_reward_5(curr_edge, next_edge, dest_edge, vehicle_id, feasible_actions, episode_checker)
        logger.info(f"Vehicle {vehicle_id} took action: {next_edge}, received reward: {reward}")

        return next_edge, reward, append_route 

    def is_dead_edge(self, next_edge):
        """
        Check if the action chosen is a dead edge
        """
        # logger.debug(f"check {next_edge} is a dead edge")
        # logger.debug("connected edges of this edge: ", self.edges[next_edge]["connected_edges"])
        # there is no connectecs for the next edge, and next edge is not a destination edge
        if not self.edges[next_edge]["connected_edges"] and next_edge not in self.destination_edges:
            logger.warning(f"Check Dead edge: {next_edge}")
            return True
        return False      
    
    def inference_get_append_route(self, next_edge, dest_edge):
        return self.SumoDB_instance.get_append_route(next_edge, dest_edge)

    def filter_edges_not_connected_dest(self, feasible_actions, dest_edge):
        """
        Filter out next feasible actions that do not lead to the destination edge.
        feasible_actions: list[int]  # 1 means allowed, 0 means not allowed
        dest_edge: str               # target edge ID
        """
        # Find indices where the action is feasible (== 1)
        feasible_idxs = [i for i, v in enumerate(feasible_actions) if v == 1]

        a = np.where(np.array(feasible_actions) == 1)[0]
        logger.warning(f"Before filtering, feasible actions indices: {[self.edges_list[idx] for idx in a]}")

        # new result
        new_feasible_actions = list(feasible_actions)

        for idx in feasible_idxs:
            next_edge = self.edges_list[idx]

            check = self.SumoDB_instance.check_edge_connected_dest(next_edge, dest_edge)
            if not check:
                new_feasible_actions[idx] = 0
                logger.warning(f"Filter out edge {next_edge} not connected to dest {dest_edge}")

        b = np.where(np.array(new_feasible_actions) == 1)[0]
        logger.warning(f"After filtering, feasible actions indices: {[self.edges_list[idx] for idx in b]}")

        return new_feasible_actions

    def penalty_agents_loops(self, not_completed_trips, episode_checker):
        """
        Apply penalty for agents if they cannot reach their destination within the maximum simulation time.
        """
       # if current mode is training, we store the exp and update policy
        if self.training:
            for vehicle_id in not_completed_trips:

                dest_edge = self.trips[vehicle_id]['to']  # Get the destination edge of the vehicle
                agent = self.groups[dest_edge]  # Get the agent for the vehicle's group

                # Get the history edges of the vehicle
                history_edges = episode_checker.get_history_edges_trip(vehicle_id)
                if not history_edges:
                    logger.warning(f"No history edges found for vehicle {vehicle_id}. Skipping penalty.")
                    continue
                else:
                    curr_edge = history_edges[-1]
                    previous_edge = history_edges[-2] if len(history_edges) > 1 else None

                if not previous_edge:
                    logger.warning(f"No previous edge found for vehicle {vehicle_id}. Skipping penalty.")
                    continue

                # Store experience for training
                encoded_current_observation = self.encode_next_oberservation(curr_edge)
                encoded_previous_observation = self.encode_next_oberservation(previous_edge)

                # action is the index of current edge
                action = self.edges_list.index(curr_edge) if curr_edge in self.edges_list else -1
                if action == -1:
                    logger.warning(f"Current edge {curr_edge} not found in edges list for vehicle {vehicle_id}. Skipping penalty.")
                    continue
                
                # we take the reward as the -T (negative expected time to dest)
                # find route from selected next edge to destination edge
                logger.debug(f"Reward type {self.reward_type}")
                route = traci.simulation.findRoute(curr_edge, dest_edge, routingMode=traci.constants.ROUTING_MODE_AGGREGATED)
                if len(route.edges) != 0:  
                    travel_time = route.travelTime

                    if self.reward_type == "R1":
                        reward = self.reward.get_penalty_loop_R1(travel_time)
                    elif self.reward_type == "R1R2":
                        reward = self.reward.get_penalty_loop_R1R2(travel_time, curr_edge, dest_edge, vehicle_id)
        
                    logger.debug(f"Penalty agent loop: {reward} from edge {previous_edge}-> edge {curr_edge}")
                else:
                    
                    if self.reward_type == "R1":
                        reward = self.reward.get_penalty_loop_R1_in_dead_edge(self.CONFIG['simulation_steps'])
                    elif self.reward_type == "R1R2":
                        reward = self.reward.get_penalty_loop_R1R2_in_dead_edge()

                    logger.debug(f"Penalty agent loop for dead edge: {reward} from edge {previous_edge}-> edge {curr_edge}")

                agent.replay_buffer.push((encoded_previous_observation, action, encoded_current_observation, reward))

                # Update the agent's policy
                # agent.update(self.plotter)

    def add_reward_reaching_dest(self, completed_trips, episode_checker):
        """
        Apply reward for agent as the travel time when they complete the trips
        """
       # if current mode is training, we store the exp and update policy
        if self.training:
            for vehicle_id in self.trips.keys():
                
                if vehicle_id not in completed_trips:
                    continue

                dest_edge = self.trips[vehicle_id]['to']  # Get the destination edge of the vehicle
                agent = self.groups[dest_edge]['agent']  # Get the agent for the vehicle's group

                # Get the history edges of the vehicle
                history_edges = episode_checker.get_history_edges_trip(vehicle_id)
                if not history_edges:
                    logger.warning(f"No history edges found for vehicle {vehicle_id}. Skipping penalty.")
                    continue
                else:
                    curr_edge = history_edges[-1]
                    previous_edge = history_edges[-2] if len(history_edges) > 1 else None

                if not previous_edge:
                    logger.warning(f"No previous edge found for vehicle {vehicle_id}. Skipping penalty.")
                    continue

                # Store experience for training
                encoded_current_observation = self.encode_next_oberservation(curr_edge)
                encoded_previous_observation = self.encode_next_oberservation(previous_edge)

                # action is the index of current edge
                action = self.edges_list.index(curr_edge) if curr_edge in self.edges_list else -1
                if action == -1:
                    logger.error(f"Current edge {curr_edge} not found in edges list for vehicle {vehicle_id}. Skipping penalty.")
                    continue
                
                # we take the reward as the travel time when agemt complete the trip
                removed_time = episode_checker.get_time_vehicle_removed(vehicle_id)
                # depart_time = traci.vehicle.getDeparture(vehicle_id)
                depart_time = self.trips[vehicle_id]['depart']  # Get the departure time of the vehicle
                reward = depart_time - removed_time
                logger.debug(f"Add reward {reward} for vehicle {vehicle_id} reaching destination")

                agent.replay_buffer.push((encoded_previous_observation, action, encoded_current_observation, reward))

                # Update the agent's policy
                agent.update(self.plotter)

    def initialize_plotter(self, max_episodes):
        """
        Initialize the plotter for visualizing agent performance.
        """
        self.plotter = MultiAgentPlotter(list(self.groups.keys()), max_episodes)  # Initialize the plotter

    def save_arrived_vehicles(self, arrived_vehicles):
        self.plotter.write_arrived_vehicles(arrived_vehicles)

    def save_all_models(self, episode=None):  
        """Save models for all agents in all groups"""  
        os.makedirs(self.saved_folder, exist_ok=True)  
        
        # Create temp directory if it doesn't exist  
        os.makedirs(self.saved_folder+"/temp", exist_ok=True)  
        
        # Read or create temp file for this episode  
        temp_file_path = self.saved_folder+f"/temp/temp_{episode}.txt"  
        try:  
            # Try to read existing index  
            with open(temp_file_path, 'r') as f:  
                N = int(f.read().strip())  
        except (FileNotFoundError, ValueError):  
            # Create new file with index 0 if file doesn't exist or has invalid content  
            N = 0  
            with open(temp_file_path, 'w') as f:  
                f.write(str(N))  
            
        # Increment index  
        index = N + 1  
        
        # Write the incremented index back to the file  
        with open(temp_file_path, 'w') as f:  
            f.write(str(index))  
        
        # Create a metadata file for this save session  
        metadata = {  
            'episode': episode,  
            'index': index,  
            'groups': {}  
        }  
        
        # Save each agent's model  
        for dest_edge, group in self.groups.items():  
            agent = group
            # Create model directory if it doesn't exist  
            os.makedirs(self.saved_folder+"/model", exist_ok=True) 
            model_path = f"{self.saved_folder}/model/agent_{dest_edge}_ep{episode}_{index}.pt"  
            agent.save_model(model_path)  
            
            # Record in metadata  
            metadata['groups'][dest_edge] = {  
                'model_path': model_path
      
                # 'avg_reward': self.plotter.get_avg_reward(agent.group_id) if self.plotter else None  
            }  
      
        # Create meta directory if it doesn't exist  
        os.makedirs(self.saved_folder+"/meta", exist_ok=True)  
        with open(f"{self.saved_folder}/meta/metadata_ep{episode}_{index}.json", 'w') as f:  
            json.dump(metadata, f, indent=4)  
            
        logger.success(f"All models saved. Episode: {episode}, Index: {index}")  
        return str(index)  # Return index as string instead of timestamp

    def load_models(self, index=None, episode=None, eval_mode=True, fine_tune_mode=False):  
        """  
        Load saved models for all agents  

        Args:  
            index: Specific index to load models from (matches the new saving logic)  
            episode: Specific episode to load models from  
            eval_mode: Whether to set agents to evaluation mode  
        """  
        # Ensure folder exists  
        if not os.path.exists(self.saved_folder):  
            logger.success(f"Saved models folder {self.saved_folder} not found")  
            return False  

        # Find the metadata file based on criteria  
        metadata_path = None  

        # If index and episode are provided (primary lookup, matches new save pattern)  
        if index and episode:  
            metadata_path = f"{self.saved_folder}/meta/metadata_ep{episode}_{index}.json"  

        # If only index is provided  
        elif index:  
            files = glob.glob(f"{self.saved_folder}/meta/metadata_ep*_{index}.json")  
            if files:  
                metadata_path = files[0]  

        # If only episode is provided  
        elif episode:  
            files = glob.glob(f"{self.saved_folder}/meta/metadata_ep{episode}_*.json")  
            if files:  
                # Sort by modification time (newest first)  
                files.sort(key=lambda x: os.path.getmtime(x), reverse=True)  
                metadata_path = files[0]  

        # Find the latest metadata file if neither is provided  
        else:  
            files = glob.glob(f"{self.saved_folder}/meta/metadata_ep*.json")  
            if files:  
                files.sort(key=lambda x: os.path.getmtime(x), reverse=True)  
                metadata_path = files[0]  

        if not metadata_path or not os.path.exists(metadata_path):  
            logger.warning(f"No saved models found matching criteria: index={index}, episode={episode}")  
            return False  

        # Load metadata  
        try:  
            with open(metadata_path, 'r') as f:  
                metadata = json.load(f)  
        except json.JSONDecodeError:  
            logger.error(f"Failed to parse metadata file {metadata_path}")  
            return False  

        logger.info(f"Loading models from: {metadata_path}")  
        logger.info(f"Episode: {metadata['episode']}, Index: {metadata.get('index', 'unknown')}")  # updated key  

        # # Load each agent's model  
        # success = True  
        # for agent_edge, group_meta in metadata['groups'].items():  
        #     if agent_edge in self.groups:  
        #         model_path = group_meta['model_path']  
        #         if os.path.exists(model_path):  
        #             logger.info(f"Loading model for agent {agent_edge} from {model_path}")  
        #             agent = self.groups[agent_edge]  
        #             if agent.load_model(model_path) and eval_mode:  
        #                 agent.set_eval_mode()  
        #                 logger.info("Set inference mode for models successuly")
        #         else:  
        #             logger.error(f"Model file not found: {model_path}")  
        #             success = False  
        #     else:  
        #         logger.warning(f"Group {agent_edge} not found in current simulation.")  
        success = True
        for agent_edge, group_meta in metadata['groups'].items():
            if agent_edge in self.groups:
                model_path = group_meta['model_path']
                if os.path.exists(model_path):
                    logger.info(f"Loading model for agent {agent_edge} from {model_path}")
                    agent = self.groups[agent_edge]

                    # NEW: pass fine_tune_mode flag
                    if agent.load_model(model_path, fine_tune_mode=fine_tune_mode):
                        if fine_tune_mode:
                            logger.info("Model loaded in fine-tune mode.")
                        elif eval_mode:
                            agent.set_eval_mode()
                            logger.info("Set inference mode for models successfully.")
                else:
                    logger.error(f"Model file not found: {model_path}")
                    success = False
            else:
                logger.warning(f"Group {agent_edge} not found in current simulation.")

        return success
    
    def load_offline_models(self, eval_mode=True):

        model_dir = self.CONFIG['offline_models_path']
        # Ensure folder exists  
        if not os.path.exists(model_dir):  
            logger.error(f"Saved offline models folder {model_dir} not found")  
            return
        
        pattern = re.compile(r'^(?P<agent_name>.+)_eps_(?P<eps>\d+)_time_.*$')

        # find models with latest episodes
        latest_models = {}
        for filename in os.listdir(model_dir):
            match = pattern.match(filename)
            if match:
                agent_name = match.group('agent_name')
                eps = int(match.group('eps'))
                filepath = os.path.join(model_dir, filename)

                # Keep only the model with the highest eps
                if agent_name not in latest_models or eps > latest_models[agent_name]['eps']:
                    latest_models[agent_name] = {'eps': eps, 'path': filepath}

        success = True
        # Load models
        for agent_name, info in latest_models.items():
            if agent_name in self.groups:
                agent = self.groups[agent_name]['agent']
                if agent.load_offline_model(info['path'], agent_name) and eval_mode:  
                        agent.set_eval_mode()  
                        logger.info("Set inference mode for model {agent_name} successuly")
            else:
                success = False
                logger.warning("Agent {agent_name} not in group of agents")

        return success
