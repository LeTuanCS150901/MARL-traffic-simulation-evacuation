import traci
import torch
import os
import json
import glob
import numpy as np
import numba
import re

from agent import Agent
from logs.logger import logger  # import our custom logger
from plotter import MultiAgentPlotter
from read_config import get_config
from utils.reward import Reward
from groups_shared_infor import GroupsSharedInfo
from database.h5_database import OfflineRLDataset

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
        self.batch_size = self.CONFIG['batch_size']  # batch size for training

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

    # def step(self, vehicle_id, curr_edge, connected_edges, disaster_move=None):
    #     """
    #     Update each agent based on their actions and interactions in the environment.
    #     """
    #     dest_edge = self.trips[vehicle_id]['to']  # Get the destination edge of the vehicle
    #     agent = self.groups[dest_edge]['agent']  # Get the agent for the vehicle's group

    #     # 1-hot encoding for feasible next edges
    #     # feasible_actions = [1 if edge in connected_edges else 0 for edge in self.edges_list]
    #     feasible_actions = [1 if (edge in connected_edges and (not self.is_dead_edge(edge) or edge in self.destination_edges))  
    #                 else 0 for edge in self.edges_list]
        
    #     # logger.error(f"feasible actions: {feasible_actions}")
    #     # np_edge = np.array(self.edges_list)
    #     # np_actions = np.array(feasible_actions)
    #     # logger.error(f"feasibles actions to feasible edges {np_edge[np_actions ==1]}")
    #     # logger.error(f"current vehicle is {vehicle_id}")
        
    #     # TODO: if there no feasible actions, it means current edge wil lead to dead road
    #     if not any(feasible_actions):
    #         logger.error(f"all feasible actions from {curr_edge} is empty")
    #         return

    #     try:
    #         # Create observation for the agent (e.g., current node, travel time)
    #         encoded_observation = self.encode_oberservation(curr_edge, connected_edges) 
    #     except Exception as e:
    #         logger.error(f"Error creating observation for vehicle {vehicle_id}: {e}")
    
    #     try:
    #         # Select action using the GREEDY policy
    #         # action = agent.select_action(encoded_observation, feasible_actions) 

    #         # Select action using the UCB policy
    #         # action = agent.UCB_select_action(encoded_observation, feasible_actions)

    #         # if choose action according to offline model inference 
    #         if self.offline_infer: 
    #             action = agent.select_action_offline(encoded_observation, feasible_actions)

    #         # otherwise  using softmax action selection
    #         action = agent.softmax_select_action(encoded_observation, feasible_actions)
    #     except Exception as e:
    #         logger.error(f"Error selecting action for vehicle {vehicle_id}: {e}")
            
    #     # Execute joint action and obtain the next observation and reward
    #     if self.training:
    #         next_edge, reward = self.execute_action(action, curr_edge, dest_edge, vehicle_id, feasible_actions, disaster_move) 
    #         self.plotter.record_reward(agent.group_id, reward)  # Record the reward for the group agent
    #     else:
    #         # in inference mode, we do not calculate the reward
    #         next_edge = self.edges_list[action]

    #     # if current mode is training, we store the exp and update policy
    #     if self.training:
    #         # Store experience for training
    #         encoded_next_observation = self.encode_next_oberservation(next_edge)
    #         agent.replay_buffer.push((encoded_observation, action, encoded_next_observation, reward))

    #         # Update the agent's policy
    #         agent.update(self.plotter)

    #     # next state is the next edge chosend, so next state = next action = the next edge chosen
    #     return next_edge


    def step(self, vehicle_id, curr_edge, connected_edges, disaster_move=None):
        """Step for individual vehicle"""
        dest_edge = self.trips[vehicle_id]['to']
        group_agent = self.groups[dest_edge]

        ############## Obsevation
        try:
            # Create observation for the agent (e.g., current node, travel time)
            encoded_observation = self.encode_oberservation(curr_edge, connected_edges) 
        except Exception as e:
            logger.error(f"Error creating observation for vehicle {vehicle_id}: {e}")
        

        ##############  Action 
        feasible_actions = [
            1 if (edge in connected_edges and 
                  (not self.is_dead_edge(edge) or edge in self.destination_edges))
            else 0 
            for edge in self.edges_list
        ]
        # TODO: if there no feasible actions, it means current edge wil lead to dead road
        if not any(feasible_actions):
            logger.warning(f"All feasible actions from {curr_edge} is empty")
            return

        try: 
            action = group_agent.softmax_select_action(encoded_observation, feasible_actions)
        except Exception as e:
            logger.error(f"Error selecting action for vehicle {vehicle_id}: {e}")  
        

        ##############  Excecute action => receive reward and next obsevation 
        if self.training:
            next_edge, individual_reward = self.execute_action(
                action, curr_edge, dest_edge, vehicle_id, feasible_actions, disaster_move
            )
            
            encoded_next_observation = self.encode_next_oberservation(next_edge)
            done = (next_edge in self.destination_edges)
            
            # Store in temporary buffer for this timestep
            self.group_timestep_buffers[dest_edge].append(
                (vehicle_id, encoded_observation, action, encoded_next_observation, done)
            )

            # Accumulate reward for this timestep of current agent group
            self.current_timestep_rewards[dest_edge].append(individual_reward)
        else:
            next_edge = self.edges_list[action]
        
        return next_edge
    
    def get_overlap_transitions(self, destination, vehs_current_edge):
        """
        Get obervations of other agent groups vehicles that have the same current edge 
        as the vehicles in this group

        Args:
        -destination: current group destination
        -vehs_current_edge: List edges of all vehicles are are traversing in this group

        Return:
        -results: List of transitions (vehicle_id, obs, action, next_obs, done) from other groups
        """
        results = []
        for other_dest, group_agent in self.groups.items():
            if other_dest != destination:
                continue
            # Get all transitions for this group in current timestep

            try:
                transitions = self.group_timestep_buffers[other_dest]
            except Exception as e:
                logger.error(f"No transition for group {destination}: {e}. Skipping.")
                continue
            
            if len(transitions) > 0:
                for vehicle_id, obs, action, next_obs, done in transitions:
                    edge = obs[0][0].item()  # get current edge index from observation
                    if edge in vehs_current_edge:
                        results.append((vehicle_id, obs, action, next_obs, done))
        
        return results

    def end_timestep(self, use_hierarchical=True):  
        """  
        Call this at the end of each simulation timestep.  
        Stores joint transitions and triggers training.  
        
        Args:  
            use_hierarchical: If True, collect group data for hierarchical training  
        """  
        if not self.training:  
            return  
        
        # Collect group data if hierarchical mode  
        if use_hierarchical:  
            self.timestep_group_data = {}  
        
        for destination, group_agent in self.groups.items():  
            try:  
                transitions = self.group_timestep_buffers[destination]  
            except Exception as e:  
                logger.error(f"No transition for group {destination}: {e}. Skipping.")  
                continue  
            
            if len(transitions) > 0:  
                global_reward = self.calculate_group_reward(destination)  
                logger.debug(f"Global reward for group {destination}: {global_reward}")  
                
                # store all transitions from this group
                vehs_current_edge = []  
    
                for vehicle_id, obs, action, next_obs, done in transitions:  
                    group_agent.store_transition_timestep(vehicle_id, obs, action, next_obs, done)  
                    current_edge = obs[0][0].item()
                    vehs_current_edge.append(current_edge)  
                
                # number of vehicles in this group at current step
                num_vehs_group = len(transitions)

                # store overlapping transitions from other groups
                overlap_transitions = self.get_overlap_transitions(destination, vehs_current_edge)  
                for vehicle_id, obs, action, next_obs, done in overlap_transitions:  
                    group_agent.store_transition_timestep(vehicle_id, obs, action, next_obs, done)  
                
                # store global reward for this timestep of curent group
                group_agent.end_timestep(global_reward, num_vehs_group)  
                
                # Store for hierarchical training or train individually  
                if use_hierarchical:  
                    self.timestep_group_data[destination] = {  
                        'agent': group_agent,  
                        'reward': global_reward
                    }  
                else:  
                    # Original individual training  
                    loss = group_agent.update(self.plotter)  
            
                if self.plotter:
                    self.plotter.record_reward(destination, global_reward)  
            self.group_timestep_buffers[destination] = []  
        
        # Hierarchical training across groups  
        if use_hierarchical and len(self.timestep_group_data) > 0:  
            self._hierarchical_train()  

    def _hierarchical_train(self):  
        """Simple hierarchical VDN training: Q_total = sum(Q_group_i)"""  
        if len(self.timestep_group_data) < 2:  
            # Only one group, use standard training  
            for data in self.timestep_group_data.values():  
                data['agent'].update(self.plotter)  
            return  
        
        # Get minimum batch size across groups  
        min_buffer_size = min(len(data['agent'].replay_buffer)   
                            for data in self.timestep_group_data.values())  
        
        if min_buffer_size < self.batch_size:  
            return  
        
        # Sample from each group  
        group_samples = {dest: data['agent'].replay_buffer.sample(self.batch_size)  
                        for dest, data in self.timestep_group_data.items()}  
        
        total_loss = 0.0  
        
        for idx in range(self.batch_size):  
            q_current_sum = 0.0  
            q_next_sum = 0.0  
            global_reward = sum(data['reward'] for data in self.timestep_group_data.values())  
            
            for dest, samples in group_samples.items():  
                agent = self.timestep_group_data[dest]['agent']  
                joint_transition = samples[idx]  
                vehicles_data = joint_transition['vehicles']  
                num_vehs_group = joint_transition['num_vehicles']
                
                if len(vehicles_data) == 0:  
                    continue  
                
                # Stack tensors  
                obs_batch = torch.stack([torch.tensor(v[1], dtype=torch.float32).squeeze()   
                                        for v in vehicles_data])  
                next_obs_batch = torch.stack([torch.tensor(v[3], dtype=torch.float32).squeeze()   
                                            for v in vehicles_data])  
                actions_batch = torch.tensor([v[2] for v in vehicles_data], dtype=torch.int64)  
                dones_batch = torch.tensor([v[4] for v in vehicles_data], dtype=torch.float32)  
                
                # Current Q for this group  
                q_values = agent.q_network(obs_batch, num_vehs_group=num_vehs_group)  
                # we only take action for vehicles in this group
                # because in the joint transition, there are also vehicles from other groups that are 
                # on the same edges as vehicles in this group
                q_chosen = q_values.gather(1, actions_batch[0:num_vehs_group].unsqueeze(1)).squeeze(1)  
                q_current_sum += q_chosen.sum()  
                
                # Next Q for this group  
                with torch.no_grad():  
                    next_q = agent.target_network(next_obs_batch)  
                    next_actions = agent.q_network(next_obs_batch).argmax(dim=1)  
                    next_q_values = next_q.gather(1, next_actions.unsqueeze(1)).squeeze(1)  
                    next_q_values = next_q_values * (1 - dones_batch)  
                    q_next_sum += next_q_values.sum()  
            
            # VDN: Q_total = sum of all group Q-values  
            with torch.no_grad():  
                target = global_reward + agent.gamma * q_next_sum  
            
            loss = (q_current_sum - target) ** 2  
            total_loss += loss  
        
        total_loss = total_loss / self.batch_size  
        
        # Backprop through all networks  
        for data in self.timestep_group_data.values():  
            data['agent'].optimizer.zero_grad()  
        
        total_loss.backward()  
        
        for data in self.timestep_group_data.values():  
            agent = data['agent']  
            torch.nn.utils.clip_grad_norm_(agent.q_network.parameters(), max_norm=10.0)  
            agent.optimizer.step()  
            
            agent.update_step_count += 1  
            if agent.update_step_count % agent.target_update_freq == 0:  
                agent.target_network.load_state_dict(agent.q_network.state_dict()) 

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
        
        return full_obs.unsqueeze(0)  # convert from 1D to 2D tensor
        ########### Add disaster info to the state: (mean x, mean y) and radius of disaster 
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
                       disaster_move=None):
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
        reward = self.reward.get_reward_5(curr_edge, next_edge, dest_edge, vehicle_id, feasible_actions)
        logger.info(f"Vehicle {action} took action: {next_edge}, received reward: {reward}")

        return next_edge, reward 

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

    def load_models(self, index=None, episode=None, eval_mode=True):  
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

        # Load each agent's model  
        success = True  
        for agent_edge, group_meta in metadata['groups'].items():  
            if agent_edge in self.groups:  
                model_path = group_meta['model_path']  
                if os.path.exists(model_path):  
                    logger.info(f"Loading model for agent {agent_edge} from {model_path}")  
                    agent = self.groups[agent_edge]  
                    if agent.load_model(model_path) and eval_mode:  
                        agent.set_eval_mode()  
                        logger.info("Set inference mode for models successuly")
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

            