from read_config import get_config
from logs.logger import logger  # import our custom logger

import numpy as np
from collections import defaultdict
# import libsumo as traci.constants
import libsumo as traci

class GroupsSharedInfo:
    def __init__(self, graph_manager):
        """
        Initializes the GroupsSharedInfo manager.

        Args:
            graph_manager: An instance of GraphManager to access network information
                           and perform path calculations.
        """
        self.CONFIG = get_config()
        self.graph = graph_manager
        
        # Global list to store agent-specific data.
        # Structure: {agent_id: {edge_id: time_to_dest, ...}}
        self.agents_data = defaultdict(lambda: defaultdict(lambda: None))
        self.destination_edges = self.CONFIG['destinations_edges']

        # parameters for travel time calculation
        self.alpha = 0.1
        self.beta = 0.5

    def initialize_agent_data(self, agent_id, current_edge):
        """
        Initializes the time_to_dest for a new agent for all possible destinations as None.

        Args:
            agent_id (str): The unique identifier for the agent (e.g., vehicle ID or group ID).
            current_edge (str): The current edge the agent is on.
        """
        for dest_edge in self.destination_edges:
            # Initialize time_to_dest for each destination as None
            self.agents_data[agent_id][(current_edge, dest_edge)] = None
        logger.info(f"Initialized data for agent {agent_id} at edge {current_edge} with destinations {self.destination_edges}")

    def get_and_update_travel_time(self, agent_id, current_edge, destination_edge, nearyby_vehs_ids,
                                   travel_time_exp_current_edge):
        """
        When accessing an agent’s dictionary at a particular edge:
        - If dict_agent[edge] is None: Compute and assign the travel time using A* from the agent’s origin to its destination.
        - Otherwise: Set dict_agent[edge] to the average of the values from other agents on the same edge.

        Args:
            agent_id (str): The unique identifier for the agent.
            current_edge (str): The current edge the agent is on.
            destination_edge (str): The destination edge for the agent.
            nearby_agents List[str]: List of other agents' ids who are in the same edge of current agent

        Returns:
            float: The travel time to destination for the given agent and edge.
        """
        key = current_edge
        if key not in self.agents_data[agent_id]:
            # If the key does not exist, initialize it
            self.agents_data[agent_id][key] = None
            # logger.debug(f"Initialized travel time for agent {agent_id} on edge {key} as None")

        # Case 1,2: A none, B none/has
        if self.agents_data[agent_id][key] is None:
            self.get_travel_time_other_agents_case_1_2(key, nearyby_vehs_ids, agent_id, 
                                                                                travel_time_exp_current_edge)
            logger.warning(f"history {self.agents_data[agent_id][key]}")
            
        # Case 3,4: A has, B none/has
        if self.agents_data[agent_id][key]:
            self.get_travel_time_other_agents_case_3_4(key, nearyby_vehs_ids, agent_id, 
                                                                                travel_time_exp_current_edge)
            
        # Finally return the travel time of history and A*
        Astar_time = self.compute_travel_time_astar(key, destination_edge)
        if Astar_time is not None:
            return Astar_time + self.agents_data[agent_id][key]
        else:
            return None
  
    def get_travel_time_other_agents_case_1_2(self, key, nearyby_vehs_ids, agent_id, travel_time_exp_current_edge):
        
        if not nearyby_vehs_ids:
            logger.warning("No nearby case 1,2")
            self.agents_data[agent_id][key] = travel_time_exp_current_edge
            return  # No nearby agents to share information with
        
       
        for other_agent_id in nearyby_vehs_ids:
             # Case 1: A none, B none
            if key not in self.agents_data[other_agent_id] or self.agents_data[other_agent_id][key] == None:
                self.agents_data[agent_id][key] = travel_time_exp_current_edge
                logger.warning(f"other agent {other_agent_id} has no data for edge {key}")
                continue
            
            # Case 2: A none, B has
            other_time = self.agents_data[other_agent_id][key]
            logger.warning(f"other time{other_time}")
            self.agents_data[agent_id][key] = (1-self.alpha) * travel_time_exp_current_edge + self.alpha * other_time
                 
    def get_travel_time_other_agents_case_3_4(self, key, nearyby_vehs_ids, agent_id, travel_time_exp_current_edge):
        previous_time = self.agents_data[agent_id][key]
  
        if not nearyby_vehs_ids:
            logger.warning("No nearby case 3,4")
            self.agents_data[agent_id][key] = travel_time_exp_current_edge
            return  
        
        for other_agent_id in nearyby_vehs_ids:
             # Case 3: A has, B none
            if key not in self.agents_data[other_agent_id] or self.agents_data[other_agent_id][key] == None:
                self.agents_data[agent_id][key] = (1-self.alpha) * travel_time_exp_current_edge + self.alpha * previous_time
                continue

            # Case 4: A has, B has
            # if other agents have this edge data, we take the average
            self.agents_data[agent_id][key] = (1-self.alpha) * travel_time_exp_current_edge + self.alpha * (self.beta*previous_time + (1-self.beta) * self.agents_data[other_agent_id][key])

    def compute_travel_time_astar(self, key, destination_edge):
        current_edge = key
        travel_time = None

        # Use A* algorithm to compute travel time from current_edge to destination_edge
        route = traci.simulation.findRoute(current_edge, destination_edge, routingMode=traci.constants.ROUTING_MODE_AGGREGATED)
        if len(route.edges) != 0:  
            travel_time = route.travelTime

        return travel_time

    