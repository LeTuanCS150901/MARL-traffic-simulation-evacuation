from read_config import get_config
from logs.logger import logger  # import our custom logger

import numpy as np
from collections import defaultdict
import traci.constants

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

    def get_and_update_travel_time(self, agent_id, current_edge, destination_edge, nearby_agents):
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
        key = (current_edge, destination_edge)

        if key not in self.agents_data[agent_id]:
            # If the key does not exist, initialize it
            self.agents_data[agent_id][key] = None
            logger.debug(f"Initialized travel time for agent {agent_id} on edge {key} as None")

        # Case 1: Current agent has no experience on current edge -> using A*
        if self.agents_data[agent_id][key] is None:
            # using A* otherwise
            travel_time = self.compute_travel_time_astar(key)
            self.agents_data[agent_id][key] = travel_time
            logger.debug(f"Computed A* travel time for agent {agent_id} from {current_edge} to {destination_edge}: {travel_time}")

            return travel_time
        else:
            # case 2: current agent has experience on current edge
            # other agents share their travel time of all edges to current agent
            self.get_travel_time_other_agents(key, nearby_agents, agent_id)

            # return the updated travel time for considered agent after the sharing mechanism
            return self.agents_data[agent_id][key]
    
    def get_travel_time_other_agents(self, key, nearby_agents, agent_id):
        previous_time = self.agents_data[agent_id][key]
        alpha = 0.01  # Consider making this configurable
        
        for other_agent_id in nearby_agents:
            # Only iterate over edges the current agent knows
            for edge_key in list(self.agents_data[agent_id].keys()):
                if edge_key[0] in self.destination_edges:
                    continue       
                # # # Skip if other agent lacks this edge data
                # if edge_key not in self.agents_data[other_agent_id]:
                #     continue
                
                #  Skip if currennt agent lacks this edge data
                if edge_key not in self.agents_data[agent_id] or self.agents_data[agent_id][edge_key] == None:
                        continue

                #  Skip if other agent lacks this edge data
                if  edge_key not in self.agents_data[other_agent_id] or self.agents_data[other_agent_id][edge_key] == None:
                    continue
                    
                other_time = self.agents_data[other_agent_id][edge_key]
                current_time = self.agents_data[agent_id][edge_key]
                updated_time = alpha * other_time + (1 - alpha) * current_time
                self.agents_data[agent_id][edge_key] = updated_time

        # after the sharing, we check if travel time from current edge->dest edge is updated?
        new_time = self.agents_data[agent_id][key]
        if new_time == previous_time:
            self.agents_data[agent_id][key] = self.compute_travel_time_astar(key)
            logger.debug(f"After sharing mechanism, the travel time is still the same, so we compute a new time with A*")
    
    def compute_travel_time_astar(self, key):
        current_edge, destination_edge = key
        travel_time = None

        # Use A* algorithm to compute travel time from current_edge to destination_edge
        route = traci.simulation.findRoute(current_edge, destination_edge, routingMode=traci.constants.ROUTING_MODE_AGGREGATED)
        if len(route.edges) != 0:  
            travel_time = route.travelTime

        return travel_time

    