from logs.logger import logger
import traci
import numpy as np
import pandas as pd 
import os

import traci.constants

class Reward:  
    def __init__(self, agents_manager):  
        # Store reference to the manager  
        self.agents_manager = agents_manager  
    
    def get_reward_MSE(self, next_edge, dest_edge):
        """
        R = (1-t/tmax)^2 + (1-d/max)^2 + I
        t: travel to next edge
        tmax: max free-flow travel time of all edges
        d: euclid distance of next edge to dest edge
        dmax: max euclid distance of an edge to dest edge
        I: -alpha if next edge is a dead edge
        """
        travel_time_reward = 0.0
        travel_time_next_edge = traci.edge.getTraveltime(next_edge)
      
        distance_reward = 0.0
        # get the distance euclid from next edge to the destination edge
        next_edge_distance = self.agents_manager.edges_distance[next_edge][dest_edge]

        # calculate travel time reward
        if travel_time_next_edge > self.agents_manager.max_travel_time:
            travel_time_reward = 1
        else:
            travel_time_reward = (travel_time_next_edge / self.agents_manager.max_travel_time)

        # calculate distance reward
        if next_edge_distance > self.agents_manager.d_max:
            distance_reward = 1
        else:
            distance_reward = (next_edge_distance / self.agents_manager.d_max)

        travel_time_reward = (1 - travel_time_reward)**2
        distance_reward = (1 - distance_reward)**2

       # check if current edge is a dead edge 
        penalty_dead_edge = 0
        if self.agents_manager.is_dead_edge(next_edge):
                penalty_dead_edge = (self.agents_manager.dead_edge)**2

        return travel_time_reward + self.agents_manager.d_weight * distance_reward + penalty_dead_edge
    
    def get_reward_based_d_weight(self, next_edge, dest_edge):
        """
        R = [alpha * (1-t/tmax)] / [(1+d/dmax)] + I 
        t: travel to next edge
        tmax: max free-flow travel time of all edges
        d: euclid distance of next edge to dest edge
        dmax: max euclid distance of an edge to dest edge
        I: -alpha if next edge is a dead edge
        """
        travel_time_reward = 0
        travel_time_next_edge = traci.edge.getTraveltime(next_edge)
      
        distance_reward = 0
        # get the distance euclid from next edge to the destination edge
        next_edge_distance = self.agents_manager.edges_distance[next_edge][dest_edge]

        # calculate travel time reward
        if travel_time_next_edge > self.agents_manager.max_travel_time:
            travel_time_reward = 1
        else:
            travel_time_reward = (travel_time_next_edge / self.agents_manager.max_travel_time)

        # calculate distance reward
        if next_edge_distance > self.agents_manager.d_max:
            distance_reward = 1
        else:
            distance_reward = (next_edge_distance / self.agents_manager.d_max)

       # check if current edge is a dead edge 
        penalty_dead_edge = 0
        if self.agents_manager.is_dead_edge(next_edge):
                penalty_dead_edge = (self.agents_manager.dead_edge)**2

        base_reward = (self.agents_manager.d_weight*(1-distance_reward)) / (1+ travel_time_reward) + penalty_dead_edge

        # TODO: add penalty if agent choose wrong destination
        # because 2 destination might near each other, so even choose desti, the reward is enough
        # so the agent keep choosing the wrong dest
        penalty_wrong_dest = 0
        if next_edge in self.agents_manager.destination_edges:
            if next_edge != dest_edge:
                logger.error(f"Vehicle {next_edge} chose wrong destination: {dest_edge}")
                penalty_wrong_dest = -self.agents_manager.d_weight
                logger.error(f"reward {base_reward + penalty_wrong_dest}")
        
        return base_reward + penalty_wrong_dest
             
    def get_reward_risk_model(self, next_edge, dest_edge):
        """
        R = [alpha * (1-t/tmax)] / [(1+d/dmax)] * Rsafe - I
        t: travel to next edge
        tmax: max free-flow travel time of all edges
        d: euclid distance of next edge to dest edge
        dmax: max euclid distance of an edge to dest edge
        Rsafe (risk reward): ds/dsmax 
             dsmax: radius of the disaster zone 
             ds: min (ds, dsmax)
        I: -alpha if agent choose wrong dest
        """
        travel_time_reward = 0.0
        travel_time_next_edge = traci.edge.getTraveltime(next_edge)
      
        distance_reward = 0.0
        # get the distance euclid from next edge to the destination edge
        next_edge_distance = self.agents_manager.edges_distance[next_edge][dest_edge]

        # calculate travel time reward
        if travel_time_next_edge > self.agents_manager.max_travel_time:
            travel_time_reward = 1
        else:
            travel_time_reward = (travel_time_next_edge / self.agents_manager.max_travel_time)

        logger.error(f"travel time:{travel_time_next_edge} for the edge chosen {next_edge}")

        # calculate distance reward
        if next_edge_distance > self.agents_manager.d_max:
            distance_reward = 1
        else:
            distance_reward = (next_edge_distance / self.agents_manager.d_max)
        logger.error(f"distance euclid to dest:{next_edge_distance} with the edge chosen {next_edge}")

        # calculate the risk reward. This is the distance to the mean of disaster
        ds_reward = 1
        # get nearest disaster zone and distance to that zone
        nearest_zone, ds = self.agents_manager.get_nearest_disaster(next_edge)
        if nearest_zone and ds:
            # if the distance to that zone > the zone's radius 
            if ds >= self.agents_manager.disaster_info[nearest_zone]['radius']:
                ds_reward = 1
            else:
                ds_reward = ds / self.agents_manager.disaster_info[nearest_zone]['radius']
        else:
            logger.error("Error when get nearest disaster infor")

        logger.error(f"ds to disaster:{ds} with the edge chosen {next_edge}")
        base_reward = ((self.agents_manager.d_weight * (1 - distance_reward)) / (1 + travel_time_reward)) * ds_reward
            
        # add penalty if agent choose wrong destination
        # because 2 destination might near each other, so even choose desti, the reward is enough
        # so the agent keep choosing the wrong dest
        penalty_wrong_dest = 0
        if next_edge in self.agents_manager.destination_edges:
            if next_edge != dest_edge:
                logger.error(f"Vehicle {next_edge} chose wrong destination: {dest_edge}")
                penalty_wrong_dest = -self.agents_manager.d_weight
                logger.error(f"based reward {base_reward + penalty_wrong_dest}")
        logger.error(f"penalty reward value {penalty_wrong_dest} with the edge chosen {next_edge}")
        return base_reward + penalty_wrong_dest
    
    def get_reward_1(self, curr_edge, next_edge, dest_edge, vehicle_id, feasible_actions):
        """
        R = [alpha * (1-t/tmin)] / [(1+d/dmax)] + I
        this reward:
        - t: expected time of option -> dest
        - tmin: min expected time of current edge -> dest 
        - d: euclid distance of option -> dest
        - dmax: (euclid distance of origin -> dest) * 2
        - I: -alpha if agent choose wrong dest
        """
        # calculate distance reward
        distance_reward = 0.0
        # get the distance euclid from next edge to the destination edge
        next_edge_distance = self.agents_manager.edges_distance[next_edge][dest_edge]
   
        origin = self.agents_manager.trips[vehicle_id]['from']
        dest = self.agents_manager.trips[vehicle_id]['to']
        try:
            dist_od = self.agents_manager.dist_OD[origin][dest] * 2
        except Exception as e: 
            logger.error("Error when accessing distance OD")

        if next_edge_distance > dist_od:
            distance_reward = 1
        else:
            distance_reward = (next_edge_distance / dist_od)
        logger.error(f"distance euclid to dest:{next_edge_distance} with the edge chosen {next_edge}")

        # calculate travel time reward
        travel_time_reward = 0.0
        # calculate the travel time from current edge to the destination edge
        min_travel_time = np.inf  
        route = traci.simulation.findRoute(curr_edge, dest)
        # if there is route connected to dest
        if len(route.edges) != 0:  
            min_travel_time = route.travelTime

        # calculate the travel time from next edge selected to the destination edge
        route_next_edge_chosen = traci.simulation.findRoute(next_edge, dest)
        # if there is no route connected to dest
        if len(route_next_edge_chosen.edges) == 0:
            travel_time_next_edge = np.inf  
        else:
            travel_time_next_edge = route_next_edge_chosen.travelTime

        travel_time_reward = travel_time_next_edge / min_travel_time
        logger.error(f"currrent vehicle is {vehicle_id}")
        logger.error(f"travel time:{travel_time_next_edge} for the edge chosen {next_edge}, at current edge {curr_edge} with travel time {min_travel_time}")
        
        base_reward = ((self.agents_manager.d_weight * (1 - distance_reward)) / (1 + travel_time_reward)) 

        # add penalty if agent choose wrong destination
        # because 2 destination might near each other, so even choose desti, the reward is enough
        # so the agent keep choosing the wrong dest
        penalty_wrong_dest = 0
        if next_edge in self.agents_manager.destination_edges:
            if next_edge != dest_edge:
                logger.error(f"Vehicle {vehicle_id} chose wrong destination: {next_edge}")
                penalty_wrong_dest = -self.agents_manager.d_weight
                logger.error(f"based reward {base_reward + penalty_wrong_dest}")
        logger.error(f"penalty reward value {penalty_wrong_dest} with the edge chosen {next_edge}")
       
        return base_reward + penalty_wrong_dest
    
    def get_reward_2(self, curr_edge, next_edge, dest_edge, vehicle_id, feasible_actions):
        """
        R = [alpha * (1-t/tmax)] / [(1+d/dmax)] + I
        - t: expected time of chosen option->dest
        - tmax: max expected time of options->dest
        - d: (euclid distance of option -> dest) * 2
        - dmax: euclid distance of origin -> dest
        """
        # calculate distance reward
        distance_reward = 0.0
        # get the distance euclid from next edge to the destination edge
        next_edge_distance = self.agents_manager.edges_distance[next_edge][dest_edge]
   
        origin = self.agents_manager.trips[vehicle_id]['from']
        dest = self.agents_manager.trips[vehicle_id]['to']
        try:
            dist_od = self.agents_manager.dist_OD[origin][dest] * 2
        except Exception as e: 
            logger.error("Error when accessing distance OD")

        if next_edge_distance > dist_od:
            distance_reward = 1
        else:
            distance_reward = (next_edge_distance / dist_od)
        logger.error(f"distance euclid to dest:{next_edge_distance} with the edge chosen {next_edge}")


        max_travel_time = 0.0
        travel_time_reward = 0.0

        indices = np.where(np.array(feasible_actions) == 1)[0] # filter indices with 1 value in feasible actions
        for edge_idx in indices:
            edge_id = self.agents_manager.edges_list[edge_idx]
            route = traci.simulation.findRoute(edge_id, dest) 
            # if there is route connected to dest
            if len(route.edges) != 0:  
                max_travel_time = max(max_travel_time, route.travelTime)

        route_next_edge = traci.simulation.findRoute(next_edge, dest)
        travel_time_next_edge = route_next_edge.travelTime
        if len(route_next_edge.edges) == 0:
            travel_time_reward = 1
        else:
            travel_time_reward = travel_time_next_edge / max_travel_time
    
        logger.error(f"currrent vehicle is {vehicle_id}")
        logger.error(f"travel time:{travel_time_next_edge} for the edge chosen {next_edge}, at current edge {curr_edge}")
    
        base_reward = ((self.agents_manager.d_weight * (1 - distance_reward)) / (1 + travel_time_reward)) 

        # add penalty if agent choose wrong destination
        # because 2 destination might near each other, so even choose desti, the reward is enough
        # so the agent keep choosing the wrong dest
        penalty_wrong_dest = 0
        if next_edge in self.agents_manager.destination_edges:
            if next_edge != dest_edge:
                logger.error(f"Vehicle {vehicle_id} chose wrong destination: {next_edge}")
                penalty_wrong_dest = -self.agents_manager.d_weight
                logger.error(f"based reward {base_reward + penalty_wrong_dest}")
        logger.error(f"penalty reward value {penalty_wrong_dest} with the edge chosen {next_edge}")
        
        return base_reward + penalty_wrong_dest
    
    def get_reward_3(self, curr_edge, next_edge, dest_edge, vehicle_id, feasible_actions):
        """
        R = -t
        t: expected travel time of chosen option->dest. This can be obtained by A* algorithm      
        Because A* already find the path to ensure current connected to dest, so if there is nothing return, 
        the option should have bad reward to avoide chooding it later  
        """
        travel_time_reward = 0.0
        BIG_PENALTY = self.agents_manager.CONFIG['simulation_steps'] * 2
        # find route from selected next edge to destination edge
        route = traci.simulation.findRoute(next_edge, dest_edge, routingMode=traci.constants.ROUTING_MODE_AGGREGATED)

        if len(route.edges) != 0:  
            travel_time = route.travelTime
            travel_time_reward = -travel_time
        else:
            travel_time_reward = -BIG_PENALTY
            logger.debug(f"penalty for no route from {next_edge} to {dest_edge}: {travel_time_reward}")
        logger.debug(f"travel time reward: {travel_time_reward} for the edge chosen {next_edge}, at current edge {curr_edge}")
        
        return travel_time_reward

    # def get_reward_4(self, curr_edge, next_edge, dest_edge, vehicle_id, feasible_actions):
    #     """
    #     R = w1 * R1 + w2 * R2
    #     R1: travel time reward
    #     R2: distance to disaster reward
    #     """
    #     # calculate R1 reward: travel time
    #     R1 = 0.0
    #     origin = self.agents_manager.trips[vehicle_id]['from']
    #     dest = self.agents_manager.trips[vehicle_id]['to']
    #     distance =  self.agents_manager.dist_OD[origin][dest]
    #     t_min_next_edge = distance / self.agents_manager.max_speed_vehs

    #     t_actual_next_edge = 0.0
    #     route = traci.simulation.findRoute(next_edge, dest_edge, routingMode=traci.constants.ROUTING_MODE_AGGREGATED)
    #     if len(route.edges) != 0:  
    #         t_actual_next_edge = route.travelTime
    #         # R1 = (0.00001 + t_min_next_edge) / (0.00001 + t_actual_next_edge)
    #         R1 =  1.0 / (1.0 + np.exp(-0.08*(t_min_next_edge - t_actual_next_edge)))
    #     else: # BIG PENALTY FOR CHOOSING DEAD EDGES
    #         R1 = -1.2
    #         logger.debug(f"penalty for no route from {next_edge} to {dest_edge}: {R1}")

    #     logger.debug(f"travel time reward: {R1} for the edge chosen {next_edge}, at current edge {curr_edge}")

    #     # calculate R2 reward: distance to disaster
    #     total_r2 = 0.0
    #     for disaster_id in self.agents_manager.disaster_info:
    #         disaster = self.agents_manager.disaster_info[disaster_id]
    #         disaster_radius = disaster['radius']
    #         # get the distance from next edge to the disaster zone
    #         ds = self.agents_manager.edges_dist_disaster[next_edge][disaster_id]

    #         logger.error(f"dist to disaster {ds}, disaster radius {disaster_radius}")
    #         r2 = (disaster_radius / ds)**2
    #         total_r2 += r2  
    #         #TODO: ADD PENALTY IF CHOOSING DISATER EDGES

    #     R2 = 1 / (1+ np.sqrt(total_r2))

    #     # TODO: LOG THIS METRIC IN OTHER FILE
    #     # self.save_not_survive_proba(disaster_radius, ds)

    #     return self.agents_manager.w1 * R1 + self.agents_manager.w2 * R2

    # def get_reward_4(self, curr_edge, next_edge, dest_edge, vehicle_id, feasible_actions):
    #     """
    #     R = w1 * R1 + w2 * R2
    #     R1: travel time reward
    #     R2: distance to disaster reward
    #     """
    #     # calculate R1 reward: travel time
    #     R1 = 0.0
    #     origin = self.agents_manager.trips[vehicle_id]['from']
    #     dest = self.agents_manager.trips[vehicle_id]['to']
    #     distance =  self.agents_manager.dist_OD[origin][dest]
    #     t_min_od = distance / self.agents_manager.max_speed_vehs

    #     t_actual_next_edge = 0.0
    #     route = traci.simulation.findRoute(next_edge, dest_edge, routingMode=traci.constants.ROUTING_MODE_AGGREGATED)
    #     if len(route.edges) != 0:  
    #         t_actual_next_edge = route.travelTime
    #         R1 = 1 / (1+ (t_actual_next_edge/t_min_od))
    #     else: # BIG PENALTY FOR CHOOSING DEAD EDGES
    #         R1 = -1.2
    #         logger.debug(f"penalty for no route from {next_edge} to {dest_edge}: {R1}")

    #     # calculate reward distance
    #     origin = self.agents_manager.trips[vehicle_id]['from']
    #     dest = self.agents_manager.trips[vehicle_id]['to']
    #     distance_od =  self.agents_manager.dist_OD[origin][dest]
    #     next_edge_distance = self.agents_manager.edges_distance[next_edge][dest_edge]
    #     R2_distance_reward = next_edge_distance / distance_od

    #     logger.error(f"Travel reward {R1}")
    #     logger.error(f"Distance reward {R2_distance_reward}")
    #     final_reward = self.agents_manager.w1 * R1 + self.agents_manager.w2 * R2_distance_reward
    #     logger.error(f"final reward {final_reward}")

    #     return final_reward
    #    

    # def _get_reward_5(self, curr_edge, next_edge, dest_edge, vehicle_id, feasible_actions):
    #     """
    #     Normalize distance to time unit
    #     [m] -> [s]
    #     """
    #     ###### travel time reward ######
    #     travel_time_reward = 0.0
    #     BIG_PENALTY = self.agents_manager.CONFIG['simulation_steps'] * 2
    #     # find route from selected next edge to destination edge
    #     route = traci.simulation.findRoute(next_edge, dest_edge, routingMode=traci.constants.ROUTING_MODE_AGGREGATED)

    #     if len(route.edges) != 0:  
    #         travel_time = route.travelTime
    #         travel_time_reward = -travel_time
    #     else:
    #         travel_time_reward = -BIG_PENALTY
    #         logger.debug(f"penalty for no route from {next_edge} to {dest_edge}: {travel_time_reward}")
    #     logger.debug(f"travel time reward: {travel_time_reward} for the edge chosen {next_edge}, at current edge {curr_edge}")
        

    #     # calculate median speed of edges in disaster zones
    #     all_speed_edges_in_disaster = []
    #     for zone_id in self.agents_manager.disaster_zones:
    #         disaster_edges = self.agents_manager.disaster_zones[zone_id]
    #         for edge_id in disaster_edges:
    #             speed = self.agents_manager.edges[edge_id]['speed']
    #             all_speed_edges_in_disaster.append(speed)
        
    #     logger.error(f"all speed edges in disaster: {all_speed_edges_in_disaster}")

    #     # ###### distance to disaster reward ######
    #     # list_r2 = np.array([])
    #     # total_r2 = 0.0
    #     # for disaster_id in self.agents_manager.disaster_info:
    #     #     disaster = self.agents_manager.disaster_info[disaster_id]
    #     #     disaster_radius = disaster['radius']
    #     #     # get the distance from next edge to the disaster zone
    #     #     ds = self.agents_manager.edges_dist_disaster[next_edge][disaster_id]

    #     #     logger.error(f"dist to disaster {ds}, disaster radius {disaster_radius}")
    #     #     total_r2 += ds  

    #     # # divide by max median distance of edges in disater
    #     # total_r2 = total_r2 / np.median(all_speed_edges_in_disaster)
    #     # logger.error(f"Convert distance to disaster reward to travel time: {total_r2}")

    #     ###### distance to disaster reward ######
    #     list_r2 = []
    #     for disaster_id in self.agents_manager.disaster_info:
    #         disaster = self.agents_manager.disaster_info[disaster_id]
    #         disaster_radius = disaster['radius']
    #         # get the distance from next edge to the disaster zone
    #         ds = self.agents_manager.edges_dist_disaster[next_edge][disaster_id]

    #         logger.error(f"dist to disaster {ds}, disaster radius {disaster_radius}")
    #         # divide by max median distance of edges in disater
    #         r2 = ds / np.median(all_speed_edges_in_disaster)
    #         list_r2.append(r2)
    #         logger.error(f"Convert distance from disaster reward to travel time: {r2}")

    #     r2_min = np.min(list_r2)
    #     logger.error(f"Take the min dinstance of disasters: {r2_min}")

    #     return self.agents_manager.w1*travel_time_reward + self.agents_manager.w2*r2_min

    # def get_reward_5(self, curr_edge, next_edge, dest_edge, vehicle_id, feasible_actions):
    #     """
    #     Sharing mechanism function
    #     Normalize distance to time unit
    #     [m] -> [s]
    #     """
    #     ###### travel time reward ######
    #     # current ohter nearby agents also at current edge to allow them share information to current considered agent
    #     nearyby_vehs_ids = list(traci.edge.getLastStepVehicleIDs(curr_edge))
        
    #     #filter only other agents with the same group (same destination edge) as the current agent
    #     # nearyby_vehs_ids = [veh_id for veh_id in nearyby_vehs_ids if self.agents_manager.trips[veh_id]['to'] == dest_edge]
    #     nearyby_vehs_ids.remove(vehicle_id) # remove curent veh from the list of nearby veh
    #     # logger.warning(f"current agent {vehicle_id}, nearby filtered agents: {nearyby_vehs_ids}")
    #     travel_time_exp_current_edge = traci.edge.getTraveltime(curr_edge)

    #     BIG_PENALTY = self.agents_manager.CONFIG['simulation_steps'] * 2
    #     travel_time_reward = self.agents_manager.shared_info.get_and_update_travel_time(vehicle_id,
    #                                                                                     next_edge, 
    #                                                                                     dest_edge,
    #                                                                                     nearyby_vehs_ids, 
    #                                                                                     travel_time_exp_current_edge)
    #     if not travel_time_reward:
    #         travel_time_reward = -BIG_PENALTY
    #         logger.debug(f"penalty for no route from {next_edge} to {dest_edge}: {travel_time_reward}")
    #     else:
    #         travel_time_reward = -travel_time_reward  # negative reward for travel time

    #     logger.debug(f"travel time reward: {travel_time_reward} for the edge chosen {next_edge}, at current edge {curr_edge}")
        

    #     if not self.agents_manager.disaster_info:
    #         logger.error("No disaster information available, returning travel time reward only.")
    #         return travel_time_reward
        
    #     # calculate median speed of edges in disaster zones
    #     all_speed_edges_in_disaster = []
    #     for zone_id in self.agents_manager.disaster_zones:
    #         disaster_edges = self.agents_manager.disaster_zones[zone_id]
    #         for edge_id in disaster_edges:
    #             speed = self.agents_manager.edges[edge_id]['speed']
    #             all_speed_edges_in_disaster.append(speed)
        
    #     logger.error(f"all speed edges in disaster: {all_speed_edges_in_disaster}")

    #     ###### distance to disaster reward ######
    #     list_r2 = []
    #     for disaster_id in self.agents_manager.disaster_info:
    #         disaster = self.agents_manager.disaster_info[disaster_id]
    #         disaster_radius = disaster['radius']
    #         # get the distance from next edge to the disaster zone
    #         ds = self.agents_manager.edges_dist_disaster[next_edge][disaster_id]

    #         logger.error(f"dist to disaster {ds}, disaster radius {disaster_radius}")
    #         # divide by max median distance of edges in disater
    #         r2 = ds / np.median(all_speed_edges_in_disaster)
    #         list_r2.append(r2)
    #         logger.error(f"Convert distance from disaster reward to travel time: {r2}")

    #     r2_min = np.min(list_r2)
    #     logger.error(f"Take the min dinstance of disasters: {r2_min}")

    #     return self.agents_manager.w1*travel_time_reward + self.agents_manager.w2*r2_min

    def get_reward_5(self, curr_edge, next_edge, dest_edge, vehicle_id, feasible_actions):
            """
            Normalize distance to time unit
            [m] -> [s]
            """
            ###### travel time reward ######
            travel_time_reward = 0.0
            BIG_PENALTY = self.agents_manager.CONFIG['simulation_steps'] 
            # find route from selected next edge to destination edge
            route = traci.simulation.findRoute(next_edge, dest_edge, routingMode=traci.constants.ROUTING_MODE_AGGREGATED)

            if len(route.edges) != 0:  
                travel_time = route.travelTime
                travel_time_reward = -travel_time
            else:
                travel_time_reward = -BIG_PENALTY
                logger.debug(f"penalty for no route from {next_edge} to {dest_edge}: {travel_time_reward}")
            logger.debug(f"travel time reward: {travel_time_reward} for the edge chosen {next_edge}, at current edge {curr_edge}")
            

            if not self.agents_manager.disaster_info:
                logger.error("No disaster information available, returning travel time reward only.")
                return travel_time_reward

            # calculate median speed of edges in disaster zones
            all_speed_edges_in_disaster = []
            for zone_id in self.agents_manager.disaster_zones:
                disaster_edges = self.agents_manager.disaster_zones[zone_id]
                for edge_id in disaster_edges:
                    speed = self.agents_manager.edges[edge_id]['speed']
                    all_speed_edges_in_disaster.append(speed)
            
            logger.error(f"all speed edges in disaster: {all_speed_edges_in_disaster}")

            ###### distance to disaster reward ######
            total_r2 = 0.0
            for disaster_id in self.agents_manager.disaster_info:
                disaster = self.agents_manager.disaster_info[disaster_id]
                disaster_radius = disaster['radius']
                # get the distance from next edge to the disaster zone
                ds = self.agents_manager.edges_dist_disaster[next_edge][disaster_id]

                logger.error(f"dist to disaster {ds}, disaster radius {disaster_radius}")
                total_r2 += ds  

            # divide by max median distance of edges in disater
            total_r2 = total_r2 / np.median(all_speed_edges_in_disaster)
            logger.error(f"Convert distance to disaster reward to travel time: {total_r2}")

            return self.agents_manager.w1*travel_time_reward + self.agents_manager.w2*total_r2

    def save_not_survive_proba(self, disaster_radius, ds):
        # Check and create folder if it doesn't exist
        output_dir = os.path.join(os.path.dirname(__file__), '../..', 'result/disaster_metric') 
        os.makedirs(output_dir, exist_ok=True)
        
        if ds > disaster_radius:
            not_survive_proba = 0
        else:
            # Calculate not survival probability
            not_survive_proba = 1 - (ds/disaster_radius)
        
        # Create new data row
        new_data = {
            'disaster_radius': disaster_radius,
            'distance': ds,
            'not_survive_probability': not_survive_proba
        }
        
        # Define file path
        filepath = os.path.join(output_dir, "not_survive_proba_R1R2.csv")
        
        # Append to existing CSV or create new one
        if os.path.exists(filepath):
            # Read existing data and append new row
            df_existing = pd.read_csv(filepath)
            df_new = pd.DataFrame([new_data])
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
            df_combined.to_csv(filepath, index=False)
        else:
            # Create new CSV file
            df_new = pd.DataFrame([new_data])
            df_new.to_csv(filepath, index=False)
        
        print(f"Result appended to: {filepath}")

    def get_penalty_loop_R1(self, actual_travel_time):
        # return -actual_travel_time
        # return -650
        max_simu_time = self.agents_manager.CONFIG['simulation_steps']
        return -max_simu_time

    def get_penalty_loop_R1R2(self, actual_travel_time, next_edge, dest_edge, vehicle_id):
        
        origin = self.agents_manager.trips[vehicle_id]['from']
        dest = self.agents_manager.trips[vehicle_id]['to']
        distance =  self.agents_manager.dist_OD[origin][dest]
        t_min_od = distance / self.agents_manager.max_speed_vehs
        # return 1 / (1+ (actual_travel_time/t_min_od))
        # return -1.2

        # TODO: BELOW IS WRONG FORMULA, NEED TO FIX
        # distance =  self.agents_manager.edges_distance[next_edge][dest_edge]
        # t_min_next_edge = distance / self.agents_manager.max_speed_vehs
        # reward  = (0.00001 + t_min_next_edge) / (0.00001 + actual_travel_time)
        # return -reward

        return -1.2

    def get_penalty_loop_R1_in_dead_edge(self, max_simu_steps):
        return -2*max_simu_steps
    
    def get_penalty_loop_R1R2_in_dead_edge(self):
        return -1.2

    def get_reward_offline_training(self, curr_edge, next_edge, dest_edge, 
                                     disaster_move):
            """
            Normalize distance to time unit
            [m] -> [s]
            This reward combine between travel time and distance to nearest disaster coordinate
            final_reward
            """
            ###### travel time reward ######
            travel_time_reward = 0.0
            BIG_PENALTY = self.agents_manager.CONFIG['simulation_steps'] 

            route = traci.simulation.findRoute(next_edge, dest_edge, routingMode=traci.constants.ROUTING_MODE_AGGREGATED)
            
            if len(route.edges) != 0:  
                travel_time = route.travelTime
                travel_time_reward = -travel_time
            else:
                travel_time_reward = -BIG_PENALTY
                logger.debug(f"penalty for no route from {next_edge} to {dest_edge}: {travel_time_reward}")
            logger.debug(f"travel time reward: {travel_time_reward} for the edge chosen {next_edge}, at current edge {curr_edge}")
            

            if not disaster_move:
                logger.error("No disaster POI file provided, returning travel time reward only.")
                return travel_time_reward

            # otherwise, we calculate the risk reward
            risk_reward = -disaster_move.get_risk_edge(next_edge)
            if not risk_reward:
                logger.error(f"Not found risk for this edge. Return only trave time")
                return travel_time_reward

            logger.debug(f"Travel time reward {travel_time_reward}, Risk reward {risk_reward}")
            final_reward = self.agents_manager.w1*travel_time_reward + self.agents_manager.w2*risk_reward
            logger.debug(f"Final reward of choosing next edge {next_edge}: {final_reward}")
            return final_reward




  