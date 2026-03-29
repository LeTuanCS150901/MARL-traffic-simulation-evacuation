import traci
import sys
import os
import time
import numpy as np

from logs.logger import logger  # import our custom logger
from utils.check_end_episode import CheckEndEpisode

# --- SETUP ---
if 'SUMO_HOME' not in os.environ:
    logger.error("Please set the SUMO_HOME environment variable")
    sys.exit("Please set the SUMO_HOME environment variable")

sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))

def set_routing_mode(trips, vehicle_ids):
    """
    Set the routing for vehicles to always routing based on smoothed travel time of traffic condition
    """
    # logger.debug("Setting mode for vehicles routing based on smoothed travel time")
    # Get the list of active vehicle IDs in the simulation.
    # vehicle_ids = traci.vehicle.getIDList()
    # filter trips that we control. Other trips background are controlled default by SUMO
    controlled_vehicle_ids = np.intersect1d(vehicle_ids, list(trips.keys())) 
    for vid in controlled_vehicle_ids:
        traci.vehicle.setRoutingMode(vid, traci.constants.ROUTING_MODE_AGGREGATED)

    return controlled_vehicle_ids

# def run_simulation(edges, trips, last_edge, multi_agents, CONFIG, simulation_steps=700, disaster_move=None):  
#     """  
#     Launch SUMO simulation using TraCI, and interact with vehicles  
#     by looking up edges in a NetworkX graph for re-routing.  
#     THis is simulation with batch step
#     """  
#     # instance to check if the episode has ended  
#     episode_checker = CheckEndEpisode(trips)  
#     latest_depart_time = max(int(info['depart'])   
#                              for info in trips.values()) + 10  
#     # run simulation  
#     for step in range(simulation_steps):  

#         logger.info(f"Step {step}")  
#         # Advance the simulation by one step.  
#         traci.simulationStep()  

#         if disaster_move:  
#             # simulate disaster evolution  
#             disaster_move.simulate_points(step)  
#             # update risk parameters after 30s interval  
#             if step % 30 == 0:  
#                 disaster_move.compute_risks_numba()  
#                 logger.success(f"Recalculated risks and Update new edge costs at step {step}")  
#                 multi_agents.apply_new_edges_weight(disaster_move)  

#         # simulate vehicles routing  
#         try:  
#             vehicle_ids = traci.vehicle.getIDList()  
#             controlled_vehicle_ids = set_routing_mode(trips, vehicle_ids)  

#             # ============================================================  
#             # PHASE 1: Collect data for all vehicles that need routing  
#             # ============================================================  
#             vehicles_to_route = []  # List of (vid, curr_edge, connected_edges) tuples  
#             vehicles_to_remove = []  # Vehicles that reached destination or dead edge  

#             current_time = traci.simulation.getTime()          
#             for vid in controlled_vehicle_ids:  
#                 if episode_checker.is_trip_completed(vid):  
#                     logger.info(f"Vehicle {vid} has already completed its trip. Skipping re-routing.")  
#                     continue  

#                 # Retrieve the current edge where the vehicle is located.  
#                 curr_edge = traci.vehicle.getRoadID(vid)  

#                 # If the current edge is internal, skip re-routing.  
#                 if curr_edge.startswith(":"):  
#                     logger.warning(f"Current edge {curr_edge} is internal. Keeping original route for vehicle {vid}.")  
#                     continue  
                
#                 # If the vehicle is still on the same edge as last time, skip.  
#                 if vid in last_edge and last_edge[vid] == curr_edge:  
#                     continue  
                
#                 # Record the current edge for the vehicle.  
#                 logger.info(f"[Step {step}] Vehicle {vid} is on edge: {curr_edge}")  
#                 last_edge[vid] = curr_edge    
#                 # Save the current edge into history edges of vehicle   
#                 episode_checker.set_edge_history_trip(vid, curr_edge)  

#                 # Check if the current edge exists within our edge mapping.  
#                 if curr_edge in edges:  
#                     # Check if the vehicle has reached its destination.  
#                     if curr_edge in CONFIG['destinations_edges']:  
#                         if curr_edge == trips[vid]['to']:  
#                             logger.success(f"Vehicle {vid} has reached its dest: {curr_edge}. Stopping vehicle.")  
#                             episode_checker.mark_trip_completed(vid)  
#                         else:  
#                             logger.warning(f"Vehicle {vid} has reached wrong dest: {curr_edge}. Stopping vehicle.")   
                        
#                         # current_time = traci.simulation.getTime()   
#                         episode_checker.mark_time_vehicle_removed(vid, current_time)  
#                         vehicles_to_remove.append(vid)  
#                         continue  

#                     # Get the list of possible next edges  
#                     connected_edges = edges[curr_edge]["connected_edges"]  
#                     if not connected_edges:  
#                         logger.warning(f"No connected edges found for {curr_edge}.")  
#                         logger.warning(f"Vehicle {vid} disappear at {curr_edge}.")  
#                         continue  
                    
#                     logger.info(f"Vehicle {vid} can take these next edges: {connected_edges}")  
                    
#                     # Add to batch processing list  
#                     vehicles_to_route.append((vid, curr_edge, connected_edges))  
                    
#                 else:  
#                     logger.warning(f"{curr_edge} not found in edge map")  
#                     continue  

#             # Remove vehicles that reached destination  
#             for vid in vehicles_to_remove:  
#                 traci.vehicle.remove(vid)  

#             # ============================================================  
#             # PHASE 2: Check offline training mode  
#             # ============================================================  
#             try:  
#                 offline_training = CONFIG['offline_training']  
#             except:  
#                 logger.warning("No offline training config found, set to False")  
#                 offline_training = False  

#             # ============================================================  
#             # PHASE 3: Batch process routing decisions  
#             # ============================================================  
#             routing_results = []  
#             if vehicles_to_route:  
#                 if offline_training:  
#                     # Offline training: collect data individually (no batching needed)  
#                     for vid, curr_edge, connected_edges in vehicles_to_route:  
#                         next_edge = multi_agents.collect_offline_data(vid, curr_edge, connected_edges, disaster_move)  
#                         routing_results.append((vid, curr_edge, next_edge, None))  
#                 else:  
#                     # Online: batch process all vehicles  
#                     batch_results = multi_agents.batch_step(  
#                         vehicles_to_route,  # List of (vid, curr_edge, connected_edges)  
#                         disaster_move  
#                     )  
#                     for i, (vid, curr_edge, connected_edges) in enumerate(vehicles_to_route):  
#                         next_edge, append_route = batch_results[i]  
#                         routing_results.append((vid, curr_edge, next_edge, append_route))  

#             # ============================================================  
#             # PHASE 4: Apply routing decisions  
#             # ============================================================  
#             for vid, curr_edge, next_edge, append_route in routing_results:  
#                 logger.info(f"selected next_edge: {next_edge}")  
                
#                 if next_edge:    
#                     # Build a new route for the vehicle consisting of the current edge followed by the chosen next edge + subsequence edges.  
#                     if append_route:  
#                         new_route = [curr_edge] + append_route  
#                         logger.success(f"Append route successfully to veh {vid}")  
#                     else:  
#                         new_route = [curr_edge, next_edge]   

#                     traci.vehicle.setRoute(vid, new_route)  

#                 else:  
#                     # If no valid next edge is found => current edge lead to dead edge  
#                     logger.warning(f"Current edge {curr_edge} of vehicle {vid} leads to dead edge. Remove vehicle from simulation")  
#                     traci.vehicle.remove(vid)              

#             # After the current simulation step, trigger training of transitions of all current vehicles  
#             multi_agents.end_timestep(time_step=current_time)  
#             multi_agents.reset_timestep_rewards()  
#         except Exception as e:  
#             # Log any exceptions that occur during the simulation step.  
#             logger.exception(f"Error during simulation step {step}: {e}")  
#             continue  
        
#         if episode_checker.is_ended_episode(trips):  
#             logger.success("-------------------All trips have arrrived. Episode ended.")  
#             logger.success(f"Break the SIMULATION TIME at {step} steps")  
#             break  

#         # TODO: if all the controlled vehicles disappear (reach dest, or go to dead loop), break the simulation loop  
#         if step > latest_depart_time:  
#             if len(np.intersect1d(vehicle_ids, list(trips.keys()))) == 0:  
#                 logger.success("-------------------All trips have disappeard. Episode ended.")  
#                 logger.success(f"Break the SIMULATION TIME at {step} steps")  
#                 break  
    
#     # TODO: After the end of episode, add -Penalty for vehicle not reach destination in time limit  
#     not_completed_trips = episode_checker.get_trips_not_completed(trips)  
#     if not_completed_trips:  
#         logger.info(f"Add Penalty for trips not completed after episode: {len(not_completed_trips)} / {len(trips)}")  
#         # multi_agents.penalty_agents_loops(not_completed_trips, episode_checker)  

#     # After the end of episode, add Reward for vehicle that reached destination in time limit  
#     completed_trips = episode_checker.get_trips_completed()  
#     if completed_trips:  
#         logger.info(f"Add reward for trips completed after episode: {len(completed_trips)} / {len(trips)}")  
#         # multi_agents.add_reward_reaching


    
#     # Write the number of arrived vehicles at the end of episode
#     # TOOD: user traci.getArrived instead
#     multi_agents.save_arrived_vehicles(len(completed_trips))
  

#     # Close the TraCI connection after the simulation ends.
#     # traci.close(wait=True)
#     traci.close()
#     multi_agents.plotter.write_survived_vehicles()

def run_simulation(edges, trips, last_edge, multi_agents, CONFIG, simulation_steps=700, disaster_move=None):
    """
    Launch SUMO simulation using TraCI, and interact with vehicles
    by looking up edges in a NetworkX graph for re-routing.
    """
    # instance to check if the episode has ended
    episode_checker = CheckEndEpisode(trips)
    latest_depart_time = max(int(info['depart']) 
                             for info in trips.values()) + 10
    # run simultion
    for step in range(simulation_steps):

        logger.info(f"Step {step}")
        # Advance the simulation by one step.
        traci.simulationStep()


        # # Check teleportinng vehs due to jam => later we add penalty for them
        # teleport_vehs= traci.vehicle.getTeleportingIDList()

        # if len(teleport_vehs) > 0:
        #     # print("Add teleporting vehicles to episode checker: ", teleport_vehs)
        #     episode_checker.add_teleport_vehicle(teleport_vehs)

        if disaster_move:
            # simulate disaster evolution
            disaster_move.simulate_points(step)
            # update risk parameters after 30s interval
            if step % 30 ==0:
                disaster_move.compute_risks_numba()
                logger.success(f"Recalculated risks and Update new edge costs at step {step}")
                multi_agents.apply_new_edges_weight(disaster_move)

        # simulate vehicles routing
        try:
            # # Get the list of active vehicle IDs in the simulation.
            # vehicle_ids = traci.vehicle.getIDList()
            # # filter trips that we control. Other trips background are controlled default by SUMO
            # controlled_vehicle_ids = np.intersect1d(vehicle_ids, list(trips.keys())) 
            vehicle_ids = traci.vehicle.getIDList()
            controlled_vehicle_ids = set_routing_mode(trips, vehicle_ids)

            current_time = traci.simulation.getTime()
            
            for vid in controlled_vehicle_ids:
                if episode_checker.is_trip_completed(vid):
                    logger.info(f"Vehicle {vid} has already completed its trip. Skipping re-routing.")
                    continue
                    
                # Retrieve the current edge (i.e., road segment) where the vehicle is located.
                curr_edge = traci.vehicle.getRoadID(vid)

                # If the current edge is internal (which starts with ":"), skip re-routing.
                if curr_edge.startswith(":"):
                    logger.warning(f"Current edge {curr_edge} is internal. Keeping original route for vehicle {vid}.")
                    continue
                
                # If the vehicle is still on the same edge as last time,
                # skip processing this vehicle for re-routing.
                if vid in last_edge and last_edge[vid] == curr_edge:
                    # logger.info(f"Vehicle {vid} still on edge {curr_edge} from previous step; skipping re-route update.")
                    continue
                # otherwise, Record the current edge for the vehicle.
                logger.info(f"[Step {step}] Vehicle {vid} is on edge: {curr_edge}")
                last_edge[vid] = curr_edge    
                # Save the current edge into history edges of vehicle 
                episode_checker.set_edge_history_trip(vid, curr_edge)

                # Check if the current edge exists within our edge mapping (created by our network graph).
                if curr_edge in edges:
                    # Check if the vehicle has reached its destination.  
                    if curr_edge in CONFIG['destinations_edges']:
                        # if this is a true dest
                        if curr_edge == trips[vid]['to']:  
                            logger.success(f"Vehicle {vid} has reached its dest: {curr_edge}. Stopping vehicle.")  
                            episode_checker.mark_trip_completed(vid) # mark trip as completed
                        # if this is a wrong dest
                        else:
                            logger.warning(f"Vehicle {vid} has reached wrong dest: {curr_edge}. Stopping vehicle.") 
                        
                        # current_time = traci.simulation.getTime() 
                        episode_checker.mark_time_vehicle_removed(vid, current_time)

                        # NEW: Clean up the history for this vehicle to free memory
                        multi_agents.cleanup_vehicle_history(vid)


                        traci.vehicle.remove(vid)  # Remove the vehicle from the simulation  
                        # mark time when vehicle is removed from simulation
                        continue  # Skip further processing since the vehicle has arrived.  

                    # Get the list of possible next edges that are connected to the current edge.
                    connected_edges = edges[curr_edge]["connected_edges"]
                    if not connected_edges:
                        logger.warning(f"No connected edges found for {curr_edge}.")
                        logger.warning(f"Vehicle {vid} dissapear at {curr_edge}.")
                        continue
                    logger.info(f"Vehicle {vid} can take these next edges: {connected_edges}")
                    # Randomly select one of the connected edges as the next edge.
                    # next_edge = random.choice(connected_edges) if connected_edges else None
                    
                    try: # check if we run offline training for data collection
                        offline_training = CONFIG['offline_training']
                    except:
                        logger.warning("No offline training config found, set to False")
                        offline_training = False

                    # if do offline training
                    if offline_training:
                        next_edge = multi_agents.collect_offline_data(vid, curr_edge, connected_edges, disaster_move)
                    else:
                        next_edge, append_route = multi_agents.step(vid, curr_edge, connected_edges, disaster_move, episode_checker)
                    # TODO: check if offline RL => then collect data for training
                    logger.info(f"selected next_edge: {next_edge}")
                    
                else:
                    # This is dead edge
                    # Log a warning if the current edge is not found in our edge mapping.
                    logger.warning(f"{curr_edge} not found in edge map")
                    continue

                if next_edge:    
                    # Build a new route for the vehicle consisting of the current edge followed by the chosen next edge + subsequence edges.
                    # this help to avoid SUMO bug of removing vehicles when we do edge-based routing
                    # but in reality, the vehicle only go to next chosen edge, and the 
                    # rerouting will be repeated again and not follow subsequence edges
                    if append_route:
                        new_route = [curr_edge] + append_route
                        logger.success(f"Append route successfully to veh {vid}")
                    else:
                        new_route = [curr_edge, next_edge] 

                    traci.vehicle.setRoute(vid, new_route)

                else:
                    # If no valid next edge is found => current edge lead to dead edge
                    logger.warning(f"Current edge {curr_edge} of vehicle {vid} leads to dead edge. Remove vehicle from simulation")
                    traci.vehicle.remove(vid)              

            # After the current simulation step, trigger training of transitions of all current vehicles
            multi_agents.end_timestep(time_step=current_time)
            multi_agents.reset_timestep_rewards()
        except Exception as e:
            # Log any exceptions that occur during the simulation step.
            logger.exception(f"Error during simulation step {step}: {e}")
            continue
        
        if episode_checker.is_ended_episode(trips):
            logger.success("-------------------All trips have arrrived. Episode ended.")
            logger.success(f"Break the SIMULATION TIME at {step} steps")
            break  

        # TODO: if all the controlled vehicles disappear (reach dest, or go to dead loop), break the simulation loop
        if step > latest_depart_time:
            if len(np.intersect1d(vehicle_ids, list(trips.keys()))) == 0:
                logger.success("-------------------All trips have disappeard. Episode ended.")
                logger.success(f"Break the SIMULATION TIME at {step} steps")
                break
    
    # TODO: After the end of episode, add -Penalty for vehicle not reach destination in time limit
    not_completed_trips = episode_checker.get_trips_not_completed(trips)
    if not_completed_trips:
        logger.info(f"Add Penalty for trips not completed after episode: {len(not_completed_trips)} / {len(trips)}")
        # multi_agents.penalty_agents_loops(not_completed_trips, episode_checker)

    # After the end of episode, add Reward for vehicle that reached destination in time limit
    completed_trips = episode_checker.get_trips_completed()
    if completed_trips:
        logger.info(f"Add reward for trips completed after episode: {len(completed_trips)} / {len(trips)}")
        # multi_agents.add_reward_reaching_dest(completed_trips, episode_checker)

    # Write the number of arrived vehicles at the end of episode
    # TOOD: user traci.getArrived instead
    multi_agents.save_arrived_vehicles(len(completed_trips))
  

    # Close the TraCI connection after the simulation ends.
    # traci.close(wait=True)
    traci.close()
    multi_agents.plotter.write_survived_vehicles()

def run_simulation_inference(edges, trips, last_edge, multi_agents, CONFIG, simulation_steps=1000, disaster_move=None):
    """
    Launch SUMO simulation using TraCI, and interact with vehicles
    by looking up edges in a NetworkX graph for re-routing.
    """
    # instance to check if the episode has ended
    episode_checker = CheckEndEpisode(trips)

    latest_depart_time = max(int(info['depart']) 
                             for info in trips.values()) 

    # run simultion
    for step in range(simulation_steps):

        logger.info(f"Step {step}")
        # Advance the simulation by one step.
        traci.simulationStep()

        if disaster_move:
            # simulate disaster evolution
            disaster_move.simulate_points(step)
            # update risk parameters after 30s interval
            if step % 30 ==0:
                disaster_move.compute_risks_numba()
                logger.success(f"Recalculated risks and Update new edge costs at step {step}")
                multi_agents.apply_new_edges_weight(disaster_move)
        try:
            # # Get the list of active vehicle IDs in the simulation.
            # vehicle_ids = traci.vehicle.getIDList()
            # # filter trips that we control. Other trips background are controlled default by SUMO
            # controlled_vehicle_ids = np.intersect1d(vehicle_ids, list(trips.keys())) 
            vehicle_ids = traci.vehicle.getIDList()
            controlled_vehicle_ids = set_routing_mode(trips, vehicle_ids)

            for vid in controlled_vehicle_ids:

                if episode_checker.is_trip_completed(vid):
                    logger.info(f"Vehicle {vid} has already completed its trip. Skipping re-routing.")
                    continue

                # Retrieve the current edge (i.e., road segment) where the vehicle is located.
                curr_edge = traci.vehicle.getRoadID(vid)
                # curr_edge =  traci.vehicle.getLaneID(vid)

                # If the current edge is internal (which starts with ":"), skip re-routing.
                if curr_edge.startswith(":"):
                    logger.warning(f"Current edge {curr_edge} is internal. Keeping original route for vehicle {vid}.")
                    continue
                
                # If the vehicle is still on the same edge as last time,
                # skip processing this vehicle for re-routing.
                if vid in last_edge and last_edge[vid] == curr_edge:
                    # logger.info(f"Vehicle {vid} still on edge {curr_edge} from previous step; skipping re-route update.")
                    continue
                # otherwise, Record the current edge for the vehicle.
                logger.info(f"[Step {step}] Vehicle {vid} is on edge: {curr_edge}")
                last_edge[vid] = curr_edge    
                # Save the current edge into history edges of vehicle 
                episode_checker.set_edge_history_trip(vid, curr_edge)

                # Check if the current edge exists within our edge mapping (created by our network graph).
                if curr_edge in edges:
                    # Check if the vehicle has reached its destination.  
                    if curr_edge in CONFIG['destinations_edges']:
                        # if this is a true dest
                        if curr_edge == trips[vid]['to']:  
                            logger.success(f"Vehicle {vid} has reached its dest: {curr_edge}. Stopping vehicle.")  
                            episode_checker.mark_trip_completed(vid) # mark trip as completed
                        # if this is a wrong dest
                        else:
                            logger.warning(f"Vehicle {vid} has reached wrong dest: {curr_edge}. Stopping vehicle.") 
                         
                        # traci.vehicle.remove(vid)  # Remove the vehicle from the simulation  
                        continue  # Skip further processing since the vehicle has arrived.  

                    # Get the list of possible next edges that are connected to the current edge.
                    connected_edges = edges[curr_edge]["connected_edges"]
                    if not connected_edges:
                        logger.warning(f"No connected edges found for {curr_edge}.")
                        logger.warning(f"Vehicle {vid} dissapear at {curr_edge}.")
                        continue
                    logger.info(f"Vehicle {vid} can take these next edges: {connected_edges}")
                    # Randomly select one of the connected edges as the next edge.
                    # next_edge = random.choice(connected_edges) if connected_edges else None
                    next_edge, append_route = multi_agents.step(vid, curr_edge, connected_edges, disaster_move)
                    logger.info(f"selected next_edge: {next_edge}")
                    
                else:
                    # This is dead edge
                    # Log a warning if the current edge is not found in our edge mapping.
                    logger.warning(f"{curr_edge} not found in edge map")
                    continue

                if next_edge:    
                     # Build a new route for the vehicle consisting of the current edge followed by the chosen next edge.
                    if append_route:
                        new_route = [curr_edge] + append_route
                        logger.success(f"Append route successfully to veh {vid}")
                    else:
                        new_route = [curr_edge, next_edge]
                                  
                    traci.vehicle.setRoute(vid, new_route)
                    logger.info(f"Vehicle {vid} choose new edge: {next_edge}")
                    
                else:
                    # If no valid next edge is found => current edge lead to dead edge
                    logger.warning(f"Current edge {curr_edge} of vehicle {vid} leads to dead edge. Remove vehicle from simulation")
                    traci.vehicle.remove(vid)              

            # Sleep briefly between simulation steps to slow execution for debugging.
            time.sleep(0.1)
        except Exception as e:
            # Log any exceptions that occur during the simulation step.
            logger.exception(f"Error during simulation step {step}: {e}")
            continue
        
        # if step >= latest_depart_time +640:
        #     logger.success("Remove remaining vehs")
        #     [traci.vehicle.remove(veh, reason=2) for veh in controlled_vehicle_ids]

        # after all trips are maked as reach destination
        if episode_checker.is_ended_episode(trips):
            # we check if all the controlled vehicle removed from the simulation or not
            done_simulation = True
            for trip in list(trips.keys()):
                if trip in controlled_vehicle_ids:
                    done_simulation = False

            if done_simulation: 
                # if all the controlled vehicle removed from the simulation
                logger.error("-------------------All trips have arrrived. Episode ended.")
                logger.error(f"Break the SIMULATION TIME at {step} steps") 
                break 
        
        # check trips not completed after simulation
        not_completed_trips = episode_checker.get_trips_not_completed(trips)
        if step > latest_depart_time and not_completed_trips:
            logger.info(f"Trips not completed: {not_completed_trips}")
                
    
    # Close the TraCI connection after the simulation ends.
    traci.close(False)

def reset_simulation(episode, CONFIG):  
    """  
    Resets the SUMO simulation by starting a new simulation instance.  
    """ 
    logger.info(f"Starting SUMO simulation at episode {episode + 1}...")
    
    sumo_cmd = [
                CONFIG['sumo_binary'] , "-c",
                CONFIG['sumo_config'], "--start", 
                "--routing-algorithm", CONFIG['routing_algorithm'],
                "--device.rerouting.adaptation-steps", "4",
                "--collision.action", "none",
                "--collision.mingap-factor", "1.0",
                "--ignore-junction-blocker", "1.0"
                # "--mesosim", "true",
                # "--meso-junction-control", "true"
            ]

    traci.start(sumo_cmd)  # Start a new instance of SUMO  

