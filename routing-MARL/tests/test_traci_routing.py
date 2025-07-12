import traci
import traci.constants as tc  # Import TraCI constants explicitly

'''
<?xml version='1.0' encoding='UTF-8'?>  
<routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"  
    xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/routes_file.xsd">  
    <vType id="type1" length="5.00" maxSpeed="70.00" accel="2.6" decel="4.5" sigma="0.5" />  
    <!-- <vType id="yellowType" vClass="emergency" color="yellow" length="5.00" maxSpeed="70.00" accel="2.6" decel="4.5" sigma="0.5">
        <param key="priority" value="1"/>
    </vType> -->

    <!-- Place flow first since it begins at time 0 -->  
    <flow id="f1" type="type1" from="gneE22" to="gneE24" begin="0" end="100" vehsPerHour="3600"/>  
    
    <!-- Then your trips in order -->  
    <trip id="trip_0" depart="20" from="gneE6" to="gneE19" type="type1" color="blue" />  
    <trip id="trip_1" depart="22" from="gneE6" to="gneE19" type="type1" color="blue" />  

</routes>

'''
def main():
    """
    Test default dijkstra algorithm on time dependent network
    """
    sumo_cmd = ["sumo-gui", "-c", "../network/sumo.sumocfg",
                 "--routing-algorithm", "astar",  # Use the A* routing algorithm
                 "--device.rerouting.adaptation-steps", "3"]
    traci.start(sumo_cmd)


    dest = "gneE13"

    net_edges = traci.edge.getIDList()
    trip_controlled = ["trip_0", "trip_1", "trip_2", "trip_3"]
    # trip_controlled = ["trip_0", "trip_1"]

    reroute_time = [1, 25, 32]  # seconds simulation time to trigger rerouting
    simulation_steps = 100

    #Add vehicles with random routes initially
    for veh_id in traci.vehicle.getIDList():
        traci.vehicle.setRoutingMode(veh_id, tc.ROUTING_MODE_AGGREGATED)

    for step in range(simulation_steps):
        traci.simulationStep()

        sim_time = traci.simulation.getTime()

        # # At reroute_time, reroute all vehicles based on current travel times
        # if step in reroute_time:
        #     print(f"*** Rerouting all vehicles at simulation time {sim_time}s ***")

        for veh_id in traci.vehicle.getIDList():
                            if veh_id not in trip_controlled:
                                continue

                            print("current vehicle id:", veh_id)
                            current_edge = traci.vehicle.getRoadID(veh_id)
                            travel_time = traci.edge.getTraveltime(current_edge)
                            # print(f"current travel time of edge {current_edge}: {travel_time}")
                            if current_edge is None or current_edge == "":
                                # vehicle might already have arrived/left network
                                continue

                            if current_edge.startswith(":"):
                                # print(f"Current edge {current_edge} is internal. Keeping original route for vehicle {veh_id}.")
                                continue
                            
                            if current_edge != dest and current_edge == "gneE5":
                                travel_time_5 = traci.edge.getTraveltime("gneE5") 
                                print(f"the travel time of edge gneE5 is {travel_time_5}")  

                                travel_time_27 = traci.edge.getTraveltime("gneE27") 
                                print(f"the travel time of edge gneE27 is {travel_time_27}")  

                                travel_time_7 = traci.edge.getTraveltime("gneE7") 
                                print(f"the travel time of edge gneE7 is {travel_time_7}")  
                                
                                travel_time_8 = traci.edge.getTraveltime("gneE8")
                                print(f"the travel time of edge gneE8 is {travel_time_8}")  

                                travel_time_9 = traci.edge.getTraveltime("gneE9")
                                print(f"the travel time of edge gneE9 is {travel_time_9}")  

                                # print(f"travel time of route 1 {travel_time_27+travel_time_7}")
                                # print(f"travel time of route 2 {travel_time_8+travel_time_9}")

                                print(f"prepare do routing")
                                # this one is dynamic travel time routing
                                # traci.vehicle.rerouteTraveltime(veh_id, currentTravelTimes=True)
                                new_route = traci.simulation.findRoute(current_edge, dest, 
                                                                       routingMode=tc.ROUTING_MODE_AGGREGATED)

                                # After rerouting, retrieve the updated route of the vehicle
                                # updated_route = traci.vehicle.getRoute(veh_id)
                                # print(f"Updated route for vehicle {veh_id}: {updated_route}")
                                # print(new_route.travelTime)
                                # print(new_route.edges)                    
                    
                                # find route with routing mode default (uses current travel times internally)
                                # new_route = traci.simulation.findRoute(current_edge, dest, 
                                #                                     routingMode=tc.ROUTING_MODE_AGGREGATED)
                                traci.vehicle.setRoute(veh_id, new_route.edges)
                                # print(f"new route for vehicle {veh_id}: {new_route}")

                                # edges = list(new_route.edges)
                                # edges[2] = 'gneE27'
                                # edges[3] = 'gneE7'
                            
                                # edges = ['gneE5', 'gneE27', 'gneE7', 'gneE12']
                                # traci.vehicle.setRoute(veh_id, tuple(edges))

                                # if new_route.edges:
                                #     print(f"Rerouting vehicle {veh_id} from {current_edge} to {dest} with route: {new_route.edges}")
                                #     traci.vehicle.setRoute(veh_id, new_route.edges)
                                # else:
                                #     print(f"No route found for vehicle {veh_id} from {current_edge} to {dest}")


    traci.close()

if __name__ == "__main__":
    main()