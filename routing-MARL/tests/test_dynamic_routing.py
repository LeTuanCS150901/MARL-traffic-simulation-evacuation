import traci
import xml.etree.ElementTree as ET

def extract_trip_destinations(xml_file, print_output=False):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    trip_destinations = {}
    for trip in root.findall('trip'):
        trip_id = trip.attrib['id']
        trip_from = trip.attrib['from']
        trip_to = trip.attrib['to']
        depart = trip.attrib.get('depart', '0')  # Default depart time to 0 if not specified
        trip_destinations[trip_id] = {
            'from': trip_from,
            'to': trip_to,
            'depart': depart
        }

    if print_output:
        print("trip_destinations = {")
        for k, v in trip_destinations.items():
            print(f' "{k}": {v},')
        print("}")
    return trip_destinations

def main(trips_to_control, trip_destinations):
    """
    Test A* routing algorithm for trips 0-9 using SUMO and TraCI.
    Ends simulation when all controlled vehicles have arrived.
    """
    rerouted = set()
    arrived = set()

    # Find the maximum departure time among all trips
    max_departure_time = max(float(trip_destinations[veh_id]['depart']) for veh_id in trips_to_control)
    """
    Dynamic routing
    """
    step = 0
    while len(arrived) < len(trips_to_control):
        traci.simulationStep()
        current_time = traci.simulation.getTime()
        step +=1 
        # Get current vehicles in simulation
        current_vehicles = set(traci.vehicle.getIDList())

        # Reroute new vehicles from our target list
        for veh_id in current_vehicles:
            if veh_id in trips_to_control:
                traci.vehicle.rerouteTraveltime(veh_id, currentTravelTimes=True)
                rerouted.add(veh_id)
                print(f"[{current_time:.1f}s] Rerouted {veh_id}")

        # Check for arrived vehicles (vehicles that were rerouted but no longer in simulation)
        for veh_id in trips_to_control:
            if step > max_departure_time:
                if veh_id not in current_vehicles:
                    arrived.add(veh_id)
                    # print(f"[{current_time:.1f}s] {veh_id} arrived at destination")

        # Optional: Break if no vehicles are expected and all our vehicles arrived
        # if traci.simulation.getMinExpectedNumber() == 0 and len(arrived) == len(rerouted):
        #     print("break")
        #     break

    print("All target vehicles have arrived. Ending simulation.")
    
if __name__ == "__main__":
    # THIS cmd is static routing with astar
    # this equivilant to sumo -c sumo.cfg
    """
    The default weight interval is 1 => do routing with current travel time
    however, in sumo cfg, it do routing only 1 time.
    For this script, i do rerouteTraveltime() for each time step, with edges as weighted for 4 previous travel times history
    => dynamic routing
    """
    cfg_file = '../network/manhattan2/manhattan2.sumocfg'
    demand_file = '../network/manhattan2/manhattan2.rou.xml'
    route_file = '../network/manhattan2/routes.xml'

    # cfg_file = '../network/toynet2/net2.sumocfg'
    # demand_file = '../network/toynet2/net2.rou.xml'
    # route_file = '../network/toynet2/routes.xml'

    sumo_cmd = ["sumo", "-c", cfg_file,
                "--routing-algorithm", "dijkstra", "--device.rerouting.adaptation-steps", "4"] #astar
    traci.start(sumo_cmd)

    # extract controlled trips
    trip_destinations = extract_trip_destinations(demand_file, print_output=True)
    trips_to_control = set(trip_destinations.keys())

    # run simulation
    main(trips_to_control, trip_destinations)
    traci.close()

