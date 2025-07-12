import traci
import xml.etree.ElementTree as ET

def extract_trip_destinations(xml_file, print_output=False):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    trip_destinations = {}
    for trip in root.findall('trip'):
        trip_id = trip.attrib['id']
        trip_to = trip.attrib['to']
        trip_destinations[trip_id] = trip_to

    if print_output:
        print("trip_destinations = {")
        for k, v in trip_destinations.items():
            print(f'    "{k}": "{v}",')
        print("}")
    return trip_destinations

def main(trips_to_control):
    rerouted = set()
    arrived = set()

    step = 0
    while len(arrived) < len(trips_to_control):
        traci.simulationStep()
        current_time = traci.simulation.getTime()
        
        # Get current vehicles in simulation
        current_vehicles = set(traci.vehicle.getIDList())
        
        # Reroute new vehicles from our target list
        for veh_id in current_vehicles:
            if veh_id in trips_to_control and veh_id not in rerouted:
                traci.vehicle.rerouteTraveltime(veh_id, currentTravelTimes=True)
                rerouted.add(veh_id)
                print(f"[{current_time:.1f}s] Rerouted {veh_id}")

        # Check for arrived vehicles (vehicles that were rerouted but no longer in simulation)
        for veh_id in rerouted:
            if veh_id not in current_vehicles and veh_id not in arrived:
                arrived.add(veh_id)
                print(f"[{current_time:.1f}s] {veh_id} arrived at destination ({len(arrived)}/{len(trips_to_control)})")

        # Optional: Break if no vehicles are expected and all our vehicles arrived
        # if traci.simulation.getMinExpectedNumber() == 0 and len(arrived) == len(rerouted):
        #     print("break")
        #     break

    print("All target vehicles have arrived. Ending simulation.")

if __name__ == "__main__":
    """
    The default weight interval is 1 => do routing with current travel time
    In sumo cfg, it do routing only 1 time.  => static (offline) routing
    For this script, i also do rerouteTraveltime() for only one first time step
    => static (offline) routing
    """
    cfg_file = '../network/manhattan/manhattan.sumocfg'
    demand_file = '../network/manhattan/manhattan.rou.xml'
    route_file = '../network/manhattan/routes.xml'

    
    # cfg_file = '../network/toynet2/net2.sumocfg'
    # demand_file = '../network/toynet2/net2.rou.xml'
    # route_file = '../network/toynet2/routes.xml'

    # THIS cmd IS static routing with astar
    # this equivilant to sumo -c sumo.cfg
    sumo_cmd = ["sumo", "-c", cfg_file,
                "--routing-algorithm", "dijkstra", "--device.rerouting.adaptation-steps", "1"] #astar
    traci.start(sumo_cmd)

    # extract controlled trips
    trip_destinations = extract_trip_destinations(demand_file, print_output=True)
    trips_to_control = set(trip_destinations.keys())

    # run simulation
    main(trips_to_control)
    traci.close()
