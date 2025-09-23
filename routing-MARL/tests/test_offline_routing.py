import traci
import xml.etree.ElementTree as ET
import traci.constants as tc  # Import TraCI constants explicitly

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

def predefine_routes(trip_destinations):
    print("Precomputing routes for trips...")
    vehicle_routes = {}
    for veh_id in trip_destinations:
        from_edge = trip_destinations[veh_id]['from']
        to_edge = trip_destinations[veh_id]['to']
        route = traci.simulation.findRoute(from_edge, to_edge).edges
        vehicle_routes[veh_id] = route
        # print(f"Computed route for {veh_id} from {from_edge} to {to_edge} with edges: {route}")
    return vehicle_routes

def main(trips_to_control, trip_destinations):
    arrived = set()
    step = 0
    # Find the maximum departure time among all trips
    max_departure_time = max(float(trip_destinations[veh_id]['depart']) for veh_id in trips_to_control)
    print(f"Maximum departure time: {max_departure_time:.1f}s")

    # Precompute routes for all vehicles
    vehicle_routes = predefine_routes(trip_destinations)
    routed_vehicles = set()

    while len(arrived) < len(trips_to_control):
        traci.simulationStep()
        current_time = traci.simulation.getTime()
        current_vehicles = set(traci.vehicle.getIDList())
        print(f"[{current_time:.1f}s]")

        # Set route and routing mode for vehicles that have appeared in the simulation
        for veh_id in current_vehicles:
            if veh_id in trips_to_control and veh_id not in routed_vehicles:
                traci.vehicle.setRoutingMode(veh_id, tc.ROUTING_MODE_DEFAULT)
                traci.vehicle.setRoute(vehID=veh_id, edgeList=vehicle_routes[veh_id])
                print(f"Set route for {veh_id} at time {current_time:.1f}s")
                routed_vehicles.add(veh_id)

        #  # Check for arrived vehicles (vehicles that were rerouted but no longer in simulation)
        for veh_id in routed_vehicles:
            if veh_id not in current_vehicles and veh_id not in arrived:
                arrived.add(veh_id)
                print(f"[{current_time:.1f}s] {veh_id} arrived at destination ({len(arrived)}/{len(trips_to_control)})")


    print("All target vehicles have arrived. Ending simulation.")

if __name__ == "__main__":
    cfg_file = '../network/manhattan2/manhattan2.sumocfg'
    demand_file = '../network/manhattan2/manhattan2.rou.xml'

    sumo_cmd = ["sumo-gui", "-c", cfg_file, "--routing-algorithm", "dijkstra"] #astar

    traci.start(sumo_cmd)

    trip_destinations = extract_trip_destinations(demand_file, print_output=True)
    trips_to_control = set(trip_destinations.keys())


    # Start simulation
    main(trips_to_control, trip_destinations)

    traci.close()
