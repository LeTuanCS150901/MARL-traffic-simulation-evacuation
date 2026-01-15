import xml.etree.ElementTree as ET
import networkx as nx
import sumolib 
import math
from read_config import get_config, read_config
from collections import defaultdict
from logs.logger import logger  # import our custom logger
import numpy as np

class GraphManager:
    def __init__(self, route_file, net_file):
        """
        Initialize the traffic network graph with routes and network information.
        
        Args:
            route_file (str): Path to the SUMO routes XML file
            net_file (str): Path to the SUMO network XML file
        """
        self.CONFIG = get_config()
        self.net = None # net object for xml structure net file of sumo
        self.trips = self.parse_route_file(route_file)

        # Maximum free-flow travel time on an edge
        self.max_travel_time = 0.0  

        self.allowable_edges = self.set_allowable_edges(net_file)
        self.network, self.edges, self.nodes = self.create_network_graph(net_file)
      
        self.origin_edges = self.CONFIG['origin_edges']
        self.dist_OD = {} # euclide dist from O->D
        # {edge: {dest id: distance}, ... }
        self.edges_distance = {} # distance of all edges to destination edges
        self.destination_edges = self.CONFIG['destinations_edges']
        # Maximum distance from an edge to destination edge 
        self.max_edge_to_dest_distance = 0.0  
        
        # {edge: {zone_id: distance}, ...}
        self.edges_dist_disaster = {} # distance of all edges to mean of all disaster zones
        self.disaster_zones = self.CONFIG['disaster']['fire']['disaster_zones'] # disaster edges
        # Store max radius and coordinate of each disaster zone
        self.disaster_info = defaultdict(dict) # {zone_id: {max_radius, coordinate}, ...}  

        # Define a list of roads not suitable for cars
        # self.non_car_road_types = [
        # "highway.cycleway",
        # "highway.footway",
        # "highway.pedestrian",
        # "highway.service",
        # "highway.steps",
        # "railway.subway"
        # ]

    def read_net_file(self, net_file):
        if self.net is None:
            self.net = sumolib.net.readNet(net_file)
        return self.net

    def parse_route_file(self, route_file):
        """
        Parse the route file and extract trip information.
        
        Returns:
            dict: A dictionary of trips with their attributes
        """
        trips = {}
        tree = ET.parse(route_file)
        root = tree.getroot()
        
        for trip in root.findall('trip'):
            trip_id = trip.get('id')
            trips[trip_id] = {
                'from': trip.get('from'),
                'to': trip.get('to'),
                'depart': float(trip.get('depart'))
            }
        
        return trips
    
    def set_allowable_edges(self, net_file):
        tree = ET.parse(net_file)
        root = tree.getroot()

        # result to check
        check_allow, check_disallow = None,None
        result = False
        vehicle = 'passenger' # we only consider passenger car for routing

        allowable_edges = defaultdict(dict)

        # iterate over all edges
        for edge in root.findall("edge"):
            edge_id = edge.get("id")

            for lane in edge.findall("lane"):  
                    
                    lane_id = lane.get("id")
                    allowed_list = None
                    disallowed_list = None
                    try:
                        allowed_list = lane.get("allow")
                        disallowed_list = lane.get("disallow")
                    except Exception as e:
                        logger.warning(f"Error getting allowed/disallowed for lane {lane_id}")

                    # print("Allowed list for edge", edge_id, ":", allowed_list)
                    # first statement check
                    if allowed_list is None:
                        check_allow = True
                    else:
                        if vehicle in allowed_list:
                            check_allow = True
                        else:
                            check_allow = False

                    # second statement check
                    if disallowed_list is None:
                            check_disallow = False 
                    else:
                        if vehicle not in disallowed_list:
                            check_disallow = False
                        else:
                            check_disallow = True

                    # final check using both statements
                    result = (check_allow == True and check_disallow == False)

            allowable_edges[edge_id][vehicle] = result

            # if edge_id == "25166922#0":
            #     print("Checking edge")    
            #     print(result, check_allow, check_disallow)
   
        # if edge.getID() == "25166922#0":
        #     print("Checking edge")    
        #     print(result, check_allow, check_disallow)
        return allowable_edges

    # def create_network_graph(self, net_file):
    #     """
    #     Build a lane-level directed graph from a SUMO .net.xml file.

    #     RETURNS
    #     -------
    #     G      : nx.DiGraph   (nodes = lane-ids, edges = connections)
    #     lanes  : dict lane_id -> attributes, incl. key "connected_edges"
    #     nodes  : list(node_ids)  (unchanged from your previous API)
    #     """

    #     # ------------- helpers ------------------------------------------
    #     def lane_id(edge_id: str, idx: int) -> str:
    #         return f"{edge_id}_{idx}"

    #     # ------------- read the net with sumolib ------------------------
    #     net = sumolib.net.readNet(net_file)

    #     G      = nx.DiGraph()
    #     lanes  = {}          # lane_id -> attribute dict
    #     nodes  = set()       # junction ids (kept for API compatibility)

    #     # ---------- 1st pass : create one graph node per lane -----------
    #     for edge in net.getEdges():
    #         e_id = edge.getID()
    #         if e_id.startswith(":"):           # skip SUMO internal edges
    #             continue

    #         from_node = edge.getFromNode().getID()
    #         to_node   = edge.getToNode().getID()
    #         nodes.update((from_node, to_node))

    #         for idx, lane in enumerate(edge.getLanes()):
    #             lid = lane_id(e_id, idx)

    #             length = lane.getLength()
    #             speed  = max(lane.getSpeed(), 1e-6)
    #             ttime  = length / speed

    #             lanes[lid] = {
    #                 "edge_id"        : e_id,
    #                 "lane_index"     : idx,
    #                 "from_node"      : from_node,
    #                 "to_node"        : to_node,
    #                 "length"         : length,
    #                 "speed"          : speed,
    #                 "travel_time"    : ttime,
    #                 "connected_edges": set(),      # filled in 2nd pass
    #             }

    #             G.add_node(lid)
    #             self.max_travel_time = max(
    #                 getattr(self, "max_travel_time", 0), ttime
    #             )

    #     # ---------- 2nd pass : follow <connection> elements -------------
    #     root = ET.parse(net_file).getroot()
    #     for conn in root.iterfind("connection"):
    #         f_edge = conn.get("from")
    #         t_edge = conn.get("to")
    #         if f_edge.startswith(":") or t_edge.startswith(":"):
    #             continue                                 # ignore internals

    #         lid_from = lane_id(f_edge, int(conn.get("fromLane")))
    #         lid_to   = lane_id(t_edge, int(conn.get("toLane")))

    #         if lid_from not in lanes or lid_to not in lanes:
    #             continue

    #         # add graph edge and store successor lane id
    #         G.add_edge(
    #             lid_from,
    #             lid_to,
    #             travel_time = lanes[lid_to]["travel_time"],
    #             turn_dir    = conn.get("dir", "s"),
    #         )
    #         lanes[lid_from]["connected_edges"].add(lid_to)

    #     # ---------- final clean-up: convert sets to lists ---------------
    #     for lane_data in lanes.values():
    #         lane_data["connected_edges"] = list(lane_data["connected_edges"])

    #     return G, lanes, list(nodes)

    def create_network_graph(self, net_file):
        """
        Create a NetworkX graph from the SUMO network file.
        
        Args:
            net_file (str): Path to the SUMO network XML file
        
        Returns:
            nx.DiGraph: Directed graph representing the network
        """
        non_car_road_types = [
        "highway.cycleway",
        "highway.footway",
        "highway.pedestrian",
        "highway.service",
        "highway.steps",
        "railway.subway"
        ]

        vehicle_allow = 'passenger' # we only consider passenger car for routing
        # non_car_road_types = [
        # "highway.cycleway",
        # "highway.footway",
        # "highway.pedestrian",
        # "highway.steps",
        # "railway.subway"
        # ]

        net = self.read_net_file(net_file)
        G = nx.DiGraph()  
        edges = {}  
        nodes = set()  

        # First pass: Collect all edges and nodes  
        for edge in net.getEdges():  
            edge_id = edge.getID()  
            edge_type = edge.getType()

            # TODO:
            # edge_allow = edge.getAllow()
            if edge_id.startswith(":"):  # Skip internal edges  
                continue  

            # if edge_type in non_car_road_types:  # Skip non-car roads
            #     continue  

            if not self.allowable_edges[edge_id][vehicle_allow]:
                continue

            from_node = edge.getFromNode().getID()  
            to_node = edge.getToNode().getID()  
            length = edge.getLength()  
            speed = edge.getSpeed() if edge.getSpeed() > 0 else 0.0  # m/s
            shape = edge.getShape()

            # if int(length) < 5: # skip the road which is too short for vehicle to cross
            #     continue
            # this is free-flow travel time
            travel_time = length / speed if speed > 0 else float('inf')  

            # 2. Get the number of lanes (CRITICAL STEP)
            # Capacity is per lane, so we must multiply by lane count later
            # (If using sumolib)
            num_lanes = len(edge.getLanes()) 
            # (If using traci and 'edge' is just an ID string, use: traci.edge.getLaneNumber(edge_id))

            # 3. Define Vehicle Parameters (Standard SUMO defaults)
            veh_len = 5.0   # meters
            min_gap = 2.5   # meters
            tau = 1.0       # seconds (reaction time)

            # 4. Calculate Gross Time Headway (The SUMO formula)
            # How long does it take to clear enough space for the next car?
            # Space needed = Vehicle Length + Minimum Gap
            space_required = veh_len + min_gap
            
            # Headway = Reaction Time + (Space Needed / Speed)
            h_gross = tau + (space_required / speed)

            # 5. Calculate Capacity
            # Capacity = (3600 / Headway) * Number of Lanes
            capacity_per_lane = 3600 / h_gross
            final_capacity = capacity_per_lane * num_lanes

            edges[edge_id] = {  
                "from_node": from_node,  
                "to_node": to_node,  
                "connected_edges": set(), 
                "travel_time": travel_time,
                'speed': speed,
                'length': length,
                'shape' : list(shape),
                'capacity': final_capacity # Use the fixed value
            }

            # Keep track of maximum travel time  
            if travel_time != float('inf') and travel_time > self.max_travel_time:  
                self.max_travel_time = travel_time  

            nodes.add(from_node)  
            nodes.add(to_node)  

        # Second pass: Populate connected_edges based on <connection> tags  
        tree = ET.parse(net_file)  
        root = tree.getroot()  
        for connection in root.findall("connection"):  
            from_edge = connection.get("from")  
            to_edge = connection.get("to")  

            if from_edge in edges and to_edge in edges:  
                edges[from_edge]["connected_edges"].add(to_edge)  

        # Add edges to the NetworkX graph  
        for edge_id, edge_data in edges.items():  
            G.add_edge(  
                edge_data["from_node"],  
                edge_data["to_node"],  
                edge_id=edge_id,  
                travel_time=edge_data["travel_time"],  
                connected_edges=list(edge_data["connected_edges"]) 
            )  

        return G, edges, list(nodes)  

    def calculate_distances(self, net_file):
        """
        Calculate the distances of all edges to destination edges
        """
        net = self.read_net_file(net_file)

        for edge in net.getEdges():  
            edge_id = edge.getID()  
            self.edges_distance[edge_id] = {}

            # Get the shape (list of points) of the current edge  
            edge_shape = edge.getShape()  
            if not edge_shape:  
                print(f"Warning: Edge {edge_id} has no shape.  Skipping distance calculation.")  
                continue  # Skip edges without shapes  

            # Skip internal edges  
            if edge_id.startswith(":"): 
                continue  
            
            # # if current edge is a destination edge, set distance to 0.0
            # if edge_id in self.destination_edges:
            #     self.edges_distance[edge_id][dest_edge_id] = 0.0
            # else:
            # Calculate the center point the edge
            x1, y1 = self.calculate_center_point(edge_shape)  
            for dest_edge_id in self.destination_edges:
                # try:  
                dest_edge = net.getEdge(dest_edge_id)  
                dest_edge_shape = dest_edge.getShape() 

                # Calculate the center point of the destination edge  
                x2, y2 = self.calculate_center_point(dest_edge_shape)   

                # Calculate Euclidean distance  
                distance = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)  
                self.edges_distance[edge_id][dest_edge_id] = distance

                # Update max distance  
                if distance > self.max_edge_to_dest_distance:  
                    self.max_edge_to_dest_distance = distance  

                # except Exception as e:  
                #     print(f"Error processing edge {edge_id} to destination edge {dest_edge_id}: {e}")
                
    def calculate_center_point(self, shape):  
        """  
        Calculates the center point (centroid) of a shape (list of points).  

        Args:  
            shape: A list of (x, y) tuples representing the shape.  

        Returns:  
            A tuple (x_center, y_center) representing the center point.  
        """  
        x_sum = 0  
        y_sum = 0  
        for x, y in shape:  
            x_sum += x  
            y_sum += y  
        x_center = x_sum / len(shape)  
        y_center = y_sum / len(shape)  
        return x_center, y_center  
    
    def calculate_distance_to_disaster_zones(self, net_file):
        """
        Calculate distances of all edges to disaster fixed zones
        """
        # escape the function if no disaster available 
        if self.disaster_zones == None:
            logger.warning("No disasters available. Not calculating dist edge->disaster zone")
            return 
        
        net = self.read_net_file(net_file)

         # First, calculate mean points for each disaster zone and their radius  
        for zone_id, disaster_edges in self.disaster_zones.items():  
            mean_x, mean_y = self.get_mean_edges(disaster_edges)  
            max_radius = self.calculate_disaster_radius(disaster_edges, mean_x, mean_y)  
            self.disaster_info[zone_id]['radius'] = max_radius  
            self.disaster_info[zone_id]['coordinate'] = (mean_x, mean_y) #  mean coordinate of the disaster zone
        # Now calculate distance from each edge to each disaster zone  
        for edge in net.getEdges():  
            edge_id = edge.getID()  
            self.edges_dist_disaster[edge_id] = {}

            # Get the shape (list of points) of the current edge  
            edge_shape = edge.getShape()  
            if not edge_shape:  
                print(f"Warning: Edge {edge_id} has no shape.  Skipping distance calculation.")  
                continue  # Skip edges without shapes  

            # Skip internal edges  
            if edge_id.startswith(":"): 
                continue  
        
            x1, y1 = self.calculate_center_point(edge_shape)  
            for zone_id, disaster_edges in self.disaster_zones.items():
                mean_x, mean_y = self.get_mean_edges(disaster_edges)  
                # print("mean---------------", mean_x, mean_y)

                # Calculate Euclidean distance  
                distance = math.sqrt((x1 - mean_x)**2 + (y1 - mean_y)**2)  
                self.edges_dist_disaster[edge_id][zone_id] = distance

    def get_mean_edges(self, edge_ids):  
        """  
        Calculate the mean (average) point (x, y) of a list of edges.  
        Parameters:  
            edge_ids (list of str): List of edge IDs  
        Returns:  
            tuple: (mean_x, mean_y) - The average x and y coordinate of all edge points  
        """  
        net = self.net  # Assumed to be pre-loaded  
        sum_x = 0.0  
        sum_y = 0.0  
        count = 0  

        for edge_id in edge_ids:  
            edge = net.getEdge(edge_id)  
            if edge is None:  
                print(f"Warning: Edge ID {edge_id} not found in network.")  
                continue  
            shape = edge.getShape()  
            if not shape:  
                print(f"Warning: Edge {edge_id} has no shape.")  
                continue  
            for x, y in shape:  
                sum_x += x  
                sum_y += y  
                count += 1  

        if count == 0:  
            print("Warning: No points found in given edges.")  
            return (0.0, 0.0)  

        mean_x = sum_x / count  
        mean_y = sum_y / count  
        return (mean_x, mean_y)
    
    def calculate_disaster_radius(self, edge_ids, mean_x, mean_y):  
        """  
        Calculate (maximum distance) from the farthest (corner) point in the disaster edges to the mean disaster point.  
        This distance represents the radius of the disaster zone.  
        
        Args:  
            edge_ids (list): List of edge IDs in the disaster zone  
            mean_x (float): X-coordinate of the mean point  
            mean_y (float): Y-coordinate of the mean point  
            
        Returns:  
            float: Maximum radius of the disaster zone  
        """  
        net = self.net  
        max_radius = 0.0  
        # check_e = []
        # check_c = []
        
        for edge_id in edge_ids:  
            edge = net.getEdge(edge_id)  
            if edge is None:  
                print(f"Warning: Edge ID {edge_id} not found in network.")  
                continue  
                
            shape = edge.getShape()  
            if not shape:  
                print(f"Warning: Edge {edge_id} has no shape.")  
                continue  
                
            # Check each point in the edge's shape  
            for x, y in shape:  
                # Calculate distance from this point to the mean point  
                distance = (math.sqrt((x - mean_x)**2 + (y - mean_y)**2))  
                if distance > max_radius:  
                    max_radius = distance  
                    # check_e.append(edge_id)
                    # check_c.append((x, y))

        # print("farthest edge", check_e[-1], "and coordinate", check_c[-1])       
        return max_radius  
    
    def calculate_dist_OD(self, net_file):  
        net = self.read_net_file(net_file)

        try:  
            for origin_id in self.origin_edges:  
                if origin_id not in self.dist_OD:  
                    self.dist_OD[origin_id] = {}  
                    
                for dest_id in self.destination_edges:  
                    if dest_id not in self.dist_OD[origin_id]:  
                        origin_edge = net.getEdge(origin_id)  
                        dest_edge = net.getEdge(dest_id)  
                        
                        # Get the shape (coordinates) of the edges  
                        origin_edge_shape = origin_edge.getShape()  
                        dest_edge_shape = dest_edge.getShape()  
                        
                        x1, y1 = self.calculate_center_point(origin_edge_shape)  
                        x2, y2 = self.calculate_center_point(dest_edge_shape)  
                        dist = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)  
                        self.dist_OD[origin_id][dest_id] = dist   
        except Exception as e:  
            logger.error(f"Error calculating dist OD: {e}")


    def euclidean_distance_between_edges(self, edge_id1, edge_id2):
        """
        Calculate the Euclidean distance between the center points of two edges.

        Args:
            edge_id1 (str): The first edge ID.
            edge_id2 (str): The second edge ID.

        Returns:
            float: The Euclidean distance between the two edge centers.
        """
        net = self.net  # Assumed to be pre-loaded
        edge1 = net.getEdge(edge_id1)
        edge2 = net.getEdge(edge_id2)
        if edge1 is None or edge2 is None:
            raise ValueError(f"Edge not found: {edge_id1 if edge1 is None else edge_id2}")
        shape1 = edge1.getShape()
        shape2 = edge2.getShape()
        if not shape1 or not shape2:
            raise ValueError(f"Edge has no shape: {edge_id1 if not shape1 else edge_id2}")
        x1, y1 = self.calculate_center_point(shape1)
        x2, y2 = self.calculate_center_point(shape2)
        return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


if __name__ == "__main__":  

    # Set the config file
    CONFIG = read_config(config_filename='manhattan2')
    
    route_file = CONFIG['route_file']  # route path
    net_file = CONFIG['net_file']  # net path
    graph = GraphManager(route_file, net_file)
    
    # print(graph.trips)
    # print(graph.groups)
    # print(graph.edges)

    # print("Nodes:")  
    # for node in nodes:  
    #     print(f"  {node}")  
    edges = graph.edges
    print(f"\nEdges: {len(edges)}")  
    # for edge in edges:  
    #     print(f" edge {edge}, from_node: {edges[edge]['from_node']}, to_node: {edges[edge]['to_node']}, "
    #           f"travel_time: {edges[edge]['travel_time']}")

    # print(edges['569345540#0']['connected_edges'])
    # lane = "420907908#0_1"               # edge 420907908#0 , lane index 1
    # # print("successor lanes:", edges[lane]["connected_edges"])
    # # print(edges)
    # print("successor lanes:", edges['1376951512_1'])
    
        
    # calculate distance of edges to destination edges
    graph.calculate_distances(net_file)
    graph.calculate_distance_to_disaster_zones(net_file)
    graph.calculate_dist_OD(net_file)
    # print("max travel time", graph.max_travel_time)
    # print("max edge distance to disaster", graph.max_edge_to_dest_distance)
    # print("disaster info\n", graph.disaster_info)
    if not graph.disaster_info:
        print("disaster info is not available" )
    # print("edges distance to disaster\n")
    # print()
    sorted_dist_disaster = sorted(graph.edges_dist_disaster.items(), key=lambda x: x[1][1])
    # Display the result
    # for k, v in sorted_dist_disaster:
        # print(f"{k}: {v}")

    # print("dist OD", graph.dist_OD)
    # print()
    # print("trips", graph.trips)
    print("check connection")
    # print(edges['561042197']['connected_edges'])  # edge 1287670369
    # print(edges['44165030#0']['connected_edges'])  # edge 1287670369
    # print(edges['386972970#0']['connected_edges'])  # edge 1287670369

    print("check allowable edges")
    # print(graph.edges['1159051925#0'])
    # print(graph.edges['-1159051925#1'])

    # print(graph.edges['25166922#0'])
    # print(graph.edges['-25166922#1'])
    print(graph.edges['786982251#0'])



    




