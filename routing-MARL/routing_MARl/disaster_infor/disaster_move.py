import traci
import xml.etree.ElementTree as ET
import math
import numpy as np
from numba import njit, prange

from logs.logger import logger
from graph_manager import GraphManager
from read_config import get_config

class DisasterMove:
    def __init__(self, poi_file, graph):
        self.CONFIG = get_config()  
        self.poi_file = poi_file
        self.graph = graph # graph object to store the network edge's infor

        self.pois = None # list to store POIs
        self.max_time = None # Maximum time for disaster events.
        self.R = 200 # safe radius of agent during a disaster event
        self.edge_risks = {} # store dynamic risk values for edges
        self.speed = self.CONFIG['disaster']['speed'] # Mean speed of disaster movement (m/s)

    def init_disaster_params(self):
        if self.is_disaster:
            self.pois = self.add_pois_to_simulation() # list to store POIs
            self.max_time = self.max_time_disaster() # Maximum time for disaster events.
            self.init_risk_parameters() # Initialize risk parameters for edges
  
    def is_disaster(self):
        """
        Return true if there is coordinate disaster file
        """
        return self.poi_file

    def add_pois_to_simulation(self):
        """
        Add POIs from an XML file to a SUMO simulation at specified times using TraCI.
        Handles continuous time values by adding POIs at the floor of their time.

        Args:
            sumo_config (str): Path to the SUMO configuration file (.sumocfg)
            poi_file (str): Path to the POI XML file
        """
        # Parse the POI XML file
        pois = []
        tree = ET.parse(self.poi_file)
        root = tree.getroot()

        # Store POIs with their attributes
        for poi in root.findall("poi"):
            poi_data = {
                "id": poi.get("id"),
                "time": float(poi.get("time")),  # Handle continuous time
                "x": float(poi.get("x")),
                "y": float(poi.get("y")),
                "color": poi.get("color"),
                "type": poi.get("type"),
                "imgFile": poi.get("imgFile"),
                "width": float(poi.get("width")),
                "height": float(poi.get("height"))
            }
            pois.append(poi_data)  
        return pois

    def max_time_disaster(self):
        return math.ceil(max(poi["time"] for poi in self.pois)) + 1

    def simulate_points(self, simulation_step):
        if simulation_step <= self.max_time:
            remaining_pois = []
            
            for poi in self.pois:
                if math.floor(poi["time"]) == simulation_step:
                    poiID = poi["id"]
                    x = poi["x"]
                    y = poi["y"]
                    color = (255,0,0)
                    type = poi["type"]
                    width = float(poi["width"])
                    height = float(poi["height"])

                    traci.poi.add(poiID, x, y, color=color,
                                poiType=type, width=width, height=height)
                else:
                    # Keep POIs that don't match
                    remaining_pois.append(poi)
            
            # Replace the entire list - O(n) operation
            self.pois = remaining_pois

    def init_risk_parameters(self):
        """
        Create a list of risk parameters for all edges in the network.
        Each edge will have a default risk value initialized to None.
        """
        for edge_id in self.graph.edges:
            self.edge_risks[edge_id] = None

    def get_risk_edge(self, edge_id):
        """
        Get the risk value for a specific edge.
        If the edge does not exist, return None.

        Args:
            edge_id (str): The ID of the edge.
        Returns:
            float or None: The risk value of the edge, or None if not found.
        """
        try:
            return self.edge_risks[edge_id]
        except KeyError as e:
            logger.exception(f"Edge ID {edge_id} not found in edge risk dictionary. Error: {e}")

    def compute_risks(self):
        """
        Compute the risks for all edges in the network based on the proximity to disaster POIs.
        """
        for edge_id, edge_data in self.graph.edges.items():
            edge_center = self.graph.calculate_center_point(edge_data['shape'])
            min_risk = float('inf')

            for poi in self.pois:
                poi_position = (poi['x'], poi['y'])
                
                distance = math.sqrt((edge_center[0] - poi_position[0]) ** 2 + (edge_center[1] - poi_position[1]) ** 2)
                if distance > self.R:
                    risk = distance
                else:
                    risk = math.exp(-distance / self.R)

                min_risk = min(min_risk, risk)

            self.edge_risks[edge_id] = min_risk
            # logger.info(f"Computed risk for edge {edge_id}: {min_risk} to POI {poi['id']}")
        logger.info("Computed risks for all edges in the network.")
            
    def compute_risks_numba(self):

        @njit(parallel=True, fastmath=True)
        def compute_risks_numba(centers, pois, R, speed):
            E = centers.shape[0]
            P = pois.shape[0]
            out = np.empty(E, dtype=np.float64)
            for i in prange(E):
                cx, cy = centers[i, 0], centers[i, 1]
                max_risk = -1e5
                for j in range(P):
                    dx = cx - pois[j, 0]
                    dy = cy - pois[j, 1]
                    d = math.hypot(dx, dy) # euclid distance
                    # risk = d if d > R else math.exp(-d / R)
                    # exp(k*(x0-x)) => function will growth exponentially as x < x0; distance (m), speed (m/s)
                    # k =0.02 means growth gently
                    # risk = d if d > R else np.exp(0.02*(R-d/speed)) 

                    risk = -d/speed # linear risk growth: distance/speed
                   
                    # if risk < 0:
                    #     # Convert to Python native types for printing
                    #     print("negative risk values:", float(risk), float(d), float(speed))

                 
                    max_risk = max(risk, max_risk)
                    # if risk < min_risk:
                    #     min_risk = risk
                out[i] = max_risk
            return out

        edge_items = list(self.graph.edges.items())
        centers = np.array([self.graph.calculate_center_point(ed['shape']) for _, ed in edge_items], dtype=np.float64)

        # if all the disaster coordinate have been added -> do nothing, all the risk are kept the same
        if len(self.pois) == 0:
            pass
        else:
            # calculate new risks
            pois = np.array([(p['x'], p['y']) for p in self.pois], dtype=np.float64)
            risks = compute_risks_numba(centers, pois, float(self.R), self.speed)
            self.edge_risks = {edge_items[i][0]: float(risks[i]) for i in range(len(edge_items))}
    
if __name__ == "__main__":
    poi_file = r"D:\PhD_Life\Work\Code\marl-traffic-simulation-evacuation\routing-MARL\network\campfire\generated_pois_large.add.xml"  # Replace with your POI XML file path
    route_file = r"D:\PhD_Life\Work\Code\marl-traffic-simulation-evacuation\routing-MARL\network\campfire\campfire.rou.xml"  # Replace with your route file path
    net_file = r"D:\PhD_Life\Work\Code\marl-traffic-simulation-evacuation\routing-MARL\network\campfire\campfire.net.xml"  # Replace with your network file path

    graph = GraphManager(route_file, net_file)
    disaster_move = DisasterMove(poi_file, graph)

    # Example simulation loop
    for step in range(disaster_move.max_time + 1):
        disaster_move.simulate_points(step)

        # Recalculate risks every 10 seconds
        if step % 10 == 0:
            disaster_move.compute_risks()
            print(f"Recalculated risks at step {step}")

    # Print computed risks for validation
    print("Computed Risks:")
    for edge_id, risk in disaster_move.edge_risks.items():
        print(f"Edge {edge_id}: Risk {risk}")