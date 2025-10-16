from logs.logger import logger  # import our custom logger
class CheckEndEpisode:  

    def __init__(self, trips):  
        self.trips_completed = set()  
        # Initial dictionary to hold the vehicle edge information
        self.edges_history = {vid: [] for vid in trips.keys()}
        self.time_vehs_removed = {vid: None for vid in trips.keys()}

    def set_edge_history_trip(self, vehicle_id, current_edge):
        try:
            self.edges_history[vehicle_id].append(current_edge)   
        except KeyError as e:
            logger.exception(f"Vehicle ID {vehicle_id} not found in edges history. Error: {e}")

    def check_edge_in_history(self, vehicle_id, current_edge):
        try:
            # return True if current_edge in self.edges_history[vehicle_id] else False
            # print("Check history edges------------------------------------------------------------------")
            return current_edge in self.edges_history[vehicle_id]
        except KeyError as e:
            logger.exception(f"Vehicle ID {vehicle_id} not found in edges history. Error: {e}")

    def get_history_edges_trip(self, vehicle_id):
        try:
            return self.edges_history[vehicle_id]  
        except KeyError as e:
            logger.exception(f"Vehicle ID {vehicle_id} not found in edges history. Error: {e}")
                                 
    def mark_trip_completed(self, vehicle_id):  
        if vehicle_id not in self.trips_completed:  
            self.trips_completed.add(vehicle_id)  
            logger.debug(f"Trip for vehicle {vehicle_id} has been marked as completed.")
     
    def mark_time_vehicle_removed(self, vehicle_id, current_time):
        try:
            if self.time_vehs_removed[vehicle_id]:
                logger.warning(f"Vehicle ID {vehicle_id} already have time removed from simulation {e}")
            else:
                self.time_vehs_removed[vehicle_id] = current_time
        except KeyError as e:
            logger.exception(f"Vehicle ID {vehicle_id} not found from list of time removed. Error {e}")   

    def get_time_vehicle_removed(self, vehicle_id):
        return self.time_vehs_removed[vehicle_id]

    def is_trip_completed(self, vehicle_id):  
        return True if vehicle_id in self.trips_completed else False  
    
    def is_ended_episode(self, trips):  
        if len(self.trips_completed) > 0 and len(trips) >0:  
            logger.debug(f"len completed trips: {len(self.trips_completed)}, len Total trips: {len(trips)}")
            return self.trips_completed == set(trips.keys())   
        return False  
    
    def get_trips_not_completed(self, trips):  
        not_completed_trips = [vid for vid in trips.keys() if vid not in self.trips_completed]  
        # logger.debug(f"Trips not completed: {not_completed_trips}")  
        return not_completed_trips
    
    def get_trips_completed(self):
        return self.trips_completed

    
