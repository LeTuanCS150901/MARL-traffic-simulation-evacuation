import numpy as np  
import matplotlib.pyplot as plt  
import os
from datetime import datetime  
import xml.etree.ElementTree as ET

from logs.logger import logger  # import our custom logger
from read_config import get_config

class MultiAgentPlotter:  
    def __init__(self, agent_group_ids, max_episodes):  
        """  
        Initialize records for multiple agents.  
        agent_ids: list of unique agent identifiers (could be strings or ints)  
        """  
        self.CONFIG = get_config()
        self.loss_records = []  
        self.reward_records = []
        self.agent_group_ids = agent_group_ids  
        self.max_episodes = max_episodes  
        self.current_episode = 0  

        self.disaster_zones = self.CONFIG['disaster']['fire']['disaster_zones']

        self.init_loss_records(agent_group_ids, max_episodes)  
        self.init_reward_records(agent_group_ids, max_episodes)

        self.loss_reward_path, self.arrival_path, self.survival_path, self.gradient_path = self.init_results_per_eps()

    def init_loss_records(self, agent_group_ids, num_episodes):  
        # Initialize loss records as a list of dictionaries  
        self.loss_records = [{group_id: [] for group_id in agent_group_ids} for _ in range(num_episodes)]  

    def init_reward_records(self, agent_group_ids, num_episodes):  
        # Initialize loss records as a list of dictionaries  
        self.reward_records = [{group_id: [] for group_id in agent_group_ids} for _ in range(num_episodes)]     

    def init_results_per_eps(self):
        """
        Init file for saving loss and reward, and arrvied vehicles in each eps during training
        """

        result_folder = os.path.join(os.path.dirname(__file__), '..', 'result')  
        if not os.path.exists(result_folder):  
            os.makedirs(result_folder)  
            logger.info(f"Created result folder: {result_folder}")  
        
        # Set up specific subfolder  
        full_result_path = os.path.join(result_folder, self.CONFIG['saved_dir'])  
        if not os.path.exists(full_result_path):  
            os.makedirs(full_result_path)  
            logger.info(f"Created subfolder: {full_result_path}")  
        
        # Create filename with timestamp  
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  
        loss_reward_path = os.path.join(full_result_path, f'training_log_{timestamp}.txt')  
        arrival_path = os.path.join(full_result_path, f'arrived_vehicles_{timestamp}.txt')
        survival_path = os.path.join(full_result_path, f'survived_vehicles_{timestamp}.txt')
        gradient_path = os.path.join(full_result_path, f'gradient_{timestamp}.txt')
        
        # Write the head for the file 
        with open(loss_reward_path, 'w') as f:  
            header = "Agent, Eps, Loss, Reward"
            f.write(header)

        with open(arrival_path, 'w') as f:  
            header = "Eps, Arrived_Vehicles"
            f.write(header)

        with open(survival_path, 'w') as f:  
            header = "Eps, Survived_Vehicles, Vehicles_Risk_IDs"
            f.write(header)

        with open(gradient_path, 'w') as f:
            header = "step,group,layer,mean,std,norm"
            f.write(header)
            
        logger.success(f"Created {loss_reward_path} for saving loss,reward at each eps")
        return loss_reward_path, arrival_path, survival_path, gradient_path

    def record_loss(self, group_id, loss):  
        """  
        Record loss value for the specific agent at specific episode.  
        """  
        # logger.debug(f"Recording loss for agent {group_id} at episode {self.current_episode}: {loss}")
        try:  
            # self.loss_records[self.current_episode][group_id].append(loss)  
            self.write_each_loss(group_id, loss)
        except Exception as e:  
            logger.exception(f"Error recording loss for agent {group_id} at episode {self.current_episode}: {e}")  

    def write_each_loss(self, group_id, loss):
        """
        Write each loss to file to track 
        """
        # Write the head for the file 
        # No write reward, so we put it as -1
        with open(self.loss_reward_path, 'a') as f:  
            value= f"{group_id}, {self.current_episode}, {loss}, -1"
            f.write("\n")
            f.write(value)
            
    def write_each_gradient(self, step, group_id, layer, mean, std, norm):
        """
        Write each gradient to file to track 
        """
        # Write the head for the file 
        with open(self.gradient_path, 'a') as f:  
            value= f"{step}, {group_id}, {layer}, {mean}, {std}, {norm}"
            f.write("\n")
            f.write(value)

    def write_each_reward(self, group_id, reward):
        """
        Write each loss to file to track 
        """
        # Write the head for the file 
        # No write loss, so we put it as -1
        with open(self.loss_reward_path, 'a') as f:  
            value= f"{group_id}, {self.current_episode}, -1, {reward}"
            f.write("\n")
            f.write(value)

    def record_reward(self, group_id, reward):
        # logger.debug(f"Recording reward for agent {group_id} at episode {self.current_episode}: {reward}")
        try:
            # self.reward_records[self.current_episode][group_id].append(reward)
            self.write_each_reward(group_id, reward)
        except Exception as e:
            logger.exception(f"Error recording reward for agent {group_id} at episode {self.current_episode}: {e}")

    def increase_episode(self):  
        """  
        Increase the current episode count.  
        """  
        if self.current_episode < self.max_episodes - 1:  # Allow an episode count to reach max_episodes  
            self.current_episode += 1  

    def plot_loss_and_reward_after_episode(self):  
        episode_to_plot = self.current_episode  
        if episode_to_plot >= self.max_episodes:  
            episode_to_plot = self.max_episodes - 1  
        # Actually, in your usage, you want to plot the most recently finished episode:  
        # if you call increase_episode() at end of iteration:  
        episode_to_plot = max(0, self.current_episode - 1)  

        fig, axs = plt.subplots(2, 1, figsize=(10, 12))   

        for group_id in self.agent_group_ids:  
            loss = self.loss_records[episode_to_plot][group_id]  
            reward = self.reward_records[episode_to_plot][group_id]  

            if len(loss) > 0:  
                axs[0].plot(loss, label=f"Agent {group_id}")  
            if len(reward) > 0:  
                axs[1].plot(reward, label=f"Agent {group_id}")  

        axs[0].set_xlabel("Training Steps")  
        axs[0].set_ylabel("Loss")  
        axs[0].set_title(f"Loss for Each Agent at Episode {episode_to_plot + 1}")  
        axs[0].grid()  
        axs[0].legend()  

        axs[1].set_xlabel("Simulation Steps")  
        axs[1].set_ylabel("Reward")  
        axs[1].set_title(f"Reward for Each Agent at Episode {episode_to_plot + 1}")  
        axs[1].grid()  
        axs[1].legend()  

        plt.tight_layout(pad=3.0)  
        plt.show()  

    def write_to_file(self):  
        """  
        At the final epoch
        Write the loss and reward records to a file in the 'result' folder.  
        """  
        # Set up base result folder  
        result_folder = os.path.join(os.path.dirname(__file__), '..', 'result')  
        if not os.path.exists(result_folder):  
            os.makedirs(result_folder)  
            logger.info(f"Created result folder: {result_folder}")  
        
        # Set up specific subfolder  
        full_result_path = os.path.join(result_folder, self.CONFIG['saved_dir'])  
        if not os.path.exists(full_result_path):  
            os.makedirs(full_result_path)  
            logger.info(f"Created subfolder: {full_result_path}")  
        
        # Create filename with timestamp  
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  
        file_path = os.path.join(full_result_path, f'training_log_{timestamp}.txt')  
        
        # Write data to file  
        with open(file_path, 'w') as f:  
            for episode in range(self.max_episodes):  
                f.write(f"Episode {episode + 1}:\n")  
                for group_id in self.agent_group_ids:  
                    f.write(f"  Agent {group_id}:\n")  
                    f.write(f"    Loss: {self.loss_records[episode][group_id]}\n")  
                    f.write(f"    Reward: {self.reward_records[episode][group_id]}\n")  

        logger.info(f"Loss and reward records written to {file_path}")

    def write_arrived_vehicles(self, arrived_vehicles):
        """
        Write each loss to file to track 
        """
        # Write the head for the file 
        # No write reward, so we put it as -1
        with open(self.arrival_path, 'a') as f:  
            value= f"{self.current_episode}, {arrived_vehicles}"
            f.write("\n")
            f.write(value)

    def write_survived_vehicles(self):
        """
        Write survival vehs 
        """

        if not self.disaster_zones:
            return 
        
        XML_FILE_PATH = self.CONFIG['saved_route']
        # The list of edges considered as disruption points
        disruption_edges = [edge for edge_list in self.disaster_zones.values() for edge in edge_list]

        # Initialize a set to store the IDs of vehicles using a disrupted edge
        disrupted_vehicle_ids = set()
        num_vehicles = 0

        # --- XML Parsing and Processing ---
        try:
            if not os.path.exists(XML_FILE_PATH):
                raise FileNotFoundError(f"The file was not found at: {XML_FILE_PATH}")

            # Use ET.parse() directly on the file path
            tree = ET.parse(XML_FILE_PATH)
            root = tree.getroot()

            # Iterate over every 'vehicle' element
            for vehicle in root.findall('vehicle'):
                vehicle_id = vehicle.get('id')
                num_vehicles +=1 
 
                # Find the nested 'route' element
                route = vehicle.find('route')
                
                if route is not None:
                    # Get the edges string and split it into a list
                    edges_string = route.get('edges', "")
                    vehicle_edges = edges_string.split()
                    
                    # Check if any vehicle edge is in the disruption list
                    if any(edge in disruption_edges for edge in vehicle_edges):
                        disrupted_vehicle_ids.add(vehicle_id)
        except Exception as e:
            logger.error(f"Error occurred when outputing survival vehicles: {e}")

        # --- Output Results ---
        num_disrupted_vehicles = len(disrupted_vehicle_ids)
        survived_vehicles = num_vehicles - num_disrupted_vehicles
        # Write the head for the file 
        # No write reward, so we put it as -1
        
        with open(self.survival_path, 'a') as f:  
            value= f"{self.current_episode}, {survived_vehicles}, {disrupted_vehicle_ids}"
            f.write("\n")
            f.write(value)
            logger.success(f"Successfully wrote survived vehicles for episode {self.current_episode}")
