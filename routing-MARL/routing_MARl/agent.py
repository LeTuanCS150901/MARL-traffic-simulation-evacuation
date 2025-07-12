from Q_network import QNetwork
from experience_buffer import ExperienceReplayBuffer
from logs.logger import logger  
from read_config import get_config

import torch
import torch.nn.functional as F 
import torch.optim as optim
import os
import random
import numpy as np
import time

class Agent:  
    def __init__(self, obs_dim, action_dim, group_id):  
        self.CONFIG = get_config()
        self.group_id = group_id  
        # switch to device processing
        self.q_network = QNetwork(obs_dim, action_dim)
        self.target_network = QNetwork(obs_dim, action_dim)
        self.target_network.load_state_dict(self.q_network.state_dict())  # Initial sync  

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.CONFIG['learning_rate'])  
        self.replay_buffer = ExperienceReplayBuffer(capacity=self.CONFIG['buffer_capacity'])  
        self.batch_size = self.CONFIG['batch_size']

        # Epsilon-greedy parameters  
        self.epsilon_start = self.CONFIG['epsilon_greedy']['initial']
        self.epsilon_final = self.CONFIG['epsilon_greedy']['final']
        self.epsilon_decay = self.CONFIG['epsilon_greedy']['decay']
        self.total_steps = 0  

        #  UCB parameters
        self.action_counts = np.zeros(action_dim, dtype=np.int64)  # Initialize with zeros
        self.c = self.CONFIG['UCB']['c']  # Exploration parameter for UCB

        # Softmax parameters
        self.tau = self.CONFIG['softmax']['tau']  # Temperature parameter for softmax action selection
    
        # Target update frequency params  
        self.target_update_freq = self.CONFIG['update_frequency']  
        self.update_step_count = 0  

        # configs for save and load model
        self.saved_folder = os.path.join(os.path.dirname(__file__), '..', 'new_models') 
        # Add training mode flag  
        self.training = True  

    def epsilon_by_step(self):  
        return self.epsilon_final + (self.epsilon_start - self.epsilon_final) * \
                np.math.exp(-1. * self.total_steps / self.epsilon_decay) 
    
    def softmax_select_action(self, observation, feasible_actions):
        """
        Select action using Softmax action selection strategy
        """
        if not self.training:
            return self.select_action_for_inference(observation, feasible_actions)

        with torch.no_grad():
            q_values = self.q_network(observation.view(1, -1))
            feasible_mask = torch.tensor(feasible_actions, dtype=torch.bool).unsqueeze(0)
            q_values_masked = q_values.clone()
            q_values_masked[~feasible_mask] = float('-inf')
            
            # For softmax: filter to feasible actions only
            feasible_indices = [i for i, f in enumerate(feasible_actions) if f]
            q_values_for_softmax = q_values[0, feasible_indices]
            
            # Subtract max(Q) for numerical stability
            q_norm = q_values_for_softmax - torch.max(q_values_for_softmax)
            exp_q = torch.exp(q_norm / self.tau)
            prob = exp_q / exp_q.sum()
            
            # Sample action
            action_index_within_feasible = torch.multinomial(prob, num_samples=1).item()
            action_index = feasible_indices[action_index_within_feasible]
            
            # Logging
            # logger.info(f"Softmax Action Selection")
            # logger.info(f"Q-values: {q_values_masked}")
            # logger.info(f"Probabilities: {prob}")
            # logger.info(f"Selected Action: {action_index}")

            self.total_steps += 1
            return action_index
    
    def UCB_select_action(self, observation, feasible_actions):
        """
        Select action using Upper Confidence Bound (UCB) strategy
        """
        # If not in training mode, use inference selection
        if not self.training:
            logger.debug("Set inference mode for UCB")
            return self.select_action_for_inference(observation, feasible_actions)

        with torch.no_grad():
            # Compute Q-values for the observation
            q_values = self.q_network(observation.view(1, -1))
            
            # Create feasible mask
            feasible_mask = torch.tensor(feasible_actions, dtype=torch.bool).unsqueeze(0)
            
            # Mask infeasible actions
            # normalize q_values as there are negative values
            # q_values_normalized = (q_values - q_values.min()) / (q_values.max() - q_values.min() + 1e-8)
            q_values_masked = q_values.clone()
            q_values_masked[~feasible_mask] = float('-inf')
            
            # Prepare UCB scores
            ucb_scores = q_values_masked.clone()
            
            # Get feasible action indices
            feasible_indices = [i for i, f in enumerate(feasible_actions) if f]
            
            # Compute UCB scores for feasible actions
            for action in feasible_indices:
                # Skip masked actions
                if q_values_masked[0, action] == float('-inf'):
                    continue

                # logger.error(f"Values of q_values_masked {q_values_masked[0, action]}")
                
                # Initial exploration for never-selected actions
                if self.action_counts[action] == 0:
                    ucb_scores[0, action] = float('inf')
                    continue

                # Standard UCB formula
                # Q-value + exploration bonus
                exploration_bonus = self.c * torch.sqrt(
                    torch.log(torch.tensor(self.total_steps + 1, dtype=torch.float)) / 
                    (self.action_counts[action] + 1e-8)
                )

                # Add exploration bonus to existing Q-value 
                ucb_scores[0, action] += exploration_bonus 
            
            # Select action with highest UCB score
            action_index = torch.argmax(ucb_scores).item()
            
            # Update action counts and total steps
            self.action_counts[action_index] += 1
            self.total_steps += 1
            
            # Logging for debugging
            logger.info(f"UCB Action Selection")
            logger.info(f"Q-values: {q_values_masked}")
            logger.info(f"UCB Scores: {ucb_scores}")
            logger.info(f"Selected Action: {action_index}")
            # output all possible actions counts for each agent. At certain time, some actions are infeasible
            # so we only select feasible actions
            logger.info(f"Action Counts: {self.action_counts}") 
            
            return action_index
    
    def select_action(self, observation, feasible_actions): 
        """
        select action based on epsilon-greedy policy
        """ 
        # if current model is inference 
        if not self.training:  
            logger.debug("Set infernece mode")
            action_index = self.select_action_for_inference(observation, feasible_actions)
            return action_index

        # Training mode: use epsilon-greedy as before   
        epsilon = self.epsilon_by_step()  
        self.total_steps += 1  
        
        with torch.no_grad():  
            # the input from size [N,1] => [B, N]
            # B is the batch size, B = 1 in here
            q_values = self.q_network(observation.view(1, -1))  
            logger.info(f"observation: {observation}")
            feasible_mask = torch.tensor(feasible_actions, dtype=torch.bool).unsqueeze(0)  
            logger.info(f"shapes: {q_values.shape}, {feasible_mask.shape}")  
            logger.info(f"q_values: {q_values}")  
            
            # Mask infeasible actions: set to -inf so they won't be picked  
            q_values_masked = q_values.clone()  
            q_values_masked[~feasible_mask] = float('-inf')  
            logger.info(f"q_values_masked: {q_values_masked}")  

            if random.random() < epsilon:  
                # Exploration: random feasible action  
                feasible_indices = [i for i, f in enumerate(feasible_actions) if f]  
                action_index = random.choice(feasible_indices)  
                logger.info(f"Exploration: chose action {action_index} with epsilon {epsilon:.4f}")  
            else:  
                # Exploitation: greedy action  
                action_index = torch.argmax(q_values_masked).item()  
                logger.info(f"Exploitation: chose action {action_index} with epsilon {epsilon:.4f}")  

            return action_index  
        
    def select_action_for_inference(self, observation, feasible_actions):
        # In evaluation mode, always be greedy (no exploration)  
        if not self.training:  
            with torch.no_grad():  
                q_values = self.q_network(observation.view(1, -1))  
                feasible_mask = torch.tensor(feasible_actions, dtype=torch.bool).unsqueeze(0)  
                q_values_masked = q_values.clone()  
                q_values_masked[~feasible_mask] = float('-inf')  
                
                # Always choose best action in evaluation mode  
                action_index = torch.argmax(q_values_masked).item()  
                logger.info(f"Inference mode: chose action {action_index}")  
                return action_index  
        
    def update(self, plotter):  
        if len(self.replay_buffer.buffer) < self.batch_size:  
            return  
        
        # if agent in inference mode, no need to run this function
        if not self.training:
            return
        
        batch = self.replay_buffer.sample(self.batch_size)
        # logger.warning(f"Current batch size ............... {len(batch)}")  
        
        # obs_batch and next_obs_batch come as [B, N, 1], flatten last two dims to [B, N]  
        obs_batch = torch.stack([torch.tensor(b[0], dtype=torch.float32) for b in batch]).view(self.batch_size, -1)  
        next_obs_batch = torch.stack([torch.tensor(b[2], dtype=torch.float32) for b in batch]).view(self.batch_size, -1)  
        
        # action_batch and reward_batch as 1D tensors [B]  
        action_batch = torch.tensor([b[1] for b in batch], dtype=torch.int64)  
        reward_batch = torch.tensor([b[3] for b in batch], dtype=torch.float32)  

        with torch.no_grad():  
            # forward pass for target values
            target_q_values = self.target_network(next_obs_batch)  # shape [B, action_dim]  
            max_target_q_values, _ = torch.max(target_q_values, dim=1)  # max over actions → [B]  
            target = reward_batch + self.CONFIG['gamma'] * max_target_q_values  # [B]  
        
        # get the Q-values for the actions taken in the current batch
        # forward pass for prediction
        current_q_values = self.q_network(obs_batch).gather(1, action_batch.unsqueeze(1)).squeeze(1)  # [B]  

        logger.info(f"Update---------------current_q_values: {current_q_values}")  
        logger.info(f"Update---------------target values: {target}")  

        loss = F.mse_loss(current_q_values, target)  
        
        self.optimizer.zero_grad()  
        loss.backward()  
        self.optimizer.step()  

        # record loss results
        plotter.record_loss(self.group_id, loss.item()) 
        
        self.update_step_count += 1  
        if self.update_step_count % self.target_update_freq == 0:  
            self.target_network.load_state_dict(self.q_network.state_dict())  
            logger.info(f"Target network updated at step {self.update_step_count}")

    def save_model(self, path=None):  
        """Save the agent's model to disk"""  
        if path is None:  
            # Create a directory for saved models if it doesn't exist  
            # Should be:  
            os.makedirs(self.saved_folder, exist_ok=True)  
            path = f"{self.saved_folder}/agent_{self.group_id}_{time.strftime('%Y%m%d_%H%M%S')}.pt"  
        
        # Save the policy network and target network  
        torch.save({  
            'policy_net_state_dict': self.q_network.state_dict(),  
            'target_net_state_dict': self.target_network.state_dict(),  
            'optimizer_state_dict': self.optimizer.state_dict(),  
            'group_id': self.group_id,  
            'total_steps': self.total_steps,  # Save steps for proper epsilon calculation  
            # 'epsilon': self.epsilon,  
            # Add any other parameters you want to save  
        }, path)  
        
        logger.debug(f"Model saved to {path}")  
        return path  
    
    def load_model(self, path):  
        """Load a saved model from disk"""  
        if not os.path.exists(path):  
            logger.debug(f"Model file not found: {path}")  
            return False  
            
        checkpoint = torch.load(path)  
        
        # Load the saved state dictionaries  
        self.q_network.load_state_dict(checkpoint['policy_net_state_dict'])  
        self.target_network.load_state_dict(checkpoint['target_net_state_dict'])  
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])  
        # self.epsilon = checkpoint['epsilon']  
        self.total_steps = checkpoint['total_steps'] 
        
        logger.debug(f"Model {checkpoint['group_id']} loaded from {path}")  
        return True  
        
    def set_eval_mode(self):  
        """Set the agent to evaluation mode (no exploration)"""  
        self.training = False  # Critical flag to disable exploration  
        self.q_network.eval()  
        self.target_network.eval()

    def set_train_mode(self):  
        """Set the agent back to training mode"""  
        self.training = True  
        self.q_network.train()  
        self.target_network.train()