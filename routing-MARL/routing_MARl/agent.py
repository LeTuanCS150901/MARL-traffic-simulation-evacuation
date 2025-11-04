from Q_network import QNetwork
from experience_buffer import JointExperienceReplayBuffer
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
    def __init__(self, obs_dim, action_dim, group_id, train_offline=False, config=None):

        # train offline from anothẻ folder
        if train_offline:
            self.CONFIG = config
        # train online with sumo interaction
        else:
            self.CONFIG = get_config()

        self.group_id = group_id  
        # switch to device processing
        self.q_network = QNetwork(obs_dim, action_dim)
        self.target_network = QNetwork(obs_dim, action_dim)
        self.target_network.load_state_dict(self.q_network.state_dict())  # Initial sync  

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.CONFIG['learning_rate'],  
                                    weight_decay=1e-4)  
        self.gamma = self.CONFIG['gamma']  # Discount factor
        # Use joint replay buffer
        self.replay_buffer = JointExperienceReplayBuffer(capacity=self.CONFIG['buffer_capacity'])
        
        # Temporary storage for current timestep
        self.current_timestep_transitions = []
        self.current_timestep = 0 

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
        # """
        if not self.training:
            return self.select_action_for_inference(observation, feasible_actions)

        with torch.no_grad():
            q_values = self.q_network(observation.view(1, -1))
            q_values = q_values.squeeze(0)
            
            # For softmax: filter to feasible actions only
            feasible_indices = [i for i, f in enumerate(feasible_actions) if f]
            q_values_for_softmax = q_values[feasible_indices]

            if not feasible_actions:
                logger.warning("No feasible actions available during softmax selection. do nothing")
                return None
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

    # def softmax_select_action(self, observation, feasible_actions):
    #     """
    #     Select action using Softmax action selection strategy.
    #     Works with TransformerQNetwork output (1D Q-values).
    #     """
    #     if not self.training:
    #         return self.select_action_for_inference(observation, feasible_actions)

    #     with torch.no_grad():
    #         # Compute Q-values: output shape [action_dim]
    #         q_values = self.q_network(observation)

    #         feasible_mask = torch.tensor(feasible_actions, dtype=torch.bool)

    #         # Mask infeasible actions
    #         q_values_masked = q_values.clone()
    #         q_values_masked[~feasible_mask] = float('-inf')

    #         # Get feasible indices
    #         feasible_indices = torch.nonzero(feasible_mask, as_tuple=False).squeeze(-1).tolist()
    #         if len(feasible_indices) == 0:
    #             logger.warning("No feasible actions available during softmax selection. Skipping.")
    #             return None

    #         # Extract feasible Q-values
    #         q_values_for_softmax = q_values[feasible_indices]

    #         # Subtract max(Q) for numerical stability
    #         q_norm = q_values_for_softmax - torch.max(q_values_for_softmax)

    #         # Softmax with temperature
    #         exp_q = torch.exp(q_norm / self.tau)
    #         prob = exp_q / exp_q.sum()

    #         # Sample action
    #         action_index_within_feasible = torch.multinomial(prob, num_samples=1).item()
    #         action_index = feasible_indices[action_index_within_feasible]

    #         self.total_steps += 1
    #         return action_index

    
    def UCB_select_action(self, observation, feasible_actions):
        """
        Select action using Upper Confidence Bound (UCB) strategy
        """
        # If not in training mode, use inference selection
        if not self.training:
            logger.success("Set inference mode for UCB")
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
                    q_values = self.q_network(observation.view(1, -1))   # [1, action_dim]
                    q_values = q_values.squeeze(0)                          # [action_dim]

                    feasible_mask = torch.tensor(feasible_actions, dtype=torch.bool)
                    q_values_masked = q_values.clone()  
                    q_values_masked[~feasible_mask] = float('-inf')  

                    # Compute probabilities only among feasible actions
                    feasible_q_values = q_values_masked[feasible_mask]
                    probs = torch.softmax(feasible_q_values, dim=0)

                    # Sample according to softmax probabilities
                    idx_within_feasible = torch.multinomial(probs, 1).item()

                    # Map back to global action index
                    feasible_indices = torch.nonzero(feasible_mask, as_tuple=True)[0]
                    action_index = feasible_indices[idx_within_feasible].item()
                        
                    # Always choose best action in evaluation mode  
                    # action_index = torch.argmax(q_values_masked).item()  
                    
                    # Sample action
                    logger.debug(f"Inference mode: chose action {action_index}")  
                    return action_index  
        
    @torch.no_grad()
    def select_action_offline(self, observation, feasible_actions):
        """
        Select an action for each vehicle within the agent group
        """
        logits = self.q_network(observation.view(1,- 1))
        feasible_mask = torch.tensor(feasible_actions, dtype=torch.bool).unsqueeze(0)  
        logits_masked = logits.clone()
        logits_masked[~feasible_mask] = float('-inf')

        probs = F.softmax(logits_masked, dim=1)
        actions = torch.argmax(logits_masked, dim=1)

        return actions.item()
    
    def store_transition_timestep(self, vehicle_id, obs, action, next_obs, done):
        """
        Store individual transition for current timestep.
        Call end_timestep() when all vehicles have acted.
        """
        self.current_timestep_transitions.append((vehicle_id, obs, action, next_obs, done))
    
    def end_timestep(self, global_reward):
        """
        Called when all vehicles in the group have acted in current timestep.
        Stores the joint transition with global reward.
        """
        if len(self.current_timestep_transitions) > 0:
            joint_transition = {
                'timestep': self.current_timestep,
                'vehicles': self.current_timestep_transitions.copy(),
                'global_reward': global_reward
            }
            self.replay_buffer.push(joint_transition)
            
            # Reset for next timestep
            self.current_timestep_transitions = []
            self.current_timestep += 1
        
    # def update(self, plotter):  
    #     if len(self.replay_buffer.buffer) < self.batch_size:  
    #         return  
        
    #     # if agent in inference mode, no need to run this function
    #     if not self.training:
    #         return
        
    #     batch = self.replay_buffer.sample(self.batch_size)
    #     # logger.warning(f"Current batch size ............... {len(batch)}")  
        
    #     # obs_batch and next_obs_batch come as [B, N, 1], flatten last two dims to [B, N]  
    #     obs_batch = torch.stack([torch.tensor(b[0], dtype=torch.float32) for b in batch]).view(self.batch_size, -1)  
    #     next_obs_batch = torch.stack([torch.tensor(b[2], dtype=torch.float32) for b in batch]).view(self.batch_size, -1)  
    #     action_batch = torch.tensor([b[1] for b in batch], dtype=torch.int64)  
    #     reward_batch = torch.tensor([b[3] for b in batch], dtype=torch.float32)  

    #     with torch.no_grad():  
    #         # forward pass for target values
    #         target_q_values = self.target_network(next_obs_batch)  # shape [B, action_dim]  
    #         max_target_q_values, _ = torch.max(target_q_values, dim=1)  # max over actions → [B]  
    #         target = reward_batch + self.CONFIG['gamma'] * max_target_q_values  # [B]  
        
    #     # get the Q-values for the actions taken in the current batch
    #     # forward pass for prediction
    #     current_q_values = self.q_network(obs_batch).gather(1, action_batch.unsqueeze(1)).squeeze(1)  # [B]  

    #     logger.debug(f"Update---------------current_q_values: {current_q_values}")  
    #     logger.debug(f"Update---------------target values: {target}")  

    #     loss = F.mse_loss(current_q_values, target)  
    #     logger.debug(f"Loss---------------: {loss}") 
        
    #     self.optimizer.zero_grad()  
    #     loss.backward()  
    #     self.optimizer.step()  

    #     # record loss results
    #     if plotter is not None:
    #         plotter.record_loss(self.group_id, loss.item()) 
        
    #     self.update_step_count += 1  
    #     if self.update_step_count % self.target_update_freq == 0:  
    #         self.target_network.load_state_dict(self.q_network.state_dict())  
    #         # logger.info(f"Target network updated at step {self.update_step_count}")

    #     return loss

    def update(self, plotter=None):
        """
        VDN Training with proper joint transitions.
        """
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        if not self.training:
            return None
        
        # Sample joint transitions
        batch = self.replay_buffer.sample(self.batch_size)
        
        total_loss = 0.0
        
        for joint_transition in batch:
            vehicles_data = joint_transition['vehicles']
            global_reward = joint_transition['global_reward']
            
            if len(vehicles_data) == 0:
                continue
            
            # Extract data for all vehicles in this timestep
            obs_list = [v[1] for v in vehicles_data]
            actions = [v[2] for v in vehicles_data]
            next_obs_list = [v[3] for v in vehicles_data]
            dones = [v[4] for v in vehicles_data]
            
            # Stack: [num_vehicles_in_timestep, obs_dim]
            obs_batch = torch.stack([torch.tensor(o, dtype=torch.float32).squeeze() for o in obs_list])
            next_obs_batch = torch.stack([torch.tensor(o, dtype=torch.float32).squeeze() for o in next_obs_list])
            actions_batch = torch.tensor(actions, dtype=torch.int64)
            dones_batch = torch.tensor(dones, dtype=torch.float32)
            
            # ===== Q_total (current) =====
            q_values_all = self.q_network(obs_batch)  # [num_vehicles, action_dim]
            q_values_chosen = q_values_all.gather(1, actions_batch.unsqueeze(1)).squeeze(1)  # [num_vehicles]
            q_total_current = q_values_chosen.sum()  # Scalar - sum over vehicles in SAME timestep
            
            # ===== Q_total_target (Double DQN) =====
            with torch.no_grad():
                # Argmax from main network
                next_q_main = self.q_network(next_obs_batch)
                next_actions = torch.argmax(next_q_main, dim=1)
                
                # Evaluate with target network
                next_q_target = self.target_network(next_obs_batch)
                next_q_values = next_q_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)
                
                # Apply done mask
                next_q_values = next_q_values * (1 - dones_batch)
                
                # Sum over vehicles in SAME timestep
                q_total_next = next_q_values.sum()
                
                # Target: r_shared + γ * Q_total_next
                target = global_reward + self.gamma * q_total_next
            
            # ===== Loss for this joint transition =====
            loss = (q_total_current - target) ** 2
            total_loss += loss
        
        # Average loss over batch
        # total_loss = total_loss / len(batch)
        logger.debug(f"Total Loss: {total_loss.item()}")
        
        # Backpropagation
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=10.0)
        self.optimizer.step()
        
        # Update target network
        self.update_step_count += 1
        if self.update_step_count % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        if plotter:
            plotter.record_loss(self.group_id, total_loss.item())
        
        return total_loss.item()
    
    
    def save_model(self, path=None):  
        """Save the agent's model to disk"""  
        if path is None:  
            # Create a directory for saved models if it doesn't exist  
            # Should be:  
            os.makedirs(self.saved_folder, exist_ok=True)  
            path = f"{self.saved_folder}/agent_{self.group_id}_{time.strftime('%Y%m%d_%H%M%S')}.pt"  
        
        try:
            torch.save({
                'policy_net_state_dict': self.q_network.cpu().state_dict(),
                'target_net_state_dict': self.target_network.cpu().state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'group_id': self.group_id,
                'total_steps': self.total_steps,
            }, path)
            logger.success(f"Model saved to {path}")
            return path
        except Exception as e:
            logger.error(f"Failed to save model at {path}: {e}")
   
        return path  
    
    def load_model(self, path):  
        """Load a saved model from disk"""  
        if not os.path.exists(path):  
            logger.warning(f"Model file not found: {path}")  
            return False  
            
        checkpoint = torch.load(path)  
        
        # Load the saved state dictionaries  
        self.q_network.load_state_dict(checkpoint['policy_net_state_dict'])  
        self.target_network.load_state_dict(checkpoint['target_net_state_dict'])  
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])  
        # self.epsilon = checkpoint['epsilon']  
        self.total_steps = checkpoint['total_steps'] 
        
        logger.info(f"Model {checkpoint['group_id']} loaded from {path}")  
        return True  
    
    def load_offline_model(self, path, agent_name):
        """Load a saved model from disk"""  
        if not os.path.exists(path):  
            logger.debug(f"Model file not found: {path}")  
            return False  

        # load the model
        checkpoint = torch.load(path)  
        # load the weights of model
        self.q_network.load_state_dict(checkpoint, map_location=torch.device('cpu'))
        # load the saved state of the model

        logger.info(f"Model {agent_name} loaded from {path}")
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