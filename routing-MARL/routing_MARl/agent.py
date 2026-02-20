from Q_network import QNetwork
from experience_buffer import JointExperienceReplayBuffer
from logs.logger import logger  
from read_config import get_config
from adaptive_layers import AdaptiveLayerControllerAuto

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

        # Set up device
        # add cuda
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.device = torch.device("cpu")
        logger.info(f"Agent {group_id} initialized using device: {self.device}")

        self.group_id = group_id  
        # switch to device processing
        # add cuda
        self.q_network = QNetwork(obs_dim, action_dim).to(self.device)
        self.target_network = QNetwork(obs_dim, action_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())  # Initial sync  

        # gradient parameters
        self.max_grad_norm = float(self.CONFIG.get('max_grad_norm', 1.0))
        self.weight_decay = float(self.CONFIG.get('weight_decay', 1e-4))

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.CONFIG['learning_rate'],  
                                    weight_decay=self.weight_decay)  
        # freeze layers and retraint the explosion layers if needed
        self.controller = AdaptiveLayerControllerAuto(model=self.q_network, optimizer=self.optimizer)

        self.gamma = self.CONFIG['gamma']  # Discount factor
        # Use joint replay buffer
        self.replay_buffer = JointExperienceReplayBuffer(capacity=self.CONFIG['buffer_capacity'])
        self.current_size_buffer = 0
        
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
        # self.target_update_freq = self.CONFIG['update_frequency']  
        self.update_step_count = 0  
        self.beta_soft = self.CONFIG.get('beta_soft', 0.01)  # Soft update parameter

        # configs for save and load model
        self.saved_folder = os.path.join(os.path.dirname(__file__), '..', 'new_models') 
        # Add training mode flag  
        self.training = True  

    def epsilon_by_step(self):  
        return self.epsilon_final + (self.epsilon_start - self.epsilon_final) * \
                np.math.exp(-1. * self.total_steps / self.epsilon_decay) 
    
    def batch_softmax_select_action(self, observations, feasible_actions_list, encoded_disaster_obs):  
        """  
        Batched softmax selection for multiple vehicles at a single timestep.  
        
        observations: [batch_size, obs_dim]  
        encoded_disaster_obs: [batch_size, dis_dim] or None  
        feasible_actions_list: list of list of 0/1 (bools or ints) of length action_dim  
        
        Returns: list of actions (one per vehicle, None if no feasible actions)  
        """  
        batch_size = observations.shape[0]  
        
        if batch_size == 0:  
            return []  
        
        actions = [None] * batch_size  
        
        # Find valid vehicles (those with any feasible actions)  
        valid_indices = []  
        for i, feasible_actions in enumerate(feasible_actions_list):  
            if any(feasible_actions):  
                valid_indices.append(i)  
            else:  
                logger.warning(f"No feasible actions available for batch index {i}. Do nothing.")  
        
        if not valid_indices:  
            return actions  
        
        # For inference mode, fall back to individual calls to preserve logic  
        if not self.training:  
            for i in valid_indices:  
                obs = observations[i:i+1]  
                feasible = feasible_actions_list[i]  
                enc_dis = encoded_disaster_obs[i:i+1] if encoded_disaster_obs is not None else None  
                actions[i] = self.select_action_for_inference(obs, feasible, enc_dis)  
            return actions  

        with torch.no_grad():  
            # Move inputs to device  
            observations = observations.to(self.device)  
            if encoded_disaster_obs is not None:  
                encoded_disaster_obs = encoded_disaster_obs.to(self.device)  

            # Forward pass through Q-network for ALL valid vehicles at once (single GPU call)  
            q_values = self.q_network(observations, encoded_disaster_obs)  # [batch_size, action_dim]  

            # Process each vehicle's action selection  
            for batch_idx, vehicle_idx in enumerate(valid_indices):  
                q_vals = q_values[batch_idx]  # [action_dim]  
                feasible_actions = feasible_actions_list[vehicle_idx]  
                
                # Softmax over feasible actions  
                feasible_indices = [j for j, f in enumerate(feasible_actions) if f]  
                q_for_softmax = q_vals[feasible_indices]  

                # Numerical stability  
                q_norm = q_for_softmax - torch.max(q_for_softmax)  
                exp_q = torch.exp(q_norm / self.tau)  
                prob = exp_q / exp_q.sum()  

                # Sample action  
                idx_within_feasible = torch.multinomial(prob, 1).item()  
                actions[vehicle_idx] = feasible_indices[idx_within_feasible]  

            self.total_steps += len(valid_indices)  
            return actions  


    
    def softmax_select_action(self, observation, feasible_actions, encoded_disaster_obs, hist_obs):
        """
        Softmax selection per vehicle for a single timestep.
        
        observation:  [1, obs_dim]
        encoded_disaster_obs:  [1, dis_dim]
        feasible_actions: list of bools of length action_dim
        """
        if not self.training:
            return self.select_action_for_inference(observation, feasible_actions, encoded_disaster_obs)

        if not feasible_actions:
            logger.warning("No feasible actions available. Do nothing.")
            return None

        with torch.no_grad():
            # Move inputs to device
            # add cuda
            observation = observation.to(self.device)
            if encoded_disaster_obs is not None:
                encoded_disaster_obs = encoded_disaster_obs.to(self.device)

            # Forward pass through Q-network
            # q_values = self.q_network(observation, encoded_disaster_obs)  # [1, action_dim]

            # Don't forget to pass hist_obs (adding a batch dimension with .unsqueeze(0))
            q_values = self.q_network(observation, encoded_disaster_obs, hist_obs=hist_obs.unsqueeze(0))
            q_values = q_values.squeeze(0)                          # [action_dim]

            # Softmax over feasible actions
            feasible_indices = [i for i, f in enumerate(feasible_actions) if f]
            # mask infeasible actions, keep feasible only
            q_for_softmax = q_values[feasible_indices]

            # Numerical stability
            q_norm = q_for_softmax - torch.max(q_for_softmax)
            exp_q = torch.exp(q_norm / self.tau)
            prob = exp_q / exp_q.sum()

            # Sample action
            idx_within_feasible = torch.multinomial(prob, 1).item()
            action = feasible_indices[idx_within_feasible]

            self.total_steps += 1
            return action

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
        
    def select_action_for_inference(self, observation, feasible_actions, encoded_disaster_obs):
        # In evaluation mode
        if not self.training:  
            with torch.no_grad():  

                # Move inputs to device
                # add cuda
                observation = observation.to(self.device)
                if encoded_disaster_obs is not None:
                    encoded_disaster_obs = encoded_disaster_obs.to(self.device)
                
                q_values = self.q_network(observation, encoded_disaster_obs)   # [1, action_dim]
                q_values = q_values.squeeze(0)                          # [action_dim]

                feasible_mask = torch.tensor(feasible_actions, dtype=torch.bool)
                q_values_masked = q_values.clone()  
                q_values_masked[~feasible_mask] = float('-inf')  

                ### SOFTMAX SELECTION DURING INFERENCE ###
                # Compute probabilities only among feasible actions
                feasible_q_values = q_values_masked[feasible_mask]
                probs = torch.softmax(feasible_q_values, dim=0)

                # Sample according to softmax probabilities
                idx_within_feasible = torch.multinomial(probs, 1).item()

                # Map back to global action index
                feasible_indices = torch.nonzero(feasible_mask, as_tuple=True)[0]
                action_index = feasible_indices[idx_within_feasible].item()
                    
                #   GREEDY SELECTION DURING INFERENCE ###   
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
    
    # def store_transition_timestep(self, vehicle_id, obs, action, next_obs, done, disaster_obs):
    #     """
    #     Store individual transition for current timestep.
    #     Call end_timestep() when all vehicles have acted.
    #     """
    #     self.current_timestep_transitions.append((vehicle_id, obs, action, next_obs, done, disaster_obs))


    def store_transition_timestep(self, vehicle_id, obs, action, next_obs, done, disaster_obs, hist_obs, next_hist_obs):
        self.current_timestep_transitions.append((vehicle_id, obs, action, next_obs, done, disaster_obs, hist_obs, next_hist_obs))
    
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


    # def update(self, plotter=None):
    #     """
    #     VDN Training with optimized batch calculation.
    #     Transfers all data to GPU in one go and computes loss using vectorized operations.
    #     """
    #     if len(self.replay_buffer) < self.batch_size or not self.training:
    #         return None
        
    #     # if len(self.replay_buffer) - self.current_size_buffer == 100: 
    #     #     # update network params every 100 samples added to buffer
    #     #     self.current_size_buffer = len(self.replay_buffer)

    #     # Sample batch
    #     batch = self.replay_buffer.sample(self.batch_size)
        
    #     # --- 1. Data Preparation (CPU Side) ---  
    #     all_obs = []  
    #     all_next_obs = []  
    #     all_actions = []  
    #     all_dones = []  
    #     all_disasters = []  
    #     batch_rewards = []  
    #     group_sizes = []  # New: track num vehicles per valid batch  

    #     valid_batch_count = 0  

    #     for joint_transition in batch:  
    #         vehicles_data = joint_transition['vehicles']  
    #         global_reward = joint_transition['global_reward']  

    #         if len(vehicles_data) == 0:  
    #             continue  

    #         batch_rewards.append(global_reward)  
    #         num_vehicles = len(vehicles_data)  
    #         group_sizes.append(num_vehicles)  

    #         # Collect per-group, squeezing out singleton dims  
    #         group_obs = []  
    #         group_next_obs = []  
    #         group_actions = []  
    #         group_dones = []  
    #         group_disasters = []  

    #         for v in vehicles_data:  
    #             # Squeeze to remove extra dims (e.g., (1, O) -> (O,))  
    #             group_obs.append(np.squeeze(v[1]))  
    #             group_next_obs.append(np.squeeze(v[3]))  
    #             group_actions.append(v[2])  # Assuming action is scalar/int, no squeeze needed  
    #             group_dones.append(v[4])    # Assuming done is scalar/float  
    #             # For disaster: squeeze if it's an array, or handle None  
    #             disaster = v[5]  
    #             group_disasters.append(np.squeeze(disaster) if disaster is not None else None)  

    #         all_obs.append(np.array(group_obs))  # Now reliably [num_vehicles, obs_dim]  
    #         all_next_obs.append(np.array(group_next_obs))  
    #         all_actions.extend(group_actions)  # Keep flat for gathering  
    #         all_dones.extend(group_dones)  
    #         # For disasters: if None, keep as None; else stack squeezed arrays  
    #         if group_disasters[0] is not None:  
    #             all_disasters.append(np.array(group_disasters))  # [num_vehicles, dis_dim]  
    #         else:  
    #             all_disasters.append(None)  

    #         valid_batch_count += 1  

    #     if not all_obs:  
    #         return 0.0  

    #     # Compute max_N across the batch  
    #     max_N = max(group_sizes)  

    #     # --- 2. Pad and Move to GPU ---  
    #     # Pad obs, next_obs to [B, max_N, obs_dim]  
    #     obs_padded = np.zeros((valid_batch_count, max_N, all_obs[0].shape[-1]), dtype=np.float32)  
    #     next_obs_padded = np.zeros((valid_batch_count, max_N, all_next_obs[0].shape[-1]), dtype=np.float32)  

    #     # For disaster: handle None case  
    #     if all(d is None for d in all_disasters):  
    #         disaster_padded = None  
    #     else:  
    #         dis_dim = all_disasters[0].shape[-1]  # Use first non-None for dim  
    #         disaster_padded = np.zeros((valid_batch_count, max_N, dis_dim), dtype=np.float32)  

    #     padding_mask = torch.ones((valid_batch_count, max_N), dtype=torch.bool)  # True = padded  

    #     for i in range(valid_batch_count):  
    #         num_v = group_sizes[i]  
    #         obs_padded[i, :num_v] = all_obs[i]  # Now 2D -> 2D assignment  
    #         next_obs_padded[i, :num_v] = all_next_obs[i]  
    #         if disaster_padded is not None and all_disasters[i] is not None:  
    #             disaster_padded[i, :num_v] = all_disasters[i]  
    #         padding_mask[i, :num_v] = False  # False = valid  

    #     # To tensors (unchanged)  
    #     obs_padded = torch.tensor(obs_padded, dtype=torch.float32, device=self.device)  
    #     next_obs_padded = torch.tensor(next_obs_padded, dtype=torch.float32, device=self.device)  
    #     if disaster_padded is not None:  
    #         disaster_padded = torch.tensor(disaster_padded, dtype=torch.float32, device=self.device)  
    #     padding_mask = padding_mask.to(self.device)  

    #     # ... (rest of the update function remains unchanged)

    #     # Actions, dones: flat [N_total]  
    #     actions_tensor = torch.tensor(all_actions, dtype=torch.int64, device=self.device)  
    #     dones_tensor = torch.tensor(all_dones, dtype=torch.float32, device=self.device)  

    #     # Rewards: [B]  
    #     rewards_tensor = torch.tensor(batch_rewards, dtype=torch.float32, device=self.device)  

    #     # batch_segment_ids: flat indices for index_add_ (0 to B-1 repeated for each vehicle's group)  
    #     batch_segment_ids = []  
    #     for i, size in enumerate(group_sizes):  
    #         batch_segment_ids.extend([i] * size)  
    #     batch_ids_tensor = torch.tensor(batch_segment_ids, dtype=torch.int64, device=self.device)  

    #     # --- 3. Forward Pass (Batch) ---  
    #     # Now outputs [B*max_N, A], but padded positions are ignored via gathering on real indices  
    #     q_values_all = self.q_network(obs_padded, disaster_padded, padding_mask=padding_mask)  
        
    #     # Select Q for actions: only real vehicles (first N_total rows, since padded are at the end per group)  
    #     N_total = sum(group_sizes)  
    #     q_values_all = q_values_all[:N_total]  # Trim padded if any (but since flattened, it's fine)  
    #     q_chosen = q_values_all.gather(1, actions_tensor.unsqueeze(1)).squeeze(1)  

    #     # --- 4. VDN Aggregation (Sum Q per batch sample) ---
        
    #     # Initialize container for Total Q per batch sample: [B]
    #     q_total_current = torch.zeros(valid_batch_count, device=self.device, dtype=torch.float32)
        
    #     # Sum individual agent Qs into their corresponding batch index
    #     # [B].index_add_(dim=0, index=[N], source=[N]) -> Result: [B]
    #     q_total_current.index_add_(0, batch_ids_tensor, q_chosen)

    #     # --- 5. Target Calculation ---  
    #     with torch.no_grad():  
    #         next_q_main = self.q_network(next_obs_padded, disaster_padded, padding_mask=padding_mask)  
    #         next_q_main = next_q_main[:N_total]  # Trim  
    #         next_actions = torch.argmax(next_q_main, dim=1)  

    #         next_q_target = self.target_network(next_obs_padded, disaster_padded, padding_mask=padding_mask)  
    #         next_q_target = next_q_target[:N_total]  # Trim  
    #         next_q_values = next_q_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)  

    #         next_q_values = next_q_values * (1 - dones_tensor)  

    #         q_total_next = torch.zeros(valid_batch_count, device=self.device, dtype=torch.float32)  
    #         q_total_next.index_add_(0, batch_ids_tensor, next_q_values)  

    #         target = rewards_tensor + self.gamma * q_total_next  

    #     # --- 6. Loss & Optimization ---
        
    #     # Huber Loss per batch sample (delta=1.0)
    #     # ([B] - [B]) -> [B]
    #     # use huber loss, replace MSE loss
    #     loss = F.smooth_l1_loss(q_total_current, target, reduction='none')
        
    #     # Sum all batch losses to get single gradient scalar
    #     # [B] -> Scalar
    #     total_loss = loss.sum()

    #     self.optimizer.zero_grad()
    #     total_loss.backward()
        
    #     # track gradients for debugging 
    #     self.track_gradients(plotter)

    #     # freeze layers and retrain explosion layers if needed
    #     self.controller.check_and_adapt()

    #     # Clip gradient norm to 1
    #     torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=self.max_grad_norm)

    #     # Clip individual gradient values to [-0.5, 0.5]
    #     # torch.nn.utils.clip_grad_value_(self.q_network.parameters(), clip_value=0.5)
    #     self.optimizer.step()

    #     # --- NEW CODE START ---
    #     # Step the scheduler based on the loss of this batch
    #     # self.scheduler.step(total_loss.item())
    #     # --- NEW CODE END ---

    #     # Update target network
    #     self.update_step_count += 1
    #     # OPTION 1: Soft Update (Recommended for continuous control or modern DQN)
    #     # logic: "Move target slightly towards policy every single step"
    #     self.soft_update(self.q_network, self.target_network)

    #     # OPTION 2: Hard Update (Old school DQN)
    #     # logic: "Copy weights entirely every N steps"
    #     # if self.update_step_count % self.target_update_freq == 0:
    #     #     self.target_network.load_state_dict(self.q_network.state_dict())

    #     if plotter:
    #         plotter.record_loss(self.group_id, total_loss.item())

    #     return total_loss.item()

    def update(self, plotter=None):
        if len(self.replay_buffer) < self.batch_size or not self.training:
            return None
        
        batch = self.replay_buffer.sample(self.batch_size)
        
        all_obs, all_next_obs, all_actions, all_dones, all_disasters = [], [], [], [], []
        batch_rewards, group_sizes = [], []
        
        # NEW
        all_hist_obs, all_next_hist_obs = [], []
        valid_batch_count = 0

        for joint_transition in batch:
            vehicles_data = joint_transition['vehicles']
            global_reward = joint_transition['global_reward']

            if len(vehicles_data) == 0: continue

            batch_rewards.append(global_reward)
            group_sizes.append(len(vehicles_data))

            group_obs, group_next_obs, group_actions, group_dones, group_disasters = [], [], [], [], []
            group_hist_obs, group_next_hist_obs = [], []

            for v in vehicles_data:
                # Unpack
                _, obs, act, n_obs, done, dis, h_o, nh_o = v
                
                group_obs.append(np.squeeze(obs))
                group_actions.append(act)
                group_next_obs.append(np.squeeze(n_obs))
                group_dones.append(done)
                group_disasters.append(np.squeeze(dis) if dis is not None else None)
                group_hist_obs.append(h_o)
                group_next_hist_obs.append(nh_o)

            all_obs.append(np.array(group_obs))
            all_next_obs.append(np.array(group_next_obs))
            all_actions.extend(group_actions)
            all_dones.extend(group_dones)
            
            if group_disasters[0] is not None:
                all_disasters.append(np.array(group_disasters))
            else:
                all_disasters.append(None)
                
            all_hist_obs.append(np.stack(group_hist_obs))
            all_next_hist_obs.append(np.stack(group_next_hist_obs))
            valid_batch_count += 1

        if not all_obs: return 0.0

        max_N = max(group_sizes)
        obs_dim = all_obs[0].shape[-1]
        seq_len = all_hist_obs[0].shape[1]

        # Padding
        obs_padded = np.zeros((valid_batch_count, max_N, obs_dim), dtype=np.float32)
        next_obs_padded = np.zeros((valid_batch_count, max_N, obs_dim), dtype=np.float32)
        
        # NEW: 4D Padding for the LSTM
        hist_obs_padded = np.zeros((valid_batch_count, max_N, seq_len, obs_dim), dtype=np.float32)
        next_hist_obs_padded = np.zeros((valid_batch_count, max_N, seq_len, obs_dim), dtype=np.float32)

        if all(d is None for d in all_disasters):
            disaster_padded = None
        else:
            disaster_padded = np.zeros((valid_batch_count, max_N, all_disasters[0].shape[-1]), dtype=np.float32)

        padding_mask = torch.ones((valid_batch_count, max_N), dtype=torch.bool)

        for i in range(valid_batch_count):
            num_v = group_sizes[i]
            obs_padded[i, :num_v] = all_obs[i]
            next_obs_padded[i, :num_v] = all_next_obs[i]
            hist_obs_padded[i, :num_v] = all_hist_obs[i]
            next_hist_obs_padded[i, :num_v] = all_next_hist_obs[i]
            
            if disaster_padded is not None and all_disasters[i] is not None:
                disaster_padded[i, :num_v] = all_disasters[i]
            padding_mask[i, :num_v] = False 

        # Convert to Tensors
        obs_padded = torch.tensor(obs_padded, dtype=torch.float32, device=self.device)
        next_obs_padded = torch.tensor(next_obs_padded, dtype=torch.float32, device=self.device)
        hist_obs_padded = torch.tensor(hist_obs_padded, dtype=torch.float32, device=self.device)
        next_hist_obs_padded = torch.tensor(next_hist_obs_padded, dtype=torch.float32, device=self.device)
        
        if disaster_padded is not None:
            disaster_padded = torch.tensor(disaster_padded, dtype=torch.float32, device=self.device)
        padding_mask = padding_mask.to(self.device)

        actions_tensor = torch.tensor(all_actions, dtype=torch.int64, device=self.device)
        dones_tensor = torch.tensor(all_dones, dtype=torch.float32, device=self.device)
        rewards_tensor = torch.tensor(batch_rewards, dtype=torch.float32, device=self.device)

        batch_segment_ids = []
        for i, size in enumerate(group_sizes):
            batch_segment_ids.extend([i] * size)
        batch_ids_tensor = torch.tensor(batch_segment_ids, dtype=torch.int64, device=self.device)

        # --- Forward Pass with hist_obs ---
        q_values_all = self.q_network(obs_padded, disaster_padded, padding_mask, hist_obs_padded)
        N_total = sum(group_sizes)
        q_values_all = q_values_all[:N_total]
        q_chosen = q_values_all.gather(1, actions_tensor.unsqueeze(1)).squeeze(1)

        q_total_current = torch.zeros(valid_batch_count, device=self.device, dtype=torch.float32)
        q_total_current.index_add_(0, batch_ids_tensor, q_chosen)

        # --- Target Calculation ---
        with torch.no_grad():
            next_q_main = self.q_network(next_obs_padded, disaster_padded, padding_mask, next_hist_obs_padded)
            next_actions = torch.argmax(next_q_main[:N_total], dim=1)

            next_q_target = self.target_network(next_obs_padded, disaster_padded, padding_mask, next_hist_obs_padded)
            next_q_values = next_q_target[:N_total].gather(1, next_actions.unsqueeze(1)).squeeze(1)
            next_q_values = next_q_values * (1 - dones_tensor)

            q_total_next = torch.zeros(valid_batch_count, device=self.device, dtype=torch.float32)
            q_total_next.index_add_(0, batch_ids_tensor, next_q_values)
            target = rewards_tensor + self.gamma * q_total_next

        # --- Loss ---
        loss = F.smooth_l1_loss(q_total_current, target, reduction='none')
        total_loss = loss.sum()

        # bacckpropagation
        self.optimizer.zero_grad()
        total_loss.backward()

        # track gradient
        self.track_gradients(plotter)

        # check freezing/explosion layers
        self.controller.check_and_adapt()

        # clip norm gradient
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=self.max_grad_norm)
       
        self.optimizer.step()

        self.update_step_count += 1
        self.soft_update(self.q_network, self.target_network)

        if plotter:
            plotter.record_loss(self.group_id, total_loss.item())
            
        return total_loss.item()

    def track_gradients(self, plotter):
        """
        Logs gradient mean, std, and norm per layer to a text file for analysis.
        Format: step, layer_name, mean, std, norm
        The aim is to detect vanishing/exploding gradients during training.
        """
        for name, param in self.q_network.named_parameters():
            if param.grad is not None:
                g = param.grad.data
                
                # Calculate stats
                mean_val = g.mean().item()
                std_val = g.std().item()
                norm_val = g.norm().item()
                
                # Write to file (txt format)
                plotter.write_each_gradient(self.update_step_count, self.group_id, name, mean_val, std_val, norm_val)
    
    def soft_update(self, source_net, target_net):
        """
        Soft update target network parameters.
        theta_target = beta * theta_target + (1 - beta) * theta_source
        """
        with torch.no_grad():
            # 1. Update Parameters (Weights/Biases)
            for target_param, source_param in zip(target_net.parameters(), source_net.parameters()):
                target_param.copy_(self.beta_soft * source_param + (1.0 - self.beta_soft) * target_param)
                
            # 2. Update Buffers (Batch Norm stats: running_mean, running_var)
            # Standard practice: Hard copy buffers (do not interpolate means/vars)
            # for target_buffer, source_buffer in zip(target_net.buffers(), source_net.buffers()):
            #     target_buffer.copy_(source_buffer)

    # # CPU calculation 
    # # def update(self, plotter=None):
    #     """
    #     VDN Training with proper joint transitions, one batch → one total loss.
    #     Each sample (timestep) in the batch is processed individually.
    #     """
    #     if len(self.replay_buffer) < self.batch_size or not self.training:
    #         return None

    #     # Sample joint transitions
    #     batch = self.replay_buffer.sample(self.batch_size)  # List of joint transitions
    #     total_loss = 0.0

    #     for joint_transition in batch:
    #         vehicles_data = joint_transition['vehicles']  # List of vehicles
    #         global_reward = joint_transition['global_reward']  # Scalar reward

    #         if len(vehicles_data) == 0:
    #             continue

    #         # Extract per-vehicle data
    #         obs_list = [v[1] for v in vehicles_data]        # Each obs: (1, obs_dim)
    #         next_obs_list = [v[3] for v in vehicles_data]   # Each next_obs: (1, obs_dim)
    #         actions = [v[2] for v in vehicles_data]         # Scalar per vehicle
    #         dones = [v[4] for v in vehicles_data]           # Scalar per vehicle
    #         disasters = [v[5] for v in vehicles_data]       # Each disaster: (1, dis_dim)
            
    #         # add cuda
    #         # Convert to tensors: squeeze first dimension (1, dim) -> (dim,)
    #         obs_tensor = torch.stack([torch.tensor(o, dtype=torch.float32).squeeze(0) for o in obs_list]).to(self.device)         # [N, obs_dim]
    #         next_obs_tensor = torch.stack([torch.tensor(o, dtype=torch.float32).squeeze(0) for o in next_obs_list]).to(self.device)  # [N, obs_dim]
    #         # disaster_tensor = torch.stack([torch.tensor(d, dtype=torch.float32).squeeze(0) for d in disasters if d is not None])     # [N, dis_dim]
    #         actions_tensor = torch.tensor(actions, dtype=torch.int64, device=self.device)                                              # [N]
    #         dones_tensor = torch.tensor(dones, dtype=torch.float32, device=self.device)                                                # [N]

    #         # if disaster is empty (meaning None)
    #         # we replace with None, then later it is replaced by -1 vector in Q_network model
    #         # add cuda
    #         if all(d is None for d in disasters):
    #             disaster_tensor = None
    #         else:
    #             disaster_tensor = torch.stack([torch.tensor(d, dtype=torch.float32).squeeze(0) for d in disasters]).to(self.device)     # [N, dis_dim]

    #         # --- Forward pass: current Q ---
    #         q_values_all = self.q_network(obs_tensor, disaster_tensor)  # [N, action_dim]
    #         q_chosen = q_values_all.gather(1, actions_tensor.unsqueeze(1)).squeeze(1)            # [N]
    #         q_total_current = q_chosen.sum()                                                     # scalar

    #         # --- Target computation (Double DQN) ---
    #         with torch.no_grad():
    #             next_q_main = self.q_network(next_obs_tensor, disaster_tensor)  # [ N, action_dim]
    #             next_actions = torch.argmax(next_q_main, dim=1)                                         # [N]

    #             next_q_target = self.target_network(next_obs_tensor, disaster_tensor)  # [N, action_dim]
                
    #             next_q_values = next_q_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)          # [N]

    #             # Mask done vehicles
    #             next_q_values = next_q_values * (1 - dones_tensor)                                      # [N]

    #             # Sum over vehicles → joint Q for this timestep
    #             q_total_next = next_q_values.sum()                                                     # scalar

    #             # Compute target
    #             target = global_reward + self.gamma * q_total_next                                      # scalar

    #         # --- Loss for this timestep ---
    #         loss = (q_total_current - target) ** 2

    #         # change to huber loss
    #         # loss = F.huber_loss(q_total_current.unsqueeze(0), target.unsqueeze(0), delta=1.0)
    #         total_loss += loss

    #     # Backpropagation
    #     self.optimizer.zero_grad()
    #     total_loss.backward()
    #     torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=10.0)
    #     self.optimizer.step()

    #     # --- Update target network periodically ---
    #     self.update_step_count += 1
    #     if self.update_step_count % self.target_update_freq == 0:
    #         self.target_network.load_state_dict(self.q_network.state_dict())

    #     if plotter:
    #         plotter.record_loss(self.group_id, total_loss.item())

    #     logger.debug(f"Total Loss for batch: {total_loss.item()}")
    #     return total_loss.item()

    def save_model(self, path=None):  
        """Save the agent's model to disk"""  
        if path is None:  
            os.makedirs(self.saved_folder, exist_ok=True)  
            path = f"{self.saved_folder}/agent_{self.group_id}_{time.strftime('%Y%m%d_%H%M%S')}.pt"  
        
        try:
            # Save optimizer state before moving model to CPU
            optimizer_state = self.optimizer.state_dict()
            # Save model state dicts on CPU
            policy_state = self.q_network.state_dict()
            target_state = self.target_network.state_dict()
            # Save everything
            torch.save({
                'policy_net_state_dict': policy_state,
                'target_net_state_dict': target_state,
                'optimizer_state_dict': optimizer_state,
                'group_id': self.group_id,
                'total_steps': self.total_steps,
            }, path)
            logger.success(f"Model saved to {path}")
            return path
        except Exception as e:
            logger.error(f"Failed to save model at {path}: {e}")
   
        return path  
    
    # def load_model(self, path):  
    #     """Load a saved model from disk"""  
    #     if not os.path.exists(path):  
    #         logger.warning(f"Model file not found: {path}")  
    #         return False  
            
    #     checkpoint = torch.load(path)  
        
    #     # Load the saved state dictionaries  
    #     self.q_network.load_state_dict(checkpoint['policy_net_state_dict'])  
    #     self.target_network.load_state_dict(checkpoint['target_net_state_dict'])  
    #     self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])  
    #     # self.epsilon = checkpoint['epsilon']  
    #     self.total_steps = checkpoint['total_steps'] 
        
    #     logger.info(f"Model {checkpoint['group_id']} loaded from {path}")  
    #     return True  

    # def load_model(self, path, fine_tune_mode=False):
    #     """Load a saved model from disk, with optional fine-tuning mode."""
    #     if not os.path.exists(path):
    #         logger.warning(f"Model file not found: {path}")
    #         return False
            
    #     checkpoint = torch.load(path)
        
    #     # Load model weights
    #     self.q_network.load_state_dict(checkpoint['policy_net_state_dict'])
    #     self.target_network.load_state_dict(checkpoint['target_net_state_dict'])
        
    #     # Load optimizer state only if not fine-tuning
    #     if not fine_tune_mode:
    #         self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
    #     self.total_steps = checkpoint.get('total_steps', 0)
    #     logger.info(f"Model {checkpoint.get('group_id', 'unknown')} loaded from {path}")

    #     # --- Fine-tuning setup ---
    #     if fine_tune_mode:
    #         self.setup_fine_tune()
          
    #     else:
    #     # otherwise, do inference mode
    #         self.set_eval_mode()

    #     return True

    def load_model(self, path, fine_tune_mode=False):
        """Load a saved model from disk, with optional fine-tuning mode."""
        if not os.path.exists(path):
            logger.warning(f"Model file not found: {path}")
            return False
            
        checkpoint = torch.load(path, map_location=self.device)
        
        # 1. Load model weights 
        try:
            self.q_network.load_state_dict(checkpoint['policy_net_state_dict'])
            self.target_network.load_state_dict(checkpoint['target_net_state_dict'])
        except RuntimeError as e:
            logger.error(f"Model architecture mismatch: {e}")
            return False
        
        # 2. Safely try to load optimizer in inference mode
        # in inference mode, we do not always need to load optimizer
        if not fine_tune_mode:
            try:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            except ValueError as e:
                # This catches your specific error and lets the code continue
                logger.warning(f"Warning: Could not load optimizer state (Size mismatch). Starting with fresh optimizer. \nDetails: {e}")
            except KeyError:
                logger.warning("Optimizer state not found in checkpoint.")
        
        self.total_steps = checkpoint.get('total_steps', 0)
        logger.info(f"Model {checkpoint.get('group_id', 'unknown')} loaded from {path}")

        # --- Fine-tuning setup ---
        if fine_tune_mode:
            self.setup_fine_tune()
        else:
            # otherwise, do inference mode
            self.set_eval_mode()

        return True
    
    def setup_fine_tune(self):
        logger.info("Fine-tune mode activated: freezing encoder layers and setting training mode.")
        self.set_train_mode()

        # Freeze encoder layers in q_network
        for name, param in self.q_network.named_parameters():
            # name of freezing layers
            if any(k in name for k in ["input_proj", "transformer", "disaster_encoder"]):
                param.requires_grad = False
            else:
                param.requires_grad = True

        # Define new fine-tuning optimizer (fresh)
        self.optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.q_network.parameters()),
            lr=self.CONFIG['learning_rate'],  
            weight_decay=1e-4
        )

    
    def load_offline_model(self, path, agent_name):
        """Load a saved model from disk"""  
        if not os.path.exists(path):  
            logger.debug(f"Model file not found: {path}")  
            return False  

        # load the model
        checkpoint = torch.load(path)  
        # load the weights of model
        self.q_network.load_state_dict(checkpoint, map_location=self.device)
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