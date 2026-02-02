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

        # self.optimizer = optim.RMSprop(self.q_network.parameters(), 
        #                        lr=self.CONFIG['learning_rate'], 
        #                        alpha=0.99,      # Smoothing constant (default 0.99)
        #                        eps=1e-8,        # Stability term
        #                        weight_decay=self.weight_decay,
        #                        momentum=0.95)    # RMSProp default is 0, Adam has built-in momentum
        
        # --- NEW CODE START ---  
        # "patience": Number of update steps to wait before reducing LR if loss doesn't improve
        # "factor": How much to reduce LR (0.5 means cut in half)
        # self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        #     self.optimizer, mode='min', factor=0.5, patience=15000
        # )
        # --- NEW CODE END ---

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
        self.target_update_freq = self.CONFIG['update_frequency']  
        self.update_step_count = 0  

        # configs for save and load model
        self.saved_folder = os.path.join(os.path.dirname(__file__), '..', 'new_models') 
        # Add training mode flag  
        self.training = True  

    def epsilon_by_step(self):  
        return self.epsilon_final + (self.epsilon_start - self.epsilon_final) * \
                np.math.exp(-1. * self.total_steps / self.epsilon_decay) 

    
    def softmax_select_action(self, observation, feasible_actions, encoded_disaster_obs):
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
            q_values = self.q_network(observation, encoded_disaster_obs)  # [1, action_dim]
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
    
    def store_transition_timestep(self, vehicle_id, obs, action, next_obs, done, disaster_obs):
        """
        Store individual transition for current timestep.
        Call end_timestep() when all vehicles have acted.
        """
        self.current_timestep_transitions.append((vehicle_id, obs, action, next_obs, done, disaster_obs))
    
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


    def update(self, plotter=None):
        """
        VDN Training with optimized batch calculation.
        Transfers all data to GPU in one go and computes loss using vectorized operations.
        """
        if len(self.replay_buffer) < self.batch_size or not self.training:
            return None
        
        # if len(self.replay_buffer) - self.current_size_buffer == 100: 
        #     # update network params every 100 samples added to buffer
        #     self.current_size_buffer = len(self.replay_buffer)

        # Sample batch
        batch = self.replay_buffer.sample(self.batch_size)
        
        # --- 1. Data Preparation (CPU Side) ---
        all_obs = []
        all_next_obs = []
        all_actions = []
        all_dones = []
        all_disasters = []
        
        batch_rewards = []
        batch_segment_ids = [] 
        
        valid_batch_count = 0 
        
        for joint_transition in batch:
            vehicles_data = joint_transition['vehicles']
            global_reward = joint_transition['global_reward']

            if len(vehicles_data) == 0:
                continue

            batch_rewards.append(global_reward)

            for v in vehicles_data:
                # v[1] is (1, O) or (O,)
                all_obs.append(v[1])
                all_next_obs.append(v[3])
                all_actions.append(v[2])
                all_dones.append(v[4])
                all_disasters.append(v[5])
                # Map this agent to the current batch sample index (0 to B-1)
                batch_segment_ids.append(valid_batch_count)
            
            valid_batch_count += 1

        if not all_obs:
            return 0.0

        # --- 2. Move to GPU (Single Transfer) ---
        # Stack observations: List of N arrays -> Tensor [N, O]
        obs_tensor = torch.tensor(np.concatenate(all_obs, axis=0), dtype=torch.float32, device=self.device)
        next_obs_tensor = torch.tensor(np.concatenate(all_next_obs, axis=0), dtype=torch.float32, device=self.device)
        
        # Actions: List of N ints -> Tensor [N]
        actions_tensor = torch.tensor(all_actions, dtype=torch.int64, device=self.device)
        
        # Dones: List of N floats -> Tensor [N]
        dones_tensor = torch.tensor(all_dones, dtype=torch.float32, device=self.device)
        
        # Rewards: List of B floats -> Tensor [B] (One global reward per timestep)
        rewards_tensor = torch.tensor(batch_rewards, dtype=torch.float32, device=self.device)
        
        # Segment IDs: List of N ints -> Tensor [N] (Used for summation grouping)
        batch_ids_tensor = torch.tensor(batch_segment_ids, dtype=torch.int64, device=self.device)

        # Disasters: [N, D] or None
        if all(d is None for d in all_disasters):
            disaster_tensor = None
        else:
            disaster_tensor = torch.tensor(np.concatenate(all_disasters, axis=0), dtype=torch.float32, device=self.device)

        # --- 3. Forward Pass (Batch) ---
        
        # Compute Q-values for ALL agents
        # Input: [N, O] -> Output: [N, A]
        # add changes
        q_values_all = self.q_network(obs_tensor, disaster_tensor   )
        
        # Select Q-value for the specific action taken
        # gather input: [N, A], index: [N, 1] -> [N, 1] -> squeeze -> [N]
        q_chosen = q_values_all.gather(1, actions_tensor.unsqueeze(1)).squeeze(1)

        # --- 4. VDN Aggregation (Sum Q per batch sample) ---
        
        # Initialize container for Total Q per batch sample: [B]
        q_total_current = torch.zeros(valid_batch_count, device=self.device, dtype=torch.float32)
        
        # Sum individual agent Qs into their corresponding batch index
        # [B].index_add_(dim=0, index=[N], source=[N]) -> Result: [B]
        q_total_current.index_add_(0, batch_ids_tensor, q_chosen)

        # --- 5. Target Calculation (Double DQN) ---
        with torch.no_grad():
            # Main Network: Get argmax action
            # Input: [N, O] -> Output: [N, A]
            # add changes
            next_q_main = self.q_network(next_obs_tensor, disaster_tensor)
            # Output: [N]
            next_actions = torch.argmax(next_q_main, dim=1)

            # Target Network: Get Q-value
            # Input: [N, O] -> Output: [N, A]
            # add changes
            next_q_target = self.target_network(next_obs_tensor, disaster_tensor)
            
            # Select Q-value based on Main Net's action
            # Output: [N]
            next_q_values = next_q_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)

            # Apply Done mask (if agent is done, next Q is 0)
            # [N] * [N] -> [N]
            next_q_values = next_q_values * (1 - dones_tensor)

            # Aggregate Next Q (Sum per batch sample)
            # Initialize: [B]
            q_total_next = torch.zeros(valid_batch_count, device=self.device, dtype=torch.float32)
            
            # Sum: [B] <- [N]
            q_total_next.index_add_(0, batch_ids_tensor, next_q_values)

            # Compute Scalar Target per batch sample
            # [B] + scalar * [B] -> [B]
            target = rewards_tensor + self.gamma * q_total_next

        # --- 6. Loss & Optimization ---
        
        # Huber Loss per batch sample (delta=1.0)
        # ([B] - [B]) -> [B]
        # use huber loss, replace MSE loss
        loss = F.smooth_l1_loss(q_total_current, target, reduction='none')
        
        # Sum all batch losses to get single gradient scalar
        # [B] -> Scalar
        total_loss = loss.sum()

        self.optimizer.zero_grad()
        total_loss.backward()
        
        # track gradients for debugging 
        self.track_gradients()

        # Clip gradient norm to 1
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=self.max_grad_norm)

        # Clip individual gradient values to [-0.5, 0.5]
        # torch.nn.utils.clip_grad_value_(self.q_network.parameters(), clip_value=0.5)
        self.optimizer.step()

        # --- NEW CODE START ---
        # Step the scheduler based on the loss of this batch
        # self.scheduler.step(total_loss.item())
        # --- NEW CODE END ---

        # Update target network
        self.update_step_count += 1
        # OPTION 1: Soft Update (Recommended for continuous control or modern DQN)
        # logic: "Move target slightly towards policy every single step"
        self.soft_update(self.q_network, self.target_network)

        # OPTION 2: Hard Update (Old school DQN)
        # logic: "Copy weights entirely every N steps"
        # if self.update_step_count % self.target_update_freq == 0:
        #     self.target_network.load_state_dict(self.q_network.state_dict())

        if plotter:
            plotter.record_loss(self.group_id, total_loss.item())

        return total_loss.item()
    

    def track_gradients(self):
        """
        Logs gradient mean, std, and norm per layer to a text file for analysis.
        Format: step, layer_name, mean, std, norm
        """
        log_file = "gradient_stats.txt"
        
        # Create header if file doesn't exist
        if not os.path.exists(log_file):
            with open(log_file, "w") as f:
                f.write("step,group,layer,mean,std,norm\n")

        with open(log_file, "a") as f:
            # Use named_parameters to identify WHICH layer is vanishing/exploding
            for name, param in self.q_network.named_parameters():
                if param.grad is not None:
                    g = param.grad.data
                    
                    # Calculate stats
                    mean_val = g.mean().item()
                    std_val = g.std().item()
                    norm_val = g.norm().item()
                    
                    # Write to file (CSV format)
                    f.write(f"{self.update_step_count},{self.group_id},{name},{mean_val:.6f},{std_val:.6f},{norm_val:.6f}\n")
                    
                    # Optional: Keep your debug logger if you want real-time stream
                    # logger.debug(f"Step {self.update_step_count} | {name} | Norm: {norm_val:.4f}")
    
    def soft_update(self, source_net, target_net, beta=0.01):
        """
        Soft update target network parameters.
        theta_target = beta * theta_target + (1 - beta) * theta_source
        """
        with torch.no_grad():
            # 1. Update Parameters (Weights/Biases)
            for target_param, source_param in zip(target_net.parameters(), source_net.parameters()):
                target_param.copy_(beta * source_param + (1.0 - beta) * target_param)
                
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
            if any(k in name for k in ["input_proj", "transformer"]):
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