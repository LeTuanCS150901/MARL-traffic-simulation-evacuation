from PPO_network import PPONetwork  # Changed import
from logs.logger import logger  
from read_config import get_config

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import os
import numpy as np
import time

class Agent:  
    def __init__(self, obs_dim, action_dim, group_id, train_offline=False, config=None):

        if train_offline:
            self.CONFIG = config
        else:
            self.CONFIG = get_config()

        self.group_id = group_id  

        # add gpu:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # --- PPO Hyperparameters ---
        self.lr = self.CONFIG.get('learning_rate', 0.000003)
        self.gamma = self.CONFIG['gamma']
        self.eps_clip = 0.2           # Clipping for PPO
        self.K_epochs = 30             # Number of update epochs per batch
        self.batch_size = self.CONFIG.get('batch_size', 2048) # Timesteps to collect for memory storage
        self.mini_batch_size = 512 # put high, depends on gpu memory. High mini batch -> reduce time cpu-gpu transfer
        self.entropy_coef = 0.3      # Explore/Exploit balance. High entropy => more exploration
        self.critic_coef = 0.8
        self.gae_lamda = 0.9
        
        # --- NEW: Adaptive KL Parameters ---
        self.target_kl = 0.03         # Target KL divergence (approx 0.01 - 0.05 is standard)
        self.kl_beta = 0.5            # Initial penalty coefficient

        # --- Network ---
        # add gpu
        self.ppo_network = PPONetwork(obs_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.ppo_network.parameters(), lr=self.lr, weight_decay=1e-4)

        # --- Memory ---
        # Replaces ExperienceReplayBuffer. Stores rollout data.
        self.memory = [] 

        # --- NEW: Learning Rate Scheduler (Decay) ---
        # Decays LR by 0.99 every update step. 
        # After 100 updates, LR will be ~36% of initial. Tune 'gamma' if decay is too fast.
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.99)

        
        # Temporary storage for the current timestep (group of vehicles)
        self.current_timestep_data = []
        self.current_timestep = 0 

        self.total_steps = 0  
        self.saved_folder = os.path.join(os.path.dirname(__file__), '..', 'new_models') 
        self.training = True  

    def softmax_select_action(self, observation, feasible_actions, disaster_obs=None): 
        """
        Select action using PPO Policy (Stochastic during training).
        """ 
        if not self.training:
            return self.select_action_for_inference(observation, feasible_actions, disaster_obs)

        with torch.no_grad():
            # 1. Ensure obs is correct shape
            if observation.dim() == 1:
                observation = observation.unsqueeze(0)

            # 2. Add gpu: MOVE INPUTS TO GPU (Crucial Step) since we perform forward to get prediction
            observation = observation.to(self.device)
            if disaster_obs is not None:
                disaster_obs = disaster_obs.to(self.device)

            # Get logits and value from network
            logits, state_value = self.ppo_network(observation, disaster_obs)
            logits = logits.squeeze(0) # [action_dim]

            # 3. CREATE MASK ON GPU
            # feasible_mask = torch.tensor(feasible_actions, dtype=torch.bool)  <-- OLD (CPU)
            feasible_mask = torch.tensor(feasible_actions, dtype=torch.bool, device=self.device) # <-- NEW (GPU)
            
            # Important: Set infeasible to -inf so probabilities are 0
            logits[~feasible_mask] = float('-inf')

            # Create distribution
            dist = Categorical(logits=logits)

            # Sample action
            action = dist.sample()
            action_logprob = dist.log_prob(action)

            # Store data 
            self.current_timestep_data.append({
                'obs': observation,          
                'disaster': disaster_obs,    
                'action': action,            
                'log_prob': action_logprob,  
                'val': state_value,          
                'mask': feasible_mask        
            })

            return action.item()
        
    def select_action_for_inference(self, observation, feasible_actions, disaster_obs=None):
        """
        Deterministic (or Greedy) selection for evaluation.
        """
        with torch.no_grad():
            if observation.dim() == 1:
                observation = observation.unsqueeze(0)
            
            logits, _ = self.ppo_network(observation, disaster_obs)
            logits = logits.squeeze(0)
            
            feasible_mask = torch.tensor(feasible_actions, dtype=torch.bool)
            logits[~feasible_mask] = float('-inf')

            # Greedy selection
            action_index = torch.argmax(logits).item()
            return action_index

    def store_transition_timestep(self, vehicle_id, obs, action, next_obs, done, disaster_obs):
        """
        Legacy function signature kept for compatibility. 
        In PPO, we store data inside select_action and end_timestep.
        """
        pass
    
    def end_timestep(self, global_reward):
        """
        Called when all vehicles in the group have acted.
        Combines individual vehicle data with the global shared reward.
        """
        if len(self.current_timestep_data) > 0:
            # Package the whole group's experience for this timestep
            timestep_record = {
                'vehicles': self.current_timestep_data, # List of dicts (one per vehicle)
                'reward': global_reward,
                'is_terminal': False # Simplified terminal handling
            }
            self.memory.append(timestep_record)
            
            self.current_timestep_data = [] # Reset for next step
            self.current_timestep += 1
            self.total_steps += 1

    def update(self, plotter=None):
        """
        PPO Update Phase with Mini-Batch Processing for GPU Optimization.

        This is default PPO with clipped objective
        For PPO with KL or adaptive KL penalty: there is no clipping anymore of objective function of actor, and
        the penalty is added into final loss function
        """
        if len(self.memory) < self.batch_size or not self.training:
            return None

        # Define optimization hyperparams locally (or move to __init__)
        # Adjust based on GPU memory (64, 128, 256)

        # --- 1. Prepare Data for GAE ---
        rewards = []
        dones = []
        values = []

        # Extract step-level data
        for step in self.memory:
            rewards.append(step['reward'])
            dones.append(step['is_terminal'])
            # Average value of all vehicles in this step (matches your original logic)
            step_vals = torch.stack([v['val'] for v in step['vehicles']]).squeeze()
            values.append(step_vals)

        # --- 2. Calculate Advantages (GAE) ---
        advantages = []
        returns = [] 
        gae = 0
        next_val = 0 

        for i in reversed(range(len(self.memory))):
            reward = rewards[i]
            val = values[i].mean().item() 
            done = dones[i]
            
            delta = reward + self.gamma * next_val * (1 - done) - val
            gae = delta + self.gamma * self.gae_lamda * (1 - done) * gae 
            
            advantages.insert(0, gae)
            returns.insert(0, gae + val)
            next_val = val

        # Normalize advantages
        adv_tensor = torch.tensor(advantages, dtype=torch.float32).to(self.device)
        adv_tensor = (adv_tensor - adv_tensor.mean()) / (adv_tensor.std() + 1e-5)
        returns_tensor = torch.tensor(returns, dtype=torch.float32).to(self.device)

        # --- 3. PRE-PROCESSING: Flatten Memory to Big Tensors ---
        # We move all data to GPU once here, instead of inside the loop
        
        all_obs = []
        all_disasters = []
        all_actions = []
        all_log_probs = []
        all_masks = []
        all_advantages = []
        all_returns = []

        has_disaster_data = False

        for i, step in enumerate(self.memory):
            vehicles = step['vehicles']
            num_vehicles = len(vehicles)

            # 1. Stack vehicle attributes
            all_obs.append(torch.stack([v['obs'] for v in vehicles]).squeeze(1))
            all_actions.append(torch.stack([v['action'] for v in vehicles]))
            all_log_probs.append(torch.stack([v['log_prob'] for v in vehicles]))
            all_masks.append(torch.stack([v['mask'] for v in vehicles]))

            # 2. Broadcast step-level Advantage/Return to all vehicles in this step
            # If step i has 5 vehicles, we repeat adv[i] 5 times
            all_advantages.append(adv_tensor[i].repeat(num_vehicles))
            all_returns.append(returns_tensor[i].repeat(num_vehicles))

            # 3. Handle Disaster (check if any vehicle has disaster data)
            dis_list = [v['disaster'] for v in vehicles]
            if not all(d is None for d in dis_list):
                has_disaster_data = True
                all_disasters.append(torch.stack(dis_list).squeeze(1))

        # Concatenate lists into single giant tensors on GPU
        full_obs = torch.cat(all_obs).to(self.device)
        full_actions = torch.cat(all_actions).to(self.device)
        full_log_probs = torch.cat(all_log_probs).to(self.device)
        full_masks = torch.cat(all_masks).to(self.device)
        full_adv = torch.cat(all_advantages).to(self.device)
        full_returns = torch.cat(all_returns).to(self.device)
        
        if has_disaster_data:
            full_disasters = torch.cat(all_disasters).to(self.device)
        else:
            full_disasters = None

        dataset_size = full_obs.size(0)
        
        total_loss_val = 0
        # total_approx_kl = 0 

        # --- 4. PPO Optimization Loop (Mini-Batches) ---
        for _ in range(self.K_epochs):
            epoch_loss = 0
            # epoch_kl = 0
            
            # Generate random indices for shuffling
            indices = torch.randperm(dataset_size).to(self.device)
            
            # Iterate through mini-batches
            num_batches = 0
            for start in range(0, dataset_size, self.mini_batch_size):
                num_batches += 1
                end = start + self.mini_batch_size
                idx = indices[start:end]

                # Slice the mini-batch (mb)
                mb_obs = full_obs[idx]
                mb_actions = full_actions[idx]
                mb_old_log_probs = full_log_probs[idx]
                mb_masks = full_masks[idx]
                mb_adv = full_adv[idx]
                mb_returns = full_returns[idx]
                
                mb_dis = full_disasters[idx] if full_disasters is not None else None

                # --- Forward Pass ---
                log_probs, state_values, dist_entropy = self.ppo_network.evaluate(
                    mb_obs, mb_dis, mb_actions, mb_masks
                )

                # --- Ratios ---
                ratios = torch.exp(log_probs - mb_old_log_probs.detach())

                # --- Approx KL Divergence ---
                # approx_kl = ((ratios - 1) - torch.log(ratios)).mean()
                # epoch_kl += approx_kl.item()

                # --- Surrogate Loss ---
                surr1 = ratios * mb_adv
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * mb_adv
                loss_actor = -torch.min(surr1, surr2).mean()

                # --- Critic Loss ---
                loss_critic = nn.SmoothL1Loss()(state_values.view(-1), mb_returns)

                # --- Entropy Loss ---
                loss_entropy = -dist_entropy.mean()

                # --- KL Penalty Loss ---
                # kl_penalty = self.kl_beta * approx_kl

                # --- Total Loss ---
                loss = loss_actor + self.critic_coef * loss_critic + self.entropy_coef * loss_entropy 

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.ppo_network.parameters(), max_norm=0.5)
                self.optimizer.step()

                epoch_loss += loss.item()

            # Aggregate stats (average over batches)
            if num_batches > 0:
                total_loss_val += (epoch_loss / num_batches)
                # total_approx_kl += (epoch_kl / num_batches)

        # --- 5. Update Adaptive KL Beta ---
        # Normalize stats by K_epochs
        avg_loss = total_loss_val / self.K_epochs
        # avg_kl_total = total_approx_kl / self.K_epochs
        
        # if avg_kl_total > 1.5 * self.target_kl:
        #     self.kl_beta *= 2 
        # elif avg_kl_total < self.target_kl / 1.5:
        #     self.kl_beta /= 2 
        
        # self.kl_beta = np.clip(self.kl_beta, 0.01, 20.0)

        # --- 6. Step Learning Rate Scheduler ---
        self.scheduler.step()

        # Clear memory
        self.memory.clear()
        
        if plotter:
            plotter.record_loss(self.group_id, avg_loss)

        return avg_loss

    # def update(self, plotter=None):
    #     """
    #     PPO Update Phase with KL Penalty and LR Decay.
    #     """
    #     if len(self.memory) < self.batch_size or not self.training:
    #         return None

    #     # --- 1. Prepare Data for GAE ---
    #     rewards = []
    #     dones = []
    #     values = []

    #     for step in self.memory:
    #         rewards.append(step['reward'])
    #         dones.append(step['is_terminal'])
    #         step_vals = torch.stack([v['val'] for v in step['vehicles']]).squeeze()
    #         values.append(step_vals)

    #     # --- 2. Calculate Advantages (GAE) ---
    #     advantages = []
    #     returns = [] 
    #     gae = 0
    #     next_val = 0 

    #     for i in reversed(range(len(self.memory))):
    #         reward = rewards[i]
    #         val = values[i].mean().item() 
    #         done = dones[i]
            
    #         delta = reward + self.gamma * next_val * (1 - done) - val
    #         gae = delta + self.gamma * self.gae_lamda * (1 - done) * gae 
            
    #         advantages.insert(0, gae)
    #         returns.insert(0, gae + val)
    #         next_val = val

    #     # Normalize advantages
    #     adv_tensor = torch.tensor(advantages, dtype=torch.float32).to(self.device)
    #     adv_tensor = (adv_tensor - adv_tensor.mean()) / (adv_tensor.std() + 1e-5)
    #     returns_tensor = torch.tensor(returns, dtype=torch.float32).to(self.device)

    #     total_loss_val = 0
    #     total_approx_kl = 0 # Track KL for beta adjustment


    #     # --- OUTSIDE the K_epochs loop (Pre-process everything) ---
    #     # 1. Flatten all steps and vehicles into huge tensors once
    #     all_obs = []
    #     all_actions = []
    #     all_log_probs = []
    #     all_advantages = []
    #     all_returns = []
    #     all_disasters = []
    #     all_masks = []

    #     # --- 3. PPO Optimization Loop ---
    #     for _ in range(self.K_epochs):
    #         epoch_loss = 0
    #         epoch_kl = 0
            
    #         for i, step in enumerate(self.memory):
    #             vehicles = step['vehicles']
                
    #             # Stack data
    #             # add gpu
    #             obs_batch = torch.stack([v['obs'] for v in vehicles]).squeeze(1).to(self.device)
                
    #             disaster_list = [v['disaster'] for v in vehicles]
    #             if all(d is None for d in disaster_list):
    #                 dis_batch = None
    #             else:
    #                 dis_batch = torch.stack([d for d in disaster_list]).squeeze(1).to(self.device)
                
    #             # add gpu
    #             old_actions = torch.stack([v['action'] for v in vehicles]).to(self.device)
    #             old_log_probs = torch.stack([v['log_prob'] for v in vehicles]).to(self.device)
    #             masks_batch = torch.stack([v['mask'] for v in vehicles]).to(self.device)

    #             # --- Forward Pass ---
    #             log_probs, state_values, dist_entropy = self.ppo_network.evaluate(
    #                 obs_batch, dis_batch, old_actions, masks_batch
    #             )

    #             # --- Ratios ---
    #             ratios = torch.exp(log_probs - old_log_probs.detach())

    #             # --- NEW: Calculate Approx KL Divergence ---
    #             # Formula: (ratio - 1) - log(ratio)
    #             # We calculate this here so it is part of the computation graph (differentiable)
    #             approx_kl = ((ratios - 1) - torch.log(ratios)).mean()
                
    #             # Track for stats (detach to save memory)
    #             epoch_kl += approx_kl.item()

    #             # --- Surrogate Loss ---
    #             adv = adv_tensor[i]
    #             surr1 = ratios * adv
    #             surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * adv
    #             loss_actor = -torch.min(surr1, surr2).mean()
                
    #             # --- Critic Loss ---
    #             target_val = returns_tensor[i].repeat(len(vehicles)) 
    #             loss_critic = nn.SmoothL1Loss()(state_values.view(-1), target_val)
                
    #             # --- Entropy Loss ---
    #             loss_entropy = -dist_entropy.mean()

    #             # --- NEW: KL Penalty Loss ---
    #             kl_penalty = self.kl_beta * approx_kl

    #             # --- Total Loss ---
    #             loss = loss_actor + self.critic_coef * loss_critic + self.entropy_coef * loss_entropy + kl_penalty
                
    #             self.optimizer.zero_grad()
    #             loss.backward()
    #             torch.nn.utils.clip_grad_norm_(self.ppo_network.parameters(), max_norm=1.0)
    #             self.optimizer.step()
                
    #             epoch_loss += loss.item()

    #         # Aggregate stats
    #         total_loss_val += (epoch_loss / len(self.memory))
    #         total_approx_kl += (epoch_kl / len(self.memory))

    #     # --- 4. NEW: Update Adaptive KL Beta ---
    #     # Adjust beta based on average KL across all epochs
    #     avg_kl_batch = total_approx_kl / self.K_epochs
        
    #     if avg_kl_batch > 1.5 * self.target_kl:
    #         self.kl_beta *= 2   # Penalize more if changed too much
    #     elif avg_kl_batch < self.target_kl / 1.5:
    #         self.kl_beta /= 2   # Penalize less if changed too little
        
    #     # Clip beta to prevent explosion
    #     self.kl_beta = np.clip(self.kl_beta, 0.01, 20.0)

    #     # --- 5. NEW: Step Learning Rate Scheduler ---
    #     self.scheduler.step()

    #     # Clear memory
    #     self.memory.clear()
        
    #     if plotter:
    #         logger.debug(f"XXXXXXXXXXXXXXXgroup {self.group_id}, loss {total_loss_val / self.K_epochs}")
    #         plotter.record_loss(self.group_id, total_loss_val / self.K_epochs)

    #     return total_loss_val / self.K_epochs
    
    # def update(self, plotter=None):
    #     """
    #     PPO Update Phase.
    #     Aggregates collected memory, calculates GAE, and updates network.
    #     """
    #     # Only update if we have collected enough data (batch_size timesteps)
    #     if len(self.memory) < self.batch_size or not self.training:
    #         return None

    #     # --- 1. Prepare Data for GAE ---
    #     rewards = []
    #     dones = []
    #     values = []

    #     # Extract sequence of rewards and values from memory
    #     for step in self.memory:
    #         rewards.append(step['reward'])
    #         dones.append(step['is_terminal'])
    #         # Collect values from all vehicles in this step. Shape: [N]
    #         step_vals = torch.stack([v['val'] for v in step['vehicles']]).squeeze()
    #         values.append(step_vals)

    #     # --- 2. Calculate Advantages (GAE) ---
    #     advantages = []
    #     returns = [] # Monte Carlo returns (for Critic target)
    #     gae = 0
    #     next_val = 0 # Assume 0 value for post-terminal state

    #     # Iterate backwards
    #     for i in reversed(range(len(self.memory))):
    #         reward = rewards[i]
    #         # Mean value of the group for global advantage approximation
    #         val = values[i].mean().item() 
    #         done = dones[i]
            
    #         delta = reward + self.gamma * next_val * (1 - done) - val
    #         gae = delta + self.gamma * self.gae_lamda * (1 - done) * gae # 0.99 is GAE lambda
            
    #         advantages.insert(0, gae)
    #         # Target for critic = Advantage + Value
    #         returns.insert(0, gae + val)
            
    #         next_val = val

    #     # Normalize advantages for stability
    #     # add gpt
    #     adv_tensor = torch.tensor(advantages, dtype=torch.float32)
    #     adv_tensor = (adv_tensor - adv_tensor.mean()) / (adv_tensor.std() + 1e-5)
    #     returns_tensor = torch.tensor(returns, dtype=torch.float32)

    #     total_loss_val = 0

    #     # --- 3. PPO Optimization Loop ---
    #     for _ in range(self.K_epochs):
    #         epoch_loss = 0
            
    #         # We iterate through collected timesteps
    #         # Note: We process one "Group Timestep" at a time to keep Transformer context valid
    #         for i, step in enumerate(self.memory):
    #             vehicles = step['vehicles']
                
    #             # Stack data for the group: [N, ...]
    #             # add gpu
    #             obs_batch = torch.stack([v['obs'] for v in vehicles]).squeeze(1)
                
    #             # Handle Disaster batching
    #             disaster_list = [v['disaster'] for v in vehicles]
    #             if all(d is None for d in disaster_list):
    #                 dis_batch = None
    #             else:
    #                 # add gpu
    #                 dis_batch = torch.stack([d for d in disaster_list]).squeeze(1)

    #             # add gpu
    #             old_actions = torch.stack([v['action'] for v in vehicles])
    #             old_log_probs = torch.stack([v['log_prob'] for v in vehicles])
    #             masks_batch = torch.stack([v['mask'] for v in vehicles])

    #             # --- Forward Pass (Re-evaluate) ---
    #             # We pass the WHOLE group obs to use Transformer attention correctly
    #             log_probs, state_values, dist_entropy = self.ppo_network.evaluate(
    #                 obs_batch, dis_batch, old_actions, masks_batch
    #             )

    #             # --- Loss Calculation ---
    #             # Ratio = pi_new / pi_old
    #             ratios = torch.exp(log_probs - old_log_probs.detach())

    #             # Surrogate Loss
    #             adv = adv_tensor[i]
    #             surr1 = ratios * adv
    #             surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * adv
                
    #             # PPO Loss
    #             loss_actor = -torch.min(surr1, surr2).mean()
                
    #             # Critic Loss (MSE)
    #             target_val = returns_tensor[i].repeat(len(vehicles)) # Broadcast return to all agents
    #             # loss_critic = nn.MSELoss()(state_values.view(-1), target_val)
    #             loss_critic = nn.SmoothL1Loss()(state_values.view(-1), target_val)
                
    #             # Entropy Loss (Bonus)
    #             loss_entropy = -dist_entropy.mean()

    #             loss = loss_actor + self.critic_coef * loss_critic + self.entropy_coef * loss_entropy
                
    #             self.optimizer.zero_grad()
    #             loss.backward()
    #             torch.nn.utils.clip_grad_norm_(self.ppo_network.parameters(), max_norm=10.0)
    #             self.optimizer.step()
                
    #             epoch_loss += loss.item()

    #         total_loss_val += (epoch_loss / len(self.memory))

    #     # --- 4. Clear Memory ---
    #     self.memory.clear()
        
    #     if plotter:
    #         plotter.record_loss(self.group_id, total_loss_val / self.K_epochs)

    #     return total_loss_val / self.K_epochs

    def save_model(self, path=None):  
        """Save the agent's model to disk"""  
        if path is None:  
            os.makedirs(self.saved_folder, exist_ok=True)  
            path = f"{self.saved_folder}/agent_ppo_{self.group_id}_{time.strftime('%Y%m%d_%H%M%S')}.pt"  
        
        try:
            torch.save({
                'policy_net_state_dict': self.ppo_network.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'group_id': self.group_id,
                'total_steps': self.total_steps,
            }, path)
            logger.success(f"Model saved to {path}")
            return path
        except Exception as e:
            logger.error(f"Failed to save model at {path}: {e}")
            return path

    def load_model(self, path, fine_tune_mode=False):
        """Load a saved model from disk"""
        if not os.path.exists(path):
            logger.warning(f"Model file not found: {path}")
            return False
            
        checkpoint = torch.load(path)
        
        try:
            
            # add gpu
            self.ppo_network.load_state_dict(checkpoint)
        except RuntimeError as e:
            logger.error(f"Model architecture mismatch: {e}")
            return False
        
        if not fine_tune_mode:
            try:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            except:
                logger.warning("Could not load optimizer state. Starting fresh.")
        
        self.total_steps = checkpoint.get('total_steps', 0)
        logger.info(f"Model {checkpoint.get('group_id', 'unknown')} loaded from {path}")

        if fine_tune_mode:
            self.setup_fine_tune()
        else:
            self.set_eval_mode()

        return True

    def setup_fine_tune(self):
        logger.info("Fine-tune mode activated: freezing encoder layers.")
        self.set_train_mode()

        # Freeze encoder layers
        for name, param in self.ppo_network.named_parameters():
            if any(k in name for k in ["input_proj", "transformer"]):
                param.requires_grad = False
            else:
                param.requires_grad = True

        self.optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, self.ppo_network.parameters()),
            lr=self.lr * 0.1,  
            weight_decay=1e-4
        )
        
    def set_eval_mode(self):  
        self.training = False  
        self.ppo_network.eval()

    def set_train_mode(self):  
        self.training = True  
        self.ppo_network.train()
    
    def load_offline_model(self, path, agent_name):
        # Kept for compatibility
        if not os.path.exists(path):  
            return False  
        checkpoint = torch.load(path)  
        self.ppo_network.load_state_dict(checkpoint)
        return True