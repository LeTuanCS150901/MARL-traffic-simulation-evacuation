from MAT_network import MATNetwork  # Renamed from Q_network
from experience_buffer import JointExperienceReplayBuffer
from logs.logger import logger  
from read_config import get_config
from adaptive_layers import AdaptiveLayerControllerAuto # New adaptive layer controller

import torch
import torch.nn.functional as F 
import torch.optim as optim
import torch.distributions as D  # For sampling
import os
import random
import numpy as np
import time

class Agent:  
    def __init__(self, obs_dim, action_dim, group_id, train_offline=False, config=None):

        # train offline from another folder
        if train_offline:
            self.CONFIG = config
        # train online with sumo interaction
        else:
            self.CONFIG = get_config()

        # Set up device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Agent {group_id} initialized using device: {self.device}")

        self.group_id = group_id  
        self.model = MATNetwork(obs_dim, action_dim).to(self.device)
        self.target_model = MATNetwork(obs_dim, action_dim).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())  # Initial sync  


        # Optimizers for encoder and decoder separately [^15][^16]
        self.encoder_optimizer = optim.Adam(self.model.encoder.parameters(), lr=self.CONFIG['learning_rate'],  
                                            weight_decay=float(self.CONFIG.get('weight_decay', 1e-4)))
        self.decoder_optimizer = optim.Adam(self.model.decoder.parameters(), lr=self.CONFIG['learning_rate'],  
                                            weight_decay=float(self.CONFIG.get('weight_decay', 1e-4)))
        
        # freeze layers and retraint the explosion layers if needed
        self.controller = AdaptiveLayerControllerAuto(model=self.model, optimizer=self.decoder_optimizer)

        
        ### PPO hyperparameters (for decoder loss) 
        self.ppo_epochs = int(self.CONFIG['model']['MAT']['ppo_epochs'])  # Usually 10 or 15
        self.clip_eps = float(self.CONFIG['model']['MAT']['clip_eps']) 
        self.entropy_coef = float(self.CONFIG['model']['MAT']['entropy_coef'])
        self.mini_batch_size = int(self.CONFIG['model']['MAT']['mini_batch_size']) # Size for each mini batch in a big batch
        self.batch_size = int(self.CONFIG['model']['MAT']['batch_size']) # big batch samples from memory buffer
        
        # epochs parameters
        self.beta_soft = float(self.CONFIG['beta_soft'])  # Soft update
        self.max_grad_norm = float(self.CONFIG['max_grad_norm']) # 
        self.gamma = float(self.CONFIG['gamma'])  # Discount factor

        self.memory_buffer = JointExperienceReplayBuffer(capacity=self.CONFIG['model']['MAT']['memory_buffer_size'])  # Adapted for trajectories
        self.current_size_buffer = 0
        
        self.current_timestep_transitions = []
        self.current_timestep = 0 

        self.total_steps = 0  
        # Configs for save and load model
        self.saved_folder = os.path.join(os.path.dirname(__file__), '..', 'new_models') 
        self.training = True  

    def softmax_select_action(self, observation, feasible_actions, encoded_disaster_obs):
        """
        observation: tensor [1, obs_dim]
        encoded_disaster_obs: tensor [1, dis_dim] or None (per-vehicle disasters)
        feasible_actions: list of bools of length action_dim
        """
        if not self.training:
            return self.select_action_for_inference(observation, feasible_actions, encoded_disaster_obs)

        if not feasible_actions:
            logger.warning("No feasible actions available. Do nothing.")
            return None

        with torch.no_grad():
            observation = observation.to(self.device)

            # Disaster input handling
            if encoded_disaster_obs is not None:
                encoded_disaster_obs = encoded_disaster_obs.to(self.device)
                disaster_input = encoded_disaster_obs.unsqueeze(0)  # [1, 1, dis_dim]
            else:
                disaster_input = None  # Let MATNetwork handle None

            # Forward through MAT in inference-like path for a single vehicle
            logits, values = self.model(observation.unsqueeze(0), disaster_input, mode='inference')  # logits: tensor, values: tensor  
            logger.debug("Action logit shape: {}".format(logits.shape if isinstance(logits, torch.Tensor) else None))  

            action_logits = logits.squeeze(0).squeeze(0)  # [action_dim]


            # Mask infeasible actions
            feasible_indices = [i for i, f in enumerate(feasible_actions) if f]
            q_for_softmax = action_logits[feasible_indices]

            prob = F.softmax(q_for_softmax, dim=-1)
            idx_within_feasible = torch.multinomial(prob, 1).item()
            action = feasible_indices[idx_within_feasible]

            return action


    def select_action_for_inference(self, observation, feasible_actions, encoded_disaster_obs):
        with torch.no_grad():  
            observation = observation.to(self.device)
            if encoded_disaster_obs is not None:
                encoded_disaster_obs = encoded_disaster_obs.to(self.device)
            
            action_logits = self.model(observation.unsqueeze(0), encoded_disaster_obs.unsqueeze(0), mode='inference')  # Autoregressive [^16]
            action_logits = action_logits.squeeze(0)

            feasible_mask = torch.tensor(feasible_actions, dtype=torch.bool)
            action_logits[~feasible_mask] = float('-inf')  

            feasible_q_values = action_logits[feasible_mask]
            probs = torch.softmax(feasible_q_values, dim=0)
            idx_within_feasible = torch.multinomial(probs, 1).item()
            feasible_indices = torch.nonzero(feasible_mask, as_tuple=True)[0]
            action_index = feasible_indices[idx_within_feasible].item()

            return action_index  

    def store_transition_timestep(self, vehicle_id, obs, action, next_obs, done, disaster_obs):
        self.current_timestep_transitions.append((vehicle_id, obs, action, next_obs, done, disaster_obs))
    
    def end_timestep(self, global_reward):
        if len(self.current_timestep_transitions) > 0:
            joint_transition = {
                'timestep': self.current_timestep,
                'vehicles': self.current_timestep_transitions.copy(),
                'global_reward': global_reward
            }
            self.memory_buffer.push(joint_transition)
            self.current_timestep_transitions = []
            self.current_timestep += 1

    def compute_gae(self, rewards, values, dones, next_values):
        """
        Computes Generalized Advantage Estimation (GAE) for a batch of trajectories.
        Paper Ref: Eq (5) mentions GAE with V(o).
        """
        gae_lambda = self.CONFIG.get('gae_lambda', 0.95)
        advantages = torch.zeros_like(rewards)
        last_gae_lam = 0
        
        # Calculate GAE reverse-order
        # Note: We assume the batch is ordered by time. If the buffer is mixed, 
        # this calculation must be done per-trajectory before batching.
        # Here we approximate by treating the whole buffer as a sequence or 
        # using the 'dones' mask to reset GAE at episode boundaries.
        
        path_len = rewards.shape[0]
        for t in reversed(range(path_len)):
            if t == path_len - 1:
                next_non_terminal = 1.0 - dones[t]
                next_value = next_values[t]
            else:
                next_non_terminal = 1.0 - dones[t]
                next_value = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value * next_non_terminal - values[t]
            last_gae_lam = delta + self.gamma * gae_lambda * next_non_terminal * last_gae_lam
            advantages[t] = last_gae_lam
            
        return advantages

    # def update(self, plotter=None):
    #     """
    #     Full MAT Update Implementation matching Algorithm 1 in Appendix A.
    #     - On-policy: Uses all data in buffer then clears it.
    #     - PPO Epochs: Iterates multiple times over the collected batch.
    #     - Joint Advantage: Uses GAE.
    #     """
    #     # 1. Check if we have enough data to perform an update
    #     if len(self.memory_buffer) < self.batch_size:
    #         return None
    

    #     # 3. PREPARE FULL BATCH (No random sampling yet, preserve order for GAE)
    #     # We assume buffer.sample(len) returns everything if len >= capacity or we force it.
    #     # Ideally, implement a buffer.get_all() method. Here we simulate it:
    #     all_transitions = self.memory_buffer.sample(len(self.memory_buffer))

    #     # --- Data Parsing & Padding (Same logic as before, applied to ALL data) ---
    #     all_obs, all_actions, all_next_obs, all_dones, all_disasters, all_rewards, group_sizes = [], [], [], [], [], [], []
        
    #     for t in all_transitions:
    #         vehicles = t['vehicles']
    #         if not vehicles: continue
            
    #         all_rewards.append(t['global_reward'])
    #         group_sizes.append(len(vehicles))
    #         all_obs.append(np.array([np.squeeze(v[1]) for v in vehicles]))
    #         all_actions.append(np.array([v[2] for v in vehicles]))
    #         all_next_obs.append(np.array([np.squeeze(v[3]) for v in vehicles]))
    #         all_dones.append(np.array([v[4] for v in vehicles]))
    #         dis_list = [v[5] for v in vehicles]
    #         all_disasters.append(np.array([np.squeeze(d) for d in dis_list]) if dis_list[0] is not None else None)

    #     if not all_obs: return 0.0

    #     B = len(all_obs)
    #     max_N = max(group_sizes)

    #     # Move to Device & Pad
    #     obs_tens = torch.zeros(B, max_N, all_obs[0].shape[-1], device=self.device)
    #     act_tens = torch.zeros(B, max_N, dtype=torch.long, device=self.device)
    #     n_obs_tens = torch.zeros(B, max_N, all_next_obs[0].shape[-1], device=self.device)
    #     dones_tens = torch.zeros(B, max_N, device=self.device)
    #     rewards_tens = torch.tensor(all_rewards, dtype=torch.float32, device=self.device)
    #     pad_mask = torch.ones(B, max_N, dtype=torch.bool, device=self.device)
        
    #     # Disaster padding handling
    #     use_disaster = all_disasters[0] is not None
    #     if use_disaster:
    #         dis_tens = torch.zeros(B, max_N, all_disasters[0].shape[-1], device=self.device)
    #     else:
    #         dis_tens = None

    #     for i in range(B):
    #         n = group_sizes[i]
    #         obs_tens[i, :n] = torch.tensor(all_obs[i], device=self.device)
    #         act_tens[i, :n] = torch.tensor(all_actions[i], device=self.device)
    #         n_obs_tens[i, :n] = torch.tensor(all_next_obs[i], device=self.device)
    #         dones_tens[i, :n] = torch.tensor(all_dones[i], device=self.device)
    #         if use_disaster:
    #             dis_tens[i, :n] = torch.tensor(all_disasters[i], device=self.device)
    #         pad_mask[i, :n] = False

    #     # 4. PRE-COMPUTE OLD LOG PROBS & VALUES (For Ratio & GAE)
    #     # We need "old" probabilities to compute r_t(theta) in Eq 5 
    #     with torch.no_grad():
    #         # Get values V(o) and logits from current policy (before update)
    #         old_logits, old_values = self.model(obs_tens, dis_tens, act_tens, pad_mask, mode='training')
            
    #         # Compute Value for Next State (for GAE)
    #         _, next_values = self.target_model(n_obs_tens, dis_tens, act_tens, pad_mask, mode='training')
            
    #         # Compute Log Probs of actions actually taken
    #         dist = D.Categorical(logits=old_logits)
    #         old_log_probs = dist.log_prob(act_tens)
    #         # Mask out padding log_probs so they don't affect mean
    #         old_log_probs[pad_mask] = 0.0

    #     # 5. COMPUTE GAE & RETURNS 
    #     # We average values across agents for the joint advantage: V_hat = mean(V(o^i))
    #     # But MAT trains per-agent advantages usually. 
    #     # Here we follow the paper's implementation logic of per-agent advantage decomposition.
        
    #     # Collapse agent dim for calculation or keep it? 
    #     # MAT paper suggests: A_joint = Sum(A_local). 
    #     # We compute advantages per agent (vectorized)
    #     advantages = self.compute_gae(
    #         rewards_tens.unsqueeze(1).repeat(1, max_N), # Broadcast global reward to all agents
    #         old_values.squeeze(-1), 
    #         dones_tens,
    #         next_values.squeeze(-1)
    #     )
        
    #     # Mask advantages for padding
    #     advantages[pad_mask] = 0.0
        
    #     # Returns = Advantage + Value (used for Encoder Loss target)
    #     returns = advantages + old_values.squeeze(-1)

    #     # Normalize Advantages (Standard PPO trick, not explicitly in paper but standard practice)
    #     adv_mean = advantages[~pad_mask].mean()
    #     adv_std = advantages[~pad_mask].std()
    #     advantages = (advantages - adv_mean) / (adv_std + 1e-8)

    #     # 6. PPO EPOCHS LOOP [cite: 442]
    #     total_loss_log = 0.0
        
    #     for _ in range(self.ppo_epochs):
    #         # Shuffle batch indices
    #         indices = torch.randperm(B)
            
    #         # Mini-batch Loop
    #         for start in range(0, B, self.mini_batch_size):
    #             end = start + self.mini_batch_size
    #             idx = indices[start:end]

    #             # Slice mini-batch data
    #             mb_obs = obs_tens[idx]
    #             mb_act = act_tens[idx]
    #             mb_pad = pad_mask[idx]
    #             mb_dis = dis_tens[idx] if use_disaster else None
    #             mb_adv = advantages[idx]
    #             mb_ret = returns[idx]
    #             mb_old_log_probs = old_log_probs[idx]

    #             # Random Agent Permutation (Data Augmentation) [cite: 17, 442]
    #             # We do this inside the mini-batch to ensure robust encoder
    #             for b_i in range(len(idx)):
    #                 actual_n = (~mb_pad[b_i]).sum()
    #                 perm = torch.randperm(actual_n)
    #                 mb_obs[b_i, :actual_n] = mb_obs[b_i, perm]
    #                 mb_act[b_i, :actual_n] = mb_act[b_i, perm]
    #                 mb_adv[b_i, :actual_n] = mb_adv[b_i, perm]
    #                 mb_ret[b_i, :actual_n] = mb_ret[b_i, perm]
    #                 mb_old_log_probs[b_i, :actual_n] = mb_old_log_probs[b_i, perm]
    #                 if mb_dis is not None:
    #                     mb_dis[b_i, :actual_n] = mb_dis[b_i, perm]

    #             # Forward Pass (New Policy)
    #             new_logits, new_values = self.model(mb_obs, mb_dis, mb_act, mb_pad, mode='training')
                
    #             # --- DECODER LOSS (PPO Clipping)  ---
    #             dist = D.Categorical(logits=new_logits)
    #             new_log_probs = dist.log_prob(mb_act)
    #             entropy = dist.entropy() # Exploration bonus

    #             # Ratio r_t
    #             ratio = torch.exp(new_log_probs - mb_old_log_probs)
                
    #             # Surrogate Objectives
    #             surr1 = ratio * mb_adv
    #             surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * mb_adv
                
    #             # Mask out padding from loss
    #             policy_loss = -torch.min(surr1, surr2)
    #             policy_loss[mb_pad] = 0.0
    #             policy_loss = policy_loss.sum() / (~mb_pad).sum() # Mean over valid agents

    #             # --- ENCODER LOSS (Value MSE)  ---
    #             # L_encoder = (Target_V - V_pred)^2
    #             value_loss = F.mse_loss(new_values.squeeze(-1)[~mb_pad], mb_ret[~mb_pad])

    #             # --- TOTAL LOSS ---
    #             # L = L_encoder + L_decoder - entropy
    #             entropy_loss = entropy[~mb_pad].mean()
    #             total_loss = policy_loss + value_loss - (self.entropy_coef * entropy_loss)

    #             # Backprop
    #             self.encoder_optimizer.zero_grad()
    #             self.decoder_optimizer.zero_grad()
    #             total_loss.backward()
                
    #             # track gradients for debugging 
    #             self.track_gradients(plotter)

    #             # freeze layers and retrain explosion layers if needed
    #             self.controller.check_and_adapt()

    #             # Clip Gradients [cite: 106]
    #             torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                
    #             self.encoder_optimizer.step()
    #             self.decoder_optimizer.step()
                
    #             total_loss_log += total_loss.item()

    #     # 7. SOFT UPDATE TARGET NETWORK
    #     self.soft_update(self.model, self.target_model)

    #     # 8. CLEAR BUFFER (On-Policy Requirement) 
    #     self.memory_buffer.clear()
    #     # logger.error(f"total loss: {total_loss_log}")

    #     avg_loss = total_loss_log / (self.ppo_epochs * (B // self.mini_batch_size + 1))
    #     if plotter:
    #         plotter.record_loss(self.group_id, avg_loss)

    #     return avg_loss

    def update(self, plotter=None):
        """
        MAT training: Encoder with Bellman loss, Decoder with PPO clip [^15][^16].
        Handles batched [B, max_N, ...] with padding.
        """
        if len(self.memory_buffer) < self.batch_size or not self.training:
            logger.warning(f"Not enough data to update for group {self.group_id}")
            return None
        
        batch = self.memory_buffer.sample(self.batch_size)
        
        # Data preparation (similar to original, but for MAT)
        all_obs = []  
        all_actions = []  
        all_next_obs = []  
        all_dones = []  
        all_disasters = []  
        batch_rewards = []  
        group_sizes = []  

        for joint_transition in batch:  
            vehicles_data = joint_transition['vehicles']  
            global_reward = joint_transition['global_reward']  

            if len(vehicles_data) == 0:  
                continue  

            batch_rewards.append(global_reward)  
            num_vehicles = len(vehicles_data)  
            group_sizes.append(num_vehicles)  

            group_obs = [np.squeeze(v[1]) for v in vehicles_data]  
            group_actions = [v[2] for v in vehicles_data]  
            group_next_obs = [np.squeeze(v[3]) for v in vehicles_data]  
            group_dones = [v[4] for v in vehicles_data]  
            group_disasters = [np.squeeze(v[5]) if v[5] is not None else None for v in vehicles_data]  

            all_obs.append(np.array(group_obs))  
            all_actions.append(np.array(group_actions))  
            all_next_obs.append(np.array(group_next_obs))  
            all_dones.append(np.array(group_dones))  
            all_disasters.append(np.array(group_disasters) if group_disasters[0] is not None else None)

        if not all_obs:  
            return 0.0  

        max_N = max(group_sizes)  
        B = len(all_obs)

        # Pad to [B, max_N, ...]
        obs_padded = torch.zeros(B, max_N, all_obs[0].shape[-1], dtype=torch.float32, device=self.device)
        actions_padded = torch.zeros(B, max_N, dtype=torch.long, device=self.device)  # Assuming discrete actions
        next_obs_padded = torch.zeros(B, max_N, all_next_obs[0].shape[-1], dtype=torch.float32, device=self.device)
        dones_padded = torch.zeros(B, max_N, dtype=torch.float32, device=self.device)
        disaster_padded = None if all(d is None for d in all_disasters) else torch.zeros(B, max_N, all_disasters[0].shape[-1], dtype=torch.float32, device=self.device)
        padding_mask = torch.ones(B, max_N, dtype=torch.bool, device=self.device)

        for i in range(B):  
            num_v = group_sizes[i]  
            obs_padded[i, :num_v] = torch.tensor(all_obs[i], device=self.device)
            actions_padded[i, :num_v] = torch.tensor(all_actions[i], device=self.device)
            next_obs_padded[i, :num_v] = torch.tensor(all_next_obs[i], device=self.device)
            dones_padded[i, :num_v] = torch.tensor(all_dones[i], device=self.device)
            if disaster_padded is not None and all_disasters[i] is not None:
                disaster_padded[i, :num_v] = torch.tensor(all_disasters[i], device=self.device)
            padding_mask[i, :num_v] = False

        rewards_tensor = torch.tensor(batch_rewards, dtype=torch.float32, device=self.device)

        # Random permutation of agent order per batch [^17]
        for b in range(B):
            perm = torch.randperm(group_sizes[b])
            obs_padded[b, :group_sizes[b]] = obs_padded[b, perm]
            actions_padded[b, :group_sizes[b]] = actions_padded[b, perm]
            next_obs_padded[b, :group_sizes[b]] = next_obs_padded[b, perm]
            dones_padded[b, :group_sizes[b]] = dones_padded[b, perm]
            if disaster_padded is not None:
                disaster_padded[b, :group_sizes[b]] = disaster_padded[b, perm]

        # Forward: Current
        action_logits, values = self.model(obs_padded, disaster_padded, actions_padded, padding_mask, mode='training')  # [B, max_N, action_dim], [B, max_N, 1]

        # Compute advantages (simple GAE placeholder; implement full GAE as needed [^16])
        with torch.no_grad():
            _, next_values = self.target_model(next_obs_padded, disaster_padded, actions_padded, padding_mask, mode='training')
        advantages = rewards_tensor.unsqueeze(1) + self.gamma * next_values.mean(dim=1) * (1 - dones_padded.mean(dim=1, keepdim=True)) - values.mean(dim=1)

        # Encoder loss: Bellman MSE [^15]
        encoder_loss = F.mse_loss(values.mean(dim=1).squeeze(1), rewards_tensor + self.gamma * next_values.mean(dim=1).squeeze(1).detach())

        # Decoder loss: PPO clip [^16]
        old_probs = F.softmax(action_logits, dim=-1).gather(2, actions_padded.unsqueeze(-1)).squeeze(-1)
        with torch.no_grad():
            old_action_logits, _ = self.target_model(obs_padded, disaster_padded, actions_padded, padding_mask, mode='training')
            old_probs_detach = F.softmax(old_action_logits, dim=-1).gather(2, actions_padded.unsqueeze(-1)).squeeze(-1)
        ratio = old_probs / (old_probs_detach + 1e-8)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages
        decoder_loss = -torch.min(surr1, surr2).mean()

        # Combine losses  
        total_loss = encoder_loss + decoder_loss  

        # Zero grads for both parameter groups  
        self.encoder_optimizer.zero_grad()  
        self.decoder_optimizer.zero_grad()  

        # Backprop once  
        total_loss.backward()  

        # Optional: clip grads for each component separately  
        torch.nn.utils.clip_grad_norm_(self.model.encoder.parameters(), max_norm=1.0)  
        torch.nn.utils.clip_grad_norm_(self.model.decoder.parameters(), max_norm=1.0) 

        # Step both optimizers  
        self.encoder_optimizer.step()  
        self.decoder_optimizer.step()  

        # Soft update target
        self.soft_update(self.model, self.target_model)

        total_loss = encoder_loss.item() + decoder_loss.item()
        if plotter:
            plotter.record_loss(self.group_id, total_loss)
    
        return total_loss

    def soft_update(self, source, target):
        # Soft update: target = beta * source + (1 - beta) * target
        with torch.no_grad():
            for t_param, s_param in zip(target.parameters(), source.parameters()):
                t_param.data.copy_(self.beta_soft * s_param.data + (1.0 - self.beta_soft) * t_param.data)
             
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

    def save_model(self, path=None):  
        if path is None:  
            os.makedirs(self.saved_folder, exist_ok=True)  
            path = f"{self.saved_folder}/agent_{self.group_id}_{time.strftime('%Y%m%d_%H%M%S')}.pt"  
        
        torch.save({
            'model_state_dict': self.model.state_dict(),    # weights
            'target_state_dict': self.target_model.state_dict(),     # weights
            'encoder_optimizer': self.encoder_optimizer.state_dict(), # internal memory for Adam opt
            'decoder_optimizer': self.decoder_optimizer.state_dict(), # internal memory for Adam opt
            'group_id': self.group_id,
            'total_steps': self.total_steps,
        }, path)
        logger.success(f"Model saved to {path}")
        return path
    
    def load_model(self, path, fine_tune_mode=False):
        if not os.path.exists(path):
            logger.warning(f"Model file not found: {path}")
            return False
            
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.target_model.load_state_dict(checkpoint['target_state_dict'])
        
        if not fine_tune_mode:
            try:
                self.encoder_optimizer.load_state_dict(checkpoint['encoder_optimizer'])
                self.decoder_optimizer.load_state_dict(checkpoint['decoder_optimizer'])
            except:
                logger.warning("Optimizer state mismatch; skipping.")

        self.total_steps = checkpoint.get('total_steps', 0)
        logger.info(f"Model loaded from {path}")

        if fine_tune_mode:
            self.setup_fine_tune(path)
        else:
            self.set_eval_mode()

        return True
    
    # def setup_fine_tune(self):
    #     logger.info("Fine-tune mode: freezing encoder.")
    #     self.set_train_mode()
    #     for param in self.model.encoder.parameters():
    #         param.requires_grad = False
    #     self.decoder_optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.CONFIG['learning_rate'])

    def setup_fine_tune(self, checkpoint_path=None):
        logger.info("Training mode: unfreezing both encoder and decoder.")
        self.set_train_mode()
        
        # 1. Unfreeze everything
        for param in self.model.parameters():
            param.requires_grad = True
            
        # 2. Create the optimizer
        # self.optimizer = optim.Adam(
        #     self.model.parameters(), 
        #     lr=self.CONFIG['learning_rate']
        # )
        # We create separate optimizers for encoder and decoder to allow for different learning rates or weight decay if needed
        self.encoder_optimizer = optim.Adam(self.model.encoder.parameters(), lr=self.CONFIG['learning_rate'],  
                                            weight_decay=float(self.CONFIG.get('weight_decay', 1e-4)))
        self.decoder_optimizer = optim.Adam(self.model.decoder.parameters(), lr=self.CONFIG['learning_rate'],  
                                            weight_decay=float(self.CONFIG.get('weight_decay', 1e-4)))
        # 3. (Optional) Load previous optimizer state if resuming
        if checkpoint_path:
            checkpoint = torch.load(checkpoint_path)
            
            # This restores Adam's internal memory
            self.encoder_optimizer.load_state_dict(checkpoint['encoder_optimizer'])
            self.decoder_optimizer.load_state_dict(checkpoint['decoder_optimizer'])
            logger.info("Loaded model and optimizer state from checkpoint.")
    
    def set_eval_mode(self):  
        self.training = False  
        self.model.eval()  
        self.target_model.eval()

    def set_train_mode(self):  
        self.training = True  
        self.model.train()  
        self.target_model.train()
