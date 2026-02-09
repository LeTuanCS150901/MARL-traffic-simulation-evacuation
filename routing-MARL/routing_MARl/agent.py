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
        self.lr = float(self.CONFIG['learning_rate'])
        self.gamma = float(self.CONFIG['gamma'])
        self.max_grad_norm = float(self.CONFIG['max_grad_norm'])
        self.weight_decay = float(self.CONFIG['weight_decay'])
        self.eps_clip = float(self.CONFIG['model']['PPO']['eps_clip'])           # Clipping for PPO
        self.K_epochs = int(self.CONFIG['model']['PPO']['K_epochs'])             # Number of update epochs per batch
        self.batch_size =  int(self.CONFIG['model']['PPO']['batch_size'])    # Timesteps to collect for memory storage
        self.mini_batch_size = int(self.CONFIG['model']['PPO']['mini_batch_size']) # put high, depends on gpu memory. High mini batch -> reduce time cpu-gpu transfer
        self.entropy_coef = float(self.CONFIG['model']['PPO']['entropy_coef'])      # Explore/Exploit balance. High entropy => more exploration
        self.critic_coef = float(self.CONFIG['model']['PPO']['critic_coef'])
        self.gae_lamda = self.CONFIG['model']['PPO']['gae_lamda']
        
        # --- Network ---
        # add gpu
        self.ppo_network = PPONetwork(obs_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.ppo_network.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        # --- Memory ---
        # Replaces ExperienceReplayBuffer. Stores rollout data.
        self.memory = [] 

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


    # ------------------------------------------------------------------  
    # HELPER – pad a list of variable-length tensors into a batch  
    # ------------------------------------------------------------------  
    @staticmethod  
    def _pad_and_batch(tensor_list, pad_value=0.0):  
        """  
        Pad variable-length tensors into a batch.  

        Handles both:  
          - 1-D inputs [N_i]           → output [B, N_max]  
          - 2-D inputs [N_i, feat_dim] → output [B, N_max, feat_dim]  
        """  
        sizes = [t.shape[0] for t in tensor_list]  
        N_max = max(sizes)  
        feat_dims = tensor_list[0].shape[1:]     # () for 1-D, (feat,) for 2-D  
        B = len(tensor_list)  

        batched = torch.full(  
            (B, N_max, *feat_dims), pad_value,  
            dtype=tensor_list[0].dtype,  
            device=tensor_list[0].device,  
        )  
        mask = torch.ones(B, N_max, dtype=torch.bool,  
                          device=tensor_list[0].device)  

        for i, t in enumerate(tensor_list):  
            n = t.shape[0]  
            batched[i, :n] = t  
            mask[i, :n] = False          # False = real token  

        return batched, mask

    def update(self, plotter=None):  
        """  
        PPO Update – mini-batches are groups of *steps* (not flattened  
        vehicles), so the transformer never mixes unrelated scenarios.  
        """  
        if len(self.memory) < self.batch_size or not self.training:  
            return None  

        num_steps = len(self.memory)  

        # ============================================================  
        # 1.  Compute per-step values for GAE  
        # ============================================================  
        rewards = []  
        dones   = []  
        values  = []  

        for step in self.memory:  
            rewards.append(step['reward'])  
            dones.append(step['is_terminal'])  
            step_vals = torch.stack([v['val'] for v in step['vehicles']]).squeeze()  
            values.append(step_vals)  

        # ============================================================  
        # 2.  GAE  
        # ============================================================  
        advantages = []  
        returns    = []  
        gae        = 0.0  
        next_val   = 0.0  

        for i in reversed(range(num_steps)):  
            val  = values[i].mean().item()  
            done = dones[i]  

            delta = rewards[i] + self.gamma * next_val * (1 - done) - val  
            gae   = delta + self.gamma * self.gae_lamda * (1 - done) * gae  

            advantages.insert(0, gae)  
            returns.insert(0, gae + val)  
            next_val = val  

        adv_tensor = torch.tensor(advantages, dtype=torch.float32, device=self.device)  
        adv_tensor = (adv_tensor - adv_tensor.mean()) / (adv_tensor.std() + 1e-5)  
        ret_tensor = torch.tensor(returns,    dtype=torch.float32, device=self.device)  

        # ============================================================  
        # 3.  Build per-step tensors (keep 2-D per step, NOT flattened)  
        #     Each element is for one step and has shape [N_i, ...].  
        # ============================================================  
          # ============================================================  
        # 3.  Build per-step tensors  
        # ============================================================  
        step_obs       = []  
        step_actions   = []  
        step_log_probs = []  
        step_masks     = []  
        step_disasters = []  
        step_adv       = []  
        step_ret       = []  

        has_disaster_data = False  

        for i, step in enumerate(self.memory):  
            vehicles     = step['vehicles']  
            num_vehicles = len(vehicles)  

            step_obs.append(  
                torch.stack([v['obs'] for v in vehicles]).squeeze(1).to(self.device)  
            )                                                          # [N_i, obs_dim]  

            # ---- FIX: use .view(num_vehicles) instead of .squeeze(-1) ----  
            step_actions.append(  
                torch.stack([v['action'] for v in vehicles]).view(num_vehicles).to(self.device)  
            )                                                          # [N_i]  

            step_log_probs.append(  
                torch.stack([v['log_prob'] for v in vehicles]).view(num_vehicles).to(self.device)  
            )                                                          # [N_i]  

            step_masks.append(  
                torch.stack([v['mask'] for v in vehicles]).to(self.device)  
            )                                                          # [N_i, action_dim]  

            step_adv.append(adv_tensor[i].expand(num_vehicles))  
            step_ret.append(ret_tensor[i].expand(num_vehicles))  

            dis = vehicles[0]['disaster']  
            if dis is not None:  
                has_disaster_data = True  
                step_disasters.append(dis.to(self.device))  
            else:  
                step_disasters.append(None)  

        # ============================================================  
        # 4.  PPO optimisation loop – mini-batches of STEPS  
        # ============================================================  
        total_loss_val = 0.0  

        for _ in range(self.K_epochs):  
            epoch_loss  = 0.0  
            num_batches = 0  

            # shuffle step indices  
            perm = torch.randperm(num_steps)  

            for mb_start in range(0, num_steps, self.mini_batch_size):  
                mb_end = min(mb_start + self.mini_batch_size, num_steps)  
                mb_idx = perm[mb_start:mb_end]  
                num_batches += 1  

                # --------------------------------------------------  
                # Collect the steps belonging to this mini-batch  
                # --------------------------------------------------  
                mb_obs_list   = [step_obs[j]       for j in mb_idx]  
                mb_act_list   = [step_actions[j]   for j in mb_idx]  
                mb_oldlp_list = [step_log_probs[j] for j in mb_idx]  
                mb_mask_list  = [step_masks[j]     for j in mb_idx]  
                mb_adv_list   = [step_adv[j]       for j in mb_idx]  
                mb_ret_list   = [step_ret[j]       for j in mb_idx]  

                # --------------------------------------------------  
                # Pad into [B, N_max, ...] tensors  
                # --------------------------------------------------  

                # obs: [B, N_max, obs_dim]   pad_mask: [B, N_max]  
                mb_obs_pad, pad_mask = self._pad_and_batch(mb_obs_list)  

                # actions: 1-D inputs [N_i] → [B, N_max]  
                mb_act_pad, _ = self._pad_and_batch(mb_act_list)  
                mb_act_pad = mb_act_pad.long()  

                # old log probs: 1-D inputs [N_i] → [B, N_max]  
                mb_oldlp_pad, _ = self._pad_and_batch(mb_oldlp_list)  

                # feasible masks: [B, N_max, action_dim]  
                mb_mask_pad, _ = self._pad_and_batch(  
                    mb_mask_list, pad_value=1.0  
                )  
                mb_mask_pad = mb_mask_pad.bool()  

                # advantages: 1-D [N_i] → [B, N_max]  
                mb_adv_pad, _ = self._pad_and_batch(mb_adv_list)  

                # returns: 1-D [N_i] → [B, N_max]  
                mb_ret_pad, _ = self._pad_and_batch(mb_ret_list)

                # --------------------------------------------------  
                # Pad disasters: [B, D_max, disaster_dim] or None  
                # --------------------------------------------------  
                mb_dis_pad = None  
                if has_disaster_data:  
                    mb_dis_list = [step_disasters[j] for j in mb_idx]  
                    if not all(d is None for d in mb_dis_list):  
                        # Replace None entries with a zero-sized placeholder  
                        # so _pad_and_batch can handle them uniformly  
                        dis_dim = None  
                        for d in mb_dis_list:  
                            if d is not None:  
                                dis_dim = d.shape[-1]  
                                break  
                        safe_dis_list = [  
                            d if d is not None  
                            else torch.zeros(1, dis_dim, device=self.device)  
                            for d in mb_dis_list  
                        ]  
                        mb_dis_pad, _ = self._pad_and_batch(safe_dis_list)  

                # --------------------------------------------------  
                # Forward pass through the network  
                # --------------------------------------------------  
                log_probs, state_values, dist_entropy = self.ppo_network.evaluate(  
                    mb_obs_pad,  
                    mb_dis_pad,  
                    mb_act_pad,  
                    mb_mask_pad,  
                    src_key_padding_mask=pad_mask,  
                )  
                # log_probs:    [B, N_max]  
                # state_values: [B, N_max]  
                # dist_entropy: [B, N_max]  

                # --------------------------------------------------  
                # Build a flat real-vehicle mask to ignore padding  
                # --------------------------------------------------  
                real_mask = ~pad_mask                       # [B, N_max], True = real  

                # Flatten only the REAL vehicle positions  
                flat_log_probs    = log_probs[real_mask]        # [total_real]  
                flat_old_lp       = mb_oldlp_pad[real_mask]     # [total_real]  
                flat_values       = state_values[real_mask]     # [total_real]  
                flat_adv          = mb_adv_pad[real_mask]       # [total_real]  
                flat_ret          = mb_ret_pad[real_mask]       # [total_real]  
                flat_entropy      = dist_entropy[real_mask]     # [total_real]  

                # --------------------------------------------------  
                # PPO losses (computed only over real vehicles)  
                # --------------------------------------------------  
                # Ratios  
                ratios = torch.exp(flat_log_probs - flat_old_lp.detach())  

                # Surrogate loss (actor)  
                surr1 = ratios * flat_adv  
                surr2 = torch.clamp(  
                    ratios, 1 - self.eps_clip, 1 + self.eps_clip  
                ) * flat_adv  
                loss_actor = -torch.min(surr1, surr2).mean()  

                # Critic loss  
                loss_critic = nn.SmoothL1Loss()(flat_values, flat_ret)  

                # Entropy loss  
                loss_entropy = -flat_entropy.mean()  

                # Total loss  
                loss = (  
                    loss_actor  
                    + self.critic_coef  * loss_critic  
                    + self.entropy_coef * loss_entropy  
                )  

                # --------------------------------------------------  
                # Backprop  
                # --------------------------------------------------  
                self.optimizer.zero_grad()  
                loss.backward()  
                torch.nn.utils.clip_grad_norm_(  
                    self.ppo_network.parameters(), max_norm=self.max_grad_norm  
                )  
                self.optimizer.step()  

                epoch_loss += loss.item()  

            # end of mini-batch loop  
            if num_batches > 0:  
                total_loss_val += (epoch_loss / num_batches)  

        # ============================================================  
        # 5.  Post-update bookkeeping  
        # ============================================================  
        avg_loss = total_loss_val / self.K_epochs  

        # Clear memory  
        self.memory.clear()  

        if plotter:  
            plotter.record_loss(self.group_id, avg_loss)  

        return avg_loss

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