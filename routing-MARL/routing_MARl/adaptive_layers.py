import torch
import torch.nn as nn
from read_config import get_config

class AdaptiveLayerControllerAuto:
    def __init__(self, model, optimizer):
        """
        model: Your QNetwork instance.
        optimizer: The optimizer (e.g., Adam).
        """
        self.CONFIG = get_config()
        self.epsilon = float(self.CONFIG['adaptive_layer_controller']['epsilon'])
        self.threshold = float(self.CONFIG['adaptive_layer_controller']['threshold'])
        self.streak_target = int(self.CONFIG['adaptive_layer_controller']['streak_target'])
        
        # --- NEW: Cooldown Configuration ---
        # Number of steps to ignore a layer after re-initialization.
        # This prevents the "loop" where high gradients from fresh weights trigger another immediate reset.
        self.cooldown_period = int(self.CONFIG['adaptive_layer_controller']['cooldown'])
        
        self.model = model
        self.optimizer = optimizer

        # 1. Identify "Leaf" Layers
        self.layer_names = []
        for name, mod in model.named_modules():
            if len(list(mod.parameters(recurse=False))) > 0:
                self.layer_names.append(name)

        # Counters
        self.converge_streak = {name: 0 for name in self.layer_names}
        self.explode_streak  = {name: 0 for name in self.layer_names}
        
        # --- NEW: Cooldown Counter ---
        self.cooldown_counter = {name: 0 for name in self.layer_names}

    def check_and_adapt(self):
        
        for name, mod in self.model.named_modules():
            if name not in self.layer_names:
                continue

            # 1. Skip if currently in Cooldown
            if self.cooldown_counter[name] > 0:
                self.cooldown_counter[name] -= 1
                continue

            # 2. OPTIMIZATION: Check if layer is ALREADY frozen
            # If all parameters are frozen, we skip the expensive norm calculation.
            if not any(p.requires_grad for p in mod.parameters()):
                continue

            # 3. Calculate Norms (Only for active layers)
            # Use torch.no_grad() to ensure this calculation doesn't add to the graph
            with torch.no_grad():
                norms = [p.grad.norm().item() for p in mod.parameters() if p.grad is not None]
            
            if not norms:
                continue
            
            avg_norm = sum(norms) / len(norms)

            # --- CASE 1: CONVERGENCE (Freeze) ---
            if avg_norm < self.epsilon:
                self.converge_streak[name] += 1
                self.explode_streak[name] = 0 # Reset explosion streak
                
                if self.converge_streak[name] >= self.streak_target:
                    print(f"[FREEZE] {name} | Norm: {avg_norm:.2e} | Steps: {self.streak_target}")
                    for p in mod.parameters():
                        p.requires_grad = False
                    self.converge_streak[name] = 0

            # --- CASE 2: EXPLOSION (Re-init) ---
            elif avg_norm > self.threshold:
                self.explode_streak[name] += 1
                self.converge_streak[name] = 0 # Reset convergence streak
                
                if self.explode_streak[name] >= self.streak_target:
                    print(f"[REINIT] {name} | Norm: {avg_norm:.2f} | Steps: {self.streak_target}")
                    
                    # A. Re-initialize weights safely
                    self._reinit_layer(mod)
                    
                    # B. Clear Optimizer State
                    self._reset_optimizer_state(mod)
                    
                    # --- NEW: Trigger Cooldown ---
                    # Set the counter so we ignore this layer for the next N steps
                    self.cooldown_counter[name] = self.cooldown_period
                    print(f"   -> Layer {name} entering cooldown for {self.cooldown_period} steps.")

                    self.explode_streak[name] = 0 # Reset counter

            # --- CASE 3: NORMAL ---
            else:
                self.converge_streak[name] = 0
                self.explode_streak[name] = 0

    def _reinit_layer(self, mod):
        """
        Architecture-aware initialization.
        """
        if isinstance(mod, (nn.Linear, nn.Conv2d, nn.Conv1d)):
            nn.init.kaiming_normal_(mod.weight, mode='fan_out', nonlinearity='relu')
            if mod.bias is not None:
                nn.init.zeros_(mod.bias)
        
        elif isinstance(mod, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d)):
            nn.init.ones_(mod.weight)
            if mod.bias is not None:
                nn.init.zeros_(mod.bias)
        
        elif isinstance(mod, nn.MultiheadAttention):
            if mod.in_proj_weight is not None:
                nn.init.xavier_uniform_(mod.in_proj_weight)
            if mod.out_proj.weight is not None:
                nn.init.xavier_uniform_(mod.out_proj.weight)
            if mod.in_proj_bias is not None:
                nn.init.zeros_(mod.in_proj_bias)
            if mod.out_proj.bias is not None:
                nn.init.zeros_(mod.out_proj.bias)

        else:
            for p in mod.parameters():
                if p.dim() > 1:
                    nn.init.xavier_normal_(p)
                else:
                    nn.init.zeros_(p)

    def _reset_optimizer_state(self, mod):
        """
        Removes the re-initialized parameters from the optimizer's state dict.
        """
        for p in mod.parameters():
            if p in self.optimizer.state:
                del self.optimizer.state[p]