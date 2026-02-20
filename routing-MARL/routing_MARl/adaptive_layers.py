import torch
import torch.nn as nn
import numpy as np
from collections import deque
import datetime

from read_config import get_config

class AdaptiveLayerControllerAuto:
    def __init__(self, model, optimizer):
        self.CONFIG = get_config()
        self.epsilon = float(self.CONFIG['adaptive_layer_controller']['epsilon'])
        self.threshold = float(self.CONFIG['adaptive_layer_controller']['threshold'])
        
        # --- NEW: Window Configuration ---
        self.window_size = int(self.CONFIG['adaptive_layer_controller']['window_size'])
        self.model = model
        self.optimizer = optimizer

        # 1. Identify "Leaf" Layers
        self.layer_names = []
        for name, mod in model.named_modules():
            if len(list(mod.parameters(recurse=False))) > 0:
                self.layer_names.append(name)

        # 2. History Buffers: Stores the last 100 average norms per layer
        self.norm_history = {name: deque(maxlen=self.window_size) for name in self.layer_names}
        
        # 3. Cooldown logic
        self.cooldown_period = int(self.CONFIG['adaptive_layer_controller']['cooldown'])
        self.cooldown_counter = {name: 0 for name in self.layer_names}

    def check_and_adapt(self):
        for name, mod in self.model.named_modules():
            if name not in self.layer_names:
                continue

            # Skip if in Cooldown
            if self.cooldown_counter[name] > 0:
                self.cooldown_counter[name] -= 1
                continue

            # Skip if frozen
            if not any(p.requires_grad for p in mod.parameters()):
                continue

            # Step 1: Calculate the AVERAGE of norms for the current step
            with torch.no_grad():
                # Filter for parameters that actually have gradients
                active_grads = [p.grad.norm() for p in mod.parameters() if p.grad is not None]
                
                if not active_grads:
                    continue
                
                # Calculate mean norm across all parameter tensors in this layer (Weight + Bias)
                current_avg_norm = torch.stack(active_grads).mean().item()

            # Step 2: Update the rolling window history
            self.norm_history[name].append(current_avg_norm)

            # Step 3: Wait until the window is full to avoid premature triggers
            if len(self.norm_history[name]) < self.window_size:
                continue

            # Step 4: Calculate the MEDIAN of the 100-step window
            # This is exactly what your plotting code does
            window_median = np.median(self.norm_history[name])

            # --- CASE 1: EXPLOSION ---
            if window_median > self.threshold:
                print(f"[REINIT] Layer: {name} | Window Median Norm: {window_median:.2f} > Threshold: {self.threshold}")

                # Write to the log file
                with open("explosion_log.txt", "a") as f:
                    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    f.write(f"[{timestamp}] Layer: {name}, Explosion,Window Median Norm: {window_median:.2f}\n")
                
                self._reinit_layer(mod)
                self._reset_optimizer_state(mod)
                
                # Clear history and start cooldown so we don't re-init again immediately
                self.norm_history[name].clear()
                self.cooldown_counter[name] = self.cooldown_period

            # --- CASE 2: CONVERGENCE ---
            elif window_median < self.epsilon:
                print(f"[FREEZE] {name} | Window Median Norm: {window_median:.2e}")
                for p in mod.parameters():
                    p.requires_grad = False
                self.norm_history[name].clear()

    
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