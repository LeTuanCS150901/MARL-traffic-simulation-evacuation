from pyexpat import model
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from read_config import get_config, read_config
from logs.logger import logger

class PPONetwork(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=256, nhead=4,
                 num_layers=2, dropout_p=0.1):
        super().__init__()

        self.hidden_dim = hidden_dim
        # --- Input projection ---
        self.input_proj = nn.Linear(obs_dim, hidden_dim)

        self.CONFIG = get_config()
        self.feature_dim_disaster = self.CONFIG['disaster']['fire']['feature_dim']
        
        # --- Disaster projection ---
        self.disaster_encoder = nn.Linear(self.feature_dim_disaster, hidden_dim)

        # --- Transformer encoder for inter-vehicle attention ---
        # Keeps your exact transformer setup
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout_p,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # --- Shared Body ---
        layer_sizes = [hidden_dim * 3, 512, 1024, 1024, 512, 256]
        self.shared_layers = nn.ModuleList([
            nn.Linear(layer_sizes[i], layer_sizes[i + 1]) for i in range(len(layer_sizes) - 1)
        ])
        self.bn_layers = nn.ModuleList([
            nn.LayerNorm(layer_sizes[i + 1]) for i in range(len(layer_sizes) - 1)
        ])
        self.drop_layers = nn.ModuleList([
            nn.Dropout(p=dropout_p) for _ in range(len(layer_sizes) - 1)
        ])

        # --- Actor Head (Action Probabilities) ---
       # --- Actor Head (Deepened) ---
        # Takes the shared 512 features -> processes in 256 -> outputs actions
        self.actor_head = nn.Sequential(
            nn.Linear(layer_sizes[-1], 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        )

        # --- Critic Head (Deepened) ---
        # Takes the shared 512 features -> processes in 256 -> outputs scalar value
        self.critic_head = nn.Sequential(
            nn.Linear(layer_sizes[-1], 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, obs, disaster=None):
        """
        Forward pass returning both Actor logits and Critic values.
        obs: [N, obs_dim]
        disaster: [N, dis_dim] or None
        """
        # Ensure obs is 2D
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)  # [1, obs_dim]
        N = obs.shape[0]
        
        obs_proj = self.input_proj(obs)  # [N, hidden_dim]

        if disaster is None:
            # Create placeholder disaster embedding if None
            # add gpu
            disaster_embed = torch.full((N, self.hidden_dim), -1.0)
        else:
            disaster_proj = self.disaster_encoder(disaster) # [N, M, hidden]
            disaster_embed = disaster_proj.mean(dim=1)      # [N, hidden_dim]

        # --- Transformer encoding ---
        # Transformer expects [Batch, Seq, Dim]. We treat the group N as the Sequence.
        x_in = obs_proj.unsqueeze(0)  # [1, N, hidden]
        x_encoded = self.transformer(x_in) 
        x_encoded = x_encoded.squeeze(0)   # [N, hidden]

        # --- Group embedding ---
        group_embed = x_encoded.mean(dim=0)       # [hidden_dim]
        group_embed = group_embed.repeat(N, 1)    # [N, hidden_dim]

        # --- Concatenate ---
        concat_input = torch.cat([obs_proj, group_embed, disaster_embed], dim=1)

        # --- Shared MLP Body ---
        h = concat_input
        for linear, norm, drop in zip(self.shared_layers, self.bn_layers, self.drop_layers):
            h = drop(torch.relu(norm(linear(h))))

        # --- Heads ---
        action_logits = self.actor_head(h)  # [N, action_dim]
        state_values = self.critic_head(h)  # [N, 1]

        return action_logits, state_values

    def evaluate(self, obs, disaster, action, feasible_masks):
        """
        Helper for PPO Update. 
        Calculates log_probs and entropy for the actions taken during rollout.
        """
        action_logits, state_values = self.forward(obs, disaster)
        
        # Mask infeasible actions (set to very small number)
        action_logits[~feasible_masks] = float('-inf')

        dist = Categorical(logits=action_logits)
        
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        
        return action_logprobs, state_values, dist_entropy
    

if __name__ == "__main__":

    # Set the config file
    CONFIG = read_config(config_filename="manhattan2")
    # test the size of the model
    obs_dim = 1578
    action_dim = 1578
    model = PPONetwork(obs_dim, action_dim)
    print(model)

    # Only trainable parameters (useful for fine-tuning)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable Parameters: {trainable_params:,}")