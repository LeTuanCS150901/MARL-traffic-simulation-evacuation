import torch
import torch.nn as nn
import torch.nn.functional as F

from read_config import get_config
from logs.logger import logger  # import our custom logger


class QNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=256, nhead=4,
                 num_layers=2, dropout_p=0.1):
        super().__init__()

        self.hidden_dim = hidden_dim
        # --- Input projection ---
        self.input_proj = nn.Linear(obs_dim, hidden_dim)

        self.CONFIG = get_config()
        try:
            self.feature_dim_disaster = self.CONFIG['disaster']['fire']['feature_dim']
        except Exception as e:
            logger.warning("No disaster feature dimension found in config, set to 1")
            self.feature_dim_disaster = 1
        # --- Disaster projection ---
        self.disaster_encoder = nn.Linear(self.feature_dim_disaster, hidden_dim)

        # --- Transformer encoder for inter-vehicle attention ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout_p,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # --- Feedforward layers per vehicle ---
        layer_sizes = [hidden_dim * 3, 512, 1024, 1024, 512, 512, 256]
        self.hidden_layers = nn.ModuleList([
            nn.Linear(layer_sizes[i], layer_sizes[i + 1]) for i in range(len(layer_sizes) - 1)
        ])
        self.bn_layers = nn.ModuleList([
            nn.LayerNorm(layer_sizes[i + 1]) for i in range(len(layer_sizes) - 1)
        ])

        # self.bn_layers = nn.ModuleList([
        #     # layer_sizes[i+1] is the number of features output by the linear layer
        #     nn.BatchNorm1d(layer_sizes[i + 1]) 
        #     for i in range(len(layer_sizes) - 1)
        # ])

        
        self.drop_layers = nn.ModuleList([
            nn.Dropout(p=dropout_p) if 0 < i < len(layer_sizes) - 2 else nn.Identity()
            for i in range(len(layer_sizes) - 1)
        ])

        # --- Output layer ---
        self.output_layer = nn.Linear(layer_sizes[-1], action_dim)

    def forward(self, obs, disaster):
        """
        Forward pass for QNetwork.

        obs: [N, obs_dim] or [1, obs_dim]  -> N vehicles in a single timestep
        disaster: [N, dis_dim] or [1, dis_dim] -> per vehicle or shared disaster feature
        returns: [N, action_dim]
        """
        # Ensure obs is 2D and on the correct device
        # device = self.input_proj.weight.device
        if obs.dim() == 1:
            obs = obs.unsqueeze(0) # if (obs_dim,) -> (1, obs_dim)
        # obs = obs.to(device)
        N = obs.shape[0]

        obs_proj = self.input_proj(obs)  # [N, hidden_dim]

        if disaster is None:
            disaster_embed = torch.full((N, self.hidden_dim), -1.0, device=obs.device)
        else:
            disaster_proj = self.disaster_encoder(disaster) # [N, M, hidden_dim]
            disaster_embed = disaster_proj.mean(dim=1) # [N, hidden_dim]

        # --- Transformer encoding ---
        x_encoded = self.transformer(obs_proj)  # [N, hidden_dim]

        # --- Group embedding ---
        # take the mean to reduce dependency on orders of observations
        group_embed = x_encoded.mean(dim=0)       # [hidden_dim]
        group_embed = group_embed.repeat(N, 1)    # [N, hidden_dim]

        # --- Concatenate obs_proj + group_embed + disaster_feat ---
        # residual connection, reduce vanishing gradient
        concat_input = torch.cat([obs_proj, group_embed, disaster_embed], dim=1)  # [N, 3*hidden_dim]

        # --- MLP ---
        h = concat_input
        for linear, norm, drop in zip(self.hidden_layers, self.bn_layers, self.drop_layers):
            h = drop(torch.relu(norm(linear(h))))

        q_values = self.output_layer(h)  # [N, action_dim]
        return q_values



    # def forward(self, obs, disaster, batch_ids=None):
        """
        The use of batch to improve training time efficiency and random sample from batches =>
        help breaking correlation between each b in Batches.
        Forward pass handling both Batch Training (Multiple Scenarios) and Inference (Single Obs). 
        THis is crucial step because each scenario (each b in Batches) can have different number of agents N.
        and difference scenarios are unique and independent, so we cannot mix their agents together.
        We use padding + masking to handle this.


        Args:
            obs: [Total_N, obs_dim] (Training) OR [obs_dim] (Inference)
            disaster: [Total_N, dis_dim] OR [dis_dim] OR None
            batch_ids: [Total_N] mapping agents to scenarios. None during inference.
        """
        # --- 1. Input Normalization (Handle Single Observation) ---
        # If input is 1D (e.g., just one vehicle during inference), make it 2D
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)  # [obs_dim] -> [1, obs_dim]
        
        Total_N = obs.shape[0]
        device = obs.device

        # If batch_ids is None (Inference), assume all inputs belong to the same single scenario (Batch 0)
        if batch_ids is None:
            batch_ids = torch.zeros(Total_N, dtype=torch.long, device=device)

        # --- 2. Initial Projection ---
        obs_proj = self.input_proj(obs)  # [Total_N, hidden_dim]

        # --- 3. Transformer with Padding (Dynamic N support) ---
        # We need to reshape/pad to [B, Max_N, H] for the Transformer
        counts = torch.bincount(batch_ids)
        max_n = counts.max().item()
        num_scenarios = counts.size(0)
        H = obs_proj.shape[-1]

        # Prepare padded buffer
        padded_obs = torch.zeros((num_scenarios, max_n, H), device=device)
        # Mask: True = Ignore (Padding), False = Attend (Real Data)
        padding_mask = torch.ones((num_scenarios, max_n), dtype=torch.bool, device=device)

        # "Scatter" data into padded tensor
        # Create indices: e.g., if counts=[2, 1], local_indices=[0, 1, 0]
        local_indices = torch.cat([torch.arange(n, device=device) for n in counts])
        
        padded_obs[batch_ids, local_indices] = obs_proj
        padding_mask[batch_ids, local_indices] = False 

        # Pass through Transformer
        # If N=1 (Inference), this is just [1, 1, H] with no masking needed
        x_encoded_padded = self.transformer(
            padded_obs, 
            src_key_padding_mask=padding_mask
        )

        # "Gather" back to flattened shape [Total_N, H]
        x_encoded = x_encoded_padded[batch_ids, local_indices]

        # --- 4. Group Embedding (Scenario-Aware Mean) ---
        # Calculate sum of features per scenario
        group_sum = torch.zeros((num_scenarios, H), device=device)
        group_sum.index_add_(0, batch_ids, x_encoded)
        
        # Calculate mean (avoid div by zero if a scenario somehow has 0 agents, though unlikely)
        curr_counts = counts.unsqueeze(1).float()
        curr_counts[curr_counts == 0] = 1.0 # Safety
        
        group_mean = group_sum / curr_counts    # [B, H]
        group_embed = group_mean[batch_ids]     # [Total_N, H] Broadcast back to agents

        # --- 5. Disaster Embedding ---
        if disaster is None:
            disaster_embed = torch.full((Total_N, self.hidden_dim), -1.0, device=device)
        else:
            # Handle Single Observation for disaster too
            if disaster.dim() == 1:
                disaster = disaster.unsqueeze(0)
            
            disaster_proj = self.disaster_encoder(disaster)
            # If encoder output is 3D [N, M, H], take mean. If 2D [N, H], keep as is.
            if disaster_proj.dim() == 3:
                disaster_embed = disaster_proj.mean(dim=1)
            else:
                disaster_embed = disaster_proj

        # --- 6. Concatenate & MLP ---
        # concat_input = torch.cat([obs_proj, group_embed, disaster_embed], dim=1)
        
        # h = concat_input
        # for linear, norm, drop in zip(self.hidden_layers, self.bn_layers, self.drop_layers):
        #     h = drop(torch.relu(norm(linear(h))))

        # q_values = self.output_layer(h) # [Total_N, action_dim]

        # --- 6. Concatenate & MLP with batch norm ---
        concat_input = torch.cat([obs_proj, group_embed, disaster_embed], dim=1)
                                 
        h = concat_input
        for linear, norm, drop in zip(self.hidden_layers, self.bn_layers, self.drop_layers):
            h = linear(h)
            
            # If batch size is 1, force this specific layer to eval mode 
            # to use running stats, then switch back if we are training.
            was_training = norm.training
            if h.shape[0] == 1:
                norm.eval()
            
            h = norm(h)
            
            # Restore original state
            if was_training:
                norm.train()
            
            h = torch.relu(h)
            h = drop(h)

        q_values = self.output_layer(h) # [Total_N, action_dim]
        
        return q_values