import torch
import torch.nn as nn
import torch.nn.functional as F

from read_config import get_config
from logs.logger import logger  # import our custom logger

class QNetwork(nn.Module):  
    def __init__(self, obs_dim, action_dim, dropout_p=0.1):  
        super().__init__()  

        self.CONFIG = get_config()  

        # LSTM encoder params
        self.num_heads = self.CONFIG['model']['encoder']['num_heads']
        self.head_dim = self.CONFIG['model']['encoder']['head_dim']
        self.hidden_dim = self.num_heads * self.head_dim  
        self.num_encoder_layers = self.CONFIG['model']['encoder']['num_encoder_layers']
        
        # Q network layers
        self.hidden_sizes = self.CONFIG['model']['q_network']['hidden_sizes']

        # --- Input projection ---  
        self.input_proj = nn.Linear(obs_dim, self.hidden_dim)  
    
        try:  
            self.feature_dim_disaster = self.CONFIG['disaster']['fire']['feature_dim']  
        except Exception as e:  
            logger.warning("No disaster feature dimension found in config, set to 1")  
            self.feature_dim_disaster = 1  
        # --- Disaster projection ---  
        self.disaster_encoder = nn.Linear(self.feature_dim_disaster, self.hidden_dim)  

        # --- LSTM encoder for inter-vehicle sequence modeling ---  
        self.lstm = nn.LSTM(
            input_size=self.hidden_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_encoder_layers,
            batch_first=True,
            dropout=dropout_p if self.num_encoder_layers > 1 else 0.0,
            bidirectional=False
        )

        # --- Feedforward layers per vehicle ---  
        layer_sizes = [self.hidden_dim * 3] + self.hidden_sizes  # Input: concat of encoded, group_embed, disaster_embed 
        self.hidden_layers = nn.ModuleList([  
            nn.Linear(layer_sizes[i], layer_sizes[i + 1]) for i in range(len(layer_sizes) - 1)  
        ])  
        self.bn_layers = nn.ModuleList([  
            nn.LayerNorm(layer_sizes[i + 1]) for i in range(len(layer_sizes) - 1)  
        ])  
        self.drop_layers = nn.ModuleList([  
            nn.Dropout(p=dropout_p) if 0 < i < len(layer_sizes) - 2 else nn.Identity()  
            for i in range(len(layer_sizes) - 1)  
        ])  

        # --- Output layer ---  
        self.output_layer = nn.Linear(layer_sizes[-1], action_dim)

    def forward(self, obs, disaster=None, padding_mask=None):  
        """  
        Forward pass for QNetwork.  

        - obs: [B, max_N, obs_dim] (batched groups, padded)  
          OR [N, obs_dim] (single group, inference) -> auto-unsqueeze to [1, N, obs_dim]  
          OR [1, obs_dim] (single vehicle) -> auto-unsqueeze to [1, 1, obs_dim]  
        - disaster: Same shape as obs, but [B, max_N, dis_dim] (or None)  
        - padding_mask: [B, max_N] (True for padded positions, optional for inference)  
        
        Returns: [N_total, action_dim] (flattened across batches for easy gathering)  
        """  
        # Handle inference cases (single group or vehicle)  
        if obs.dim() == 2:  # [N, obs_dim] -> [1, N, obs_dim]  
            obs = obs.unsqueeze(0)  
            if disaster is not None:  
                disaster = disaster.unsqueeze(0)  
            if padding_mask is not None:  
                padding_mask = padding_mask.unsqueeze(0)  
        elif obs.dim() == 1:  # [obs_dim] -> [1, 1, obs_dim]  
            obs = obs.unsqueeze(0).unsqueeze(0)  
            if disaster is not None:  
                disaster = disaster.unsqueeze(0).unsqueeze(0)  
            if padding_mask is not None:  
                padding_mask = padding_mask.unsqueeze(0).unsqueeze(0)  

        B, max_N, obs_dim = obs.shape  
        device = self.input_proj.weight.device  
        obs = obs.to(device)  
        if disaster is not None:  
            disaster = disaster.to(device)  
        if padding_mask is not None:  
            padding_mask = padding_mask.to(device)  

        # Project obs: [B, max_N, hidden_dim]  
        obs_proj = self.input_proj(obs)  

        # --- LSTM encoding with padding mask ---  
        if padding_mask is not None:
            # Pack padded sequences for efficient LSTM processing
            lengths = (~padding_mask).sum(dim=1).cpu()  # [B], number of valid vehicles per batch
            lengths = lengths.clamp(min=1)  # Ensure at least 1 to avoid errors
            packed = nn.utils.rnn.pack_padded_sequence(
                obs_proj, lengths, batch_first=True, enforce_sorted=False
            )
            lstm_out_packed, _ = self.lstm(packed)
            x_encoded, _ = nn.utils.rnn.pad_packed_sequence(
                lstm_out_packed, batch_first=True, total_length=max_N
            )  # [B, max_N, hidden_dim]
        else:
            x_encoded, _ = self.lstm(obs_proj)  # [B, max_N, hidden_dim]

        # --- Group embedding (mean over valid vehicles, ignoring padding) ---  
        if padding_mask is not None:  
            # Mask padded positions by setting to 0 before mean  
            masked_encoded = x_encoded.masked_fill(padding_mask.unsqueeze(-1), 0.0)  
            group_embed = masked_encoded.sum(dim=1) / (max_N - padding_mask.sum(dim=1, keepdim=True) + 1e-8)  # [B, hidden_dim]  
        else:  
            group_embed = x_encoded.mean(dim=1)  # [B, hidden_dim]  
        group_embed = group_embed.unsqueeze(1).repeat(1, max_N, 1)  # [B, max_N, hidden_dim]  

        # --- Disaster embedding ---  
        if disaster is None:  
            disaster_embed = torch.full((B, max_N, self.hidden_dim), -1.0, device=device)  
        else:  
            disaster_proj = self.disaster_encoder(disaster)  # [B, max_N, hidden_dim]  
            # Assuming mean over disaster features if multi-dimensional; adjust if needed  
            disaster_embed = disaster_proj  # Or .mean(dim=2) if disaster is [B, max_N, dis_dim] >1  

        # --- Concatenate x_encoded + group_embed + disaster_embed ---  
        concat_input = torch.cat([x_encoded, group_embed, disaster_embed], dim=2)  # [B, max_N, 3*hidden_dim]  
        # Optional: true residual connection  
        # concat_input = concat_input + obs_proj.repeat(1, 1, 3)  # Match dims if needed  

        # --- MLP (process per vehicle) ---  
        h = concat_input.view(B * max_N, -1)  # Flatten to [B*max_N, 3*hidden_dim] for efficient MLP  
        for linear, norm, drop in zip(self.hidden_layers, self.bn_layers, self.drop_layers):  
            # change the order here to LayerNorm -> ReLU -> Dropout

            h = norm(torch.relu(linear(h)))
            # h = drop(torch.relu(norm(linear(h))))  
        q_values = self.output_layer(h)  # [B*max_N, action_dim]  

        # Return flattened [N_total, action_dim] (including padded, but you'll ignore them via gathering)  
        return q_values