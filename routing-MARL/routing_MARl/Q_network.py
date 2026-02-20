import torch
import torch.nn as nn
import torch.nn.functional as F

from read_config import get_config
from logs.logger import logger

class QNetwork(nn.Module):  
    def __init__(self, obs_dim, action_dim, dropout_p=0.1):  
        super().__init__()  

        self.CONFIG = get_config()  

        # Transformer encoder params
        self.num_heads = self.CONFIG['model']['encoder']['num_heads']
        self.head_dim = self.CONFIG['model']['encoder']['head_dim']
        self.hidden_dim = self.num_heads * self.head_dim  
        self.num_encoder_layers = self.CONFIG['model']['encoder']['num_encoder_layers']
        
        # Q network layers
        self.hidden_sizes = self.CONFIG['model']['q_network']['hidden_sizes']

        # --- NEW: LSTM for History ---
        # # The input to the LSTM is obs_dim (the size of a single state S)
        self.lstm = nn.LSTM(input_size=obs_dim, hidden_size=self.hidden_dim, batch_first=True)

        # --- NEW: 2-Layer LSTM for History ---
        # The input to the LSTM is obs_dim (the size of a single state S)
        # self.lstm = nn.LSTM(
        #     input_size=obs_dim, 
        #     hidden_size=self.hidden_dim, 
        #     num_layers=2,               # <-- Added 2 layers
        #     dropout=dropout_p,          # <-- Added dropout between LSTM layers
        #     batch_first=True
        # )
        

        # Linear layer after the LSTM
        self.lstm_proj = nn.Linear(self.hidden_dim, self.hidden_dim)

        # --- MODIFIED: Input projection ---  
        # Now accepts obs_dim (S_t) + hidden_dim (LSTM output)
        self.input_proj = nn.Linear(obs_dim + self.hidden_dim, self.hidden_dim)  
    
        try:  
            self.feature_dim_disaster = self.CONFIG['disaster']['fire']['feature_dim']  
        except Exception as e:  
            logger.warning("No disaster feature dimension found in config, set to 1")  
            self.feature_dim_disaster = 1  
            
        # --- Disaster projection ---  
        self.disaster_encoder = nn.Linear(self.feature_dim_disaster, self.hidden_dim)  

        # --- Transformer encoder ---  
        encoder_layer = nn.TransformerEncoderLayer(  
            d_model=self.hidden_dim,  
            nhead=self.num_heads,  
            dim_feedforward=self.hidden_dim * 4,  
            dropout=dropout_p,  
            batch_first=True  
        )  
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=self.num_encoder_layers)  

        # --- Feedforward layers ---  
        layer_sizes = [self.hidden_dim * 3] + self.hidden_sizes  
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

    # NEW: Added hist_obs as an argument
    def forward(self, obs, disaster=None, padding_mask=None, hist_obs=None):  
        if obs.dim() == 2:   
            obs = obs.unsqueeze(0)  
            if disaster is not None: disaster = disaster.unsqueeze(0)  
            if padding_mask is not None: padding_mask = padding_mask.unsqueeze(0)  
            if hist_obs is not None: hist_obs = hist_obs.unsqueeze(0)
        elif obs.dim() == 1: 
            obs = obs.unsqueeze(0).unsqueeze(0)  
            if disaster is not None: disaster = disaster.unsqueeze(0).unsqueeze(0)  
            if padding_mask is not None: padding_mask = padding_mask.unsqueeze(0).unsqueeze(0)  
            if hist_obs is not None: hist_obs = hist_obs.unsqueeze(0).unsqueeze(0)

        B, max_N, obs_dim = obs.shape  
        device = self.input_proj.weight.device  
        
        obs = obs.to(device)  
        if disaster is not None: disaster = disaster.to(device)  
        if padding_mask is not None: padding_mask = padding_mask.to(device)  

        # --- NEW: Process History (LSTM -> Linear -> ReLU) ---
        if hist_obs is not None:
            hist_obs = hist_obs.to(device)
            T = hist_obs.size(2) # seq_len
            hist_flat = hist_obs.view(B * max_N, T, -1)
            
            _, (h_n, _) = self.lstm(hist_flat)
            lstm_out = h_n[-1] # Extract the last hidden state [B*max_N, hidden_dim]
            
            # Linear + ReLU
            lstm_embed = F.relu(self.lstm_proj(lstm_out)) 
            lstm_embed = lstm_embed.view(B, max_N, self.hidden_dim)
        else:
            lstm_embed = torch.zeros((B, max_N, self.hidden_dim), device=device)

        # --- MODIFIED: Concatenate S_t with the LSTM output before the Transformer ---
        enriched_obs = torch.cat([obs, lstm_embed], dim=-1) # Shape: [B, max_N, obs_dim + hidden_dim]
        
        # Project enriched obs: [B, max_N, hidden_dim]  
        obs_proj = self.input_proj(enriched_obs)  

        # --- Transformer encoding ---  
        x_encoded = self.transformer(obs_proj, src_key_padding_mask=padding_mask)  

        # --- Group embedding ---  
        if padding_mask is not None:  
            masked_encoded = x_encoded.masked_fill(padding_mask.unsqueeze(-1), 0.0)  
            group_embed = masked_encoded.sum(dim=1) / (max_N - padding_mask.sum(dim=1, keepdim=True) + 1e-8)  
        else:  
            group_embed = x_encoded.mean(dim=1)   
        group_embed = group_embed.unsqueeze(1).repeat(1, max_N, 1)   

        # --- Disaster embedding ---  
        if disaster is None:  
            disaster_embed = torch.full((B, max_N, self.hidden_dim), -1.0, device=device)  
        else:  
            disaster_embed = self.disaster_encoder(disaster)   

        # --- Concatenate x_encoded + group_embed + disaster_embed ---  
        concat_input = torch.cat([x_encoded, group_embed, disaster_embed], dim=2)  

        # --- MLP ---  
        h = concat_input.view(B * max_N, -1)  
        for linear, norm, drop in zip(self.hidden_layers, self.bn_layers, self.drop_layers):  
            # h = drop(torch.relu(norm(linear(h)))) # standard layer order
            # h = drop(torch.relu(norm(linear(h)))) 
            h = norm(torch.relu(linear(h)))
            
        q_values = self.output_layer(h)  
        return q_values