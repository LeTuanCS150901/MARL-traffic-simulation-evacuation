import torch
import torch.nn as nn
import torch.nn.functional as F

# class QNetwork(nn.Module):  
#     def __init__(self, obs_dim, action_dim):
#         super(QNetwork, self).__init__()

#         # hidđen layers
#         # toy net
#         # layer_sizes = [obs_dim, 128, 128, 256, 128, 64, 32]
        
#         # manhattan network
#         # layer_sizes = [obs_dim, 512, 512, 512, 512, 128, 128]

#         # manhattan2 big network
#         layer_sizes = [obs_dim, 512, 1024, 1024, 1024, 512, 256]
#         self.hiddenlayer = nn.ModuleList([
#             nn.Linear(layer_sizes[i], layer_sizes[i+1]) for i in range(len(layer_sizes)-1)
#         ])

#         # LayerNorm after each Linear (except output)
#         self.bn_layers = nn.ModuleList([
#             nn.LayerNorm(layer_sizes[i+1]) for i in range(len(layer_sizes)-1)
#         ])
#         # output layer
#         self.outputlayer = nn.Linear(layer_sizes[-1], action_dim)  
#         self.sigmoid = nn.Sigmoid()
    
#     def forward(self, x):  
#         # hidden -> batch -> relu
#         for hl, bn in zip(self.hiddenlayer, self.bn_layers):  
#             x = hl(x)
#             x = bn(x)
#             x = torch.relu(x)  
#         return self.outputlayer(x)


# import torch
# import torch.nn as nn

# class QNetwork(nn.Module):
#     def __init__(self, obs_dim, action_dim, dropout_p=0.2):
#         super(QNetwork, self).__init__()

#         layer_sizes = [obs_dim, 512, 1024, 1024, 1024, 512, 256]

#         self.hiddenlayer = nn.ModuleList([
#             nn.Linear(layer_sizes[i], layer_sizes[i+1]) for i in range(len(layer_sizes)-1)
#         ])

#         self.bn_layers = nn.ModuleList([
#             nn.LayerNorm(layer_sizes[i+1]) for i in range(len(layer_sizes)-1)
#         ])

#         # dropout modules (skip first and last)
#         self.drop_layers = nn.ModuleList([
#             nn.Dropout(p=dropout_p) if 0 < i < len(layer_sizes)-2 else nn.Identity()
#             for i in range(len(layer_sizes)-1)
#         ])

#         self.outputlayer = nn.Linear(layer_sizes[-1], action_dim)

#     def forward(self, x):
#         for hl, bn, dr in zip(self.hiddenlayer, self.bn_layers, self.drop_layers):
#             x = hl(x)
#             x = bn(x)
#             x = torch.relu(x)
#             x = dr(x)  # Apply dropout after activation
#         return self.outputlayer(x)


# import torch
# import torch.nn as nn

# class QNetwork(nn.Module):
#     """
#     Q network with LSTM encoder followed by feedforward layers.
    
#     """
#     def __init__(self, obs_dim, action_dim, hidden_dim=128, dropout_p=0.2):
#         super(QNetwork, self).__init__()

#         # --- LSTM encoder ---
#         self.lstm = nn.LSTM(
#             input_size=obs_dim,
#             hidden_size=hidden_dim,
#             num_layers=2,
#             batch_first=True
#         )

#         # --- Feedforward network ---
#         layer_sizes = [hidden_dim, 512, 1024, 1024, 512, 256]

#         self.hiddenlayer = nn.ModuleList([
#             nn.Linear(layer_sizes[i], layer_sizes[i + 1]) for i in range(len(layer_sizes) - 1)
#         ])

#         self.bn_layers = nn.ModuleList([
#             nn.LayerNorm(layer_sizes[i + 1]) for i in range(len(layer_sizes) - 1)
#         ])

#         self.drop_layers = nn.ModuleList([
#             nn.Dropout(p=dropout_p) if 0 < i < len(layer_sizes) - 2 else nn.Identity()
#             for i in range(len(layer_sizes) - 1)
#         ])

#         self.outputlayer = nn.Linear(layer_sizes[-1], action_dim)

#     def forward(self, x):
#         """
#         x: [N, obs_dim] where N = number of vehicles (treated as time steps)
#         """
#         # Add batch dimension (batch=1)
#         x = x.unsqueeze(0)  # [1, N, obs_dim]

#         # LSTM encoding
#         lstm_out, (h_n, c_n) = self.lstm(x)   # lstm_out: [1, N, hidden_dim]
#         x = lstm_out.squeeze(0)               # [N, hidden_dim]

#         # Feedforward MLP
#         for hl, bn, dr in zip(self.hiddenlayer, self.bn_layers, self.drop_layers):
#             x = hl(x)
#             x = bn(x)
#             x = torch.relu(x)
#             # x = dr(x)

#         return self.outputlayer(x)  # [N, action_dim]



import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """
    Optimized Transformer Q-network with variable number of vehicles.
    Uses layer_sizes list to define feedforward layers dynamically.
    Each vehicle has its own Q-values: [num_vehicles, action_dim].
    """
    def __init__(self, obs_dim, action_dim, hidden_dim=256, nhead=4,
                 num_layers=2, dropout_p=0.1):
        super().__init__()

        # --- Input projection ---
        self.input_proj = nn.Linear(obs_dim, hidden_dim)

        # --- Transformer encoder for inter-vehicle attention ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim*4,
            dropout=dropout_p,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # --- Feedforward layers per vehicle ---
        layer_sizes = [hidden_dim, 512, 1024, 1024, 512, 256]
        self.hidden_layers = nn.ModuleList([
            nn.Linear(layer_sizes[i], layer_sizes[i+1]) for i in range(len(layer_sizes)-1)
        ])
        self.bn_layers = nn.ModuleList([
            nn.LayerNorm(layer_sizes[i+1]) for i in range(len(layer_sizes)-1)
        ])
        self.drop_layers = nn.ModuleList([
            nn.Dropout(p=dropout_p) if 0 < i < len(layer_sizes)-2 else nn.Identity()
            for i in range(len(layer_sizes)-1)
        ])

        # --- Output layer ---
        self.output_layer = nn.Linear(layer_sizes[-1], action_dim)

    def forward(self, obs_batch):
        """
        obs_batch: [num_vehicles, obs_dim] 
        returns: [num_vehicles, action_dim]
        """
        # Project input
        x = self.input_proj(obs_batch)       # [num_vehicles, hidden_dim]
        x = x.unsqueeze(0)                   # [1, num_vehicles, hidden_dim] for transformer
        x = self.transformer(x)              # [1, num_vehicles, hidden_dim]
        x = x.squeeze(0)                     # [num_vehicles, hidden_dim]

        # Feedforward per vehicle
        for fc, bn, dr in zip(self.hidden_layers, self.bn_layers, self.drop_layers):
            x = dr(F.relu(bn(fc(x))))

        # Output Q-values per vehicle
        q_values = self.output_layer(x)      # [num_vehicles, action_dim]
        return q_values