# MATNetwork.py
# A self-contained implementation of the Multi-Agent Transformer (MAT) network
# that robustly handles per-vehicle observations and per-vehicle disaster inputs
# including None, a single disaster, or multiple disasters per vehicle.

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D

from read_config import get_config
from logs.logger import logger  # optional: used for logging if needed

class MATNetwork(nn.Module):  
    def __init__(self, obs_dim, action_dim, dropout_p: float = 0.01):  
        super().__init__()  

        # Configuration (with safe defaults if keys are missing)  
        self.CONFIG = get_config()  

        self.obs_dim = int(obs_dim)  
        self.action_dim = int(action_dim)  

        # Architecture hyperparameters  
        self.nhead = int(self.CONFIG['model']['MAT']['num_heads'])  
        self.head_dim = int(self.CONFIG['model']['MAT']['head_dim'])  
        self.d_model = self.nhead * self.head_dim   
        self.num_encoder_layers = int(self.CONFIG['model']['MAT']['num_encoder_layers'])  
        self.num_decoder_layers = int(self.CONFIG['model']['MAT']['num_decoder_layers'])  
        self.dim_feedforward = self.d_model * 4  

        self.hidden_dim = self.d_model  
        self.feature_dim_disaster = int(  
            self.CONFIG.get('disaster', {}).get('feature_dim', 1)  
        )  

        # Modules  
        self.input_proj = nn.Linear(self.obs_dim, self.hidden_dim)  
        self.disaster_encoder = nn.Linear(self.feature_dim_disaster, self.hidden_dim)  

        encoder_layer = nn.TransformerEncoderLayer(  
            d_model=self.hidden_dim,  
            nhead=self.nhead,  
            dim_feedforward=self.dim_feedforward,  
            dropout=dropout_p,  
            batch_first=True  
        )  
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_encoder_layers)  

        self.value_layers = []  
        for size in self.CONFIG['model']['MAT']['value_layers']:  
            self.value_layers.append(nn.Linear(self.hidden_dim, size))  
            self.value_layers.append(nn.ReLU())  
            self.value_layers.append(nn.LayerNorm(size))  
            self.hidden_dim = size  # Update hidden dim for the next layer  
        self.value_layers = nn.Sequential(*self.value_layers)  
        self.value_head = nn.Linear(self.hidden_dim, 1)  

        # self.value_head = nn.Sequential(
        #     nn.Linear(self.hidden_dim, 512),
        #     nn.ReLU(),
        #     nn.LayerNorm(512),
        #     nn.Linear(512, 256),
        #     nn.ReLU(),
        #     nn.LayerNorm(256),
        #     nn.Linear(256, 1)
        # )

        self.action_embed = nn.Embedding(action_dim + 1, self.hidden_dim)  

        decoder_layer = nn.TransformerDecoderLayer(  
            d_model=self.hidden_dim,  
            nhead=self.nhead,  
            dim_feedforward=self.dim_feedforward,  
            dropout=dropout_p,  
            batch_first=True  
        )  
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=self.num_decoder_layers)  

        self.policy_layers = []  
        for size in self.CONFIG['model']['MAT']['policy_layers']:  
            self.policy_layers.append(nn.Linear(self.hidden_dim, size))  
            self.policy_layers.append(nn.ReLU())  
            self.policy_layers.append(nn.LayerNorm(size))  
            self.hidden_dim = size  # Update hidden dim for the next layer  
        self.policy_layers = nn.Sequential(*self.policy_layers)  
        self.policy_head = nn.Linear(self.hidden_dim, action_dim)  

    def _normalize_disaster_input(self, disaster, B, max_N, device):
        """
        Normalize disaster input to [B, max_N, dis_dim] or None.

        Acceptable shapes:
        - None
        - [B, max_N, dis_dim]
        - [B, max_N, D, dis_dim]  -> mean over D -> [B, max_N, dis_dim]
        """
        if disaster is None:
            return None

        if disaster.dim() == 4:
            # [B, max_N, D, dis_dim] -> mean over D
            return disaster.mean(dim=2)
        elif disaster.dim() == 3:
            # [B, max_N, dis_dim]
            return disaster
        else:
            raise ValueError(f"Unsupported disaster input shape: {disaster.shape}")

    def forward(self, obs, disaster=None, actions=None, padding_mask=None, mode: str = 'inference'):
        """
        Forward through MAT.

        Params
        - obs: [B, max_N, obs_dim] or possibly [B, max_N, 1, obs_dim] (we'll squeeze)
        - disaster: None or [B, max_N, dis_dim] or [B, max_N, D, dis_dim]
        - actions: [B, max_N] long tensor (only used in training mode)
        - padding_mask: [B, max_N] (True = pad)
        - mode: 'inference' or 'training'

        Returns
        - If mode == 'inference': (action_logits, values)
          - action_logits: [B, max_N, action_dim]
          - values: [B, max_N, 1]
        - If mode == 'training': (action_logits, values)
          - action_logits: [B, max_N, action_dim]
          - values: [B, max_N, 1]
        """
        # Normalize 4D vs 3D observation shapes
        if obs.dim() == 4 and obs.size(-2) == 1:
            obs = obs.squeeze(-2)  # -> [B, max_N, obs_dim]
        if obs.dim() != 3:
            raise ValueError(f"obs must be [B, max_N, obs_dim], got {obs.shape}")

        B, max_N, obs_dim = obs.shape
        device = obs.device

        # 1) Observations projection
        obs_emb = self.input_proj(obs)  # [B, max_N, hidden_dim]

        # 2) Disaster input normalization and projection
        disaster_norm = self._normalize_disaster_input(disaster, B, max_N, device)  # [B, max_N, dis_dim] or None
        if disaster_norm is None:
            disaster_emb = torch.zeros_like(obs_emb)
        else:
            disaster_norm = disaster_norm.to(device)
            disaster_emb = self.disaster_encoder(disaster_norm)  # [B, max_N, hidden_dim]

        # 3) Combine observation and disaster embeddings
        src = obs_emb + disaster_emb  # [B, max_N, hidden_dim]

        # 4) Encode with Transformer encoder
        encoded = self.encoder(src, src_key_padding_mask=padding_mask)

        # 5) Value estimation
        # values = self.value_head(encoded)  # [B, max_N, 1]

        # 5) Value estimation with MLP head
        value_input = self.value_layers(encoded)  # [B, max_N, hidden_dim]
        values = self.value_head(value_input)  # [B, max_N, 1]

        if mode == 'inference':
            # Autoregressive decoding for each step in sequence
            action_logits = torch.zeros(B, max_N, self.policy_head.out_features, device=device)  # [B, max_N, action_dim]
            act_input = torch.full((B, 1), self.action_dim, dtype=torch.long, device=device)  # Start token index

            for t in range(max_N):
                act_emb = self.action_embed(act_input)  # [B, t+1, hidden_dim]

                # Triangular mask for decoder (only attend to previous tokens)
                tgt_mask = self.generate_triangular_mask(t + 1).to(device)

                decoded = self.decoder(
                    act_emb,
                    encoded,
                    tgt_mask=tgt_mask,
                    tgt_key_padding_mask=padding_mask[:, : t + 1] if padding_mask is not None else None
                )

                # logits = self.policy_head(decoded[:, -1, :])  # [B, action_dim]
                # action_logits[:, t, :] = logits


                # Policy head with MLP
                # Compute the policy logits  
                policy_output = self.policy_layers(decoded[:, -1, :])  # Use the output from the last time step  
                logits = self.policy_head(policy_output)  # [B, action_dim]  
                action_logits[:, t, :] = logits

                # Sample an action (training-time randomness may be controlled by caller)
                action = D.Categorical(logits=logits).sample()  # [B]
                act_input = torch.cat([act_input, action.unsqueeze(1)], dim=1)

            return action_logits, values

        else:  # Training: Parallel with ground-truth actions [^16]
            act_input = torch.cat([torch.full((B, 1), self.action_dim, device=device), actions], dim=1)  # Prepend start
            act_emb = self.action_embed(act_input)  # [B, max_N+1, hidden_dim]

            # Target sequence length is max_N+1 (start token + max_N actions)
            tgt_mask = self.generate_triangular_mask(max_N + 1).to(device)

            # Build a proper tgt_key_padding_mask of shape [B, max_N+1]
            # padding_mask is [B, max_N]; add a leading column for the start token (which is not padding)
            if padding_mask is not None:
                leading = torch.zeros((B, 1), dtype=padding_mask.dtype, device=device)  # start token is not padding
                tgt_key_padding_mask = torch.cat([leading, padding_mask], dim=1)  # [B, max_N+1]
            else:
                tgt_key_padding_mask = None

            decoded = self.decoder(act_emb, encoded, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask)

            # Skip the start token position  
            # Compute policy logits from the output of the policy layers skipping the first token  
            policy_output = self.policy_layers(decoded[:, 1:, :])  # Use the output for tokens starting from the second  
            action_logits = self.policy_head(policy_output)  # [B, max_N-1, action_dim]


            # Skip the start token position
            # action_logits = self.policy_head(decoded[:, 1:, :])  # [B, max_N, action_dim]
            return action_logits, values


    def generate_triangular_mask(self, sz: int) -> torch.Tensor:
        """
        Generate a [sz, sz] triangular mask where positions above the diagonal are -inf
        and diagonal is 0 (unused since we always mask future positions).
        This is compatible with batch_first Transformer usage.
        """
        return torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1)


# Usage example (very lightweight, for illustration only)
# model = MATNetwork(obs_dim=OBS_DIM, action_dim=ACTION_DIM)
# logits, vals = model(obs_batch, disaster_batch, padding_mask=pad_mask, mode='inference')
