from pyexpat import model
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from read_config import get_config, read_config
from logs.logger import logger

# class PPONetwork(nn.Module):  
#     def __init__(self, obs_dim, action_dim, hidden_dim=256, nhead=4,  
#                  num_layers=2, dropout_p=0.1):  
#         super().__init__()  

#         self.hidden_dim = hidden_dim  
#         self.input_proj = nn.Linear(obs_dim, hidden_dim)  

#         self.CONFIG = get_config()  
#         self.feature_dim_disaster = self.CONFIG['disaster']['fire']['feature_dim']  

#         self.disaster_encoder = nn.Linear(self.feature_dim_disaster, hidden_dim)  

#         encoder_layer = nn.TransformerEncoderLayer(  
#             d_model=hidden_dim,  
#             nhead=nhead,  
#             dim_feedforward=hidden_dim * 4,  
#             dropout=dropout_p,  
#             batch_first=True,  
#         )  
#         self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)  

#         layer_sizes = [hidden_dim * 3, 512, 1024, 1024, 512, 256]  
#         self.shared_layers = nn.ModuleList([  
#             nn.Linear(layer_sizes[i], layer_sizes[i + 1])  
#             for i in range(len(layer_sizes) - 1)  
#         ])  
#         self.bn_layers = nn.ModuleList([  
#             nn.LayerNorm(layer_sizes[i + 1])  
#             for i in range(len(layer_sizes) - 1)  
#         ])  
#         self.drop_layers = nn.ModuleList([  
#             nn.Dropout(p=dropout_p)  
#             for _ in range(len(layer_sizes) - 1)  
#         ])  

#         self.actor_head = nn.Sequential(  
#             nn.Linear(layer_sizes[-1], 256),  
#             nn.LayerNorm(256),  
#             nn.Tanh(),  
#             nn.Linear(256, 256),  
#             nn.LayerNorm(256),  
#             nn.Tanh(),  
#             nn.Linear(256, action_dim),  
#         )  

#         self.critic_head = nn.Sequential(  
#             nn.Linear(layer_sizes[-1], 256),  
#             nn.LayerNorm(256),  
#             nn.Tanh(),  
#             nn.Linear(256, 256),  
#             nn.LayerNorm(256),  
#             nn.Tanh(),  
#             nn.Linear(256, 1),  
#         )  

#     # ------------------------------------------------------------------  
#     # FORWARD  
#     # ------------------------------------------------------------------  
#     def forward(self, obs, disaster=None, src_key_padding_mask=None):  
#         """  
#         Handles BOTH calling conventions automatically:  

#         A) Rollout (single vehicle):  
#              obs     : [1, obs_dim]              (2-D)  
#              disaster: [D, disaster_dim] or None  (2-D)  
#              → returns action_logits [1, action_dim], value [1, 1]  

#         B) Batched update (padded steps):  
#              obs     : [B, N_max, obs_dim]                    (3-D)  
#              disaster: [B, D_max, disaster_dim] or None       (3-D)  
#              src_key_padding_mask: [B, N_max]  (True = pad)  
#              → returns action_logits [B, N_max, action_dim], value [B, N_max, 1]  
#         """  
#         # ------------------------------------------------------  
#         # Auto-detect 2-D (rollout) vs 3-D (batched update)  
#         # ------------------------------------------------------  
#         squeezed = False  
#         if obs.dim() == 1:  
#             obs = obs.unsqueeze(0)                     # [obs_dim] → [1, obs_dim]  

#         if obs.dim() == 2:  
#             # Rollout mode: [N, obs_dim] → [1, N, obs_dim]  
#             # N = 1 for single-vehicle calls  
#             squeezed = True  
#             obs = obs.unsqueeze(0)                     # [1, N, obs_dim]  
#             if disaster is not None and disaster.dim() == 2:  
#                 disaster = disaster.unsqueeze(0)       # [1, D, disaster_dim]  

#         B, N_max, _ = obs.shape  
#         device = obs.device  

#         # --- Project vehicle observations ---  
#         obs_proj = self.input_proj(obs)                # [B, N_max, hidden]  

#         # --- Disaster embedding ---  
#         if disaster is None:  
#             disaster_embed = torch.full(  
#                 (B, N_max, self.hidden_dim), -1.0, device=device  
#             )  
#         else:  
#             dis_proj = self.disaster_encoder(disaster)            # [B, D_max, hidden]  
#             disaster_embed = dis_proj.mean(dim=1, keepdim=True)   # [B, 1, hidden]  
#             disaster_embed = disaster_embed.expand(B, N_max, self.hidden_dim)  

#         # --- Transformer encoding (inter-vehicle attention) ---  
#         x_encoded = self.transformer(  
#             obs_proj,  
#             src_key_padding_mask=src_key_padding_mask,  
#         )                                              # [B, N_max, hidden]  

#         # --- Group embedding (mean over real vehicles only) ---  
#         if src_key_padding_mask is not None:  
#             real_mask = (~src_key_padding_mask).unsqueeze(-1).float()  # [B, N_max, 1]  
#             group_embed = (x_encoded * real_mask).sum(dim=1, keepdim=True) \
#                           / real_mask.sum(dim=1, keepdim=True).clamp(min=1)  
#         else:  
#             group_embed = x_encoded.mean(dim=1, keepdim=True)         # [B, 1, hidden]  
#         group_embed = group_embed.expand(B, N_max, self.hidden_dim)  

#         # --- Concatenate ---  
#         concat_input = torch.cat(  
#             [obs_proj, group_embed, disaster_embed], dim=-1  
#         )                                              # [B, N_max, hidden*3]  

#         # --- Shared MLP Body ---  
#         h = concat_input  
#         for linear, norm, drop in zip(  
#             self.shared_layers, self.bn_layers, self.drop_layers  
#         ):  
#             h = drop(torch.relu(norm(linear(h))))  

#         # --- Heads ---  
#         action_logits = self.actor_head(h)             # [B, N_max, action_dim]  
#         state_values  = self.critic_head(h)            # [B, N_max, 1]  

#         # ------------------------------------------------------  
#         # If we auto-wrapped from 2-D, unwrap back to 2-D  
#         # so the caller sees the same shapes as before  
#         # ------------------------------------------------------  
#         if squeezed:  
#             action_logits = action_logits.squeeze(0)   # [N, action_dim]  
#             state_values  = state_values.squeeze(0)    # [N, 1]  

#         return action_logits, state_values  

#     # ------------------------------------------------------------------  
#     # EVALUATE  –  always called from update() with 3-D padded batches  
#     # ------------------------------------------------------------------  
#     def evaluate(self, obs, disaster, actions, feasible_masks,  
#                  src_key_padding_mask=None):  
#         """  
#         Parameters  
#         ----------  
#         obs              : [B, N_max, obs_dim]  
#         disaster         : [B, D_max, disaster_dim] or None  
#         actions          : [B, N_max]             (LongTensor, padded with 0)  
#         feasible_masks   : [B, N_max, action_dim] (Bool, True = feasible)  
#         src_key_padding_mask : [B, N_max]         (True = padding)  

#         Returns  
#         -------  
#         log_probs   : [B, N_max]  
#         values      : [B, N_max]  
#         entropy     : [B, N_max]  
#         """  
#         # forward will see 3-D obs → stays in 3-D mode  
#         action_logits, state_values = self.forward(  
#             obs, disaster, src_key_padding_mask  
#         )                                              # [B, N_max, action_dim], [B, N_max, 1]  

#         # mask infeasible actions  
#         action_logits[~feasible_masks] = float('-inf')  

#         dist = Categorical(logits=action_logits)  
#         log_probs = dist.log_prob(actions)             # [B, N_max]  
#         entropy   = dist.entropy()                     # [B, N_max]  

#         return log_probs, state_values.squeeze(-1), entropy

# if __name__ == "__main__":

#     # Set the config file
#     CONFIG = read_config(config_filename="manhattan2")
#     # test the size of the model
#     obs_dim = 1578
#     action_dim = 1578
#     model = PPONetwork(obs_dim, action_dim)
#     print(model)

#     # Only trainable parameters (useful for fine-tuning)
#     trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
#     print(f"Trainable Parameters: {trainable_params:,}")

class PPONetwork(nn.Module):  
    def __init__(self, obs_dim, action_dim):  
        super().__init__()  

        # Extract config values  
        self.CONFIG = get_config()
        self.num_heads = self.CONFIG['model']['encoder']['num_heads']  
        self.head_dim = self.CONFIG['model']['encoder']['head_dim']  
        self.hidden_dim = self.num_heads * self.head_dim  # As per your YAML note (e.g., 4*64=256)  
        self.num_layers = self.CONFIG['model']['encoder']['num_encoder_layers']  
        self.dropout_p = self.CONFIG['model']['encoder']['dropout_p']

        self.input_proj = nn.Linear(obs_dim, self.hidden_dim)  

        try:  
            self.feature_dim_disaster = self.CONFIG['disaster']['fire']['feature_dim']  
        except Exception as e:  
            logger.warning("No disaster feature dimension found in config, set to 1")  
            self.feature_dim_disaster = 1  

        self.disaster_encoder = nn.Linear(self.feature_dim_disaster, self.hidden_dim)  

        encoder_layer = nn.TransformerEncoderLayer(  
            d_model=self.hidden_dim,  
            nhead=self.num_heads,  
            dim_feedforward=self.hidden_dim * 4,  
            batch_first=True,  
        )  
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)  

        # Build shared MLP dynamically from config  
        # Note that first dim of shared sizes must match the concat input concat size (hidden_dim*3)
        shared_sizes = [self.hidden_dim * 3] + self.CONFIG['model']['PPO']['shared_mlp']  
        assert len(shared_sizes) >= 2, "Shared MLP must have at least input and one output size."  
        self.shared_layers = nn.ModuleList([  
            nn.Linear(shared_sizes[i], shared_sizes[i + 1])  
            for i in range(len(shared_sizes) - 1)  
        ])  
        self.shared_norms = nn.ModuleList([  
            nn.LayerNorm(shared_sizes[i + 1])  
            for i in range(len(shared_sizes) - 1)  
        ])  
        self.shared_drops = nn.ModuleList([  
            nn.Dropout(p=self.dropout_p)  
            for _ in range(len(shared_sizes) - 1)  
        ])  

        # Build actor head dynamically from self.CONFIG  
        actor_sizes = self.CONFIG['model']['PPO']['actor_layers'] + [action_dim]  # Append final output size  
        assert actor_sizes[0] == shared_sizes[-1], f"Actor layers input ({actor_sizes[0]}) must match shared output ({shared_sizes[-1]})."  
        actor_modules = []  
        for i in range(len(actor_sizes) - 1):  
            actor_modules.append(nn.Linear(actor_sizes[i], actor_sizes[i + 1]))  
            if i < len(actor_sizes) - 2:  # No norm/ReLU on final layer  

                actor_modules.append(nn.ReLU())  
                actor_modules.append(nn.LayerNorm(actor_sizes[i + 1]))  
        self.actor_head = nn.Sequential(*actor_modules)  

        # Build critic head dynamically from self.CONFIG  
        critic_sizes = self.CONFIG['model']['PPO']['critic_layers'] + [1]  # Append final output size (scalar)  
        assert critic_sizes[0] == shared_sizes[-1], f"Critic layers input ({critic_sizes[0]}) must match shared output ({shared_sizes[-1]})."  
        critic_modules = []  
        for i in range(len(critic_sizes) - 1):  
            critic_modules.append(nn.Linear(critic_sizes[i], critic_sizes[i + 1]))  
            if i < len(critic_sizes) - 2:  # No norm/ReLU on final layer  

                critic_modules.append(nn.ReLU())  
                critic_modules.append(nn.LayerNorm(critic_sizes[i + 1]))  
        self.critic_head = nn.Sequential(*critic_modules)  

        # Orthogonal initialization for better stability in RL  
        self.apply(self._init_weights)  

    def _init_weights(self, module):  
        if isinstance(module, nn.Linear):  
            nn.init.orthogonal_(module.weight.data, gain=nn.init.calculate_gain('relu'))  # Gain for ReLU  
            if module.bias is not None:  
                module.bias.data.fill_(0.0)  

    def forward(self, obs, disaster=None, src_key_padding_mask=None):  
        """  
        Handles BOTH calling conventions automatically:  

        A) Rollout (single vehicle):  
             obs     : [1, obs_dim]              (2-D)  
             disaster: [D, disaster_dim] or None  (2-D)  
             → returns action_logits [1, action_dim], value [1, 1]  

        B) Batched update (padded steps):  
             obs     : [B, N_max, obs_dim]                    (3-D)  
             disaster: [B, D_max, disaster_dim] or None       (3-D)  
             src_key_padding_mask: [B, N_max]  (True = pad)  
             → returns action_logits [B, N_max, action_dim], value [B, N_max, 1]  
        """  
        # Auto-detect 2-D (rollout) vs 3-D (batched update)  
        squeezed = False  
        if obs.dim() == 1:  
            obs = obs.unsqueeze(0)  # [obs_dim] → [1, obs_dim]  

        if obs.dim() == 2:  
            # Rollout mode: [N, obs_dim] → [1, N, obs_dim]  
            # N = 1 for single-vehicle calls  
            squeezed = True  
            obs = obs.unsqueeze(0)  # [1, N, obs_dim]  
            if disaster is not None and disaster.dim() == 2:  
                disaster = disaster.unsqueeze(0)  # [1, D, disaster_dim]  

        B, N_max, _ = obs.shape  
        device = obs.device  

        # Project vehicle observations  
        obs_proj = self.input_proj(obs)  # [B, N_max, hidden]  

        # Disaster embedding  
        if disaster is None:  
            disaster_embed = torch.full(  
                (B, N_max, self.hidden_dim), -1.0, device=device  
            )  
        else:  
            dis_proj = self.disaster_encoder(disaster)  # [B, D_max, hidden]  
            disaster_embed = dis_proj.mean(dim=1, keepdim=True)  # [B, 1, hidden]  
            disaster_embed = disaster_embed.expand(B, N_max, self.hidden_dim)  

        # Transformer encoding (inter-vehicle attention)  
        x_encoded = self.transformer(  
            obs_proj,  
            src_key_padding_mask=src_key_padding_mask,  
        )  # [B, N_max, hidden]  

        # Group embedding (mean over real vehicles only)  
        if src_key_padding_mask is not None:  
            real_mask = (~src_key_padding_mask).unsqueeze(-1).float()  # [B, N_max, 1]  
            group_embed = (x_encoded * real_mask).sum(dim=1, keepdim=True) \
                          / real_mask.sum(dim=1, keepdim=True).clamp(min=1)  
        else:  
            group_embed = x_encoded.mean(dim=1, keepdim=True)  # [B, 1, hidden]  
        group_embed = group_embed.expand(B, N_max, self.hidden_dim)  

        # Concatenate  
        concat_input = torch.cat(  
            [obs_proj, group_embed, disaster_embed], dim=-1  
        )  # [B, N_max, hidden*3]  

        # Shared MLP Body  
        h = concat_input  
        for linear, norm, drop in zip(  
            self.shared_layers, self.shared_norms, self.shared_drops  
        ):  
            # h = drop(torch.relu(norm(linear(h))))  
            h = norm(torch.relu(linear(h)))

        # Heads  
        action_logits = self.actor_head(h)  # [B, N_max, action_dim]  
        state_values = self.critic_head(h)  # [B, N_max, 1]  

        # If we auto-wrapped from 2-D, unwrap back to 2-D  
        if squeezed:  
            action_logits = action_logits.squeeze(0)  # [N, action_dim]  
            state_values = state_values.squeeze(0)  # [N, 1]  

        return action_logits, state_values

    # ------------------------------------------------------------------  
    # EVALUATE  –  always called from update() with 3-D padded batches  
    # ------------------------------------------------------------------  
    def evaluate(self, obs, disaster, actions, feasible_masks,  
                 src_key_padding_mask=None):  
        """  
        Parameters  
        ----------  
        obs              : [B, N_max, obs_dim]  
        disaster         : [B, D_max, disaster_dim] or None  
        actions          : [B, N_max]             (LongTensor, padded with 0)  
        feasible_masks   : [B, N_max, action_dim] (Bool, True = feasible)  
        src_key_padding_mask : [B, N_max]         (True = padding)  

        Returns  
        -------  
        log_probs   : [B, N_max]  
        values      : [B, N_max]  
        entropy     : [B, N_max]  
        """  
        # forward will see 3-D obs → stays in 3-D mode  
        action_logits, state_values = self.forward(  
            obs, disaster, src_key_padding_mask  
        )                                              # [B, N_max, action_dim], [B, N_max, 1]  

        # mask infeasible actions  
        action_logits[~feasible_masks] = float('-inf')  

        dist = Categorical(logits=action_logits)  
        log_probs = dist.log_prob(actions)             # [B, N_max]  
        entropy   = dist.entropy()                     # [B, N_max]  

        return log_probs, state_values.squeeze(-1), entropy

    