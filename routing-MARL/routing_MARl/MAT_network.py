import logging
from typing import Optional, Tuple, List, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from read_config import get_config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# Small utility: build an MLP head without changing transformer d_model
# ---------------------------------------------------------------------
def make_mlp(input_dim: int, hidden_layers: List[int], dropout_p: float = 0.0):
    """
    Build an MLP block and return (module, output_dim).
    If hidden_layers is empty, returns Identity and output_dim=input_dim.
    """
    hidden_layers = hidden_layers or []
    if len(hidden_layers) == 0:
        return nn.Identity(), input_dim

    layers = []
    prev_dim = input_dim
    for h in hidden_layers:
        layers.append(nn.Linear(prev_dim, h))
        layers.append(nn.ReLU())
        layers.append(nn.LayerNorm(h))
        if dropout_p > 0:
            layers.append(nn.Dropout(dropout_p))
        prev_dim = h

    return nn.Sequential(*layers), prev_dim


# ---------------------------------------------------------------------
# MAT Network
# ---------------------------------------------------------------------
class MATNetwork(nn.Module):
    """
    Multi-Agent Transformer with:
    - Transformer encoder over agent observations
    - Transformer decoder over autoregressive action tokens
    - Value head from encoder output
    - Policy head from decoder output
    - Disaster input handled as per-agent set/tensor of shape [B, N, K, F]
      where:
        B = batch size
        N = number of agents
        K = number of disaster items
        F = disaster feature dimension

    Important shape convention for your project:
        disaster_obs for one vehicle = (5, 1)
    which means:
        K = 5 disasters, F = 1 feature per disaster.
    """

    def __init__(self, obs_dim: int, action_dim: int, dropout_p: float = 0.01):
        super().__init__()

        self.CONFIG = get_config()

        mat_cfg = self.CONFIG.get("model", {}).get("MAT", {})
        disaster_cfg = self.CONFIG.get("disaster", {})

        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)

        self.num_heads = int(mat_cfg.get("num_heads", 4))
        self.head_dim = int(mat_cfg.get("head_dim", 64))
        self.d_model = self.num_heads * self.head_dim

        self.num_encoder_layers = int(mat_cfg.get("num_encoder_layers", 2))
        self.num_decoder_layers = int(mat_cfg.get("num_decoder_layers", 2))
        self.dim_feedforward = int(mat_cfg.get("dim_feedforward", 4 * self.d_model))

        # This is the disaster FEATURE dimension, not the number of disasters.
        # In your case this should be 1.
        self.disaster_feat_dim = int(disaster_cfg.get("feature_dim", 1))

        value_layers = list(mat_cfg.get("value_layers", [512, 256]))
        policy_layers = list(mat_cfg.get("policy_layers", [512, 256]))

        # --------------------------------------------------------------
        # Input encoders
        # --------------------------------------------------------------
        self.obs_encoder = nn.Sequential(
            nn.Linear(self.obs_dim, self.d_model),
            nn.LayerNorm(self.d_model),
        )

        # Encode each disaster item independently, then pool across disaster items.
        # bias=False helps keep padded zero disaster items mapped closer to zero.
        self.disaster_item_encoder = nn.Sequential(
            nn.Linear(self.disaster_feat_dim, self.d_model, bias=False),
            nn.ReLU(),
            nn.Linear(self.d_model, self.d_model, bias=False),
        )

        self.fusion_norm = nn.LayerNorm(self.d_model)

        # --------------------------------------------------------------
        # Transformer encoder
        # --------------------------------------------------------------
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.num_heads,
            dim_feedforward=self.dim_feedforward,
            dropout=dropout_p,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.num_encoder_layers,
        )

        # --------------------------------------------------------------
        # Value head
        # --------------------------------------------------------------
        self.value_mlp, value_out_dim = make_mlp(
            input_dim=self.d_model,
            hidden_layers=value_layers,
            dropout_p=dropout_p,
        )
        self.value_head = nn.Linear(value_out_dim, 1)

        # --------------------------------------------------------------
        # Action token embedding and decoder
        # --------------------------------------------------------------
        # Extra token index = start token
        self.start_token_idx = self.action_dim
        self.action_embed = nn.Embedding(self.action_dim + 1, self.d_model)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.d_model,
            nhead=self.num_heads,
            dim_feedforward=self.dim_feedforward,
            dropout=dropout_p,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=self.num_decoder_layers,
        )

        # --------------------------------------------------------------
        # Policy head
        # --------------------------------------------------------------
        self.policy_mlp, policy_out_dim = make_mlp(
            input_dim=self.d_model,
            hidden_layers=policy_layers,
            dropout_p=dropout_p,
        )
        self.policy_head = nn.Linear(policy_out_dim, self.action_dim)

    # -----------------------------------------------------------------
    # Helper masks
    # -----------------------------------------------------------------
    @staticmethod
    def generate_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Standard autoregressive causal mask for decoder self-attention.
        """
        return torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=device),
            diagonal=1,
        )

    # -----------------------------------------------------------------
    # Helper: normalize observation shape
    # -----------------------------------------------------------------
    def _normalize_obs(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Accept:
            [B, N, obs_dim]
            [B, obs_dim]        -> treated as N=1
            [B, N, 1, obs_dim]  -> squeeze singleton
        Return:
            [B, N, obs_dim]
        """
        if obs.dim() == 4 and obs.size(-2) == 1:
            obs = obs.squeeze(-2)

        if obs.dim() == 2:
            obs = obs.unsqueeze(1)

        if obs.dim() != 3:
            raise ValueError(f"obs must be [B, N, obs_dim], got {tuple(obs.shape)}")

        if obs.size(-1) != self.obs_dim:
            raise ValueError(
                f"obs last dim must be {self.obs_dim}, got {obs.size(-1)}"
            )

        return obs

    # -----------------------------------------------------------------
    # Helper: encode disaster input
    # -----------------------------------------------------------------
    def _encode_disaster(
        self,
        disaster: Optional[torch.Tensor],
        B: int,
        N: int,
        device: torch.device,
        disaster_item_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Accept:
            None
            [B, N, F]        -> one disaster item per agent
            [B, N, K, F]     -> K disaster items per agent
            [K, F]           -> only for single-agent helper use; wrapped to [1,1,K,F]

        disaster_item_mask:
            None or [B, N, K] boolean mask, True for valid disaster items.

        Return:
            disaster_emb of shape [B, N, d_model]
        """
        if disaster is None:
            return torch.zeros(B, N, self.d_model, device=device)

        if disaster.dim() == 2:
            # Single-agent convenience path: [K, F] -> [1, 1, K, F]
            disaster = disaster.unsqueeze(0).unsqueeze(0)

        elif disaster.dim() == 3:
            # [B, N, F] -> [B, N, 1, F]
            disaster = disaster.unsqueeze(2)

        if disaster.dim() != 4:
            raise ValueError(
                f"disaster must be None, [B,N,F], [B,N,K,F], or [K,F], got {tuple(disaster.shape)}"
            )

        if disaster.size(0) != B or disaster.size(1) != N:
            raise ValueError(
                f"disaster batch/agent dims mismatch: expected ({B}, {N}, ...), got {tuple(disaster.shape)}"
            )

        if disaster.size(-1) != self.disaster_feat_dim:
            raise ValueError(
                f"disaster feature dim must be {self.disaster_feat_dim}, got {disaster.size(-1)}"
            )

        B_, N_, K, F_ = disaster.shape
        flat = disaster.reshape(B_ * N_ * K, F_)
        item_emb = self.disaster_item_encoder(flat).reshape(B_, N_, K, self.d_model)

        if disaster_item_mask is not None:
            if disaster_item_mask.dim() != 3:
                raise ValueError(
                    f"disaster_item_mask must be [B,N,K], got {tuple(disaster_item_mask.shape)}"
                )
            if disaster_item_mask.shape != (B_, N_, K):
                raise ValueError(
                    f"disaster_item_mask shape mismatch. Expected {(B_, N_, K)}, got {tuple(disaster_item_mask.shape)}"
                )

            mask = disaster_item_mask.unsqueeze(-1).to(item_emb.dtype)  # [B,N,K,1]
            pooled = (item_emb * mask).sum(dim=2) / mask.sum(dim=2).clamp(min=1.0)
        else:
            pooled = item_emb.mean(dim=2)

        return pooled

    # -----------------------------------------------------------------
    # Helper: encode observations + disasters
    # -----------------------------------------------------------------
    def encode_agents(
        self,
        obs: torch.Tensor,
        disaster: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        disaster_item_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            encoded: [B, N, d_model]
            values:  [B, N, 1]
        """
        obs = self._normalize_obs(obs)
        B, N, _ = obs.shape
        device = obs.device

        if padding_mask is None:
            padding_mask = torch.zeros(B, N, dtype=torch.bool, device=device)

        obs_emb = self.obs_encoder(obs)  # [B, N, d_model]
        disaster_emb = self._encode_disaster(
            disaster=disaster,
            B=B,
            N=N,
            device=device,
            disaster_item_mask=disaster_item_mask,
        )

        src = self.fusion_norm(obs_emb + disaster_emb)
        encoded = self.encoder(src, src_key_padding_mask=padding_mask)

        value_features = self.value_mlp(encoded)
        values = self.value_head(value_features)

        return encoded, values

    # -----------------------------------------------------------------
    # Forward
    # -----------------------------------------------------------------
    def forward(
        self,
        obs: torch.Tensor,
        disaster: Optional[torch.Tensor] = None,
        actions: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        disaster_item_mask: Optional[torch.Tensor] = None,
        mode: str = "inference",
        deterministic: bool = False,
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        """
        Modes:
            - "value_only": return (None, values)
            - "training": teacher-forced decoder with provided actions
            - "inference": autoregressive decoder

        Returns:
            action_logits: [B, N, action_dim] or None in value_only mode
            values:        [B, N, 1]
        """
        obs = self._normalize_obs(obs)
        B, N, _ = obs.shape
        device = obs.device

        if padding_mask is None:
            padding_mask = torch.zeros(B, N, dtype=torch.bool, device=device)

        encoded, values = self.encode_agents(
            obs=obs,
            disaster=disaster,
            padding_mask=padding_mask,
            disaster_item_mask=disaster_item_mask,
        )

        if mode == "value_only":
            return None, values

        if mode == "training":
            if actions is None:
                raise ValueError("actions must be provided in training mode")

            if actions.dim() != 2 or actions.shape != (B, N):
                raise ValueError(
                    f"actions must be [B, N] = {(B, N)}, got {tuple(actions.shape)}"
                )

            start_tokens = torch.full(
                (B, 1),
                fill_value=self.start_token_idx,
                dtype=torch.long,
                device=device,
            )
            decoder_tokens = torch.cat([start_tokens, actions], dim=1)  # [B, N+1]
            tgt = self.action_embed(decoder_tokens)  # [B, N+1, d_model]

            tgt_mask = self.generate_causal_mask(decoder_tokens.size(1), device=device)

            # Start token is always valid
            start_pad = torch.zeros(B, 1, dtype=torch.bool, device=device)
            tgt_key_padding_mask = torch.cat([start_pad, padding_mask], dim=1)

            decoded = self.decoder(
                tgt=tgt,
                memory=encoded,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=padding_mask,
            )

            # Drop the start token output; keep only agent positions
            decoded_agent = decoded[:, 1:, :]  # [B, N, d_model]
            policy_features = self.policy_mlp(decoded_agent)
            action_logits = self.policy_head(policy_features)  # [B, N, action_dim]
            return action_logits, values

        if mode == "inference":
            action_logits = torch.zeros(B, N, self.action_dim, device=device)

            generated_tokens = torch.full(
                (B, 1),
                fill_value=self.start_token_idx,
                dtype=torch.long,
                device=device,
            )

            for t in range(N):
                tgt = self.action_embed(generated_tokens)
                tgt_mask = self.generate_causal_mask(generated_tokens.size(1), device=device)

                decoded = self.decoder(
                    tgt=tgt,
                    memory=encoded,
                    tgt_mask=tgt_mask,
                    memory_key_padding_mask=padding_mask,
                )

                last_hidden = decoded[:, -1, :]  # [B, d_model]
                policy_features = self.policy_mlp(last_hidden)
                logits_t = self.policy_head(policy_features)  # [B, action_dim]
                action_logits[:, t, :] = logits_t

                # Sample only to build the autoregressive prefix.
                # For padded positions we inject a dummy token 0.
                if deterministic:
                    sampled = torch.argmax(logits_t, dim=-1)
                else:
                    sampled = Categorical(logits=logits_t).sample()

                if padding_mask is not None:
                    valid_t = ~padding_mask[:, t]
                    sampled = torch.where(valid_t, sampled, torch.zeros_like(sampled))

                generated_tokens = torch.cat([generated_tokens, sampled.unsqueeze(1)], dim=1)

            return action_logits, values

        raise ValueError(f"Unsupported mode='{mode}'")