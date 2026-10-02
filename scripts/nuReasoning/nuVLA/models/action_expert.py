"""
GR00T-style Flow-Matching DiT Action Expert with layer-wise VLM KV injection.

Modified from the public nuReasoning implementation.
Main change:
  - Planner cross-attention no longer receives final VLM hidden states.
  - Selected VLM attention K/V caches are injected directly into planner
    cross-attention blocks.
  - Planner hidden states generate Q; VLM K/V are NOT re-projected.

The original StateEncoder, ActionEncoder, flow-matching objective, trajectory
normalization, decoder, and inference integration are otherwise kept.
"""

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta

logger = logging.getLogger(__name__)

LayerKV = Dict[int, Dict[str, torch.Tensor]]


@dataclass
class ActionExpertConfig:
    # Kept for checkpoint/config compatibility with public nuVLA.
    vlm_feature_dim: int = 2048
    ego_state_dim: int = 4
    max_history_traj_points: int = 6
    history_traj_dim: int = 3
    num_waypoints: int = 10
    trajectory_dim: int = 3

    # DiT architecture
    hidden_dim: int = 512
    num_heads: int = 8
    num_dit_layers: int = 12
    dropout: float = 0.1
    mlp_ratio: float = 4.0
    interleave_self_attention: bool = True

    # Layer-wise VLM KV injection
    kv_layer_indices: List[int] = field(default_factory=lambda: [3, 8, 13, 18, 23, 27])
    kv_num_heads: int = 8
    kv_head_dim: int = 128

    # Flow matching
    num_inference_steps: int = 10
    num_timestep_buckets: int = 1000
    noise_beta_alpha: float = 1.5
    noise_beta_beta: float = 2.5
    noise_s: float = 0.999
    sigma_min: float = 1e-4
    trajectory_norm_scale: Tuple[float, float, float] = (50.0, 20.0, math.pi)

    add_pos_embed: bool = True
    max_seq_len: int = 64


# ============================================================================
# Building blocks
# ============================================================================


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        timesteps = timesteps.float()
        half = self.dim // 2
        exponent = -torch.arange(half, dtype=torch.float, device=timesteps.device) * (
            math.log(10000.0) / half
        )
        freqs = timesteps.unsqueeze(-1) * exponent.exp()
        return torch.cat([torch.sin(freqs), torch.cos(freqs)], dim=-1)


class TimestepEncoder(nn.Module):
    def __init__(self, hidden_dim: int, freq_dim: int = 256):
        super().__init__()
        self.freq_dim = freq_dim
        self.mlp = nn.Sequential(
            nn.Linear(freq_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half = self.freq_dim // 2
        exponent = -math.log(10000.0) / (half - 1)
        freqs = torch.exp(
            torch.arange(half, device=timesteps.device, dtype=torch.float) * exponent
        )
        args = timesteps.float().unsqueeze(-1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return self.mlp(emb.to(next(self.mlp.parameters()).dtype))


class FixedSinusoidalPositionEmbedding(nn.Module):
    def __init__(self, dim: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, dim, 2, dtype=torch.float) * -(math.log(10000.0) / dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.shape[1]].to(x.dtype)


class AdaLayerNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(dim, dim * 2)
        self.norm = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)

    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        emb = self.linear(self.silu(temb))
        scale, shift = emb.chunk(2, dim=-1)
        return self.norm(x) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class FeedForward(nn.Module):
    def __init__(self, dim: int, mult: float = 4.0, dropout: float = 0.0):
        super().__init__()
        inner = int(dim * mult)
        self.net = nn.Sequential(
            nn.Linear(dim, inner),
            nn.GELU(approximate="tanh"),
            nn.Dropout(dropout),
            nn.Linear(inner, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class InjectedKVAttention(nn.Module):
    """Cross-attention with planner Q and raw VLM K/V.

    VLM K/V are consumed exactly in cache space [B, Hkv, T, Dh]. They are not
    passed through planner-side k_proj/v_proj. To make dimensions compatible,
    only the planner query is projected into Hkv * Dh and the attended result is
    projected back to planner hidden_dim.
    """

    def __init__(
        self,
        query_dim: int,
        kv_num_heads: int,
        kv_head_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.kv_num_heads = kv_num_heads
        self.kv_head_dim = kv_head_dim
        self.attn_dim = kv_num_heads * kv_head_dim
        self.scale = kv_head_dim ** -0.5

        self.q_proj = nn.Linear(query_dim, self.attn_dim, bias=True)
        self.out_proj = nn.Linear(self.attn_dim, query_dim, bias=True)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        kv_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, S, _ = query_states.shape
        if key_states.ndim != 4 or value_states.ndim != 4:
            raise RuntimeError(
                f"Expected [B,Hkv,T,Dh], got K={tuple(key_states.shape)}, "
                f"V={tuple(value_states.shape)}"
            )
        if key_states.shape != value_states.shape:
            raise RuntimeError("Injected K and V must have identical shapes.")
        if key_states.shape[0] != B:
            raise RuntimeError("Planner batch size and VLM KV batch size differ.")
        if key_states.shape[1] != self.kv_num_heads:
            raise RuntimeError(
                f"Expected {self.kv_num_heads} KV heads, got {key_states.shape[1]}"
            )
        if key_states.shape[-1] != self.kv_head_dim:
            raise RuntimeError(
                f"Expected KV head dim {self.kv_head_dim}, got {key_states.shape[-1]}"
            )

        q = self.q_proj(query_states)
        q = q.view(B, S, self.kv_num_heads, self.kv_head_dim).transpose(1, 2)
        # q: [B,Hkv,S,Dh], k/v: [B,Hkv,T,Dh]
        scores = torch.matmul(q, key_states.transpose(-2, -1)) * self.scale

        if kv_mask is not None:
            if kv_mask.ndim != 2:
                raise RuntimeError(f"kv_mask must be [B,T], got {tuple(kv_mask.shape)}")
            mask = kv_mask[:, None, None, :].bool()
            scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)

        probs = F.softmax(scores.float(), dim=-1).to(scores.dtype)
        probs = self.dropout(probs)
        attended = torch.matmul(probs, value_states)
        attended = attended.transpose(1, 2).contiguous().view(B, S, self.attn_dim)
        return self.out_proj(attended)


class TransformerBlock(nn.Module):
    """One planner block: injected cross-attention or ordinary self-attention."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        is_cross_attention: bool,
        kv_num_heads: int,
        kv_head_dim: int,
        dropout: float = 0.0,
        mlp_ratio: float = 4.0,
        max_seq_len: int = 512,
    ):
        super().__init__()
        self.is_cross_attention = is_cross_attention
        self.norm1 = AdaLayerNorm(dim)
        self.pos_embed = FixedSinusoidalPositionEmbedding(dim, max_len=max_seq_len)

        if self.is_cross_attention:
            self.attn = InjectedKVAttention(
                query_dim=dim,
                kv_num_heads=kv_num_heads,
                kv_head_dim=kv_head_dim,
                dropout=dropout,
            )
        else:
            self.attn = nn.MultiheadAttention(
                embed_dim=dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True,
            )

        self.norm2 = nn.LayerNorm(dim)
        self.ff = FeedForward(dim, mult=mlp_ratio, dropout=dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: torch.Tensor,
        injected_kv: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        normed = self.pos_embed(self.norm1(hidden_states, temb))

        if self.is_cross_attention:
            if injected_kv is None:
                raise RuntimeError("Cross-attention planner block did not receive VLM KV.")
            attn_out = self.attn(
                normed,
                injected_kv["key"],
                injected_kv["value"],
                injected_kv.get("mask"),
            )
        else:
            attn_out = self.attn(normed, normed, normed, need_weights=False)[0]

        hidden_states = hidden_states + attn_out
        hidden_states = hidden_states + self.ff(self.norm2(hidden_states))
        return hidden_states


# ============================================================================
# Encoders
# ============================================================================


class ActionEncoder(nn.Module):
    def __init__(self, action_dim: int, hidden_dim: int):
        super().__init__()
        self.W1 = nn.Linear(action_dim, hidden_dim)
        self.W2 = nn.Linear(2 * hidden_dim, hidden_dim)
        self.W3 = nn.Linear(hidden_dim, hidden_dim)
        self.pos_encoding = SinusoidalPositionalEncoding(hidden_dim)

    def forward(self, actions: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        B, T, _ = actions.shape
        t_expanded = timesteps.unsqueeze(1).expand(-1, T)
        a_emb = self.W1(actions)
        tau_emb = self.pos_encoding(t_expanded).to(a_emb.dtype)
        x = torch.cat([a_emb, tau_emb], dim=-1)
        x = self.W2(x)
        x = x * torch.sigmoid(x)
        return self.W3(x)


class StateEncoder(nn.Module):
    def __init__(
        self,
        ego_state_dim: int,
        max_history_points: int,
        history_point_dim: int,
        hidden_dim: int,
    ):
        super().__init__()
        state_dim = ego_state_dim + max_history_points * history_point_dim
        self.mlp = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(
        self, ego_state: torch.Tensor, history_trajectory: torch.Tensor,
    ) -> torch.Tensor:
        hist_flat = history_trajectory.reshape(ego_state.shape[0], -1)
        return self.mlp(torch.cat([ego_state, hist_flat], dim=-1)).unsqueeze(1)


# ============================================================================
# DiT
# ============================================================================


class GR00TDiT(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        num_layers: int,
        kv_layer_indices: List[int],
        kv_num_heads: int,
        kv_head_dim: int,
        dropout: float = 0.1,
        mlp_ratio: float = 4.0,
        interleave_self_attention: bool = True,
        max_seq_len: int = 512,
    ):
        super().__init__()
        self.interleave_self_attention = interleave_self_attention
        self.kv_layer_indices = list(kv_layer_indices)
        if not self.kv_layer_indices:
            raise ValueError("kv_layer_indices must contain at least one VLM layer.")

        self.timestep_encoder = TimestepEncoder(hidden_dim)

        blocks = []
        self.cross_block_indices: List[int] = []
        for idx in range(num_layers):
            is_self = (idx % 2 == 1) and interleave_self_attention
            is_cross = not is_self
            if is_cross:
                self.cross_block_indices.append(idx)
            blocks.append(
                TransformerBlock(
                    dim=hidden_dim,
                    num_heads=num_heads,
                    is_cross_attention=is_cross,
                    kv_num_heads=kv_num_heads,
                    kv_head_dim=kv_head_dim,
                    dropout=dropout,
                    mlp_ratio=mlp_ratio,
                    max_seq_len=max_seq_len,
                )
            )
        self.transformer_blocks = nn.ModuleList(blocks)

        # Map cross-attention depth monotonically across selected VLM depths.
        # If counts match this is exactly 1:1; otherwise nearest depth is reused.
        n_cross = len(self.cross_block_indices)
        n_kv = len(self.kv_layer_indices)
        self.cross_block_to_kv_layer: Dict[int, int] = {}
        for rank, block_idx in enumerate(self.cross_block_indices):
            if n_cross == 1:
                src_pos = n_kv - 1
            else:
                src_pos = int(round(rank * (n_kv - 1) / (n_cross - 1)))
            self.cross_block_to_kv_layer[block_idx] = self.kv_layer_indices[src_pos]

        logger.info("Planner cross-block -> VLM KV layer map: %s", self.cross_block_to_kv_layer)

        self.norm_out = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.proj_out_1 = nn.Linear(hidden_dim, 2 * hidden_dim)
        self.proj_out_2 = nn.Linear(hidden_dim, hidden_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        layer_kv: LayerKV,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        temb = self.timestep_encoder(timestep)

        for idx, block in enumerate(self.transformer_blocks):
            if block.is_cross_attention:
                vlm_layer = self.cross_block_to_kv_layer[idx]
                if vlm_layer not in layer_kv:
                    raise KeyError(
                        f"Planner block {idx} needs VLM KV layer {vlm_layer}, "
                        f"available={sorted(layer_kv.keys())}"
                    )
                hidden_states = block(hidden_states, temb, layer_kv[vlm_layer])
            else:
                hidden_states = block(hidden_states, temb)

        shift, scale = self.proj_out_1(F.silu(temb)).chunk(2, dim=-1)
        hidden_states = (
            self.norm_out(hidden_states) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        )
        return self.proj_out_2(hidden_states)


# ============================================================================
# Action Expert
# ============================================================================


class FlowMatchingDiTActionExpert(nn.Module):
    def __init__(self, config: ActionExpertConfig):
        super().__init__()
        self.config = config

        self.state_encoder = StateEncoder(
            ego_state_dim=config.ego_state_dim,
            max_history_points=config.max_history_traj_points,
            history_point_dim=config.history_traj_dim,
            hidden_dim=config.hidden_dim,
        )
        self.action_encoder = ActionEncoder(
            action_dim=config.trajectory_dim,
            hidden_dim=config.hidden_dim,
        )

        if config.add_pos_embed:
            self.position_embedding = nn.Embedding(config.max_seq_len, config.hidden_dim)
            nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)

        self.dit = GR00TDiT(
            hidden_dim=config.hidden_dim,
            num_heads=config.num_heads,
            num_layers=config.num_dit_layers,
            kv_layer_indices=config.kv_layer_indices,
            kv_num_heads=config.kv_num_heads,
            kv_head_dim=config.kv_head_dim,
            dropout=config.dropout,
            mlp_ratio=config.mlp_ratio,
            interleave_self_attention=config.interleave_self_attention,
            max_seq_len=config.max_seq_len,
        )

        self.action_decoder = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.trajectory_dim),
        )

        self.beta_dist = Beta(config.noise_beta_alpha, config.noise_beta_beta)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.zeros_(self.action_decoder[-1].weight)
        nn.init.zeros_(self.action_decoder[-1].bias)

    def _norm_scale_tensor(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return torch.tensor(self.config.trajectory_norm_scale, device=device, dtype=dtype)

    def normalize_trajectory(self, waypoints: torch.Tensor) -> torch.Tensor:
        return waypoints / self._norm_scale_tensor(waypoints.device, waypoints.dtype)

    def denormalize_trajectory(self, waypoints: torch.Tensor) -> torch.Tensor:
        return waypoints * self._norm_scale_tensor(waypoints.device, waypoints.dtype)

    def _sample_time(
        self, batch_size: int, device: torch.device, dtype: torch.dtype,
    ) -> torch.Tensor:
        sample = self.beta_dist.sample([batch_size]).to(device, dtype=dtype)
        return (1 - sample) * self.config.noise_s

    def _forward_dit(
        self,
        noisy_trajectory: torch.Tensor,
        timestep_discrete: torch.Tensor,
        layer_kv: LayerKV,
        ego_state: torch.Tensor,
        history_trajectory: torch.Tensor,
    ) -> torch.Tensor:
        state_tokens = self.state_encoder(ego_state, history_trajectory)
        action_tokens = self.action_encoder(noisy_trajectory, timestep_discrete)

        if self.config.add_pos_embed:
            T = action_tokens.shape[1]
            pos_ids = torch.arange(T, dtype=torch.long, device=action_tokens.device)
            action_tokens = action_tokens + self.position_embedding(pos_ids).unsqueeze(0)

        sa_tokens = torch.cat([state_tokens, action_tokens], dim=1)
        dit_out = self.dit(sa_tokens, layer_kv, timestep_discrete)
        action_out = dit_out[:, -noisy_trajectory.shape[1]:]
        return self.action_decoder(action_out)

    def _compute_loss(
        self,
        x_1: torch.Tensor,
        layer_kv: LayerKV,
        ego_state: torch.Tensor,
        history_trajectory: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        x_1 = self.normalize_trajectory(x_1)
        B = x_1.shape[0]
        device = x_1.device

        t = self._sample_time(B, device, x_1.dtype)
        t_expand = t[:, None, None]
        noise = torch.randn_like(x_1)
        x_t = (1 - t_expand) * noise + t_expand * x_1
        velocity_target = x_1 - noise

        t_discrete = (t * self.config.num_timestep_buckets).long()
        v_pred = self._forward_dit(
            x_t, t_discrete, layer_kv, ego_state, history_trajectory
        )
        loss = F.mse_loss(v_pred, velocity_target)

        with torch.no_grad():
            clean_pred = x_t + (1.0 - t_expand) * v_pred
            clean_pred = self.denormalize_trajectory(clean_pred.float())
            clean_target = self.denormalize_trajectory(x_1.float())

            x_err = clean_pred[..., 0] - clean_target[..., 0]
            y_err = clean_pred[..., 1] - clean_target[..., 1]
            theta_err = clean_pred[..., 2] - clean_target[..., 2]
            theta_err = torch.atan2(torch.sin(theta_err), torch.cos(theta_err))

            per_dim_mse = torch.stack([
                (x_err ** 2).mean(),
                (y_err ** 2).mean(),
                (theta_err ** 2).mean(),
            ])

        return {
            "loss": loss,
            "mse_x": per_dim_mse[0],
            "mse_y": per_dim_mse[1],
            "mse_theta": per_dim_mse[2],
        }

    def forward(
        self,
        x_1: torch.Tensor,
        layer_kv: LayerKV,
        ego_state: torch.Tensor,
        history_trajectory: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        return self._compute_loss(
            x_1=x_1,
            layer_kv=layer_kv,
            ego_state=ego_state,
            history_trajectory=history_trajectory,
        )

    @torch.no_grad()
    def sample(
        self,
        layer_kv: LayerKV,
        ego_state: torch.Tensor,
        history_trajectory: torch.Tensor,
        num_steps: Optional[int] = None,
    ) -> torch.Tensor:
        if num_steps is None:
            num_steps = self.config.num_inference_steps
        num_steps = max(int(num_steps), 1)

        first = layer_kv[sorted(layer_kv.keys())[0]]["key"]
        B = first.shape[0]
        device = first.device
        dtype = first.dtype

        actions = torch.randn(
            B, self.config.num_waypoints, self.config.trajectory_dim,
            device=device, dtype=dtype,
        )

        dt = 1.0 / num_steps
        for step in range(num_steps):
            t_cont = step / float(num_steps)
            t_discrete = int(t_cont * self.config.num_timestep_buckets)
            t_tensor = torch.full((B,), t_discrete, device=device, dtype=torch.long)
            v_pred = self._forward_dit(
                actions, t_tensor, layer_kv, ego_state, history_trajectory,
            )
            actions = actions + dt * v_pred

        actions = self.denormalize_trajectory(actions)
        actions[..., 2] = torch.atan2(torch.sin(actions[..., 2]), torch.cos(actions[..., 2]))
        return actions


# ============================================================================
# Trajectory metrics
# ============================================================================


def compute_trajectory_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> Dict[str, float]:
    pos_error = torch.sqrt(
        (pred[..., 0] - target[..., 0]) ** 2
        + (pred[..., 1] - target[..., 1]) ** 2
    )

    ade = pos_error.mean().item()
    fde = pos_error[:, -1].mean().item()

    heading_error = torch.abs(pred[..., 2] - target[..., 2])
    heading_error = torch.min(heading_error, 2 * math.pi - heading_error)
    mean_heading_error = heading_error.mean().item()

    return {
        "ADE_m": ade,
        "FDE_m": fde,
        "heading_error_rad": mean_heading_error,
        "heading_error_deg": math.degrees(mean_heading_error),
    }
