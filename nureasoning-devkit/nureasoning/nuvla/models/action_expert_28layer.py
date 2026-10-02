#!/usr/bin/env python3
"""
Alpamayo-referenced, layer-matched Flow-Matching Action Expert for nuVLA.

What is kept from the current nuVLA design
------------------------------------------
- Flow-matching trajectory objective.
- ego_state + history trajectory conditioning.
- 10 future waypoints, (x, y, heading).
- 28 Action Expert blocks matched 1:1 with 28 Qwen3-VL language layers.
- Frozen VLM; gradients stop at the VLM K/V cache.
- Raw VLM K/V are not re-projected on the planner side.
- Sparse planner self-attention every N blocks.

What is changed using Alpamayo as the reference
-----------------------------------------------
- Qwen3-VL K cache is assumed to be POST-RoPE.
- Planner cross-attention Q is therefore normalized and rotated with the
  SAME Qwen3-VL MRoPE coordinates before attending to cached K.
- Expert positions are supplied externally as Qwen3-VL RoPE cos/sin values.
  The bridge module builds them from:
      valid_prompt_length + rope_delta + expert_token_index
  matching the positional-continuation idea used by Alpamayo.
- The previous FixedSinusoidalPositionEmbedding before cross-attention is
  removed to avoid mixing a second positional coordinate system into Q.
- Learned action-token position embeddings are retained because the planner
  has a different hidden width from Qwen and sparse local self-attention.

Expected layer_kv entry
-----------------------
layer_kv[layer_idx]["key"]   : [B, Hkv, Tkv, Dh]  (post-RoPE Qwen K)
layer_kv[layer_idx]["value"] : [B, Hkv, Tkv, Dh]
layer_kv[layer_idx]["mask"]  : [B, Tkv]            (True = valid)

Expected expert RoPE
--------------------
rope_cos : [B, S_expert, Dh]
rope_sin : [B, S_expert, Dh]

where S_expert = 1 state token + num_waypoints action tokens.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta


LayerKV = Dict[int, Dict[str, torch.Tensor]]


@dataclass
class ActionExpertConfig:
    # nuVLA I/O
    vlm_feature_dim: int = 2048
    ego_state_dim: int = 4
    max_history_traj_points: int = 6
    history_traj_dim: int = 3
    num_waypoints: int = 10
    trajectory_dim: int = 3

    # ------------------------------------------------------------------
    # Current compact 28-layer AE
    # ORIGINAL AE:
    #   hidden_dim=512
    #   num_heads=8
    #   num_dit_layers=12
    #   mlp_ratio=4.0 -> FFN=2048
    #
    # CURRENT:
    #   hidden_dim=384
    #   num_heads=6
    #   num_dit_layers=28
    #   mlp_ratio=3.0 -> FFN=1152
    # ------------------------------------------------------------------
    hidden_dim: int = 384       # ORIGINAL: 512
    num_heads: int = 6          # ORIGINAL: 8
    num_dit_layers: int = 28    # ORIGINAL: 12
    dropout: float = 0.05       # ORIGINAL: 0.05
    mlp_ratio: float = 3.0      # ORIGINAL: 4.0

    # Sparse planner-local self-attention.
    # 4 -> layers 4,8,12,...,28 => 7 self-attention blocks.
    self_attention_every: int = 4

    # One VLM layer for every AE block.
    kv_layer_indices: List[int] = field(default_factory=lambda: list(range(28)))

    # Qwen3-VL-2B text KV geometry. These MUST match the actual cache.
    # Cross-attention Q is projected to Hkv * Dh so it can directly dot-product
    # with the frozen Qwen K cache.
    kv_num_heads: int = 8
    kv_head_dim: int = 128

    # Qwen-style per-head RMS normalization for AE cross-attention Q.
    qk_norm_eps: float = 1e-6

    # Flow matching
    num_inference_steps: int = 5
    num_timestep_buckets: int = 1000
    noise_beta_alpha: float = 1.5
    noise_beta_beta: float = 2.5
    noise_s: float = 0.999
    sigma_min: float = 1e-4
    trajectory_norm_scale: Tuple[float, float, float] = (50.0, 20.0, math.pi)

    # Planner-local token identity for sparse self-attention.
    # This is NOT used for VLM/AE cross-attention positional alignment.
    add_action_pos_embed: bool = True
    max_seq_len: int = 64


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Qwen/HF-compatible rotate_half."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_qwen_rope_to_q(
    q: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """
    Apply Qwen3-VL rotary embedding to planner Q only.

    q   : [B, H, S, Dh]
    cos : [B, S, Dh]
    sin : [B, S, Dh]
    """
    if cos.ndim != 3 or sin.ndim != 3:
        raise RuntimeError(
            f"Expected RoPE cos/sin [B,S,Dh], got cos={tuple(cos.shape)} "
            f"sin={tuple(sin.shape)}"
        )
    if cos.shape != sin.shape:
        raise RuntimeError("RoPE cos/sin shapes differ.")
    if q.shape[0] != cos.shape[0] or q.shape[2] != cos.shape[1]:
        raise RuntimeError(
            f"Q/RoPE batch or sequence mismatch: q={tuple(q.shape)}, "
            f"cos={tuple(cos.shape)}"
        )
    if q.shape[-1] != cos.shape[-1]:
        raise RuntimeError(
            f"Q/RoPE head_dim mismatch: q Dh={q.shape[-1]}, "
            f"RoPE Dh={cos.shape[-1]}"
        )

    cos = cos.unsqueeze(1).to(device=q.device, dtype=q.dtype)
    sin = sin.unsqueeze(1).to(device=q.device, dtype=q.dtype)
    return (q * cos) + (rotate_half(q) * sin)


class HeadRMSNorm(nn.Module):
    """
    RMSNorm over a single attention head dimension.

    Qwen3-VL normalizes q/k per head before RoPE. Frozen cached K has already
    passed Qwen's k_norm; this gives planner Q the corresponding normalization.
    """

    def __init__(self, head_dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(head_dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        xf = x.float()
        var = xf.pow(2).mean(dim=-1, keepdim=True)
        xf = xf * torch.rsqrt(var + self.eps)
        return self.weight.to(xf.dtype) * xf.to(in_dtype)


class SinusoidalPositionalEncoding(nn.Module):
    """Used only for the diffusion timestep embedding inside ActionEncoder."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        timesteps = timesteps.float()
        half = self.dim // 2
        exponent = -torch.arange(
            half, dtype=torch.float, device=timesteps.device
        ) * (math.log(10000.0) / half)
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
            torch.arange(
                half, device=timesteps.device, dtype=torch.float
            ) * exponent
        )
        args = timesteps.float().unsqueeze(-1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return self.mlp(emb.to(next(self.mlp.parameters()).dtype))


class AdaLayerNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(dim, 2 * dim)
        self.norm = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)

    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        scale, shift = self.linear(self.silu(temb)).chunk(2, dim=-1)
        return self.norm(x) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class FeedForward(nn.Module):
    def __init__(self, dim: int, mult: float, dropout: float):
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
    """
    Planner Q attends directly to frozen, post-RoPE Qwen3-VL K/V.

    Critical positional rule:
    - key_states already contain Qwen RoPE.
    - q receives Qwen-generated expert RoPE cos/sin before SDPA.
    """

    def __init__(
        self,
        query_dim: int,
        kv_num_heads: int,
        kv_head_dim: int,
        dropout: float,
        qk_norm_eps: float,
    ):
        super().__init__()
        self.kv_num_heads = kv_num_heads
        self.kv_head_dim = kv_head_dim
        self.attn_dim = kv_num_heads * kv_head_dim

        self.q_proj = nn.Linear(query_dim, self.attn_dim, bias=False)
        self.q_norm = HeadRMSNorm(kv_head_dim, eps=qk_norm_eps)
        self.out_proj = nn.Linear(self.attn_dim, query_dim, bias=False)
        self.dropout_p = float(dropout)

    def forward(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        kv_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if key_states.ndim != 4 or value_states.ndim != 4:
            raise RuntimeError(
                f"Expected K/V [B,Hkv,T,Dh], got "
                f"K={tuple(key_states.shape)} V={tuple(value_states.shape)}"
            )
        if key_states.shape != value_states.shape:
            raise RuntimeError("Injected K and V shapes differ.")

        B, S, _ = query_states.shape
        if key_states.shape[0] != B:
            raise RuntimeError("Action/VLM batch sizes differ.")
        if key_states.shape[1] != self.kv_num_heads:
            raise RuntimeError(
                f"Expected {self.kv_num_heads} KV heads, "
                f"got {key_states.shape[1]}"
            )
        if key_states.shape[-1] != self.kv_head_dim:
            raise RuntimeError(
                f"Expected KV head_dim={self.kv_head_dim}, "
                f"got {key_states.shape[-1]}"
            )

        # [B,S,Hkv,Dh] -> [B,Hkv,S,Dh]
        q = self.q_proj(query_states)
        q = q.view(B, S, self.kv_num_heads, self.kv_head_dim)
        q = self.q_norm(q)
        q = q.transpose(1, 2)

        # Alpamayo/Qwen positional continuity:
        # VLM K is already post-RoPE, so AE Q must use the same Qwen RoPE frame.
        q = apply_qwen_rope_to_q(q, rope_cos, rope_sin)

        attn_mask = None
        if kv_mask is not None:
            if kv_mask.ndim != 2:
                raise RuntimeError(
                    f"kv_mask must be [B,T], got {tuple(kv_mask.shape)}"
                )
            # PyTorch SDPA bool mask: True = allowed.
            attn_mask = kv_mask[:, None, None, :].bool()

        attended = F.scaled_dot_product_attention(
            q,
            key_states,
            value_states,
            attn_mask=attn_mask,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=False,
        )
        attended = (
            attended.transpose(1, 2)
            .contiguous()
            .view(B, S, self.attn_dim)
        )
        return self.out_proj(attended)


class MatchedKVBlock(nn.Module):
    """
    One Action Expert depth matched to exactly one VLM depth.

    Every block:
      1) mandatory VLM K/V cross-attention
      2) optional sparse planner-local self-attention
      3) FFN

    Cross-attention Q uses Qwen RoPE.
    """

    def __init__(
        self,
        dim: int,
        planner_heads: int,
        kv_num_heads: int,
        kv_head_dim: int,
        dropout: float,
        mlp_ratio: float,
        add_self_attention: bool,
        qk_norm_eps: float,
    ):
        super().__init__()
        self.cross_norm = AdaLayerNorm(dim)
        self.cross_attn = InjectedKVAttention(
            query_dim=dim,
            kv_num_heads=kv_num_heads,
            kv_head_dim=kv_head_dim,
            dropout=dropout,
            qk_norm_eps=qk_norm_eps,
        )

        self.add_self_attention = add_self_attention
        if add_self_attention:
            self.self_norm = nn.LayerNorm(dim)
            self.self_attn = nn.MultiheadAttention(
                embed_dim=dim,
                num_heads=planner_heads,
                dropout=dropout,
                batch_first=True,
            )

        self.ff_norm = nn.LayerNorm(dim)
        self.ff = FeedForward(dim, mult=mlp_ratio, dropout=dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: torch.Tensor,
        injected_kv: Dict[str, torch.Tensor],
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
    ) -> torch.Tensor:
        # No fixed sinusoidal position embedding here.
        # Qwen MRoPE is applied to Q inside InjectedKVAttention.
        x = self.cross_norm(hidden_states, temb)
        hidden_states = hidden_states + self.cross_attn(
            x,
            injected_kv["key"],
            injected_kv["value"],
            rope_cos=rope_cos,
            rope_sin=rope_sin,
            kv_mask=injected_kv.get("mask"),
        )

        if self.add_self_attention:
            x = self.self_norm(hidden_states)
            hidden_states = hidden_states + self.self_attn(
                x, x, x, need_weights=False
            )[0]

        hidden_states = hidden_states + self.ff(self.ff_norm(hidden_states))
        return hidden_states


class LayerMatchedDiT(nn.Module):
    def __init__(self, config: ActionExpertConfig):
        super().__init__()

        if len(config.kv_layer_indices) != config.num_dit_layers:
            raise ValueError(
                "Layer-matched mode requires exactly one VLM KV layer per "
                f"Action Expert block: got {len(config.kv_layer_indices)} KV "
                f"layers vs {config.num_dit_layers} DiT layers."
            )

        self.kv_layer_indices = list(config.kv_layer_indices)
        self.timestep_encoder = TimestepEncoder(config.hidden_dim)

        blocks = []
        for idx in range(config.num_dit_layers):
            add_sa = (
                config.self_attention_every > 0
                and (idx + 1) % config.self_attention_every == 0
            )
            blocks.append(
                MatchedKVBlock(
                    dim=config.hidden_dim,
                    planner_heads=config.num_heads,
                    kv_num_heads=config.kv_num_heads,
                    kv_head_dim=config.kv_head_dim,
                    dropout=config.dropout,
                    mlp_ratio=config.mlp_ratio,
                    add_self_attention=add_sa,
                    qk_norm_eps=config.qk_norm_eps,
                )
            )
        self.blocks = nn.ModuleList(blocks)

        self.norm_out = nn.LayerNorm(
            config.hidden_dim, elementwise_affine=False
        )
        self.proj_out_1 = nn.Linear(
            config.hidden_dim, 2 * config.hidden_dim
        )
        self.proj_out_2 = nn.Linear(
            config.hidden_dim, config.hidden_dim
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        layer_kv: LayerKV,
        timestep: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
    ) -> torch.Tensor:
        temb = self.timestep_encoder(timestep)

        if rope_cos.shape[1] != hidden_states.shape[1]:
            raise RuntimeError(
                "Expert RoPE length must equal state+action token length: "
                f"rope={rope_cos.shape[1]}, hidden={hidden_states.shape[1]}"
            )

        for idx, block in enumerate(self.blocks):
            vlm_layer = self.kv_layer_indices[idx]
            if vlm_layer not in layer_kv:
                raise KeyError(
                    f"Action block {idx} requires VLM layer {vlm_layer}; "
                    f"available={sorted(layer_kv.keys())}"
                )
            hidden_states = block(
                hidden_states,
                temb,
                layer_kv[vlm_layer],
                rope_cos=rope_cos,
                rope_sin=rope_sin,
            )

        shift, scale = self.proj_out_1(F.silu(temb)).chunk(2, dim=-1)
        hidden_states = (
            self.norm_out(hidden_states)
            * (1 + scale.unsqueeze(1))
            + shift.unsqueeze(1)
        )
        return self.proj_out_2(hidden_states)


class ActionEncoder(nn.Module):
    def __init__(self, action_dim: int, hidden_dim: int):
        super().__init__()
        self.W1 = nn.Linear(action_dim, hidden_dim)
        self.W2 = nn.Linear(2 * hidden_dim, hidden_dim)
        self.W3 = nn.Linear(hidden_dim, hidden_dim)
        self.pos_encoding = SinusoidalPositionalEncoding(hidden_dim)

    def forward(
        self,
        actions: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        _, T, _ = actions.shape
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
        state_dim = (
            ego_state_dim + max_history_points * history_point_dim
        )
        self.mlp = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(
        self,
        ego_state: torch.Tensor,
        history_trajectory: torch.Tensor,
    ) -> torch.Tensor:
        hist_flat = history_trajectory.reshape(
            ego_state.shape[0], -1
        )
        return self.mlp(
            torch.cat([ego_state, hist_flat], dim=-1)
        ).unsqueeze(1)


class FlowMatchingDiTActionExpert(nn.Module):
    def __init__(self, config: ActionExpertConfig):
        super().__init__()
        self.config = config

        self.state_encoder = StateEncoder(
            config.ego_state_dim,
            config.max_history_traj_points,
            config.history_traj_dim,
            config.hidden_dim,
        )
        self.action_encoder = ActionEncoder(
            config.trajectory_dim,
            config.hidden_dim,
        )

        # Local action-token identity for the sparse planner self-attention.
        # Cross-attention positional alignment is handled only by Qwen MRoPE.
        if config.add_action_pos_embed:
            self.action_position_embedding = nn.Embedding(
                config.max_seq_len, config.hidden_dim
            )
            nn.init.normal_(
                self.action_position_embedding.weight,
                mean=0.0,
                std=0.02,
            )

        self.dit = LayerMatchedDiT(config)

        self.action_decoder = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, config.trajectory_dim),
        )

        self.beta_dist = Beta(
            config.noise_beta_alpha,
            config.noise_beta_beta,
        )
        self._init_weights()

    @property
    def num_expert_tokens(self) -> int:
        # 1 state token + future action tokens
        return 1 + self.config.num_waypoints

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.zeros_(self.action_decoder[-1].weight)
        nn.init.zeros_(self.action_decoder[-1].bias)

    def _scale(self, device, dtype):
        return torch.tensor(
            self.config.trajectory_norm_scale,
            device=device,
            dtype=dtype,
        )

    def normalize_trajectory(
        self, waypoints: torch.Tensor
    ) -> torch.Tensor:
        return waypoints / self._scale(
            waypoints.device, waypoints.dtype
        )

    def denormalize_trajectory(
        self, waypoints: torch.Tensor
    ) -> torch.Tensor:
        return waypoints * self._scale(
            waypoints.device, waypoints.dtype
        )

    def _sample_time(self, batch_size, device, dtype):
        sample = self.beta_dist.sample([batch_size]).to(
            device=device, dtype=dtype
        )
        return (1 - sample) * self.config.noise_s

    def _forward_dit(
        self,
        noisy_trajectory: torch.Tensor,
        timestep_discrete: torch.Tensor,
        layer_kv: LayerKV,
        ego_state: torch.Tensor,
        history_trajectory: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
    ) -> torch.Tensor:
        state_tokens = self.state_encoder(
            ego_state, history_trajectory
        )
        action_tokens = self.action_encoder(
            noisy_trajectory, timestep_discrete
        )

        if self.config.add_action_pos_embed:
            T = action_tokens.shape[1]
            pos_ids = torch.arange(
                T, device=action_tokens.device
            )
            action_tokens = action_tokens + (
                self.action_position_embedding(pos_ids).unsqueeze(0)
            )

        hidden = torch.cat(
            [state_tokens, action_tokens], dim=1
        )

        if rope_cos.shape[1] != hidden.shape[1]:
            raise RuntimeError(
                f"Need RoPE for {hidden.shape[1]} expert tokens, "
                f"got {rope_cos.shape[1]}."
            )

        hidden = self.dit(
            hidden,
            layer_kv,
            timestep_discrete,
            rope_cos=rope_cos,
            rope_sin=rope_sin,
        )
        action_hidden = hidden[
            :, -noisy_trajectory.shape[1] :
        ]
        return self.action_decoder(action_hidden)

    def forward(
        self,
        x_1: torch.Tensor,
        layer_kv: LayerKV,
        ego_state: torch.Tensor,
        history_trajectory: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        x_1 = self.normalize_trajectory(x_1)
        B = x_1.shape[0]

        t = self._sample_time(
            B, x_1.device, x_1.dtype
        )
        t_expand = t[:, None, None]

        noise = torch.randn_like(x_1)
        x_t = (1 - t_expand) * noise + t_expand * x_1
        velocity_target = x_1 - noise

        t_discrete = (
            t * self.config.num_timestep_buckets
        ).long()

        v_pred = self._forward_dit(
            x_t,
            t_discrete,
            layer_kv,
            ego_state,
            history_trajectory,
            rope_cos,
            rope_sin,
        )

        loss = F.mse_loss(v_pred, velocity_target)

        with torch.no_grad():
            clean_pred = x_t + (
                1.0 - t_expand
            ) * v_pred
            clean_pred = self.denormalize_trajectory(
                clean_pred.float()
            )
            clean_target = self.denormalize_trajectory(
                x_1.float()
            )

            x_err = (
                clean_pred[..., 0]
                - clean_target[..., 0]
            )
            y_err = (
                clean_pred[..., 1]
                - clean_target[..., 1]
            )
            theta_err = (
                clean_pred[..., 2]
                - clean_target[..., 2]
            )
            theta_err = torch.atan2(
                torch.sin(theta_err),
                torch.cos(theta_err),
            )

            per_dim_mse = torch.stack(
                [
                    (x_err ** 2).mean(),
                    (y_err ** 2).mean(),
                    (theta_err ** 2).mean(),
                ]
            )

        return {
            "loss": loss,
            "mse_x": per_dim_mse[0],
            "mse_y": per_dim_mse[1],
            "mse_theta": per_dim_mse[2],
        }

    @torch.no_grad()
    def sample(
        self,
        layer_kv: LayerKV,
        ego_state: torch.Tensor,
        history_trajectory: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        num_steps: Optional[int] = None,
    ) -> torch.Tensor:
        num_steps = max(
            int(
                num_steps
                or self.config.num_inference_steps
            ),
            1,
        )

        first = layer_kv[
            self.config.kv_layer_indices[0]
        ]["key"]
        B = first.shape[0]
        device = first.device
        dtype = first.dtype

        actions = torch.randn(
            B,
            self.config.num_waypoints,
            self.config.trajectory_dim,
            device=device,
            dtype=dtype,
        )

        dt = 1.0 / num_steps
        for step in range(num_steps):
            t_cont = step / float(num_steps)
            t_discrete = int(
                t_cont
                * self.config.num_timestep_buckets
            )
            t_tensor = torch.full(
                (B,),
                t_discrete,
                device=device,
                dtype=torch.long,
            )

            velocity = self._forward_dit(
                actions,
                t_tensor,
                layer_kv,
                ego_state,
                history_trajectory,
                rope_cos,
                rope_sin,
            )
            actions = actions + dt * velocity

        actions = self.denormalize_trajectory(actions)
        actions[..., 2] = torch.atan2(
            torch.sin(actions[..., 2]),
            torch.cos(actions[..., 2]),
        )
        return actions


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

    heading_error = torch.abs(
        pred[..., 2] - target[..., 2]
    )
    heading_error = torch.minimum(
        heading_error,
        2 * math.pi - heading_error,
    )
    heading = heading_error.mean().item()

    return {
        "ADE_m": ade,
        "FDE_m": fde,
        "heading_error_rad": heading,
        "heading_error_deg": math.degrees(heading),
    }
