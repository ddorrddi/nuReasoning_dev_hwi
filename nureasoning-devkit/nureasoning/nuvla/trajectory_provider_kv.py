from __future__ import annotations

import argparse
import json
import logging
import math
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from nureasoning.nuvla.models.data_loader import (
    DEFAULT_FRAME_RATE_HZ,
    camera_history_stride,
    frame_stride_for_dt,
    infer_clip_frame_rate_hz,
    waypoint_dt_s,
)

logger = logging.getLogger(__name__)

PLANNING_PROMPT_AUTO = "auto"
PLANNING_PROMPT_STRUCTURED = "structured"
PLANNING_PROMPT_MINIMAL = "minimal"
PLANNING_PROMPT_MODES = (
    PLANNING_PROMPT_AUTO,
    PLANNING_PROMPT_STRUCTURED,
    PLANNING_PROMPT_MINIMAL,
)
REASONING_FORMATS = (
    "driving",
    "spatial_driving",
    "spatial_driving_counterfactual",
    "spatial",
    "driving_counterfactual",
)

# Fallbacks match ``nureasoning.nuvla.train.parse_args`` defaults.
_TRAIN_DEFAULTS: Dict[str, Any] = {
    "num_history_steps": 1,
    "history_stride": 10,
    "current_res_w": 448,
    "current_res_h": 448,
    "history_res_w": 448,
    "history_res_h": 448,
    "frame_rate_hz": DEFAULT_FRAME_RATE_HZ,
    "trajectory_future_seconds": 5.0,
    "num_trajectory_points": 10,
    "max_history_traj_points": 6,
    "num_inference_steps": 5,
    "noise_beta_alpha": 2.5,
    "noise_beta_beta": 1.5,
    "reasoning_format": "spatial_driving_counterfactual",
    "reasoning_mode": "structured",
    "lora_rank": 0,
    "lora_alpha": 0,
    "lora_dropout": 0.0,
    "load_in_4bit": False,
    "action_hidden_dim": 384,
    "num_dit_heads": 6,
    "mlp_ratio": 3.0,
    "self_attention_every": 4,
    "kv_source": "teacher_forced",
    "generation_max_new_tokens": 256,
}


def resolve_planning_prompt_kind(
    planning_prompt: str,
    reasoning_mode: str = "structured",
) -> str:
    """Return ``structured`` or ``minimal`` for a planning user prompt.

    ``auto`` follows the checkpoint: structured-trained VLMs keep the
    annotation-style prompt; VQA-trained VLMs use the scene-only prompt
    because they never saw Spatial / Driving / Counterfactual as the user turn.
    """
    mode = str(planning_prompt or PLANNING_PROMPT_AUTO).lower().strip()
    if mode == PLANNING_PROMPT_AUTO:
        trained = str(reasoning_mode or "structured").lower().strip()
        return (
            PLANNING_PROMPT_MINIMAL
            if trained == "vqa"
            else PLANNING_PROMPT_STRUCTURED
        )
    if mode in (PLANNING_PROMPT_STRUCTURED, PLANNING_PROMPT_MINIMAL):
        return mode
    raise ValueError(
        f"Unknown planning_prompt {planning_prompt!r}; "
        f"expected one of {PLANNING_PROMPT_MODES}"
    )


def build_planning_user_prompt(
    vlm: Any,
    *,
    num_history_steps: int,
    num_current_cameras: int,
    mission_command: str,
    planning_prompt: str = PLANNING_PROMPT_AUTO,
    reasoning_mode: str = "structured",
    reasoning_format: Optional[str] = None,
) -> str:
    """Build the planning user prompt for a VLM backbone instance."""
    kind = resolve_planning_prompt_kind(planning_prompt, reasoning_mode)
    return vlm.build_multiview_prompt(
        num_history_steps=num_history_steps,
        num_current_cameras=num_current_cameras,
        mission_command=mission_command,
        include_reasoning_prefix=(kind == PLANNING_PROMPT_STRUCTURED),
        reasoning_format=reasoning_format,
    )


def add_planning_prompt_arguments(
    parser: argparse.ArgumentParser,
    *,
    hyphen: bool = False,
) -> None:
    """Add ``--planning_prompt`` / ``--planning_reasoning_format`` to a CLI."""
    prompt_flag = "--planning-prompt" if hyphen else "--planning_prompt"
    format_flag = "--planning-reasoning-format" if hyphen else "--planning_reasoning_format"
    parser.add_argument(
        prompt_flag,
        dest="planning_prompt",
        default=PLANNING_PROMPT_AUTO,
        choices=list(PLANNING_PROMPT_MODES),
        help=(
            "User prompt for planning inference. 'auto' uses the structured "
            "multi-view prompt when the checkpoint was trained with "
            "--reasoning_mode structured, and the scene-only prompt when it "
            "was trained with vqa. 'structured' always requests Spatial / "
            "Driving / Counterfactual sections. 'minimal' is cameras + mission "
            "only."
        ),
    )
    parser.add_argument(
        format_flag,
        dest="planning_reasoning_format",
        default=None,
        choices=list(REASONING_FORMATS),
        help=(
            "Override which structured sections appear in the planning prompt "
            "(default: the checkpoint's --reasoning_format)."
        ),
    )


def _load_metadata(clip_path: str) -> Dict[str, Any]:
    with open(os.path.join(clip_path, "metadata.json"), "r") as f:
        return json.load(f)


def load_vla_training_config(checkpoint_dir: str) -> Dict[str, Any]:
    """Load ``training_config.json`` saved by ``nureasoning.nuvla.train``.

    The trainer writes the file next to checkpoints (workspace root). Older
    runs sometimes copied it into the checkpoint directory itself.
    """
    ckpt_dir = os.path.normpath(checkpoint_dir)
    parent_dir = os.path.dirname(ckpt_dir)
    config_path = os.path.join(parent_dir, "training_config.json")
    if not os.path.isfile(config_path):
        config_path = os.path.join(ckpt_dir, "training_config.json")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(
            f"nuVLA training_config.json not found in {parent_dir} or {ckpt_dir}"
        )
    with open(config_path, "r", encoding="utf-8") as handle:
        cfg = json.load(handle)
    return {**_TRAIN_DEFAULTS, **cfg}


def vlm_backbone_config_kwargs(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Reconstruct VLMBackboneConfig from Stage-1/Stage-2 training config."""
    return {
        "model_name_or_path": cfg.get(
            "vlm_model_path",
            "Qwen/Qwen3-VL-2B-Instruct",
        ),
        "freeze_vision_encoder": True,
        "lora_rank": int(
            cfg.get("vlm_lora_rank", cfg.get("lora_rank", 0)) or 0
        ),
        "lora_alpha": int(
            cfg.get("vlm_lora_alpha", cfg.get("lora_alpha", 0)) or 0
        ),
        "lora_dropout": float(
            cfg.get("vlm_lora_dropout", cfg.get("lora_dropout", 0.0)) or 0.0
        ),
        "load_in_4bit": bool(cfg.get("load_in_4bit", False)),
        "current_resolution": (
            int(cfg.get("current_res_w", 448)),
            int(cfg.get("current_res_h", 448)),
        ),
        "history_resolution": (
            int(cfg.get("history_res_w", 448)),
            int(cfg.get("history_res_h", 448)),
        ),
        "reasoning_format": cfg.get(
            "reasoning_format",
            "spatial_driving_counterfactual",
        ),
    }


def _resolve_adapter_dir(
    checkpoint_dir: str,
    cfg: Dict[str, Any],
) -> str:
    """Resolve the frozen VLM adapter used by Stage-2 Action Expert training."""
    candidates: List[str] = []

    # New Stage-2 trainer records the Stage-1 checkpoint path.
    stage1 = cfg.get("vlm_checkpoint")
    if stage1:
        stage1 = os.path.expanduser(str(stage1))
        candidates.extend(
            [
                os.path.join(stage1, "vlm_adapter"),
                stage1,
            ]
        )

    # Backward compatibility with old joint checkpoints.
    candidates.extend(
        [
            os.path.join(checkpoint_dir, "vlm_adapter"),
            os.path.join(os.path.dirname(checkpoint_dir), "vlm_adapter"),
        ]
    )

    for candidate in candidates:
        if not candidate or not os.path.isdir(candidate):
            continue
        if (
            os.path.isfile(os.path.join(candidate, "adapter_model.safetensors"))
            or os.path.isfile(os.path.join(candidate, "adapter_model.bin"))
            or os.path.isfile(os.path.join(candidate, "vlm_backbone.pt"))
        ):
            return candidate

    raise FileNotFoundError(
        "Could not resolve the VLM adapter used by this Action Expert. "
        f"Checked: {candidates}"
    )


def _resolve_camera_names(cfg: Dict[str, Any]) -> List[str]:
    """Use exactly the same cameras that were used during training."""
    resolved = cfg.get("resolved_cameras")
    if isinstance(resolved, list) and resolved:
        return [str(x) for x in resolved]

    raw = cfg.get("cameras", "front,front_left,front_right")
    if isinstance(raw, str):
        names = [x.strip() for x in raw.split(",") if x.strip()]
        if names:
            return names

    if isinstance(raw, (list, tuple)) and raw:
        return [str(x) for x in raw]

    return ["front", "front_left", "front_right"]


def _resolve_selected_kv_layers(
    cfg: Dict[str, Any],
    vlm_num_layers: int,
) -> List[int]:
    """Return zero-based VLM layers used by the matched Action Expert."""
    selected_1based = cfg.get("selected_kv_layers_1based")
    if isinstance(selected_1based, list) and selected_1based:
        selected = [int(x) - 1 for x in selected_1based]
    else:
        raw = str(cfg.get("kv_layers", "all")).strip().lower()
        if raw == "all":
            selected = list(range(vlm_num_layers))
        else:
            selected = [
                int(x.strip()) - 1
                for x in raw.split(",")
                if x.strip()
            ]

    if not selected:
        raise ValueError("No VLM KV layers are configured.")
    if min(selected) < 0 or max(selected) >= vlm_num_layers:
        raise ValueError(
            f"Configured KV layers {selected} are invalid for "
            f"{vlm_num_layers} VLM layers."
        )
    return selected


def _load_frame_images(
    clip_path: str,
    frame: Dict[str, Any],
    resolution: Tuple[int, int],
    cam_names: Sequence[str],
) -> Dict[str, Any]:
    from PIL import Image

    cameras = (frame.get("sensors") or {}).get("cameras") or {}
    result: Dict[str, Any] = {}
    for cam in cam_names:
        rel = cameras.get(cam, "")
        if not rel:
            result[cam] = Image.new("RGB", resolution, (0, 0, 0))
            continue
        abs_path = os.path.join(clip_path, rel)
        try:
            with Image.open(abs_path) as raw:
                result[cam] = raw.convert("RGB").resize(resolution, Image.LANCZOS)
        except Exception:
            result[cam] = Image.new("RGB", resolution, (0, 0, 0))
    return result


def load_vlm_observation(
    clip_path: str,
    key_frame_idx: int,
    cfg: Optional[Dict[str, Any]] = None,
    cam_names: Optional[Sequence[str]] = None,
) -> Tuple[List[Any], List[str], str]:
    """Load multi-view history+current images the same way ``VLATrainer`` does.

    Returns ``(images, image_contexts, mission_command)`` with time-major then
    camera-major order: ``[t=-H ... t=0] × CAMERA_NAMES``.
    """
    from nureasoning.nuvla.models.vlm_backbone import CAMERA_NAMES

    cfg = {**_TRAIN_DEFAULTS, **(cfg or {})}
    names = list(cam_names or CAMERA_NAMES)
    meta = _load_metadata(clip_path)
    frames = meta.get("frames") or []
    if not frames or key_frame_idx < 0 or key_frame_idx >= len(frames):
        return [], [], "LANE_FOLLOW"

    num_hist = int(cfg.get("num_history_steps", 1) or 0)
    hist_stride = int(cfg.get("history_stride", 10) or 1)
    current_res = (
        int(cfg.get("current_res_w", 448)),
        int(cfg.get("current_res_h", 448)),
    )
    history_res = (
        int(cfg.get("history_res_w", 448)),
        int(cfg.get("history_res_h", 448)),
    )
    clip_hz = infer_clip_frame_rate_hz(
        meta, frames, float(cfg.get("frame_rate_hz", DEFAULT_FRAME_RATE_HZ)),
    )
    scaled_hist_stride = camera_history_stride(hist_stride, clip_hz)

    frame = frames[key_frame_idx]
    current_images = _load_frame_images(clip_path, frame, current_res, names)
    flat_images: List[Any] = []
    for step in range(num_hist, 0, -1):
        hist_idx = max(key_frame_idx - step * scaled_hist_stride, 0)
        hist_images = _load_frame_images(
            clip_path, frames[hist_idx], history_res, names
        )
        for cam in names:
            flat_images.append(hist_images.get(cam))
    for cam in names:
        flat_images.append(current_images.get(cam))
    flat_images = [img for img in flat_images if img is not None]

    image_contexts: List[str] = []
    for t in range(-num_hist, 1):
        t_label = f"t={t}" if t < 0 else "t=0 (current)"
        for cam in names:
            image_contexts.append(f"{t_label}, {cam} camera.")
    image_contexts = image_contexts[: len(flat_images)]

    mission = frame.get("mission_goal") or {}
    mission_cmd = (
        mission.get("command", "LANE_FOLLOW")
        if isinstance(mission, dict)
        else "LANE_FOLLOW"
    )
    return flat_images, image_contexts, str(mission_cmd or "LANE_FOLLOW")



class VLATrajectoryProvider:
    """
    Load a frozen VLM + layer-matched KV Action Expert and produce global-frame
    trajectories compatible with the benchmark trajectory_provider callback.

    Reasoning sources at deployment:
      generated   : generate reasoning, then re-forward it to extract full-context KV
      prompt_only : skip generation and use image+prompt full-context KV

    A Stage-2 checkpoint trained with teacher_forced KV defaults to generated
    reasoning here, because ground-truth VQA answers are not available at deployment.
    """

    def __init__(
        self,
        checkpoint_dir: str,
        num_inference_steps: Optional[int] = None,
        device: str = "cuda",
        planning_prompt: str = PLANNING_PROMPT_AUTO,
        reasoning_format: Optional[str] = None,
        reasoning_source: Optional[str] = None,
        generation_max_new_tokens: Optional[int] = None,
    ):
        import torch
        from nureasoning.nuvla.models.action_expert_28layer import (
            ActionExpertConfig,
            FlowMatchingDiTActionExpert,
        )
        from nureasoning.nuvla.models.vlm_backbone import (
            VLMBackbone,
            VLMBackboneConfig,
        )

        self._torch = torch
        self.device = torch.device(
            device if torch.cuda.is_available() else "cpu"
        )
        self.planning_prompt = (
            planning_prompt or PLANNING_PROMPT_AUTO
        )
        self.planning_reasoning_format = reasoning_format

        checkpoint_dir = os.path.normpath(
            os.path.expanduser(checkpoint_dir)
        )
        self.cfg = load_vla_training_config(checkpoint_dir)
        cfg = self.cfg

        self._cam_names = _resolve_camera_names(cfg)

        vlm_cfg = VLMBackboneConfig(
            **vlm_backbone_config_kwargs(cfg)
        )
        self.vlm = VLMBackbone(vlm_cfg)

        # Quantized models are already device-mapped.
        if not bool(cfg.get("load_in_4bit", False)):
            self.vlm = self.vlm.to(self.device)

        adapter_dir = _resolve_adapter_dir(
            checkpoint_dir,
            cfg,
        )
        self.vlm.load_adapter(
            adapter_dir,
            device=self.device,
        )

        for param in self.vlm.parameters():
            param.requires_grad_(False)
        self.vlm.eval()

        (
            self.vlm_num_layers,
            self.kv_num_heads,
            self.kv_head_dim,
        ) = self.vlm.get_kv_spec()
        self.selected_kv_layers = _resolve_selected_kv_layers(
            cfg,
            self.vlm_num_layers,
        )

        self.planning_prompt_kind = resolve_planning_prompt_kind(
            self.planning_prompt,
            cfg.get("reasoning_mode", "vqa"),
        )

        trained_source = str(
            cfg.get("kv_source", "teacher_forced")
        ).strip().lower()

        if reasoning_source is None:
            # GT reasoning does not exist during benchmark inference.
            self.reasoning_source = (
                "generated"
                if trained_source == "teacher_forced"
                else trained_source
            )
        else:
            self.reasoning_source = str(
                reasoning_source
            ).strip().lower()

        if self.reasoning_source not in (
            "generated",
            "prompt_only",
        ):
            raise ValueError(
                "Inference reasoning_source must be "
                "'generated' or 'prompt_only'."
            )

        if (
            trained_source != self.reasoning_source
            and trained_source != "teacher_forced"
        ):
            logger.warning(
                "Inference KV source (%s) differs from training KV source (%s).",
                self.reasoning_source,
                trained_source,
            )

        if trained_source == "teacher_forced":
            logger.warning(
                "Action Expert was trained with teacher-forced VQA context. "
                "Deployment has no GT answer, so provider uses %s context.",
                self.reasoning_source,
            )

        self.generation_max_new_tokens = int(
            generation_max_new_tokens
            or cfg.get("generation_max_new_tokens", 256)
            or 256
        )

        configured_steps = int(
            cfg.get("num_inference_steps", 5) or 5
        )
        self.num_inference_steps = int(
            num_inference_steps
            if num_inference_steps is not None
            else configured_steps
        )

        action_config = ActionExpertConfig(
            vlm_feature_dim=self.vlm.get_feature_dim(),
            ego_state_dim=4,
            max_history_traj_points=int(
                cfg.get("max_history_traj_points", 6)
            ),
            history_traj_dim=3,
            num_waypoints=int(
                cfg.get("num_trajectory_points", 10)
            ),
            trajectory_dim=3,
            hidden_dim=int(
                cfg.get("action_hidden_dim", 384)
            ),
            num_heads=int(
                cfg.get("num_dit_heads", 6)
            ),
            num_dit_layers=len(self.selected_kv_layers),
            dropout=0.0,
            mlp_ratio=float(
                cfg.get("mlp_ratio", 3.0)
            ),
            self_attention_every=int(
                cfg.get("self_attention_every", 4)
            ),
            kv_layer_indices=self.selected_kv_layers,
            kv_num_heads=self.kv_num_heads,
            kv_head_dim=self.kv_head_dim,
            num_inference_steps=self.num_inference_steps,
            num_timestep_buckets=int(
                cfg.get("num_timestep_buckets", 1000)
            ),
            noise_beta_alpha=float(
                cfg.get("noise_beta_alpha", 1.5)
            ),
            noise_beta_beta=float(
                cfg.get("noise_beta_beta", 2.5)
            ),
        )

        self.action_expert = FlowMatchingDiTActionExpert(
            action_config
        ).to(self.device)
        self.action_expert.eval()

        action_path = os.path.join(
            checkpoint_dir,
            "action_expert.pt",
        )
        if not os.path.isfile(action_path):
            raise FileNotFoundError(
                f"Action Expert checkpoint not found: {action_path}"
            )

        state = torch.load(
            action_path,
            map_location=self.device,
        )
        self.action_expert.load_state_dict(state)

        logger.info(
            "VLA ready: cameras=%s, VLM_layers=%d, "
            "KV_layers=%s, AE_layers=%d, reasoning_source=%s, "
            "flow_steps=%d",
            self._cam_names,
            self.vlm_num_layers,
            [x + 1 for x in self.selected_kv_layers],
            len(self.selected_kv_layers),
            self.reasoning_source,
            self.num_inference_steps,
        )

    def build_planning_prompt(
        self,
        num_history_steps: int,
        num_current_cameras: int,
        mission_command: str,
    ) -> str:
        return build_planning_user_prompt(
            self.vlm,
            num_history_steps=num_history_steps,
            num_current_cameras=num_current_cameras,
            mission_command=mission_command,
            planning_prompt=self.planning_prompt,
            reasoning_mode=str(
                self.cfg.get("reasoning_mode") or "vqa"
            ),
            reasoning_format=(
                self.planning_reasoning_format
                or self.cfg.get("reasoning_format")
            ),
        )

    @staticmethod
    def _ego_to_global(
        ego_traj: np.ndarray,
        ego_x: float,
        ego_y: float,
        ego_yaw: float,
    ) -> np.ndarray:
        cos_y = math.cos(ego_yaw)
        sin_y = math.sin(ego_yaw)
        out = np.zeros_like(ego_traj)
        for i in range(len(ego_traj)):
            dx, dy, dtheta = ego_traj[i]
            out[i, 0] = ego_x + dx * cos_y - dy * sin_y
            out[i, 1] = ego_y + dx * sin_y + dy * cos_y
            out[i, 2] = ego_yaw + dtheta
        return out

    @staticmethod
    def _global_points_to_ego(
        global_points: list,
        ego_x: float,
        ego_y: float,
        ego_yaw: float,
    ) -> np.ndarray:
        cos_neg = math.cos(-ego_yaw)
        sin_neg = math.sin(-ego_yaw)
        ego_points = []
        for pt in global_points:
            if (
                not isinstance(pt, (list, tuple, np.ndarray))
                or len(pt) < 3
            ):
                ego_points.append([0.0, 0.0, 0.0])
                continue

            gx, gy, gyaw = (
                float(pt[0]),
                float(pt[1]),
                float(pt[2]),
            )
            dx_g = gx - ego_x
            dy_g = gy - ego_y
            ego_points.append(
                [
                    dx_g * cos_neg - dy_g * sin_neg,
                    dx_g * sin_neg + dy_g * cos_neg,
                    ((gyaw - ego_yaw) + math.pi)
                    % (2 * math.pi)
                    - math.pi,
                ]
            )
        return np.asarray(
            ego_points,
            dtype=np.float32,
        )

    @classmethod
    def _build_history_trajectory(
        cls,
        hist_raw: Any,
        max_hist_pts: int,
        ego_x: float,
        ego_y: float,
        ego_yaw: float,
        *,
        frame_rate_hz: float = DEFAULT_FRAME_RATE_HZ,
        waypoint_interval_s: float = 0.5,
    ) -> np.ndarray:
        zeros = np.zeros(
            (max_hist_pts, 3),
            dtype=np.float32,
        )
        traj_history = list(hist_raw or [])
        if not traj_history:
            return zeros

        stride = frame_stride_for_dt(
            waypoint_interval_s,
            frame_rate_hz,
        )
        if stride > 1:
            traj_history = traj_history[::stride]

        if len(traj_history) > max_hist_pts:
            traj_history = traj_history[-max_hist_pts:]
        elif (
            len(traj_history) < max_hist_pts
            and len(traj_history) >= 2
        ):
            while len(traj_history) < max_hist_pts:
                first = traj_history[0]
                second = traj_history[1]
                dx = float(first[0]) - float(second[0])
                dy = float(first[1]) - float(second[1])
                dyaw = float(first[2]) - float(second[2])
                dyaw = (
                    (dyaw + math.pi)
                    % (2 * math.pi)
                    - math.pi
                )
                traj_history.insert(
                    0,
                    [
                        float(first[0]) + dx,
                        float(first[1]) + dy,
                        float(first[2]) + dyaw,
                    ],
                )

        ego_history = cls._global_points_to_ego(
            traj_history,
            ego_x,
            ego_y,
            ego_yaw,
        )
        if len(ego_history) >= max_hist_pts:
            return ego_history[:max_hist_pts]

        zeros[: len(ego_history)] = ego_history
        return zeros

    @staticmethod
    def _smooth_ego_trajectory(
        ego_traj: np.ndarray,
        cfg: Dict[str, Any],
    ) -> np.ndarray:
        traj = np.asarray(
            ego_traj,
            dtype=np.float64,
        )
        if (
            traj.ndim != 2
            or traj.shape[1] < 3
            or len(traj) == 0
        ):
            return traj

        horizon_s = float(
            cfg.get("trajectory_future_seconds", 5.0)
        )
        smooth_dt_s = float(
            cfg.get("trajectory_smoothing_dt_s", 0.1)
        )
        if horizon_s <= 0.0 or smooth_dt_s <= 0.0:
            return traj[:, :3]

        source_times = (
            np.arange(
                1,
                len(traj) + 1,
                dtype=np.float64,
            )
            * (horizon_s / len(traj))
        )
        source_times = np.concatenate(
            ([0.0], source_times)
        )
        source_traj = np.vstack(
            (
                np.zeros((1, 3), dtype=np.float64),
                traj[:, :3],
            )
        )
        source_traj[:, 2] = np.unwrap(
            source_traj[:, 2]
        )

        num_steps = int(
            round(horizon_s / smooth_dt_s)
        ) + 1
        target_times = np.linspace(
            0.0,
            horizon_s,
            max(num_steps, 2),
            dtype=np.float64,
        )

        if len(source_times) < 4:
            smoothed = np.column_stack(
                [
                    np.interp(
                        target_times,
                        source_times,
                        source_traj[:, dim],
                    )
                    for dim in range(3)
                ]
            )
        else:
            from scipy.interpolate import CubicSpline

            smoothed = np.column_stack(
                [
                    CubicSpline(
                        source_times,
                        source_traj[:, dim],
                        bc_type="natural",
                    )(target_times)
                    for dim in range(3)
                ]
            )

        smoothed[:, 2] = (
            (smoothed[:, 2] + math.pi)
            % (2 * math.pi)
            - math.pi
        )
        smoothed[0] = 0.0
        return smoothed

    def _move_inputs(
        self,
        inputs: Dict[str, Any],
    ) -> Dict[str, Any]:
        torch = self._torch
        result: Dict[str, Any] = {}
        for key, value in inputs.items():
            result[key] = (
                value.to(self.device)
                if isinstance(value, torch.Tensor)
                else value
            )
        if "prompt_length" in result:
            result["prompt_lengths"] = result.pop(
                "prompt_length"
            ).to(self.device)
        return result

    def _prepare_inputs(
        self,
        images: List[Any],
        prompt: str,
        image_contexts: List[str],
        *,
        assistant_response: Optional[str],
    ) -> Dict[str, Any]:
        return self._move_inputs(
            self.vlm.prepare_inputs(
                images=images,
                text_prompt=prompt,
                image_contexts=image_contexts,
                assistant_response=assistant_response,
            )
        )

    def _generate_reasoning(
        self,
        prompt_inputs: Dict[str, Any],
    ) -> str:
        torch = self._torch

        kwargs: Dict[str, Any] = {
            "input_ids": prompt_inputs["input_ids"],
            "attention_mask": prompt_inputs["attention_mask"],
            "max_new_tokens": self.generation_max_new_tokens,
            "do_sample": False,
            "use_cache": True,
        }
        for key in (
            "pixel_values",
            "image_grid_thw",
            "mm_token_type_ids",
        ):
            value = prompt_inputs.get(key)
            if value is not None:
                kwargs[key] = value

        use_cuda_amp = self.device.type == "cuda"
        with torch.no_grad(), torch.amp.autocast(
            "cuda",
            enabled=use_cuda_amp,
            dtype=torch.bfloat16,
        ):
            output_ids = self.vlm.model.generate(**kwargs)

        prompt_len = int(
            prompt_inputs["input_ids"].shape[1]
        )
        new_ids = output_ids[:, prompt_len:]
        text = self.vlm.processor.batch_decode(
            new_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()

        if not text:
            raise RuntimeError(
                "Frozen VLM generated an empty reasoning response."
            )
        return text

    def _extract_layer_kv(
        self,
        images: List[Any],
        image_contexts: List[str],
        prompt: str,
    ) -> Dict[int, Dict[str, Any]]:
        torch = self._torch

        if self.reasoning_source == "generated":
            prompt_inputs = self._prepare_inputs(
                images,
                prompt,
                image_contexts,
                assistant_response=None,
            )
            reasoning_text = self._generate_reasoning(
                prompt_inputs
            )
            vlm_inputs = self._prepare_inputs(
                images,
                prompt,
                image_contexts,
                assistant_response=reasoning_text,
            )
        else:
            vlm_inputs = self._prepare_inputs(
                images,
                prompt,
                image_contexts,
                assistant_response=None,
            )

        use_cuda_amp = self.device.type == "cuda"
        with torch.no_grad(), torch.amp.autocast(
            "cuda",
            enabled=use_cuda_amp,
            dtype=torch.bfloat16,
        ):
            outputs = self.vlm(
                input_ids=vlm_inputs["input_ids"],
                attention_mask=vlm_inputs["attention_mask"],
                pixel_values=vlm_inputs.get("pixel_values"),
                image_grid_thw=vlm_inputs.get("image_grid_thw"),
                mm_token_type_ids=vlm_inputs.get(
                    "mm_token_type_ids"
                ),
                labels=None,
                prompt_lengths=vlm_inputs.get(
                    "prompt_lengths"
                ),
                reasoning_token_mask=vlm_inputs.get(
                    "reasoning_token_mask"
                ),
                selected_kv_layers=self.selected_kv_layers,
                return_features=True,
            )

        layer_kv = outputs.get("layer_kv")
        if not isinstance(layer_kv, dict):
            raise RuntimeError(
                "Patched VLMBackbone returned no layer_kv."
            )

        for layer_idx in self.selected_kv_layers:
            if layer_idx not in layer_kv:
                raise RuntimeError(
                    f"Missing KV layer {layer_idx}; "
                    f"available={sorted(layer_kv.keys())}"
                )

            entry = layer_kv[layer_idx]
            entry["key"] = entry["key"].detach()
            entry["value"] = entry["value"].detach()
            if "mask" in entry:
                entry["mask"] = entry["mask"].detach()

        return layer_kv

    def __call__(
        self,
        clip_path: str,
        key_frame_idx: int,
        ego_state: Any,
    ) -> Optional[np.ndarray]:
        torch = self._torch
        cfg = self.cfg

        meta = _load_metadata(clip_path)
        frames = meta.get("frames", [])
        if key_frame_idx >= len(frames):
            return None

        num_hist = int(
            cfg.get("num_history_steps", 1) or 0
        )
        clip_hz = infer_clip_frame_rate_hz(
            meta,
            frames,
            float(
                cfg.get(
                    "frame_rate_hz",
                    DEFAULT_FRAME_RATE_HZ,
                )
            ),
        )

        flat_images, image_contexts, mission_cmd = (
            load_vlm_observation(
                clip_path,
                key_frame_idx,
                cfg,
                cam_names=self._cam_names,
            )
        )
        if not flat_images:
            return None

        prompt = self.build_planning_prompt(
            num_history_steps=num_hist,
            num_current_cameras=len(self._cam_names),
            mission_command=mission_cmd,
        )

        layer_kv = self._extract_layer_kv(
            flat_images,
            image_contexts,
            prompt,
        )

        def state_field(name: str) -> Any:
            if isinstance(ego_state, dict):
                return ego_state.get(name)
            return getattr(ego_state, name, None)

        vel_raw = state_field("velocity")
        acc_raw = state_field("acceleration")
        vel = (
            vel_raw
            if isinstance(vel_raw, dict)
            else {}
        )
        acc = (
            acc_raw
            if isinstance(acc_raw, dict)
            else {}
        )

        ego_dyn = torch.tensor(
            [
                [
                    float(vel.get("vx", 0)),
                    float(vel.get("vy", 0)),
                    float(acc.get("ax", 0)),
                    float(acc.get("ay", 0)),
                ]
            ],
            dtype=torch.float32,
            device=self.device,
        )

        max_hist_pts = int(
            cfg.get("max_history_traj_points", 6)
        )
        hist_raw = (
            state_field("trajectory_history") or []
        )
        pose_raw = state_field("pose")
        pose = (
            pose_raw
            if isinstance(pose_raw, dict)
            else {}
        )

        ego_x = float(pose.get("x", 0))
        ego_y = float(pose.get("y", 0))
        ego_yaw = float(pose.get("yaw", 0))

        hist_pts = self._build_history_trajectory(
            hist_raw,
            max_hist_pts,
            ego_x,
            ego_y,
            ego_yaw,
            frame_rate_hz=clip_hz,
            waypoint_interval_s=waypoint_dt_s(
                float(
                    cfg.get(
                        "trajectory_future_seconds",
                        5.0,
                    )
                ),
                int(
                    cfg.get(
                        "num_trajectory_points",
                        10,
                    )
                    or 10
                ),
            ),
        )

        ego_hist_traj = torch.tensor(
            np.array([hist_pts]),
            dtype=torch.float32,
            device=self.device,
        )

        use_cuda_amp = self.device.type == "cuda"
        with torch.no_grad(), torch.amp.autocast(
            "cuda",
            enabled=use_cuda_amp,
            dtype=torch.bfloat16,
        ):
            pred_traj = self.action_expert.sample(
                layer_kv=layer_kv,
                ego_state=ego_dyn,
                history_trajectory=ego_hist_traj,
                num_steps=self.num_inference_steps,
            )

        ego_traj_np = (
            pred_traj[0]
            .float()
            .cpu()
            .numpy()
        )
        ego_traj_np = self._smooth_ego_trajectory(
            ego_traj_np,
            cfg,
        )

        horizon_s = float(
            cfg.get("trajectory_future_seconds", 5.0)
        )
        smooth_dt_s = float(
            cfg.get("trajectory_smoothing_dt_s", 0.1)
        )
        expected_steps = int(
            round(horizon_s / smooth_dt_s)
        ) + 1

        if ego_traj_np.shape != (
            expected_steps,
            3,
        ):
            raise ValueError(
                f"nuVLA produced trajectory shape "
                f"{ego_traj_np.shape}; expected "
                f"({expected_steps}, 3)"
            )

        if not np.all(np.isfinite(ego_traj_np)):
            raise ValueError(
                "nuVLA produced non-finite trajectory values"
            )

        return self._ego_to_global(
            ego_traj_np,
            ego_x,
            ego_y,
            ego_yaw,
        )
