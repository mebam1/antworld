#!/usr/bin/env python3
"""Evaluate and render autoregressive WestWorld Ant predictions.

Evaluation follows the zero-shot long-horizon setup:
  - split each episode into non-overlapping 150-step segments,
  - use the first 50 states as history,
  - autoregressively predict the next 100 states with GT actions,
  - report MAE and MSE against the GT future trajectory.

When rendering is enabled, the selected segment is shown side by side as GT
versus the autoregressive prediction.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import torch


os.environ.setdefault("MUJOCO_GL", "egl")

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from generate_ant_data import (  # noqa: E402
    ACT_DIM,
    DEFAULT_OUT,
    DEFAULT_XML,
    OBS_DIM,
    AntBackend,
    rot6d_to_quat,
)
from ppo_collect_ant_data import (  # noqa: E402
    ActorCritic,
    clamp_backend_joint_angles,
    rollout_policy_episode,
)
from render_westworld_prediction import (  # noqa: E402
    add_label,
    denormalize_tensor,
    find_episode_file,
    find_stats_file,
    load_cfg,
    load_model,
    normalize_tensor,
    render_sequence,
    resolve_repo_path,
)


PPO_DEFAULT_OUT = DEFAULT_OUT.parent / "ant_running_ppo"


@dataclass
class EvalSegment:
    source: str
    episode_index: int
    start: int
    obs_raw: torch.Tensor
    action_raw: torch.Tensor
    qpos: torch.Tensor | None
    qvel: torch.Tensor | None


def resolve_ppo_checkpoint(path: Path | str) -> Path:
    resolved = resolve_repo_path(path)
    if resolved.is_file():
        return resolved
    if not resolved.is_dir():
        raise FileNotFoundError(resolved)

    candidates = sorted(resolved.glob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"No PPO *.pt checkpoints found under {resolved}")
    return candidates[0]


def load_ppo_policy(
    checkpoint_path: Path,
    device: torch.device,
) -> Tuple[ActorCritic, dict, int]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "model_state_dict" not in checkpoint:
        raise KeyError(f"{checkpoint_path} does not contain model_state_dict")

    saved_args = checkpoint.get("args", {})
    hidden_size = int(saved_args.get("hidden_size", 256))
    layers = int(saved_args.get("layers", 2))

    policy = ActorCritic(OBS_DIM, ACT_DIM, hidden_size=hidden_size, layers=layers).to(device)
    policy.load_state_dict(checkpoint["model_state_dict"])
    policy.eval()

    update = int(checkpoint.get("update", -1))
    print(
        f"[load PPO] {checkpoint_path} update={update} "
        f"hidden_size={hidden_size} layers={layers}"
    )
    return policy, saved_args, update


def raw_obs_from_episode(episode: dict, stats: dict, *, input_is_raw: bool) -> torch.Tensor:
    obs = episode["obs"].float()
    if input_is_raw:
        return obs
    return denormalize_tensor(obs, stats["obs_min"].float(), stats["obs_max"].float()).float()


def raw_action_from_episode(episode: dict, stats: dict, *, input_is_raw: bool) -> torch.Tensor:
    action = episode["action"].float()
    if input_is_raw:
        return action
    return denormalize_tensor(action, stats["action_min"].float(), stats["action_max"].float()).float()


def load_episode_list(path: Path) -> list[dict]:
    episodes = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(episodes, list):
        raise TypeError(f"Expected {path} to contain a list of episodes, got {type(episodes)!r}")
    return episodes


def find_episode_files(root: Path) -> list[Path]:
    if root.is_file():
        return [root]
    files = sorted(root.glob("**/episodes_*.pt"))
    if not files:
        raise FileNotFoundError(f"No episodes_*.pt found under {root}")
    return files


def check_stats_dims(stats: dict) -> None:
    obs_dim = int(stats["obs_min"].numel())
    action_dim = int(stats["action_min"].numel())
    if obs_dim != OBS_DIM:
        raise ValueError(f"Expected stats obs dim {OBS_DIM}, got {obs_dim}")
    if action_dim != ACT_DIM:
        raise ValueError(f"Expected stats action dim {ACT_DIM}, got {action_dim}")


def segments_from_episode(
    episode: dict,
    *,
    stats: dict,
    input_is_raw: bool,
    segment_length: int,
    source: str,
    episode_index: int,
) -> list[EvalSegment]:
    obs_raw = raw_obs_from_episode(episode, stats, input_is_raw=input_is_raw)
    action_raw = raw_action_from_episode(episode, stats, input_is_raw=input_is_raw)
    if obs_raw.shape[-1] != OBS_DIM:
        raise ValueError(f"{source} episode={episode_index}: expected obs dim {OBS_DIM}, got {obs_raw.shape[-1]}")
    if action_raw.shape[-1] != ACT_DIM:
        raise ValueError(f"{source} episode={episode_index}: expected action dim {ACT_DIM}, got {action_raw.shape[-1]}")

    total_steps = min(int(obs_raw.shape[0]), int(action_raw.shape[0]))
    usable_steps = (total_steps // segment_length) * segment_length
    qpos = episode.get("qpos")
    qvel = episode.get("qvel")
    out: list[EvalSegment] = []
    for start in range(0, usable_steps, segment_length):
        end = start + segment_length
        out.append(
            EvalSegment(
                source=source,
                episode_index=episode_index,
                start=start,
                obs_raw=obs_raw[start:end].float(),
                action_raw=action_raw[start:end].float(),
                qpos=qpos[start:end].float() if qpos is not None else None,
                qvel=qvel[start:end].float() if qvel is not None else None,
            )
        )
    return out


def collect_segments_from_files(
    files: Sequence[Path],
    *,
    stats: dict,
    input_is_raw: bool,
    segment_length: int,
    episode_index: int | None,
    max_segments: int | None,
) -> list[EvalSegment]:
    segments: list[EvalSegment] = []
    for path in files:
        episodes = load_episode_list(path)
        indices = [episode_index] if episode_index is not None else list(range(len(episodes)))
        for ep_idx in indices:
            if ep_idx < 0 or ep_idx >= len(episodes):
                raise IndexError(f"episode_index={ep_idx} out of range for {len(episodes)} episodes in {path}")
            segments.extend(
                segments_from_episode(
                    episodes[ep_idx],
                    stats=stats,
                    input_is_raw=input_is_raw,
                    segment_length=segment_length,
                    source=str(path),
                    episode_index=ep_idx,
                )
            )
            if max_segments is not None and len(segments) >= max_segments:
                return segments[:max_segments]
    return segments


def collect_fresh_ppo_episode(
    *,
    args: argparse.Namespace,
    policy: ActorCritic,
    saved_args: dict,
    update: int,
    device: torch.device,
) -> dict:
    backend = AntBackend.load(args.xml.resolve())
    if backend.nu != ACT_DIM:
        raise RuntimeError(f"Expected {ACT_DIM} actuators, got {backend.nu}")
    clamp_joint_angles = not args.no_joint_angle_clamp
    if clamp_joint_angles:
        clamp_backend_joint_angles(backend)

    rollout_steps = int(args.rollout_steps if args.rollout_steps is not None else saved_args.get("max_steps", 500))
    reset_qpos_noise = float(args.qpos_noise if args.qpos_noise is not None else saved_args.get("qpos_noise", 0.05))
    reset_qvel_noise = float(args.qvel_noise if args.qvel_noise is not None else saved_args.get("qvel_noise", 0.05))
    rng = np.random.default_rng(args.seed)
    episode = rollout_policy_episode(
        backend,
        policy,
        rng,
        device,
        max_steps=rollout_steps,
        qpos_noise=reset_qpos_noise,
        qvel_noise=reset_qvel_noise,
        deterministic=not args.ppo_stochastic,
        policy_update=update,
        clamp_joint_angles=clamp_joint_angles,
    )
    print(
        f"[PPO rollout] ckpt={args.ppo_ckpt} update={update} "
        f"steps={episode['obs'].shape[0]} deterministic={not args.ppo_stochastic}"
    )
    return episode


def autoregressive_predict_future(
    model,
    segment: EvalSegment,
    *,
    stats: dict,
    history_steps: int,
    future_steps: int,
    device: torch.device,
    clip_pred_to_stats: bool,
) -> torch.Tensor:
    obs_min = stats["obs_min"].float().to(device)
    obs_max = stats["obs_max"].float().to(device)
    action_min = stats["action_min"].float().to(device)
    action_max = stats["action_max"].float().to(device)

    obs_raw = segment.obs_raw.to(device)
    action_raw = segment.action_raw.to(device)
    obs_norm_gt = normalize_tensor(obs_raw, obs_min, obs_max)
    action_norm = normalize_tensor(action_raw, action_min, action_max)

    total_steps = int(obs_norm_gt.shape[0])
    work_obs = torch.zeros_like(obs_norm_gt)
    work_obs[:history_steps] = obs_norm_gt[:history_steps]

    batch = {
        "obs": work_obs.unsqueeze(0),
        "action": action_norm.unsqueeze(0),
        "reward": torch.zeros((1, total_steps), dtype=torch.float32, device=device),
        "task": torch.full((1, total_steps), 131, dtype=torch.long, device=device),
        "obs_mask": torch.ones((1, total_steps, OBS_DIM), dtype=torch.float32, device=device),
        "action_mask": torch.ones((1, total_steps, ACT_DIM), dtype=torch.float32, device=device),
    }

    original_prefix = getattr(model, "eval_prefix_T", None)
    with torch.no_grad():
        for next_idx in range(history_steps, history_steps + future_steps):
            model.eval_prefix_T = next_idx
            batch["obs"][0].copy_(work_obs)
            pred_norm, _, _ = model(batch)
            next_obs = pred_norm[0, next_idx - 1]
            if clip_pred_to_stats:
                next_obs = next_obs.clamp(0.0, 1.0)
            work_obs[next_idx] = next_obs
    if original_prefix is not None:
        model.eval_prefix_T = original_prefix

    pred_future = denormalize_tensor(work_obs[history_steps : history_steps + future_steps], obs_min, obs_max)
    return pred_future.detach().cpu().float()


def evaluate_segments(
    model,
    segments: Sequence[EvalSegment],
    *,
    stats: dict,
    history_steps: int,
    future_steps: int,
    device: torch.device,
    clip_pred_to_stats: bool,
) -> Tuple[float, float, list[torch.Tensor]]:
    mae_sum = 0.0
    mse_sum = 0.0
    count = 0
    predictions: list[torch.Tensor] = []

    for idx, segment in enumerate(segments):
        pred_future = autoregressive_predict_future(
            model,
            segment,
            stats=stats,
            history_steps=history_steps,
            future_steps=future_steps,
            device=device,
            clip_pred_to_stats=clip_pred_to_stats,
        )
        gt_future = segment.obs_raw[history_steps : history_steps + future_steps].float()
        diff = pred_future - gt_future
        mae_sum += float(diff.abs().sum().item())
        mse_sum += float(diff.pow(2).sum().item())
        count += int(diff.numel())
        predictions.append(pred_future)
        if (idx + 1) == 1 or (idx + 1) == len(segments) or (idx + 1) % max(1, len(segments) // 10) == 0:
            print(f"[eval] segment {idx + 1}/{len(segments)}")

    if count == 0:
        raise RuntimeError("No valid prediction elements were evaluated.")
    return mae_sum / count, mse_sum / count, predictions


def ant_obs_to_qpos_qvel(
    obs_raw: np.ndarray,
    *,
    qpos_dim: int,
    qvel_dim: int,
) -> Tuple[np.ndarray, np.ndarray]:
    obs = np.nan_to_num(obs_raw.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    qpos = np.zeros((qpos_dim,), dtype=np.float64)
    qvel = np.zeros((qvel_dim,), dtype=np.float64)

    if qpos_dim < 15 or qvel_dim < 14:
        raise RuntimeError(f"Unexpected Ant state sizes: qpos={qpos_dim}, qvel={qvel_dim}")

    qpos[2] = np.clip(obs[0], 0.05, 5.0)
    qpos[3:7] = rot6d_to_quat(obs[1:7]).astype(np.float64)
    qpos[7:15] = obs[13:21]

    qvel[0:3] = obs[7:10]
    qvel[3:6] = obs[10:13]
    qvel[6:14] = obs[21:29]
    return qpos, qvel


def reconstruct_qpos_qvel_from_obs(
    obs_raw: torch.Tensor,
    *,
    qpos_dim: int,
    qvel_dim: int,
    timestep: float,
) -> Tuple[np.ndarray, np.ndarray]:
    qpos = []
    qvel = []
    for row in obs_raw.numpy():
        qpos_t, qvel_t = ant_obs_to_qpos_qvel(row, qpos_dim=qpos_dim, qvel_dim=qvel_dim)
        qpos.append(qpos_t)
        qvel.append(qvel_t)
    qpos_np = np.stack(qpos, axis=0)
    qvel_np = np.stack(qvel, axis=0)
    for i in range(1, qpos_np.shape[0]):
        qpos_np[i, 0:2] = qpos_np[i - 1, 0:2] + qvel_np[i - 1, 0:2] * timestep
    return qpos_np, qvel_np


def predicted_render_states(
    segment: EvalSegment,
    pred_future: torch.Tensor,
    *,
    history_steps: int,
    future_steps: int,
    timestep: float,
    xy_mode: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    total_steps = history_steps + future_steps
    obs_pred = segment.obs_raw[:total_steps].clone()
    obs_pred[history_steps:total_steps] = pred_future[:future_steps]

    if segment.qpos is not None and segment.qvel is not None:
        gt_qpos = segment.qpos[:total_steps].numpy().astype(np.float64)
        gt_qvel = segment.qvel[:total_steps].numpy().astype(np.float64)
    else:
        gt_qpos, gt_qvel = reconstruct_qpos_qvel_from_obs(
            segment.obs_raw[:total_steps],
            qpos_dim=15,
            qvel_dim=14,
            timestep=timestep,
        )

    pred_qpos = gt_qpos.copy()
    pred_qvel = gt_qvel.copy()
    for t in range(history_steps, total_steps):
        qpos_t, qvel_t = ant_obs_to_qpos_qvel(
            obs_pred[t].numpy(),
            qpos_dim=gt_qpos.shape[1],
            qvel_dim=gt_qvel.shape[1],
        )
        if xy_mode == "gt":
            qpos_t[0:2] = gt_qpos[t, 0:2]
        elif xy_mode == "integrate":
            if t == history_steps:
                qpos_t[0:2] = gt_qpos[t, 0:2]
            else:
                qpos_t[0:2] = pred_qpos[t - 1, 0:2] + pred_qvel[t - 1, 0:2] * timestep
        else:
            raise ValueError(f"Unsupported xy_mode: {xy_mode}")
        pred_qpos[t] = qpos_t
        pred_qvel[t] = qvel_t

    return gt_qpos, gt_qvel, pred_qpos, pred_qvel


def combine_frames(gt_frames: List[np.ndarray], pred_frames: List[np.ndarray]) -> List[np.ndarray]:
    n = min(len(gt_frames), len(pred_frames))
    out = []
    for i in range(n):
        left = add_label(gt_frames[i], "GT segment")
        right = add_label(pred_frames[i], "WestWorld AR prediction")
        out.append(np.concatenate([left, right], axis=1))
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate 150/50/100 autoregressive Ant WestWorld predictions.")
    parser.add_argument("--config-name", default="config_ant_running")
    parser.add_argument(
        "--ckpt",
        type=Path,
        default=None,
        help="WestWorld checkpoint file/directory, relative to the repository root.",
    )
    parser.add_argument(
        "--gt-source",
        choices=["episode", "ppo"],
        default="episode",
        help="Evaluate saved episodes or collect one fresh real PPO rollout.",
    )
    parser.add_argument(
        "--ppo-ckpt",
        type=Path,
        default=None,
        help="PPO checkpoint file/directory required when --gt-source ppo.",
    )
    parser.add_argument("--episodes", type=Path, default=PPO_DEFAULT_OUT, help="Saved PPO episodes path.")
    parser.add_argument("--episode-index", type=int, default=None, help="Restrict saved-episode eval to one episode index.")
    parser.add_argument("--stats", type=Path, default=None, help="minmax_*.pt path.")
    parser.add_argument("--input-raw", action="store_true", help="Set if saved episode obs/action are raw, not normalized.")
    parser.add_argument("--xml", type=Path, default=DEFAULT_XML, help="MuJoCo XML path.")
    parser.add_argument("--out", type=Path, default=SCRIPT_DIR / "renders" / "westworld_closed_loop_eval.mp4")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--rollout-steps", type=int, default=None, help="Fresh PPO rollout horizon.")
    parser.add_argument("--qpos-noise", type=float, default=None)
    parser.add_argument("--qvel-noise", type=float, default=None)
    parser.add_argument("--ppo-stochastic", action="store_true")
    parser.add_argument("--no-joint-angle-clamp", action="store_true")
    parser.add_argument("--segment-length", type=int, default=150)
    parser.add_argument("--history-steps", type=int, default=50)
    parser.add_argument("--future-steps", type=int, default=100)
    parser.add_argument("--max-eval-segments", type=int, default=None)
    parser.add_argument("--render-segment-index", type=int, default=0)
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--clip-pred-to-stats", action="store_true")
    parser.add_argument(
        "--clip-sim-obs-to-stats",
        action="store_true",
        help="Deprecated alias for --clip-pred-to-stats.",
    )
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--camera", default="track", help="MuJoCo camera name. Use empty string for default camera.")
    parser.add_argument("--xy-mode", choices=["integrate", "gt"], default="integrate")
    parser.add_argument("overrides", nargs="*", help="Optional Hydra overrides.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.xml = resolve_repo_path(args.xml)
    args.out = resolve_repo_path(args.out)
    if args.ckpt is not None:
        args.ckpt = resolve_repo_path(args.ckpt)
    if args.stats is not None:
        args.stats = resolve_repo_path(args.stats)
    if args.episodes is not None:
        args.episodes = resolve_repo_path(args.episodes)
    if args.ppo_ckpt is not None:
        args.ppo_ckpt = resolve_ppo_checkpoint(args.ppo_ckpt)

    if args.segment_length <= 0 or args.history_steps <= 0 or args.future_steps <= 0:
        raise ValueError("--segment-length, --history-steps, and --future-steps must be positive")
    if args.history_steps + args.future_steps > args.segment_length:
        raise ValueError("--history-steps + --future-steps must be <= --segment-length")
    if args.stride <= 0:
        raise ValueError("--stride must be positive")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device)

    if args.gt_source == "ppo":
        if args.ppo_ckpt is None:
            raise ValueError("--ppo-ckpt is required when --gt-source ppo")
        policy, saved_args, update = load_ppo_policy(args.ppo_ckpt, device)
        episode_file = None
        stats_file = find_stats_file(args.episodes, args.stats)
    else:
        episode_files = find_episode_files(args.episodes)
        episode_file = find_episode_file(args.episodes)
        stats_file = find_stats_file(episode_file, args.stats)
        policy = None
        saved_args = {}
        update = -1

    stats = torch.load(stats_file, map_location="cpu", weights_only=False)
    check_stats_dims(stats)

    cfg = load_cfg(args.config_name, args.overrides)
    cfg.eval_prefix_T = args.history_steps
    world_model = load_model(cfg, args.ckpt, device)
    for param in world_model.parameters():
        param.requires_grad_(False)

    if args.gt_source == "ppo":
        assert policy is not None
        episode = collect_fresh_ppo_episode(
            args=args,
            policy=policy,
            saved_args=saved_args,
            update=update,
            device=device,
        )
        segments = segments_from_episode(
            episode,
            stats=stats,
            input_is_raw=True,
            segment_length=args.segment_length,
            source=str(args.ppo_ckpt),
            episode_index=0,
        )
        if args.max_eval_segments is not None:
            segments = segments[: args.max_eval_segments]
    else:
        segments = collect_segments_from_files(
            episode_files,
            stats=stats,
            input_is_raw=args.input_raw,
            segment_length=args.segment_length,
            episode_index=args.episode_index,
            max_segments=args.max_eval_segments,
        )

    if not segments:
        raise RuntimeError(
            f"No complete {args.segment_length}-step segments found. "
            "Increase rollout length or use longer saved episodes."
        )

    clip_pred_to_stats = bool(args.clip_pred_to_stats or args.clip_sim_obs_to_stats)
    mae, mse, predictions = evaluate_segments(
        world_model,
        segments,
        stats=stats,
        history_steps=args.history_steps,
        future_steps=args.future_steps,
        device=device,
        clip_pred_to_stats=clip_pred_to_stats,
    )

    print(f"[load stats] {stats_file}")
    print(
        "[closed-loop eval] "
        f"segments={len(segments)} segment_length={args.segment_length} "
        f"history={args.history_steps} future={args.future_steps} "
        f"MAE={mae:.6f} MSE={mse:.6f}"
    )

    if args.no_render:
        return

    render_idx = max(0, min(int(args.render_segment_index), len(segments) - 1))
    segment = segments[render_idx]
    pred_future = predictions[render_idx]

    import mujoco  # type: ignore

    mj_model = mujoco.MjModel.from_xml_path(str(args.xml.resolve()))
    timestep = float(mj_model.opt.timestep)

    gt_qpos, gt_qvel, pred_qpos, pred_qvel = predicted_render_states(
        segment,
        pred_future,
        history_steps=args.history_steps,
        future_steps=args.future_steps,
        timestep=timestep,
        xy_mode=args.xy_mode,
    )

    if args.max_frames is not None:
        limit = args.max_frames * max(1, args.stride)
        gt_qpos, gt_qvel = gt_qpos[:limit], gt_qvel[:limit]
        pred_qpos, pred_qvel = pred_qpos[:limit], pred_qvel[:limit]

    camera = args.camera if args.camera else None
    gt_frames = render_sequence(
        args.xml.resolve(),
        gt_qpos,
        gt_qvel,
        width=args.width,
        height=args.height,
        camera=camera,
        stride=args.stride,
    )
    pred_frames = render_sequence(
        args.xml.resolve(),
        pred_qpos,
        pred_qvel,
        width=args.width,
        height=args.height,
        camera=camera,
        stride=args.stride,
    )
    combined = combine_frames(gt_frames, pred_frames)
    if not combined:
        raise RuntimeError("No frames rendered.")

    import imageio.v2 as imageio

    args.out.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(args.out, combined, fps=args.fps)
    print(
        f"[save] {args.out} frames={len(combined)} fps={args.fps} "
        f"segment={render_idx} source={segment.source} episode={segment.episode_index} start={segment.start}"
    )


if __name__ == "__main__":
    main()
