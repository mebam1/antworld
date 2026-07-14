#!/usr/bin/env python3
"""Render GT Ant trajectory and WestWorld prediction side by side.

This script expects PPO-collected episodes from Ant/ppo_collect_ant_data.py.
Those episodes contain both WestWorld training tensors (obs/action/task) and raw
MuJoCo states (qpos/qvel). WestWorld predicts observation channels, so the
predicted render reconstructs qpos/qvel from the predicted Ant observation.
Root x/y is not an observation channel; by default it is integrated from the
predicted linear velocity starting at the GT prefix boundary.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch


os.environ.setdefault("MUJOCO_GL", "egl")

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from generate_ant_data import DEFAULT_OUT, DEFAULT_XML, OBS_DIM, ACT_DIM  # noqa: E402


PPO_DEFAULT_OUT = DEFAULT_OUT.parent / "ant_running_ppo"


def find_episode_file(root: Path) -> Path:
    if root.is_file():
        return root
    files = sorted(root.glob("**/episodes_*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        raise FileNotFoundError(f"No episodes_*.pt found under {root}")
    return files[0]


def find_stats_file(episodes_path: Path, explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit
    root = episodes_path.parent if episodes_path.is_file() else episodes_path
    files = sorted(root.glob("minmax_*.pt"))
    if not files:
        raise FileNotFoundError(f"No minmax_*.pt found under {root}; pass --stats explicitly.")
    return files[0]


def load_episode(path: Path, episode_index: int) -> dict:
    episodes = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(episodes, list):
        raise TypeError(f"Expected {path} to contain a list of episodes, got {type(episodes)!r}")
    if episode_index < 0 or episode_index >= len(episodes):
        raise IndexError(f"episode_index={episode_index} out of range for {len(episodes)} episodes")
    episode = episodes[episode_index]
    required = {"obs", "action", "task", "qpos", "qvel"}
    missing = sorted(required.difference(episode.keys()))
    if missing:
        raise KeyError(f"Episode is missing required keys for prediction rendering: {missing}")
    return episode


def load_cfg(config_name: str, overrides: List[str]):
    from hydra import compose, initialize
    from omegaconf import OmegaConf

    with initialize(version_base=None, config_path="../configs"):
        cfg = compose(config_name=config_name, overrides=overrides)
    OmegaConf.set_struct(cfg, False)
    return cfg


def load_model(cfg, ckpt_path: str | None, device: torch.device):
    from models import build_model

    model = build_model(cfg).to(device)
    model.eval()

    ckpt = ckpt_path or cfg.get("ckpt_path", None)
    if not ckpt:
        raise ValueError("No checkpoint path supplied. Use --ckpt or set ckpt_path in the config.")
    if not os.path.exists(ckpt):
        raise FileNotFoundError(ckpt)

    state = torch.load(ckpt, map_location=device)
    state_dict = state.get("state_dict", state)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[load ckpt] missing keys: {len(missing)}")
    if unexpected:
        print(f"[load ckpt] unexpected keys: {len(unexpected)}")
    print(f"[load ckpt] {ckpt}")
    return model


def normalize_tensor(x: torch.Tensor, mn: torch.Tensor, mx: torch.Tensor) -> torch.Tensor:
    return ((x - mn) / (mx - mn).clamp_min(1e-6)).clamp(0.0, 1.0)


def denormalize_tensor(x: torch.Tensor, mn: torch.Tensor, mx: torch.Tensor) -> torch.Tensor:
    return x * (mx - mn).clamp_min(1e-6) + mn


def make_batch(episode: dict, stats: dict, *, input_is_raw: bool, device: torch.device) -> dict:
    obs = episode["obs"].float()
    action = episode["action"].float()
    task = episode["task"].long()

    if input_is_raw:
        obs = normalize_tensor(obs, stats["obs_min"].float(), stats["obs_max"].float())
        action = normalize_tensor(action, stats["action_min"].float(), stats["action_max"].float())

    if obs.shape[-1] != OBS_DIM:
        raise ValueError(f"Expected obs dim {OBS_DIM}, got {obs.shape[-1]}")
    if action.shape[-1] != ACT_DIM:
        raise ValueError(f"Expected action dim {ACT_DIM}, got {action.shape[-1]}")

    batch = {
        "obs": obs.unsqueeze(0).to(device),
        "action": action.unsqueeze(0).to(device),
        "reward": episode.get("reward", torch.zeros(obs.shape[0])).float().unsqueeze(0).to(device),
        "task": task.unsqueeze(0).to(device),
        "obs_mask": torch.ones((1, obs.shape[0], obs.shape[-1]), dtype=torch.float32, device=device),
        "action_mask": torch.ones((1, action.shape[0], action.shape[-1]), dtype=torch.float32, device=device),
    }
    return batch


def predict_obs_raw(model, batch: dict, stats: dict) -> torch.Tensor:
    with torch.no_grad():
        pred_norm, _, _ = model(batch)
    obs_min = stats["obs_min"].float().to(pred_norm.device)
    obs_max = stats["obs_max"].float().to(pred_norm.device)
    pred_raw = denormalize_tensor(pred_norm[0], obs_min, obs_max)
    return pred_raw.detach().cpu()


def normalize_quat_np(q: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    norm = np.maximum(norm, 1e-8)
    out = q / norm
    bad = ~np.isfinite(out).all(axis=-1)
    if bad.any():
        out[bad] = np.array([1.0, 0.0, 0.0, 0.0], dtype=out.dtype)
    return out


def pred_obs_to_qpos_qvel(
    pred_obs_raw: torch.Tensor,
    gt_qpos: torch.Tensor,
    *,
    prefix_t: int,
    timestep: float,
    xy_mode: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert predicted raw Ant observations to MuJoCo qpos/qvel.

    pred_obs_raw[t-1] predicts observation at environment step t.
    Returned arrays align with GT qpos/qvel[prefix_t:].
    """
    pred = pred_obs_raw[prefix_t - 1 :].numpy().astype(np.float64)
    gt_qpos_np = gt_qpos.numpy().astype(np.float64)
    frames = min(pred.shape[0], max(0, gt_qpos_np.shape[0] - prefix_t))
    pred = pred[:frames]
    gt_aligned = gt_qpos_np[prefix_t : prefix_t + frames]

    qpos = np.zeros((frames, gt_qpos_np.shape[1]), dtype=np.float64)
    qvel = np.zeros((frames, 14), dtype=np.float64)
    if frames == 0:
        return qpos, qvel

    qpos[:, 2] = np.clip(pred[:, 0], 0.05, 5.0)
    qpos[:, 3:7] = normalize_quat_np(pred[:, 1:5])
    qpos[:, 7:15] = pred[:, 11:19]

    qvel[:, 0:3] = pred[:, 5:8]
    qvel[:, 3:6] = pred[:, 8:11]
    qvel[:, 6:14] = pred[:, 19:27]

    if xy_mode == "gt":
        qpos[:, 0:2] = gt_aligned[:, 0:2]
    elif xy_mode == "integrate":
        qpos[0, 0:2] = gt_aligned[0, 0:2]
        for i in range(1, frames):
            qpos[i, 0:2] = qpos[i - 1, 0:2] + qvel[i - 1, 0:2] * timestep
    else:
        raise ValueError(f"Unsupported xy_mode: {xy_mode}")

    qpos = np.nan_to_num(qpos, nan=0.0, posinf=0.0, neginf=0.0)
    qvel = np.nan_to_num(qvel, nan=0.0, posinf=0.0, neginf=0.0)
    return qpos, qvel


def render_sequence(
    xml_path: Path,
    qpos: np.ndarray,
    qvel: np.ndarray,
    *,
    width: int,
    height: int,
    camera: str | None,
    stride: int,
) -> List[np.ndarray]:
    import mujoco  # type: ignore

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), int(width))
    model.vis.global_.offheight = max(int(model.vis.global_.offheight), int(height))
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=height, width=width)
    frames: List[np.ndarray] = []

    for idx in range(0, qpos.shape[0], stride):
        data.qpos[:] = qpos[idx]
        data.qvel[:] = qvel[idx]
        mujoco.mj_forward(model, data)
        if camera:
            renderer.update_scene(data, camera=camera)
        else:
            renderer.update_scene(data)
        frames.append(renderer.render().copy())

    renderer.close()
    return frames


def add_label(frame: np.ndarray, label: str) -> np.ndarray:
    from PIL import Image, ImageDraw

    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image)
    draw.rectangle((8, 8, 8 + len(label) * 9 + 12, 34), fill=(0, 0, 0))
    draw.text((14, 14), label, fill=(255, 255, 255))
    return np.asarray(image)


def combine_frames(gt_frames: List[np.ndarray], pred_frames: List[np.ndarray]) -> List[np.ndarray]:
    n = min(len(gt_frames), len(pred_frames))
    out = []
    for i in range(n):
        left = add_label(gt_frames[i], "GT PPO trajectory")
        right = add_label(pred_frames[i], "WestWorld prediction")
        out.append(np.concatenate([left, right], axis=1))
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render PPO GT trajectory and WestWorld prediction side by side.")
    parser.add_argument("--config-name", default="config_ant_running")
    parser.add_argument("--ckpt", default=None, help="WestWorld checkpoint path. Defaults to config ckpt_path.")
    parser.add_argument("--episodes", type=Path, default=PPO_DEFAULT_OUT, help="PPO episodes_*.pt file or directory.")
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--stats", type=Path, default=None, help="minmax_*.pt for denormalizing WestWorld predictions.")
    parser.add_argument("--input-raw", action="store_true", help="Set if the episode obs/action are raw, not normalized.")
    parser.add_argument("--xml", type=Path, default=DEFAULT_XML)
    parser.add_argument("--out", type=Path, default=SCRIPT_DIR / "renders" / "westworld_vs_gt.mp4")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--camera", default="track", help="MuJoCo camera name. Use empty string for default camera.")
    parser.add_argument("--xy-mode", choices=["integrate", "gt"], default="integrate")
    parser.add_argument("overrides", nargs="*", help="Optional Hydra overrides, e.g. eval_prefix_T=25")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    episode_file = find_episode_file(args.episodes)
    stats_file = find_stats_file(episode_file, args.stats)
    episode = load_episode(episode_file, args.episode_index)
    stats = torch.load(stats_file, map_location="cpu", weights_only=False)

    cfg = load_cfg(args.config_name, args.overrides)
    device = torch.device(args.device)
    model = load_model(cfg, args.ckpt, device)

    batch = make_batch(episode, stats, input_is_raw=args.input_raw, device=device)
    pred_obs_raw = predict_obs_raw(model, batch, stats)

    total_steps = int(episode["obs"].shape[0])
    prefix_t = max(1, min(int(cfg.get("eval_prefix_T", 50)), total_steps // 2))
    import mujoco  # type: ignore

    mj_model = mujoco.MjModel.from_xml_path(str(args.xml.resolve()))
    timestep = float(mj_model.opt.timestep)

    pred_qpos, pred_qvel = pred_obs_to_qpos_qvel(
        pred_obs_raw,
        episode["qpos"].float(),
        prefix_t=prefix_t,
        timestep=timestep,
        xy_mode=args.xy_mode,
    )
    frames = pred_qpos.shape[0]
    gt_qpos = episode["qpos"][prefix_t : prefix_t + frames].numpy().astype(np.float64)
    gt_qvel = episode["qvel"][prefix_t : prefix_t + frames].numpy().astype(np.float64)

    if args.max_frames is not None:
        limit = args.max_frames * max(1, args.stride)
        gt_qpos, gt_qvel = gt_qpos[:limit], gt_qvel[:limit]
        pred_qpos, pred_qvel = pred_qpos[:limit], pred_qvel[:limit]

    print(f"[load episode] {episode_file} episode={args.episode_index}")
    print(f"[load stats] {stats_file}")
    print(f"[render] prefix_t={prefix_t} frames={min(len(gt_qpos), len(pred_qpos))} xy_mode={args.xy_mode}")

    camera = args.camera if args.camera else None
    stride = max(1, args.stride)
    gt_frames = render_sequence(
        args.xml.resolve(),
        gt_qpos,
        gt_qvel,
        width=args.width,
        height=args.height,
        camera=camera,
        stride=stride,
    )
    pred_frames = render_sequence(
        args.xml.resolve(),
        pred_qpos,
        pred_qvel,
        width=args.width,
        height=args.height,
        camera=camera,
        stride=stride,
    )
    combined = combine_frames(gt_frames, pred_frames)
    if not combined:
        raise RuntimeError("No frames rendered.")

    import imageio.v2 as imageio

    args.out.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(args.out, combined, fps=args.fps)
    print(f"[save] {args.out} frames={len(combined)} fps={args.fps}")


if __name__ == "__main__":
    main()
