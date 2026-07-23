#!/usr/bin/env python3
"""Render GT PPO and closed-loop WestWorld PPO rollouts side by side.

Unlike render_westworld_prediction.py, this script does not feed the GT PPO
action sequence to WestWorld. The two rollouts share only the first state:

  GT:         real obs_t -> PPO -> MuJoCo -> real obs_{t+1}
  WestWorld: pred obs_t -> PPO -> WestWorld -> pred obs_{t+1}

Root x/y is not part of the Ant observation, so the WestWorld render integrates
predicted linear velocity from the shared initial root position by default.
"""

from __future__ import annotations

import argparse
import os
import random
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

from generate_ant_data import (  # noqa: E402
    ACT_DIM,
    DEFAULT_OUT,
    DEFAULT_XML,
    OBS_DIM,
    AntBackend,
)
from ppo_collect_ant_data import (  # noqa: E402
    ActorCritic,
    clamp_backend_joint_angles,
    rollout_policy_episode,
)
from ppo_train_westworld_env import (  # noqa: E402
    WestWorldOneStepSimulator,
    sanitize_ant_obs,
)
from render_westworld_prediction import (  # noqa: E402
    add_label,
    denormalize_tensor,
    find_episode_file,
    find_stats_file,
    load_cfg,
    load_episode,
    load_model,
    normalize_quat_np,
    render_sequence,
    resolve_repo_path,
)


PPO_DEFAULT_OUT = DEFAULT_OUT.parent / "ant_running_ppo"


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


def infer_ppo_action(
    policy: ActorCritic,
    obs_raw: np.ndarray,
    device: torch.device,
    *,
    deterministic: bool,
) -> np.ndarray:
    obs_t = torch.as_tensor(obs_raw, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        action_t, _, _, _ = policy.action_and_value(obs_t, deterministic=deterministic)
    return action_t.squeeze(0).detach().cpu().numpy().astype(np.float32)


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
    qpos[3:7] = normalize_quat_np(obs[None, 1:5])[0]
    qpos[7:15] = obs[11:19]

    qvel[0:3] = obs[5:8]
    qvel[3:6] = obs[8:11]
    qvel[6:14] = obs[19:27]
    return qpos, qvel


def closed_loop_westworld_rollout(
    simulator: WestWorldOneStepSimulator,
    policy: ActorCritic,
    *,
    initial_obs_raw: np.ndarray,
    initial_qpos: np.ndarray,
    initial_qvel: np.ndarray,
    gt_qpos: np.ndarray,
    timestep: float,
    states: int,
    xy_mode: str,
    device: torch.device,
    deterministic: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    obs = sanitize_ant_obs(initial_obs_raw, simulator.joint_low, simulator.joint_high)

    pred_obs: List[np.ndarray] = [obs.astype(np.float32)]
    pred_actions: List[np.ndarray] = []
    pred_qpos: List[np.ndarray] = [initial_qpos.astype(np.float64).copy()]
    pred_qvel: List[np.ndarray] = [initial_qvel.astype(np.float64).copy()]

    for step_idx in range(states - 1):
        action = infer_ppo_action(policy, obs, device, deterministic=deterministic)
        next_obs = simulator.predict_next_obs(obs, action)
        next_qpos, next_qvel = ant_obs_to_qpos_qvel(
            next_obs,
            qpos_dim=pred_qpos[0].shape[0],
            qvel_dim=pred_qvel[0].shape[0],
        )

        if xy_mode == "gt":
            next_qpos[0:2] = gt_qpos[step_idx + 1, 0:2]
        elif xy_mode == "integrate":
            next_qpos[0:2] = pred_qpos[-1][0:2] + pred_qvel[-1][0:2] * timestep
        else:
            raise ValueError(f"Unsupported xy_mode: {xy_mode}")

        pred_actions.append(action)
        pred_obs.append(next_obs.astype(np.float32))
        pred_qpos.append(next_qpos)
        pred_qvel.append(next_qvel)
        obs = next_obs

    return (
        np.stack(pred_qpos, axis=0),
        np.stack(pred_qvel, axis=0),
        np.stack(pred_obs, axis=0),
        np.stack(pred_actions, axis=0) if pred_actions else np.zeros((0, ACT_DIM), dtype=np.float32),
    )


def combine_frames(gt_frames: List[np.ndarray], pred_frames: List[np.ndarray]) -> List[np.ndarray]:
    n = min(len(gt_frames), len(pred_frames))
    out = []
    for i in range(n):
        left = add_label(gt_frames[i], "GT PPO MuJoCo")
        right = add_label(pred_frames[i], "WestWorld PPO rollout")
        out.append(np.concatenate([left, right], axis=1))
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render PPO GT trajectory and closed-loop WestWorld PPO rollout side by side."
    )
    parser.add_argument("--config-name", default="config_ant_running")
    parser.add_argument(
        "--ckpt",
        type=Path,
        default=None,
        help="WestWorld checkpoint file/directory, relative to the repository root.",
    )
    parser.add_argument(
        "--ppo-ckpt",
        type=Path,
        required=True,
        help="PPO checkpoint file/directory used for both GT rollout and WestWorld-side action inference.",
    )
    parser.add_argument(
        "--gt-source",
        choices=["ppo", "episode"],
        default="ppo",
        help="Use a fresh PPO MuJoCo rollout or a saved episode as the GT side.",
    )
    parser.add_argument(
        "--episodes",
        type=Path,
        default=PPO_DEFAULT_OUT,
        help="Saved PPO episodes path for --gt-source episode, relative to the repository root.",
    )
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument(
        "--stats",
        type=Path,
        default=None,
        help="minmax_*.pt path, relative to the repository root.",
    )
    parser.add_argument("--input-raw", action="store_true", help="Set if saved episode obs/action are raw, not normalized.")
    parser.add_argument("--xml", type=Path, default=DEFAULT_XML, help="MuJoCo XML path, relative to the repository root.")
    parser.add_argument(
        "--out",
        type=Path,
        default=SCRIPT_DIR / "renders" / "westworld_closed_loop_vs_gt.mp4",
        help="Output path, relative to the repository root.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=43, help="MuJoCo reset and stochastic-policy seed.")
    parser.add_argument("--rollout-steps", type=int, default=None, help="GT rollout horizon; defaults to checkpoint max_steps or 500.")
    parser.add_argument("--qpos-noise", type=float, default=None, help="Reset qpos noise; defaults to the PPO checkpoint value.")
    parser.add_argument("--qvel-noise", type=float, default=None, help="Reset qvel noise; defaults to the PPO checkpoint value.")
    parser.add_argument("--ppo-stochastic", action="store_true", help="Sample PPO actions instead of using the deterministic mean action.")
    parser.add_argument("--no-joint-angle-clamp", action="store_true")
    parser.add_argument("--clip-sim-obs-to-stats", action="store_true")
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
    args.ppo_ckpt = resolve_ppo_checkpoint(args.ppo_ckpt)
    if args.ckpt is not None:
        args.ckpt = resolve_repo_path(args.ckpt)
    if args.stats is not None:
        args.stats = resolve_repo_path(args.stats)
    if args.episodes is not None:
        args.episodes = resolve_repo_path(args.episodes)
    if args.stride <= 0:
        raise ValueError("--stride must be positive")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device)
    policy, saved_args, update = load_ppo_policy(args.ppo_ckpt, device)

    episode_file = find_episode_file(args.episodes) if args.gt_source == "episode" else None
    stats_file = find_stats_file(episode_file, args.stats)
    stats = torch.load(stats_file, map_location="cpu", weights_only=False)

    cfg = load_cfg(args.config_name, args.overrides)
    cfg.eval_prefix_T = 1
    world_model = load_model(cfg, args.ckpt, device)
    for param in world_model.parameters():
        param.requires_grad_(False)

    backend = AntBackend.load(args.xml.resolve())
    if backend.nu != ACT_DIM:
        raise RuntimeError(f"Expected {ACT_DIM} actuators, got {backend.nu}")
    clamp_joint_angles = not args.no_joint_angle_clamp
    if clamp_joint_angles:
        clamp_backend_joint_angles(backend)

    if args.gt_source == "ppo":
        rollout_steps = int(args.rollout_steps if args.rollout_steps is not None else saved_args.get("max_steps", 500))
        reset_qpos_noise = float(args.qpos_noise if args.qpos_noise is not None else saved_args.get("qpos_noise", 0.05))
        reset_qvel_noise = float(args.qvel_noise if args.qvel_noise is not None else saved_args.get("qvel_noise", 0.05))
        if rollout_steps <= 1:
            raise ValueError("--rollout-steps must be greater than 1")

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
        input_is_raw = True
        print(
            f"[PPO rollout] ckpt={args.ppo_ckpt} update={update} "
            f"steps={episode['obs'].shape[0]} deterministic={not args.ppo_stochastic}"
        )
    else:
        assert episode_file is not None
        episode = load_episode(episode_file, args.episode_index)
        input_is_raw = args.input_raw
        print(f"[load episode] {episode_file} episode={args.episode_index}")

    raw_obs = raw_obs_from_episode(episode, stats, input_is_raw=input_is_raw)
    raw_action = raw_action_from_episode(episode, stats, input_is_raw=input_is_raw)
    total_states = min(int(raw_obs.shape[0]), int(episode["qpos"].shape[0]), int(episode["qvel"].shape[0]))
    if args.gt_source == "episode" and args.rollout_steps is not None:
        total_states = min(total_states, int(args.rollout_steps))
    if args.max_frames is not None:
        total_states = min(total_states, int(args.max_frames) * args.stride)
    if total_states <= 1:
        raise RuntimeError("Need at least two states to render a rollout.")

    import mujoco  # type: ignore

    mj_model = mujoco.MjModel.from_xml_path(str(args.xml.resolve()))
    timestep = float(mj_model.opt.timestep)

    gt_qpos = episode["qpos"][:total_states].numpy().astype(np.float64)
    gt_qvel = episode["qvel"][:total_states].numpy().astype(np.float64)

    simulator = WestWorldOneStepSimulator(
        backend=backend,
        world_model=world_model,
        stats=stats,
        device=device,
        max_steps=total_states,
        qpos_noise=0.0,
        qvel_noise=0.0,
        clamp_joint_angles=clamp_joint_angles,
        clip_sim_obs_to_stats=args.clip_sim_obs_to_stats,
    )

    pred_qpos, pred_qvel, pred_obs, pred_actions = closed_loop_westworld_rollout(
        simulator,
        policy,
        initial_obs_raw=raw_obs[0].numpy(),
        initial_qpos=gt_qpos[0],
        initial_qvel=gt_qvel[0],
        gt_qpos=gt_qpos,
        timestep=timestep,
        states=total_states,
        xy_mode=args.xy_mode,
        device=device,
        deterministic=not args.ppo_stochastic,
    )

    gt_actions = raw_action[: pred_actions.shape[0]].numpy().astype(np.float32)
    if pred_actions.shape[0] > 0:
        action_l2 = np.linalg.norm(pred_actions - gt_actions, axis=1)
        first_l2 = float(action_l2[0])
        later_mean = float(action_l2[1:].mean()) if action_l2.shape[0] > 1 else 0.0
        print(f"[action divergence] first={first_l2:.6f} later_mean={later_mean:.6f}")

    print(f"[load stats] {stats_file}")
    print(
        "[render] "
        f"shared_state_steps=1 states={total_states} "
        f"actions_recomputed={pred_actions.shape[0]} xy_mode={args.xy_mode}"
    )

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
    print(f"[save] {args.out} frames={len(combined)} fps={args.fps}")


if __name__ == "__main__":
    main()
