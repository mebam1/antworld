#!/usr/bin/env python3
"""Train an Ant PPO policy inside a frozen WestWorld simulator.

The real MuJoCo Ant environment is used only to sample initial states. After
reset, each transition is generated as:

  current obs -> PPO action -> WestWorld(obs, action) -> next obs

The PPO policy is trained from scratch against rewards computed from the
WestWorld-predicted next observation.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from generate_ant_data import (  # noqa: E402
    ACT_DIM,
    DEFAULT_OUT,
    DEFAULT_XML,
    OBS_DIM,
    TASK_ID,
    AntBackend,
    ant_done,
    ant_reward,
    canonicalize_rot6d,
    rot6d_to_mat,
)
from ppo_collect_ant_data import (  # noqa: E402
    ActorCritic,
    clamp_backend_joint_angles,
    ppo_update,
    reset_backend,
)


PPO_DEFAULT_OUT = DEFAULT_OUT.parent / "ant_running_ppo"
DEFAULT_POLICY_DIR = SCRIPT_DIR / "ppo_westworld_checkpoints"


def find_stats_file(explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit

    candidates = [
        PPO_DEFAULT_OUT / "minmax_ant_running_ppo.pt",
        DEFAULT_OUT / "minmax_ant_running.pt",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate

    for root in [PPO_DEFAULT_OUT, DEFAULT_OUT]:
        files = sorted(root.glob("minmax_*.pt"))
        if files:
            return files[0]

    raise FileNotFoundError("No minmax_*.pt found. Pass --stats explicitly.")


def load_cfg(config_name: str, overrides: list[str]):
    from hydra import compose, initialize
    from omegaconf import OmegaConf

    with initialize(version_base=None, config_path="../configs"):
        cfg = compose(config_name=config_name, overrides=overrides)
    OmegaConf.set_struct(cfg, False)
    cfg.eval_prefix_T = 1
    return cfg


def load_world_model(cfg, ckpt_path: str | None, device: torch.device):
    from models import build_model

    # MambaConfig otherwise defaults its newly-created layers to CUDA even when
    # the caller explicitly selected CPU (or a non-default CUDA device).
    if cfg.method.get("mamba_cfg", None) is not None:
        cfg.method.mamba_cfg.device = str(device)

    model = build_model(cfg).to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)

    ckpt = ckpt_path or cfg.get("ckpt_path", None)
    if not ckpt:
        raise ValueError("No WestWorld checkpoint supplied. Use --ckpt or set ckpt_path in the config.")
    if not Path(ckpt).is_file():
        raise FileNotFoundError(ckpt)

    state = torch.load(ckpt, map_location=device)
    state_dict = state.get("state_dict", state)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[load world model] missing keys: {len(missing)}")
    if unexpected:
        print(f"[load world model] unexpected keys: {len(unexpected)}")
    print(f"[load world model] {ckpt}")
    return model


def normalize_tensor(x: torch.Tensor, mn: torch.Tensor, mx: torch.Tensor) -> torch.Tensor:
    return ((x - mn) / (mx - mn).clamp_min(1e-6)).clamp(0.0, 1.0)


def denormalize_tensor(x: torch.Tensor, mn: torch.Tensor, mx: torch.Tensor) -> torch.Tensor:
    return x * (mx - mn).clamp_min(1e-6) + mn


def joint_angle_bounds_from_backend(backend: AntBackend) -> Tuple[np.ndarray, np.ndarray]:
    low = np.full((ACT_DIM,), -np.inf, dtype=np.float32)
    high = np.full((ACT_DIM,), np.inf, dtype=np.float32)

    model = backend.model
    jnt_limited = np.asarray(getattr(model, "jnt_limited", []))
    jnt_range = np.asarray(getattr(model, "jnt_range", []), dtype=np.float64)
    jnt_qposadr = np.asarray(getattr(model, "jnt_qposadr", []), dtype=np.int64)
    joint_count = min(jnt_limited.shape[0], jnt_range.shape[0], jnt_qposadr.shape[0])

    for joint_idx in range(joint_count):
        if not bool(jnt_limited[joint_idx]):
            continue
        adr = int(jnt_qposadr[joint_idx])
        if 7 <= adr < 15:
            low[adr - 7] = float(jnt_range[joint_idx, 0])
            high[adr - 7] = float(jnt_range[joint_idx, 1])

    return low, high


def sanitize_ant_obs(obs: np.ndarray, joint_low: np.ndarray, joint_high: np.ndarray) -> np.ndarray:
    out = np.nan_to_num(obs.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    out[0] = np.clip(out[0], 0.05, 5.0)
    out[1:7] = canonicalize_rot6d(out[1:7])
    out[13:21] = np.clip(out[13:21], joint_low, joint_high)

    rot = rot6d_to_mat(out[1:7])
    out[29] = float(rot[2, 2])
    out[30] = float(rot[0, 0])
    return out


class WestWorldOneStepSimulator:
    def __init__(
        self,
        *,
        backend: AntBackend,
        world_model,
        stats: dict,
        device: torch.device,
        max_steps: int,
        qpos_noise: float,
        qvel_noise: float,
        clamp_joint_angles: bool,
        clip_sim_obs_to_stats: bool,
    ) -> None:
        self.backend = backend
        self.world_model = world_model
        self.device = device
        self.max_steps = int(max_steps)
        self.qpos_noise = float(qpos_noise)
        self.qvel_noise = float(qvel_noise)
        self.clamp_joint_angles = bool(clamp_joint_angles)
        self.clip_sim_obs_to_stats = bool(clip_sim_obs_to_stats)
        self.ep_step = 0

        self.obs_min = stats["obs_min"].float().to(device)
        self.obs_max = stats["obs_max"].float().to(device)
        self.action_min = stats["action_min"].float().to(device)
        self.action_max = stats["action_max"].float().to(device)

        self.joint_low, self.joint_high = joint_angle_bounds_from_backend(backend)
        self.obs_seq = torch.zeros((1, 2, OBS_DIM), dtype=torch.float32, device=device)
        self.action_seq = torch.zeros((1, 2, ACT_DIM), dtype=torch.float32, device=device)
        self.reward_seq = torch.zeros((1, 2), dtype=torch.float32, device=device)
        self.task_seq = torch.full((1, 2), TASK_ID, dtype=torch.long, device=device)
        self.obs_mask = torch.ones((1, 2, OBS_DIM), dtype=torch.float32, device=device)
        self.action_mask = torch.ones((1, 2, ACT_DIM), dtype=torch.float32, device=device)

    def reset(self, rng: np.random.Generator) -> np.ndarray:
        obs = reset_backend(
            self.backend,
            rng,
            self.qpos_noise,
            self.qvel_noise,
            self.clamp_joint_angles,
        )
        self.ep_step = 0
        return sanitize_ant_obs(obs, self.joint_low, self.joint_high)

    def predict_next_obs(self, obs_raw: np.ndarray, action_raw: np.ndarray) -> np.ndarray:
        obs_t = torch.as_tensor(obs_raw, dtype=torch.float32, device=self.device)
        action_t = torch.as_tensor(action_raw, dtype=torch.float32, device=self.device)
        obs_norm = normalize_tensor(obs_t, self.obs_min, self.obs_max)
        action_norm = normalize_tensor(action_t, self.action_min, self.action_max)

        self.obs_seq.zero_()
        self.action_seq.zero_()
        self.reward_seq.zero_()
        self.obs_seq[0, 0] = obs_norm
        self.obs_seq[0, 1] = obs_norm
        self.action_seq[0, 0] = action_norm
        self.action_seq[0, 1] = action_norm

        batch = {
            "obs": self.obs_seq,
            "action": self.action_seq,
            "reward": self.reward_seq,
            "task": self.task_seq,
            "obs_mask": self.obs_mask,
            "action_mask": self.action_mask,
        }
        with torch.no_grad():
            pred_norm, _, _ = self.world_model(batch)

        next_obs = denormalize_tensor(pred_norm[0, 0], self.obs_min, self.obs_max)
        if self.clip_sim_obs_to_stats:
            next_obs = torch.minimum(torch.maximum(next_obs, self.obs_min), self.obs_max)
        return sanitize_ant_obs(next_obs.detach().cpu().numpy(), self.joint_low, self.joint_high)

    def step(self, obs_raw: np.ndarray, action_raw: np.ndarray) -> Tuple[np.ndarray, float, bool]:
        next_obs = self.predict_next_obs(obs_raw, action_raw)
        reward = ant_reward(next_obs)
        done = ant_done(next_obs, self.ep_step, self.max_steps)
        self.ep_step += 1
        return next_obs, float(reward), bool(done)


def collect_westworld_rollout(
    simulator: WestWorldOneStepSimulator,
    policy: ActorCritic,
    rng: np.random.Generator,
    device: torch.device,
    *,
    rollout_steps: int,
) -> Dict[str, torch.Tensor]:
    obs_buf = torch.zeros((rollout_steps, OBS_DIM), dtype=torch.float32, device=device)
    act_buf = torch.zeros((rollout_steps, ACT_DIM), dtype=torch.float32, device=device)
    logp_buf = torch.zeros(rollout_steps, dtype=torch.float32, device=device)
    rew_buf = torch.zeros(rollout_steps, dtype=torch.float32, device=device)
    done_buf = torch.zeros(rollout_steps, dtype=torch.float32, device=device)
    val_buf = torch.zeros(rollout_steps, dtype=torch.float32, device=device)

    obs = simulator.reset(rng)
    for step in range(rollout_steps):
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            action_t, logp_t, _, value_t = policy.action_and_value(obs_t)

        action_np = action_t.squeeze(0).detach().cpu().numpy().astype(np.float32)
        next_obs, reward, done = simulator.step(obs, action_np)

        obs_buf[step] = obs_t.squeeze(0)
        act_buf[step] = action_t.squeeze(0)
        logp_buf[step] = logp_t.squeeze(0)
        rew_buf[step] = reward
        done_buf[step] = float(done)
        val_buf[step] = value_t.squeeze(0)

        obs = simulator.reset(rng) if done else next_obs

    with torch.no_grad():
        next_value = policy.value(torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)).squeeze(0)

    return {
        "obs": obs_buf,
        "action": act_buf,
        "logp": logp_buf,
        "reward": rew_buf,
        "done": done_buf,
        "value": val_buf,
        "next_value": next_value.detach(),
    }


def save_policy_checkpoint(
    path: Path,
    *,
    update_idx: int,
    policy: ActorCritic,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "update": update_idx,
            "model_state_dict": policy.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "args": vars(args),
        },
        path,
    )
    print(f"[save] {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Ant PPO in a frozen WestWorld-simulated environment.")
    parser.add_argument("--config-name", default="config_ant_running")
    parser.add_argument("--ckpt", default=None, help="WestWorld checkpoint path. Defaults to config ckpt_path.")
    parser.add_argument("--stats", type=Path, default=None, help="minmax_*.pt used to normalize WestWorld inputs.")
    parser.add_argument("--xml", type=Path, default=DEFAULT_XML, help="Ant XML used only for initial real states.")
    parser.add_argument("--policy-dir", type=Path, default=DEFAULT_POLICY_DIR)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument(
        "--device",
        default="cuda",
        help="Training device. The repository's current mamba_ssm build requires CUDA.",
    )
    parser.add_argument("--total-updates", type=int, default=100)
    parser.add_argument("--rollout-steps", type=int, default=1024)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-coef", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--qpos-noise", type=float, default=0.05)
    parser.add_argument("--qvel-noise", type=float, default=0.05)
    parser.add_argument("--save-interval", type=int, default=10)
    parser.add_argument("--no-joint-angle-clamp", action="store_true")
    parser.add_argument("--clip-sim-obs-to-stats", action="store_true")
    parser.add_argument("overrides", nargs="*", help="Optional Hydra overrides, e.g. ckpt_path=...")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.total_updates <= 0:
        raise ValueError("--total-updates must be positive")
    if args.rollout_steps <= 0:
        raise ValueError("--rollout-steps must be positive")
    if args.save_interval <= 0:
        raise ValueError("--save-interval must be positive")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("The current WestWorld mamba_ssm kernels require a CUDA device.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Run this training script in a GPU-enabled environment.")

    stats_file = find_stats_file(args.stats)
    stats = torch.load(stats_file, map_location="cpu", weights_only=False)
    print(f"[load stats] {stats_file}")

    cfg = load_cfg(args.config_name, args.overrides)
    world_model = load_world_model(cfg, args.ckpt, device)

    backend = AntBackend.load(args.xml.resolve())
    if backend.nu != ACT_DIM:
        raise RuntimeError(f"Expected {ACT_DIM} actuators, got {backend.nu}")
    clamp_joint_angles = not args.no_joint_angle_clamp
    if clamp_joint_angles:
        clamp_backend_joint_angles(backend)
    print(f"[real reset env] {args.xml.resolve()}")
    print(f"[joint angle clamp] enabled={clamp_joint_angles}")

    simulator = WestWorldOneStepSimulator(
        backend=backend,
        world_model=world_model,
        stats=stats,
        device=device,
        max_steps=args.max_steps,
        qpos_noise=args.qpos_noise,
        qvel_noise=args.qvel_noise,
        clamp_joint_angles=clamp_joint_angles,
        clip_sim_obs_to_stats=args.clip_sim_obs_to_stats,
    )

    policy = ActorCritic(OBS_DIM, ACT_DIM, hidden_size=args.hidden_size, layers=args.layers).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr, eps=1e-5)
    args.policy_dir.mkdir(parents=True, exist_ok=True)

    print(
        "[train] "
        f"updates={args.total_updates} rollout_steps={args.rollout_steps} "
        f"world_model_sim=True policy_scratch=True"
    )
    for update_idx in range(1, args.total_updates + 1):
        rollout = collect_westworld_rollout(
            simulator,
            policy,
            rng,
            device,
            rollout_steps=args.rollout_steps,
        )
        losses = ppo_update(
            policy,
            optimizer,
            rollout,
            ppo_epochs=args.ppo_epochs,
            minibatch_size=args.minibatch_size,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            clip_coef=args.clip_coef,
            ent_coef=args.ent_coef,
            vf_coef=args.vf_coef,
            max_grad_norm=args.max_grad_norm,
        )

        rollout_return = float(rollout["reward"].sum().item())
        done_count = int(rollout["done"].sum().item())
        print(
            f"[update {update_idx:04d}/{args.total_updates}] "
            f"rollout_return={rollout_return:.3f} "
            f"done={done_count} "
            f"policy_loss={losses['policy_loss']:.4f} "
            f"value_loss={losses['value_loss']:.4f} "
            f"entropy={losses['entropy']:.4f}"
        )

        if update_idx % args.save_interval == 0 or update_idx == args.total_updates:
            save_policy_checkpoint(
                args.policy_dir / f"ppo_westworld_ant_update_{update_idx:04d}.pt",
                update_idx=update_idx,
                policy=policy,
                optimizer=optimizer,
                args=args,
            )


if __name__ == "__main__":
    main()
