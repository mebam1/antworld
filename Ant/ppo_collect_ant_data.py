#!/usr/bin/env python3
"""Train a small PPO policy for Ant and collect intermediate-policy data.

The saved files keep the UniTraj-style keys used by WestWorld:
  obs, action, reward, task

For visualization, each episode also stores raw MuJoCo states:
  qpos, qvel, policy_update

The dataset converter ignores those extra keys, so the output remains training
compatible while still being renderable.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal


SCRIPT_DIR = Path(__file__).resolve().parent
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
    minmax_from_episodes,
    save_chunks,
    save_stats,
    scale_01,
)


PPO_DEFAULT_OUT = DEFAULT_OUT.parent / "ant_running_ppo"


def atanh(x: torch.Tensor) -> torch.Tensor:
    x = x.clamp(-0.999999, 0.999999)
    return 0.5 * (torch.log1p(x) - torch.log1p(-x))


def mlp(input_dim: int, output_dim: int, hidden_size: int, layers: int) -> nn.Sequential:
    blocks: List[nn.Module] = []
    last = input_dim
    for _ in range(layers):
        blocks.append(nn.Linear(last, hidden_size))
        blocks.append(nn.Tanh())
        last = hidden_size
    blocks.append(nn.Linear(last, output_dim))
    return nn.Sequential(*blocks)


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_size: int, layers: int):
        super().__init__()
        self.actor_mean = mlp(obs_dim, act_dim, hidden_size, layers)
        self.critic = mlp(obs_dim, 1, hidden_size, layers)
        self.log_std = nn.Parameter(torch.full((act_dim,), -0.5))

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0.0)
        nn.init.orthogonal_(self.actor_mean[-1].weight, gain=0.01)
        nn.init.orthogonal_(self.critic[-1].weight, gain=1.0)

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(obs).squeeze(-1)

    def action_and_value(
        self,
        obs: torch.Tensor,
        action: torch.Tensor | None = None,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mean = self.actor_mean(obs)
        std = self.log_std.exp().expand_as(mean)
        dist = Normal(mean, std)

        if action is None:
            pre_tanh = mean if deterministic else dist.rsample()
            action = torch.tanh(pre_tanh)
        else:
            pre_tanh = atanh(action)

        log_prob = dist.log_prob(pre_tanh) - torch.log(1.0 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        value = self.value(obs)
        return action, log_prob, entropy, value


def backend_state(backend: AntBackend) -> Tuple[np.ndarray, np.ndarray]:
    return np.asarray(backend.data.qpos).copy(), np.asarray(backend.data.qvel).copy()


def forward_backend(backend: AntBackend) -> None:
    if backend.kind == "mujoco":
        import mujoco  # type: ignore

        mujoco.mj_forward(backend.model, backend.data)
    else:
        assert backend.sim is not None
        backend.sim.forward()


def clamp_backend_joint_angles(backend: AntBackend) -> int:
    """Clamp limited scalar joint qpos values to the compiled MuJoCo ranges."""
    model = backend.model
    qpos = np.asarray(backend.data.qpos)
    jnt_limited = np.asarray(getattr(model, "jnt_limited", []))
    jnt_range = np.asarray(getattr(model, "jnt_range", []), dtype=np.float64)
    jnt_qposadr = np.asarray(getattr(model, "jnt_qposadr", []), dtype=np.int64)
    if jnt_limited.size == 0 or jnt_range.size == 0 or jnt_qposadr.size == 0:
        return 0

    clipped = 0
    joint_count = min(jnt_limited.shape[0], jnt_range.shape[0], jnt_qposadr.shape[0])
    for joint_idx in range(joint_count):
        if not bool(jnt_limited[joint_idx]):
            continue
        adr = int(jnt_qposadr[joint_idx])
        if adr < 0 or adr >= qpos.shape[0]:
            continue
        lo, hi = float(jnt_range[joint_idx, 0]), float(jnt_range[joint_idx, 1])
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            continue
        before = float(qpos[adr])
        after = float(np.clip(before, lo, hi))
        if after != before:
            qpos[adr] = after
            clipped += 1

    if clipped:
        forward_backend(backend)
    return clipped


def reset_backend(
    backend: AntBackend,
    rng: np.random.Generator,
    qpos_noise: float,
    qvel_noise: float,
    clamp_joint_angles: bool,
) -> np.ndarray:
    backend.reset(rng, qpos_noise=qpos_noise, qvel_noise=qvel_noise)
    if clamp_joint_angles:
        clamp_backend_joint_angles(backend)
    return backend.obs()


def collect_ppo_rollout(
    backend: AntBackend,
    model: ActorCritic,
    rng: np.random.Generator,
    device: torch.device,
    *,
    rollout_steps: int,
    max_episode_steps: int,
    qpos_noise: float,
    qvel_noise: float,
    clamp_joint_angles: bool,
) -> Dict[str, torch.Tensor]:
    obs_buf = torch.zeros((rollout_steps, OBS_DIM), dtype=torch.float32, device=device)
    act_buf = torch.zeros((rollout_steps, ACT_DIM), dtype=torch.float32, device=device)
    logp_buf = torch.zeros(rollout_steps, dtype=torch.float32, device=device)
    rew_buf = torch.zeros(rollout_steps, dtype=torch.float32, device=device)
    done_buf = torch.zeros(rollout_steps, dtype=torch.float32, device=device)
    val_buf = torch.zeros(rollout_steps, dtype=torch.float32, device=device)

    obs = reset_backend(backend, rng, qpos_noise, qvel_noise, clamp_joint_angles)
    ep_step = 0

    for step in range(rollout_steps):
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            action_t, logp_t, _, value_t = model.action_and_value(obs_t)

        action_np = action_t.squeeze(0).cpu().numpy().astype(np.float32)
        backend.step(action_np)
        if clamp_joint_angles:
            clamp_backend_joint_angles(backend)
        next_obs = backend.obs()
        reward = ant_reward(next_obs)
        done = ant_done(next_obs, ep_step, max_episode_steps)

        obs_buf[step] = obs_t.squeeze(0)
        act_buf[step] = action_t.squeeze(0)
        logp_buf[step] = logp_t.squeeze(0)
        rew_buf[step] = float(reward)
        done_buf[step] = float(done)
        val_buf[step] = value_t.squeeze(0)

        ep_step += 1
        if done:
            obs = reset_backend(backend, rng, qpos_noise, qvel_noise, clamp_joint_angles)
            ep_step = 0
        else:
            obs = next_obs

    with torch.no_grad():
        next_value = model.value(torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)).squeeze(0)

    return {
        "obs": obs_buf,
        "action": act_buf,
        "logp": logp_buf,
        "reward": rew_buf,
        "done": done_buf,
        "value": val_buf,
        "next_value": next_value.detach(),
    }


def compute_gae(
    rewards: torch.Tensor,
    dones: torch.Tensor,
    values: torch.Tensor,
    next_value: torch.Tensor,
    gamma: float,
    gae_lambda: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    advantages = torch.zeros_like(rewards)
    last_gae = torch.zeros((), dtype=torch.float32, device=rewards.device)

    for t in reversed(range(rewards.shape[0])):
        if t == rewards.shape[0] - 1:
            next_nonterminal = 1.0 - dones[t]
            next_values = next_value
        else:
            next_nonterminal = 1.0 - dones[t]
            next_values = values[t + 1]
        delta = rewards[t] + gamma * next_values * next_nonterminal - values[t]
        last_gae = delta + gamma * gae_lambda * next_nonterminal * last_gae
        advantages[t] = last_gae

    returns = advantages + values
    return advantages, returns


def ppo_update(
    model: ActorCritic,
    optimizer: torch.optim.Optimizer,
    rollout: Dict[str, torch.Tensor],
    *,
    ppo_epochs: int,
    minibatch_size: int,
    gamma: float,
    gae_lambda: float,
    clip_coef: float,
    ent_coef: float,
    vf_coef: float,
    max_grad_norm: float,
) -> Dict[str, float]:
    advantages, returns = compute_gae(
        rollout["reward"],
        rollout["done"],
        rollout["value"],
        rollout["next_value"],
        gamma=gamma,
        gae_lambda=gae_lambda,
    )
    advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

    obs = rollout["obs"]
    actions = rollout["action"]
    old_logp = rollout["logp"]
    old_values = rollout["value"]

    batch_size = obs.shape[0]
    minibatch_size = min(minibatch_size, batch_size)
    losses = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
    updates = 0

    for _ in range(ppo_epochs):
        indices = torch.randperm(batch_size, device=obs.device)
        for start in range(0, batch_size, minibatch_size):
            mb = indices[start : start + minibatch_size]
            _, new_logp, entropy, new_values = model.action_and_value(obs[mb], action=actions[mb])
            log_ratio = new_logp - old_logp[mb]
            ratio = log_ratio.exp()

            pg_loss1 = -advantages[mb] * ratio
            pg_loss2 = -advantages[mb] * torch.clamp(ratio, 1.0 - clip_coef, 1.0 + clip_coef)
            policy_loss = torch.max(pg_loss1, pg_loss2).mean()

            value_pred_clipped = old_values[mb] + (new_values - old_values[mb]).clamp(-clip_coef, clip_coef)
            value_losses = (new_values - returns[mb]).pow(2)
            value_losses_clipped = (value_pred_clipped - returns[mb]).pow(2)
            value_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()

            entropy_loss = entropy.mean()
            loss = policy_loss + vf_coef * value_loss - ent_coef * entropy_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

            losses["policy_loss"] += float(policy_loss.item())
            losses["value_loss"] += float(value_loss.item())
            losses["entropy"] += float(entropy_loss.item())
            updates += 1

    if updates:
        for key in losses:
            losses[key] /= updates
    return losses


def rollout_policy_episode(
    backend: AntBackend,
    model: ActorCritic,
    rng: np.random.Generator,
    device: torch.device,
    *,
    max_steps: int,
    qpos_noise: float,
    qvel_noise: float,
    deterministic: bool,
    policy_update: int,
    clamp_joint_angles: bool,
) -> dict:
    obs = reset_backend(backend, rng, qpos_noise, qvel_noise, clamp_joint_angles)
    obs_list: List[np.ndarray] = []
    act_list: List[np.ndarray] = []
    rew_list: List[float] = []
    qpos_list: List[np.ndarray] = []
    qvel_list: List[np.ndarray] = []

    for step_idx in range(max_steps):
        qpos, qvel = backend_state(backend)
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            action_t, _, _, _ = model.action_and_value(obs_t, deterministic=deterministic)
        action = action_t.squeeze(0).cpu().numpy().astype(np.float32)

        backend.step(action)
        if clamp_joint_angles:
            clamp_backend_joint_angles(backend)
        obs_after = backend.obs()
        reward = ant_reward(obs_after)

        obs_list.append(obs.astype(np.float32))
        act_list.append(action)
        rew_list.append(float(reward))
        qpos_list.append(qpos.astype(np.float32))
        qvel_list.append(qvel.astype(np.float32))

        if ant_done(obs_after, step_idx, max_steps):
            break
        obs = obs_after

    obs_tensor = torch.from_numpy(np.stack(obs_list, axis=0)).float()
    act_tensor = torch.from_numpy(np.stack(act_list, axis=0)).float()
    reward_tensor = torch.tensor(rew_list, dtype=torch.float32)
    task_tensor = torch.full((obs_tensor.shape[0],), TASK_ID, dtype=torch.long)
    update_tensor = torch.full((obs_tensor.shape[0],), int(policy_update), dtype=torch.long)

    return {
        "obs": obs_tensor,
        "action": act_tensor,
        "reward": reward_tensor,
        "task": task_tensor,
        "qpos": torch.from_numpy(np.stack(qpos_list, axis=0)).float(),
        "qvel": torch.from_numpy(np.stack(qvel_list, axis=0)).float(),
        "policy_update": update_tensor,
    }


def normalize_episodes_preserve_extra(episodes: Sequence[dict], stats: dict) -> List[dict]:
    normalized = []
    for ep in episodes:
        out = dict(ep)
        out["obs"] = scale_01(ep["obs"], stats["obs_min"], stats["obs_max"]).float()
        out["action"] = scale_01(ep["action"], stats["action_min"], stats["action_max"]).float()
        out["reward"] = scale_01(
            ep["reward"].reshape(-1, 1),
            stats["reward_min"],
            stats["reward_max"],
        ).reshape(-1).float()
        out["task"] = ep["task"].long()
        normalized.append(out)
    return normalized


def collect_snapshot(
    backend: AntBackend,
    model: ActorCritic,
    rng: np.random.Generator,
    device: torch.device,
    *,
    update_idx: int,
    episodes: int,
    max_steps: int,
    qpos_noise: float,
    qvel_noise: float,
    deterministic: bool,
    clamp_joint_angles: bool,
) -> List[dict]:
    collected = []
    returns = []
    lengths = []
    for _ in range(episodes):
        ep = rollout_policy_episode(
            backend,
            model,
            rng,
            device,
            max_steps=max_steps,
            qpos_noise=qpos_noise,
            qvel_noise=qvel_noise,
            deterministic=deterministic,
            policy_update=update_idx,
            clamp_joint_angles=clamp_joint_angles,
        )
        collected.append(ep)
        returns.append(float(ep["reward"].sum().item()))
        lengths.append(int(ep["obs"].shape[0]))
    print(
        f"[snapshot update={update_idx}] episodes={episodes} "
        f"avg_len={np.mean(lengths):.1f} avg_return={np.mean(returns):.3f}"
    )
    return collected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO on Ant and collect intermediate policy episodes.")
    parser.add_argument("--xml", type=Path, default=DEFAULT_XML, help="Path to ant.xml.")
    parser.add_argument("--out-dir", type=Path, default=PPO_DEFAULT_OUT, help="Output directory for episodes_*.pt.")
    parser.add_argument("--prefix", default="ant_running_ppo", help="Output filename prefix.")
    parser.add_argument("--policy-dir", type=Path, default=SCRIPT_DIR / "ppo_checkpoints", help="Directory for PPO checkpoints.")
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--total-updates", type=int, default=30)
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
    parser.add_argument("--collect-interval", type=int, default=5, help="Collect data every N PPO updates.")
    parser.add_argument("--episodes-per-snapshot", type=int, default=20)
    parser.add_argument("--no-collect-initial", action="store_true", help="Skip collecting the untrained initial policy.")
    parser.add_argument("--deterministic-collection", action="store_true", help="Use mean action during snapshot collection.")
    parser.add_argument("--qpos-noise", type=float, default=0.05)
    parser.add_argument("--qvel-noise", type=float, default=0.05)
    parser.add_argument(
        "--no-joint-angle-clamp",
        action="store_true",
        help="Disable explicit clamping of limited MuJoCo joint angles after reset/step.",
    )
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument("--no-normalize", action="store_true", help="Save raw obs/actions/rewards.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.total_updates < 0:
        raise ValueError("--total-updates must be non-negative")
    if args.collect_interval <= 0:
        raise ValueError("--collect-interval must be positive")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device(args.device)

    backend = AntBackend.load(args.xml.resolve())
    if backend.nu != ACT_DIM:
        raise RuntimeError(f"Expected {ACT_DIM} actuators, got {backend.nu}")
    clamp_joint_angles = not args.no_joint_angle_clamp
    print(f"[joint angle clamp] enabled={clamp_joint_angles}")

    model = ActorCritic(OBS_DIM, ACT_DIM, hidden_size=args.hidden_size, layers=args.layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, eps=1e-5)

    all_episodes: List[dict] = []
    args.policy_dir.mkdir(parents=True, exist_ok=True)

    if not args.no_collect_initial:
        all_episodes.extend(
            collect_snapshot(
                backend,
                model,
                rng,
                device,
                update_idx=0,
                episodes=args.episodes_per_snapshot,
                max_steps=args.max_steps,
                qpos_noise=args.qpos_noise,
                qvel_noise=args.qvel_noise,
                deterministic=args.deterministic_collection,
                clamp_joint_angles=clamp_joint_angles,
            )
        )

    for update_idx in range(1, args.total_updates + 1):
        rollout = collect_ppo_rollout(
            backend,
            model,
            rng,
            device,
            rollout_steps=args.rollout_steps,
            max_episode_steps=args.max_steps,
            qpos_noise=args.qpos_noise,
            qvel_noise=args.qvel_noise,
            clamp_joint_angles=clamp_joint_angles,
        )
        losses = ppo_update(
            model,
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
        print(
            f"[update {update_idx:04d}/{args.total_updates}] "
            f"rollout_return={rollout_return:.3f} "
            f"policy_loss={losses['policy_loss']:.4f} "
            f"value_loss={losses['value_loss']:.4f} "
            f"entropy={losses['entropy']:.4f}"
        )

        should_collect = update_idx % args.collect_interval == 0 or update_idx == args.total_updates
        if should_collect:
            ckpt_path = args.policy_dir / f"ppo_ant_update_{update_idx:04d}.pt"
            torch.save(
                {
                    "update": update_idx,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "args": vars(args),
                },
                ckpt_path,
            )
            print(f"[save] {ckpt_path}")
            all_episodes.extend(
                collect_snapshot(
                    backend,
                    model,
                    rng,
                    device,
                    update_idx=update_idx,
                    episodes=args.episodes_per_snapshot,
                    max_steps=args.max_steps,
                    qpos_noise=args.qpos_noise,
                    qvel_noise=args.qvel_noise,
                    deterministic=args.deterministic_collection,
                    clamp_joint_angles=clamp_joint_angles,
                )
            )

    if not all_episodes:
        raise RuntimeError("No episodes collected. Increase --total-updates or enable initial collection.")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stats = minmax_from_episodes(all_episodes)
    save_stats(stats, args.out_dir, args.prefix)

    episodes_to_save = all_episodes if args.no_normalize else normalize_episodes_preserve_extra(all_episodes, stats)
    save_chunks(episodes_to_save, args.out_dir, args.chunk_size, args.prefix)
    total_steps = sum(int(ep["obs"].shape[0]) for ep in all_episodes)
    print(
        "[done] "
        f"episodes={len(all_episodes)} steps={total_steps} "
        f"normalized={not args.no_normalize} out_dir={args.out_dir}"
    )


if __name__ == "__main__":
    main()
