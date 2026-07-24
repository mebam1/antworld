#!/usr/bin/env python3
"""Generate Ant Running episodes in the UniTraj-style PT format.

Output files are lists of plain dict episodes:
  obs:    FloatTensor [T, 31]
  action: FloatTensor [T, 8]
  reward: FloatTensor [T]
  task:   LongTensor  [T] filled with task id 131

The observation/action order matches task 131 in
robotics_structure_xml/general_task_specific.yaml.
"""

from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_XML = REPO_ROOT / "robotics_structure_xml" / "ant.xml"
DEFAULT_OUT = REPO_ROOT / "Trajworld_data" / "UniTraj_pt" / "ant_running_pt" / "ant_running"
TASK_ID = 131
OBS_DIM = 31
ACT_DIM = 8


def quat_to_mat(q: np.ndarray) -> np.ndarray:
    """Convert MuJoCo quaternion [w, x, y, z] to a 3x3 rotation matrix."""
    w, x, y, z = [float(v) for v in q]
    n = math.sqrt(w * w + x * x + y * y + z * z)
    if n <= 1e-12:
        return np.eye(3, dtype=np.float32)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def quat_to_rot6d(q: np.ndarray) -> np.ndarray:
    """Convert MuJoCo quaternion [w, x, y, z] to a 6D rotation representation."""
    rot = quat_to_mat(q)
    return np.concatenate([rot[:, 0], rot[:, 1]], axis=0).astype(np.float32)


def rot6d_to_mat(rot6d: np.ndarray) -> np.ndarray:
    """Project a 6D rotation representation back to a 3x3 rotation matrix."""
    x = np.asarray(rot6d, dtype=np.float64).reshape(-1)
    if x.shape[0] != 6 or not np.isfinite(x).all():
        return np.eye(3, dtype=np.float32)

    a1 = x[:3]
    a2 = x[3:6]
    n1 = float(np.linalg.norm(a1))
    if n1 <= 1e-8:
        return np.eye(3, dtype=np.float32)
    b1 = a1 / n1

    a2_ortho = a2 - float(np.dot(b1, a2)) * b1
    n2 = float(np.linalg.norm(a2_ortho))
    if n2 <= 1e-8:
        return np.eye(3, dtype=np.float32)
    b2 = a2_ortho / n2
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=1).astype(np.float32)


def mat_to_quat(rot: np.ndarray) -> np.ndarray:
    """Convert a 3x3 rotation matrix to a MuJoCo quaternion [w, x, y, z]."""
    m = np.asarray(rot, dtype=np.float64).reshape(3, 3)
    trace = float(np.trace(m))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        quat = np.array(
            [
                0.25 * s,
                (m[2, 1] - m[1, 2]) / s,
                (m[0, 2] - m[2, 0]) / s,
                (m[1, 0] - m[0, 1]) / s,
            ],
            dtype=np.float64,
        )
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(max(1.0 + m[0, 0] - m[1, 1] - m[2, 2], 1e-12)) * 2.0
        quat = np.array(
            [
                (m[2, 1] - m[1, 2]) / s,
                0.25 * s,
                (m[0, 1] + m[1, 0]) / s,
                (m[0, 2] + m[2, 0]) / s,
            ],
            dtype=np.float64,
        )
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(max(1.0 + m[1, 1] - m[0, 0] - m[2, 2], 1e-12)) * 2.0
        quat = np.array(
            [
                (m[0, 2] - m[2, 0]) / s,
                (m[0, 1] + m[1, 0]) / s,
                0.25 * s,
                (m[1, 2] + m[2, 1]) / s,
            ],
            dtype=np.float64,
        )
    else:
        s = math.sqrt(max(1.0 + m[2, 2] - m[0, 0] - m[1, 1], 1e-12)) * 2.0
        quat = np.array(
            [
                (m[1, 0] - m[0, 1]) / s,
                (m[0, 2] + m[2, 0]) / s,
                (m[1, 2] + m[2, 1]) / s,
                0.25 * s,
            ],
            dtype=np.float64,
        )

    return normalize_quat(quat).astype(np.float32)


def rot6d_to_quat(rot6d: np.ndarray) -> np.ndarray:
    return mat_to_quat(rot6d_to_mat(rot6d))


def rot6d_to_quat_batch(rot6d: np.ndarray) -> np.ndarray:
    arr = np.asarray(rot6d, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 6:
        raise ValueError(f"Expected [N, 6] rot6d array, got {arr.shape}")
    return np.stack([rot6d_to_quat(row) for row in arr], axis=0).astype(np.float64)


def canonicalize_rot6d(rot6d: np.ndarray) -> np.ndarray:
    rot = rot6d_to_mat(rot6d)
    return np.concatenate([rot[:, 0], rot[:, 1]], axis=0).astype(np.float32)


def ant_observation(qpos: np.ndarray, qvel: np.ndarray) -> np.ndarray:
    """Build the 31D Ant Running observation from MuJoCo qpos/qvel."""
    h = qpos[2:3]
    quat = qpos[3:7]
    rot6d = quat_to_rot6d(quat)
    v_lin = qvel[0:3]
    v_ang = qvel[3:6]
    joint_pos = qpos[7:15]
    joint_vel = qvel[6:14]

    rot = quat_to_mat(quat)
    p_up = np.array([rot[2, 2]], dtype=np.float32)
    p_heading = np.array([rot[0, 0]], dtype=np.float32)

    obs = np.concatenate(
        [h, rot6d, v_lin, v_ang, joint_pos, joint_vel, p_up, p_heading],
        axis=0,
    ).astype(np.float32)
    if obs.shape != (OBS_DIM,):
        raise RuntimeError(f"Ant observation shape mismatch: {obs.shape}, expected {(OBS_DIM,)}")
    return obs


def ant_reward(obs_after: np.ndarray) -> float:
    vx = float(obs_after[7])
    p_up = float(obs_after[29])
    p_heading = float(obs_after[30])
    return vx + 0.1 * p_up + p_heading


def ant_done(obs_after: np.ndarray, step_index: int, max_steps: int) -> bool:
    h = float(obs_after[0])
    return step_index + 1 >= max_steps or h < 0.3


@dataclass
class AntBackend:
    kind: str
    model: object
    data: object
    sim: Optional[object] = None

    @classmethod
    def load(cls, xml_path: Path) -> "AntBackend":
        try:
            import mujoco  # type: ignore

            model = mujoco.MjModel.from_xml_path(str(xml_path))
            data = mujoco.MjData(model)
            return cls(kind="mujoco", model=model, data=data)
        except Exception as mujoco_error:
            try:
                import mujoco_py  # type: ignore

                model = mujoco_py.load_model_from_path(str(xml_path))
                sim = mujoco_py.MjSim(model)
                return cls(kind="mujoco_py", model=model, data=sim.data, sim=sim)
            except Exception as mujoco_py_error:
                raise RuntimeError(
                    "Could not import/load either `mujoco` or `mujoco_py`. "
                    "Install one of them in the WestWorld environment."
                ) from mujoco_py_error

    @property
    def nq(self) -> int:
        return int(self.model.nq)

    @property
    def nv(self) -> int:
        return int(self.model.nv)

    @property
    def nu(self) -> int:
        return int(self.model.nu)

    def reset(self, rng: np.random.Generator, qpos_noise: float, qvel_noise: float) -> None:
        init_qpos = np.zeros(self.nq, dtype=np.float64)
        if self.nq >= 15:
            init_qpos[:15] = np.array(
                [0.0, 0.0, 0.55, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, -1.0, 0.0, -1.0],
                dtype=np.float64,
            )
        init_qvel = np.zeros(self.nv, dtype=np.float64)

        init_qpos += rng.uniform(-qpos_noise, qpos_noise, size=self.nq)
        init_qpos[3:7] = normalize_quat(init_qpos[3:7])
        init_qvel += rng.uniform(-qvel_noise, qvel_noise, size=self.nv)

        if self.kind == "mujoco":
            import mujoco  # type: ignore

            mujoco.mj_resetData(self.model, self.data)
            self.data.qpos[:] = init_qpos
            self.data.qvel[:] = init_qvel
            self.data.ctrl[:] = 0.0
            mujoco.mj_forward(self.model, self.data)
        else:
            assert self.sim is not None
            self.sim.reset()
            self.sim.data.qpos[:] = init_qpos
            self.sim.data.qvel[:] = init_qvel
            self.sim.data.ctrl[:] = 0.0
            self.sim.forward()

    def step(self, action: np.ndarray) -> None:
        self.data.ctrl[:] = action
        if self.kind == "mujoco":
            import mujoco  # type: ignore

            mujoco.mj_step(self.model, self.data)
        else:
            assert self.sim is not None
            self.sim.step()

    def obs(self) -> np.ndarray:
        return ant_observation(np.asarray(self.data.qpos), np.asarray(self.data.qvel))


def normalize_quat(q: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(q))
    if n <= 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return q / n


class ActionPolicy:
    def __init__(self, name: str, rng: np.random.Generator, noise_std: float, smooth: float):
        self.name = name
        self.rng = rng
        self.noise_std = float(noise_std)
        self.smooth = float(smooth)
        self.prev = np.zeros(ACT_DIM, dtype=np.float32)
        self.phase = rng.uniform(0.0, 2.0 * np.pi, size=ACT_DIM).astype(np.float32)
        self.freq = rng.uniform(0.03, 0.12, size=ACT_DIM).astype(np.float32)

    def __call__(self, t: int) -> np.ndarray:
        if self.name == "random":
            raw = self.rng.uniform(-1.0, 1.0, size=ACT_DIM).astype(np.float32)
        elif self.name == "smooth_random":
            raw = self.rng.normal(0.0, self.noise_std, size=ACT_DIM).astype(np.float32)
            raw = self.smooth * self.prev + (1.0 - self.smooth) * raw
        elif self.name == "sinusoidal":
            raw = np.sin(self.phase + t * self.freq).astype(np.float32)
            raw += self.rng.normal(0.0, self.noise_std, size=ACT_DIM).astype(np.float32)
        else:
            raise ValueError(f"Unknown policy: {self.name}")
        self.prev = np.clip(raw, -1.0, 1.0).astype(np.float32)
        return self.prev


def rollout_episode(
    backend: AntBackend,
    rng: np.random.Generator,
    policy_name: str,
    max_steps: int,
    qpos_noise: float,
    qvel_noise: float,
    noise_std: float,
    smooth: float,
) -> dict:
    backend.reset(rng, qpos_noise=qpos_noise, qvel_noise=qvel_noise)
    policy = ActionPolicy(policy_name, rng, noise_std=noise_std, smooth=smooth)

    obs_list: List[np.ndarray] = []
    act_list: List[np.ndarray] = []
    rew_list: List[float] = []

    for t in range(max_steps):
        obs_t = backend.obs()
        action = policy(t)
        backend.step(action)
        obs_after = backend.obs()
        reward = ant_reward(obs_after)

        obs_list.append(obs_t)
        act_list.append(action)
        rew_list.append(reward)

        if ant_done(obs_after, t, max_steps):
            break

    obs = torch.from_numpy(np.stack(obs_list, axis=0)).float()
    action = torch.from_numpy(np.stack(act_list, axis=0)).float()
    reward = torch.tensor(rew_list, dtype=torch.float32)
    task = torch.full((obs.shape[0],), TASK_ID, dtype=torch.long)

    return {"obs": obs, "action": action, "reward": reward, "task": task}


def minmax_from_episodes(episodes: Sequence[dict]) -> dict:
    obs = torch.cat([ep["obs"] for ep in episodes], dim=0)
    action = torch.cat([ep["action"] for ep in episodes], dim=0)
    reward = torch.cat([ep["reward"].reshape(-1, 1) for ep in episodes], dim=0)
    return {
        "obs_min": obs.min(dim=0).values,
        "obs_max": obs.max(dim=0).values,
        "action_min": action.min(dim=0).values,
        "action_max": action.max(dim=0).values,
        "reward_min": reward.min(dim=0).values,
        "reward_max": reward.max(dim=0).values,
    }


def scale_01(x: torch.Tensor, mn: torch.Tensor, mx: torch.Tensor) -> torch.Tensor:
    denom = (mx - mn).clamp_min(1e-6)
    return ((x - mn) / denom).clamp(0.0, 1.0)


def normalize_episodes(episodes: Sequence[dict], stats: dict) -> List[dict]:
    out = []
    for ep in episodes:
        out.append(
            {
                "obs": scale_01(ep["obs"], stats["obs_min"], stats["obs_max"]).float(),
                "action": scale_01(ep["action"], stats["action_min"], stats["action_max"]).float(),
                "reward": scale_01(
                    ep["reward"].reshape(-1, 1),
                    stats["reward_min"],
                    stats["reward_max"],
                )
                .reshape(-1)
                .float(),
                "task": ep["task"].long(),
            }
        )
    return out


def save_chunks(episodes: Sequence[dict], out_dir: Path, chunk_size: int, prefix: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob(f"episodes_{prefix}_chunk*_E*.pt"):
        old.unlink()

    for start in range(0, len(episodes), chunk_size):
        chunk = list(episodes[start : start + chunk_size])
        chunk_idx = start // chunk_size + 1
        path = out_dir / f"episodes_{prefix}_chunk{chunk_idx}_E{len(chunk)}.pt"
        torch.save(chunk, path)
        print(f"[save] {path} ({len(chunk)} episodes)")


def save_stats(stats: dict, out_dir: Path, prefix: str) -> None:
    payload = {
        **stats,
        "task_id": TASK_ID,
        "dims": {"obs": OBS_DIM, "action": ACT_DIM, "reward": 1},
        "obs_order": [
            "height",
            "rot6d_col0_x",
            "rot6d_col0_y",
            "rot6d_col0_z",
            "rot6d_col1_x",
            "rot6d_col1_y",
            "rot6d_col1_z",
            "v_x",
            "v_y",
            "v_z",
            "w_x",
            "w_y",
            "w_z",
            "hip_1_pos",
            "ankle_1_pos",
            "hip_2_pos",
            "ankle_2_pos",
            "hip_3_pos",
            "ankle_3_pos",
            "hip_4_pos",
            "ankle_4_pos",
            "hip_1_vel",
            "ankle_1_vel",
            "hip_2_vel",
            "ankle_2_vel",
            "hip_3_vel",
            "ankle_3_vel",
            "hip_4_vel",
            "ankle_4_vel",
            "p_up",
            "p_heading",
        ],
        "action_order": ["hip_4", "ankle_4", "hip_1", "ankle_1", "hip_2", "ankle_2", "hip_3", "ankle_3"],
    }
    path = out_dir / f"minmax_{prefix}.pt"
    torch.save(payload, path)
    print(f"[save] {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Ant Running episodes for WestWorld.")
    parser.add_argument("--xml", type=Path, default=DEFAULT_XML, help="Path to ant.xml.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT, help="Output directory for episodes_*.pt.")
    parser.add_argument("--episodes", type=int, default=100, help="Number of episodes to collect.")
    parser.add_argument("--max-steps", type=int, default=500, help="Maximum episode length.")
    parser.add_argument("--seed", type=int, default=43, help="Random seed.")
    parser.add_argument("--chunk-size", type=int, default=1000, help="Episodes per saved PT chunk.")
    parser.add_argument(
        "--policy",
        choices=["random", "smooth_random", "sinusoidal"],
        default="smooth_random",
        help="Behavior policy used for data collection.",
    )
    parser.add_argument("--noise-std", type=float, default=0.6, help="Policy noise scale.")
    parser.add_argument("--smooth", type=float, default=0.92, help="Smoothing for smooth_random policy.")
    parser.add_argument("--qpos-noise", type=float, default=0.05, help="Initial qpos uniform noise.")
    parser.add_argument("--qvel-noise", type=float, default=0.05, help="Initial qvel uniform noise.")
    parser.add_argument("--prefix", default="ant_running", help="Output filename prefix.")
    parser.add_argument("--no-normalize", action="store_true", help="Save raw observations/actions/rewards.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.episodes <= 0:
        raise ValueError("--episodes must be positive")
    if args.max_steps <= 0:
        raise ValueError("--max-steps must be positive")

    xml_path = args.xml.resolve()
    if not xml_path.is_file():
        raise FileNotFoundError(xml_path)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    backend = AntBackend.load(xml_path)
    if backend.nu != ACT_DIM:
        raise RuntimeError(f"Expected {ACT_DIM} actuators, got {backend.nu}")
    if backend.nq < 15 or backend.nv < 14:
        raise RuntimeError(f"Unexpected Ant state sizes: nq={backend.nq}, nv={backend.nv}")

    episodes = []
    lengths = []
    returns = []
    for ep_idx in range(args.episodes):
        ep = rollout_episode(
            backend=backend,
            rng=rng,
            policy_name=args.policy,
            max_steps=args.max_steps,
            qpos_noise=args.qpos_noise,
            qvel_noise=args.qvel_noise,
            noise_std=args.noise_std,
            smooth=args.smooth,
        )
        episodes.append(ep)
        lengths.append(int(ep["obs"].shape[0]))
        returns.append(float(ep["reward"].sum().item()))
        if (ep_idx + 1) == 1 or (ep_idx + 1) == args.episodes or (ep_idx + 1) % max(1, args.episodes // 20) == 0:
            print(
                f"[collect] {ep_idx + 1}/{args.episodes} "
                f"len={lengths[-1]} return={returns[-1]:.3f}"
            )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stats = minmax_from_episodes(episodes)
    save_stats(stats, args.out_dir, args.prefix)

    if not args.no_normalize:
        episodes = normalize_episodes(episodes, stats)

    save_chunks(episodes, args.out_dir, args.chunk_size, args.prefix)
    print(
        "[done] "
        f"episodes={len(episodes)} avg_len={np.mean(lengths):.1f} "
        f"avg_return={np.mean(returns):.3f} normalized={not args.no_normalize}"
    )


if __name__ == "__main__":
    main()
