#!/usr/bin/env python3
"""Render collected Ant episode data to an MP4.

The PPO collection path stores exact MuJoCo states (`qpos`/`qvel`), which are
used directly. Older/random-policy UniTraj-style files may only contain `obs`;
those are rendered by reconstructing Ant pose/velocity from the 31D observation.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch


os.environ.setdefault("MUJOCO_GL", "egl")

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from generate_ant_data import DEFAULT_OUT, DEFAULT_XML, OBS_DIM, rot6d_to_quat_batch  # noqa: E402


PPO_DEFAULT_OUT = DEFAULT_OUT.parent / "ant_running_ppo"


@dataclass(frozen=True)
class LoadedEpisode:
    path: Path
    local_index: int
    global_index: int
    episode: dict


def resolve_repo_path(path: Path | str) -> Path:
    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        resolved = REPO_ROOT / resolved
    return resolved.resolve()


def episode_file_sort_key(path: Path) -> tuple[str, int, str]:
    match = re.search(r"_chunk(\d+)_E\d+\.pt$", path.name)
    chunk = int(match.group(1)) if match else 10**12
    return (str(path.parent), chunk, path.name)


def find_episode_files(root: Path) -> list[Path]:
    if root.is_file():
        return [root]
    files = sorted(root.glob("**/episodes_*.pt"), key=episode_file_sort_key)
    if not files:
        raise FileNotFoundError(f"No episodes_*.pt found under {root}")
    return files


def load_episode_list(path: Path) -> list[dict]:
    episodes = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(episodes, list):
        raise TypeError(f"Expected {path} to contain a list of episodes, got {type(episodes)!r}")
    return episodes


def load_episode_by_global_index(files: Sequence[Path], global_index: int) -> LoadedEpisode:
    if global_index < 0:
        raise ValueError("--episode-index must be non-negative")

    offset = 0
    for path in files:
        episodes = load_episode_list(path)
        next_offset = offset + len(episodes)
        if offset <= global_index < next_offset:
            local_index = global_index - offset
            return LoadedEpisode(path, local_index, global_index, episodes[local_index])
        offset = next_offset

    raise IndexError(f"episode-index={global_index} out of range for {offset} collected episodes")


def find_stats_file(episodes_path: Path, explicit: Path | None) -> Path:
    if explicit is not None:
        if not explicit.is_file():
            raise FileNotFoundError(explicit)
        return explicit

    roots = [episodes_path.parent if episodes_path.is_file() else episodes_path, PPO_DEFAULT_OUT, DEFAULT_OUT]
    for root in roots:
        files = sorted(root.glob("minmax_*.pt"))
        if files:
            return files[0]
    raise FileNotFoundError("No minmax_*.pt found. Pass --stats or use --input-raw.")


def as_numpy_2d(value: object, name: str) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        arr = value.detach().cpu().numpy()
    else:
        arr = np.asarray(value)
    if arr.ndim != 2:
        raise ValueError(f"Expected {name} to be 2D [T, D], got shape {arr.shape}")
    return np.asarray(arr, dtype=np.float64)


def denormalize_obs(obs: np.ndarray, stats_path: Path) -> np.ndarray:
    stats = torch.load(stats_path, map_location="cpu", weights_only=False)
    obs_min = stats["obs_min"].detach().cpu().numpy().astype(np.float64)
    obs_max = stats["obs_max"].detach().cpu().numpy().astype(np.float64)
    if obs_min.shape != (OBS_DIM,) or obs_max.shape != (OBS_DIM,):
        raise ValueError(f"Expected stats obs shape {(OBS_DIM,)}, got {obs_min.shape} and {obs_max.shape}")
    return obs * np.maximum(obs_max - obs_min, 1e-6) + obs_min


def reconstruct_states_from_obs(
    obs_raw: np.ndarray,
    *,
    qpos_dim: int,
    qvel_dim: int,
    timestep: float,
) -> tuple[np.ndarray, np.ndarray]:
    if obs_raw.shape[1] != OBS_DIM:
        raise ValueError(f"Expected obs dim {OBS_DIM}, got {obs_raw.shape[1]}")
    if qpos_dim < 15 or qvel_dim < 14:
        raise RuntimeError(f"Unexpected Ant state sizes: qpos={qpos_dim}, qvel={qvel_dim}")

    obs = np.nan_to_num(obs_raw.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    qpos = np.zeros((obs.shape[0], qpos_dim), dtype=np.float64)
    qvel = np.zeros((obs.shape[0], qvel_dim), dtype=np.float64)

    qpos[:, 2] = np.clip(obs[:, 0], 0.05, 5.0)
    qpos[:, 3:7] = rot6d_to_quat_batch(obs[:, 1:7])
    qpos[:, 7:15] = obs[:, 13:21]
    qvel[:, 0:3] = obs[:, 7:10]
    qvel[:, 3:6] = obs[:, 10:13]
    qvel[:, 6:14] = obs[:, 21:29]

    for idx in range(1, qpos.shape[0]):
        qpos[idx, 0:2] = qpos[idx - 1, 0:2] + qvel[idx - 1, 0:2] * timestep
    return qpos, qvel


def episode_to_states(
    episode: dict,
    *,
    model,
    input_raw: bool,
    stats_path: Path | None,
) -> tuple[np.ndarray, np.ndarray, str]:
    if "qpos" in episode and "qvel" in episode:
        qpos = as_numpy_2d(episode["qpos"], "qpos")
        qvel = as_numpy_2d(episode["qvel"], "qvel")
        source = "qpos/qvel"
    else:
        if "obs" not in episode:
            raise KeyError("Episode has neither qpos/qvel nor obs.")
        obs = as_numpy_2d(episode["obs"], "obs")
        if not input_raw:
            if stats_path is None:
                raise ValueError("Obs-only episode needs --stats unless --input-raw is set.")
            obs = denormalize_obs(obs, stats_path)
        qpos, qvel = reconstruct_states_from_obs(
            obs,
            qpos_dim=int(model.nq),
            qvel_dim=int(model.nv),
            timestep=float(model.opt.timestep),
        )
        source = "reconstructed obs"

    if qpos.shape[0] != qvel.shape[0]:
        raise ValueError(f"qpos/qvel length mismatch: {qpos.shape[0]} vs {qvel.shape[0]}")
    if qpos.shape[1] != int(model.nq):
        raise ValueError(f"Expected qpos dim {model.nq}, got {qpos.shape[1]}")
    if qvel.shape[1] != int(model.nv):
        raise ValueError(f"Expected qvel dim {model.nv}, got {qvel.shape[1]}")
    return qpos, qvel, source


def slice_states(
    qpos: np.ndarray,
    qvel: np.ndarray,
    *,
    start_step: int,
    num_steps: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    if start_step < 0:
        raise ValueError("--start-step must be non-negative")
    end = None if num_steps is None else start_step + max(0, num_steps)
    qpos = qpos[start_step:end]
    qvel = qvel[start_step:end]
    if qpos.shape[0] == 0:
        raise RuntimeError("No states selected. Check --start-step/--num-steps.")
    return qpos, qvel


def render_to_mp4(
    xml_path: Path,
    qpos: np.ndarray,
    qvel: np.ndarray,
    out_path: Path,
    *,
    width: int,
    height: int,
    fps: int,
    camera: str | None,
    stride: int,
) -> int:
    import imageio.v2 as imageio
    import mujoco  # type: ignore

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), int(width))
    model.vis.global_.offheight = max(int(model.vis.global_.offheight), int(height))
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=height, width=width)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    frame_count = 0
    try:
        with imageio.get_writer(out_path, fps=fps) as writer:
            for idx in range(0, qpos.shape[0], stride):
                data.qpos[:] = qpos[idx]
                data.qvel[:] = qvel[idx]
                mujoco.mj_forward(model, data)
                if camera:
                    renderer.update_scene(data, camera=camera)
                else:
                    renderer.update_scene(data)
                writer.append_data(renderer.render())
                frame_count += 1
    finally:
        renderer.close()

    if frame_count == 0:
        raise RuntimeError("No frames rendered.")
    return frame_count


def print_episode_listing(files: Sequence[Path]) -> None:
    total = 0
    for file_idx, path in enumerate(files):
        episodes = load_episode_list(path)
        keys = sorted(episodes[0].keys()) if episodes else []
        print(f"[{file_idx}] {path} episodes={len(episodes)} global_range={total}:{total + len(episodes)} keys={keys}")
        total += len(episodes)
    print(f"[total] episodes={total}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render collected Ant episode data to MP4.")
    parser.add_argument("--episodes", type=Path, default=PPO_DEFAULT_OUT, help="episodes_*.pt file or directory.")
    parser.add_argument(
        "--episode-index",
        type=int,
        default=0,
        help="Global episode index across discovered files. For a single file, this is the local index.",
    )
    parser.add_argument("--stats", type=Path, default=None, help="minmax_*.pt for obs-only normalized episodes.")
    parser.add_argument("--input-raw", action="store_true", help="Set when obs-only episodes are already raw.")
    parser.add_argument("--xml", type=Path, default=DEFAULT_XML)
    parser.add_argument("--out", type=Path, default=SCRIPT_DIR / "renders" / "collected_ant_episode.mp4")
    parser.add_argument("--start-step", type=int, default=0)
    parser.add_argument("--num-steps", type=int, default=None)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--stride", type=int, default=2, help="Render every Nth state.")
    parser.add_argument("--camera", default="track", help="MuJoCo camera name. Use empty string for default camera.")
    parser.add_argument("--list", action="store_true", help="List discovered episode files and exit.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    episodes_path = resolve_repo_path(args.episodes)
    xml_path = resolve_repo_path(args.xml)
    out_path = resolve_repo_path(args.out)
    stats_path = resolve_repo_path(args.stats) if args.stats is not None else None
    stride = max(1, int(args.stride))
    camera = args.camera if args.camera else None

    files = find_episode_files(episodes_path)
    if args.list:
        print_episode_listing(files)
        return

    import mujoco  # type: ignore

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    selected = load_episode_by_global_index(files, int(args.episode_index))
    if "qpos" not in selected.episode or "qvel" not in selected.episode:
        stats_path = find_stats_file(episodes_path, stats_path) if not args.input_raw else stats_path

    qpos, qvel, state_source = episode_to_states(
        selected.episode,
        model=model,
        input_raw=bool(args.input_raw),
        stats_path=stats_path,
    )
    qpos, qvel = slice_states(qpos, qvel, start_step=int(args.start_step), num_steps=args.num_steps)

    print(
        f"[load] {selected.path} local_episode={selected.local_index} "
        f"global_episode={selected.global_index} states={qpos.shape[0]} source={state_source}"
    )
    frames = render_to_mp4(
        xml_path,
        qpos,
        qvel,
        out_path,
        width=int(args.width),
        height=int(args.height),
        fps=int(args.fps),
        camera=camera,
        stride=stride,
    )
    print(f"[save] {out_path} frames={frames} fps={args.fps} stride={stride}")


if __name__ == "__main__":
    main()
