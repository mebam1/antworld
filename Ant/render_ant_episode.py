#!/usr/bin/env python3
"""Render saved Ant episodes that include raw MuJoCo qpos/qvel states."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List

import numpy as np
import torch


os.environ.setdefault("MUJOCO_GL", "egl")

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from generate_ant_data import DEFAULT_OUT, DEFAULT_XML  # noqa: E402


def find_episode_file(root: Path) -> Path:
    if root.is_file():
        return root
    files = sorted(root.glob("**/episodes_*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        raise FileNotFoundError(f"No episodes_*.pt found under {root}")
    return files[0]


def load_episode(path: Path, episode_index: int) -> dict:
    episodes = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(episodes, list):
        raise TypeError(f"Expected {path} to contain a list of episodes, got {type(episodes)!r}")
    if episode_index < 0 or episode_index >= len(episodes):
        raise IndexError(f"episode_index={episode_index} out of range for {len(episodes)} episodes")
    episode = episodes[episode_index]
    if "qpos" not in episode or "qvel" not in episode:
        raise KeyError(
            "Selected episode does not contain qpos/qvel. "
            "Use Ant/ppo_collect_ant_data.py to generate renderable episodes."
        )
    return episode


def render_with_mujoco(
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


def render_with_mujoco_py(
    xml_path: Path,
    qpos: np.ndarray,
    qvel: np.ndarray,
    *,
    width: int,
    height: int,
    camera: str | None,
    stride: int,
) -> List[np.ndarray]:
    import mujoco_py  # type: ignore

    model = mujoco_py.load_model_from_path(str(xml_path))
    sim = mujoco_py.MjSim(model)
    frames: List[np.ndarray] = []

    for idx in range(0, qpos.shape[0], stride):
        sim.data.qpos[:] = qpos[idx]
        sim.data.qvel[:] = qvel[idx]
        sim.forward()
        frame = sim.render(width, height, camera_name=camera)
        frames.append(np.asarray(frame).copy())

    return frames


def render_episode(
    xml_path: Path,
    episode: dict,
    *,
    width: int,
    height: int,
    camera: str | None,
    stride: int,
    max_frames: int | None,
) -> List[np.ndarray]:
    qpos = episode["qpos"].detach().cpu().numpy()
    qvel = episode["qvel"].detach().cpu().numpy()
    if max_frames is not None:
        qpos = qpos[: max_frames * stride]
        qvel = qvel[: max_frames * stride]

    try:
        return render_with_mujoco(
            xml_path,
            qpos,
            qvel,
            width=width,
            height=height,
            camera=camera,
            stride=stride,
        )
    except Exception as mujoco_error:
        print(f"[warn] mujoco renderer failed: {mujoco_error}")
        print("[warn] falling back to mujoco_py renderer")
        return render_with_mujoco_py(
            xml_path,
            qpos,
            qvel,
            width=width,
            height=height,
            camera=camera,
            stride=stride,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render an Ant PPO-collected episode to video.")
    parser.add_argument("--episodes", type=Path, default=DEFAULT_OUT, help="episodes_*.pt file or directory.")
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--xml", type=Path, default=DEFAULT_XML)
    parser.add_argument("--out", type=Path, default=SCRIPT_DIR / "renders" / "ant_episode.mp4")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--camera", default="track", help="MuJoCo camera name. Use empty string for default camera.")
    parser.add_argument("--stride", type=int, default=2, help="Render every Nth saved state.")
    parser.add_argument("--max-frames", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    episode_file = find_episode_file(args.episodes)
    camera = args.camera if args.camera else None
    episode = load_episode(episode_file, args.episode_index)

    print(f"[load] {episode_file} episode={args.episode_index}")
    frames = render_episode(
        args.xml.resolve(),
        episode,
        width=args.width,
        height=args.height,
        camera=camera,
        stride=max(1, args.stride),
        max_frames=args.max_frames,
    )
    if not frames:
        raise RuntimeError("No frames rendered.")

    import imageio.v2 as imageio

    args.out.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(args.out, frames, fps=args.fps)
    print(f"[save] {args.out} frames={len(frames)} fps={args.fps}")


if __name__ == "__main__":
    main()
