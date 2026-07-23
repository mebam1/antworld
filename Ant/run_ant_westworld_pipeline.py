#!/usr/bin/env python3
"""Run the Ant PPO data -> WestWorld training -> render comparison pipeline."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PPO_ROOT = REPO_ROOT / "Trajworld_data" / "UniTraj_pt" / "ant_running_pt"


class StageError(RuntimeError):
    pass


def rel_cli_path(path: Path) -> str:
    path = path.resolve()
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def run_command(stage: str, cmd: list[str]) -> None:
    printable = " ".join(cmd)
    print(f"\n[{stage}] running:\n{printable}\n", flush=True)
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    try:
        subprocess.run(cmd, cwd=REPO_ROOT, env=env, check=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"command exited with code {exc.returncode}: {printable}") from exc


def run_stage(number: int, name: str, fn) -> None:
    stage = f"Stage {number}: {name}"
    try:
        fn(stage)
    except Exception as exc:
        message = f"[{stage}] failed: {exc}"
        raise StageError(message) from exc


def load_episode_update(episode: dict) -> int:
    policy_update = episode.get("policy_update")
    if policy_update is None:
        raise KeyError("episode does not contain policy_update")
    if hasattr(policy_update, "reshape"):
        return int(policy_update.reshape(-1)[0].item())
    if isinstance(policy_update, (list, tuple)):
        return int(policy_update[0])
    return int(policy_update)


def episode_files(root: Path, prefix: str) -> list[Path]:
    files = sorted(root.glob(f"episodes_{prefix}_chunk*_E*.pt"))
    if not files:
        files = sorted(root.glob("**/episodes_*.pt"))
    if not files:
        raise FileNotFoundError(f"No episodes_*.pt found under {root}")
    return files


def select_episode_for_update(root: Path, prefix: str, target_update: int, out_file: Path) -> tuple[int, Path]:
    import torch

    best_key: tuple[int, int] | None = None
    best_episode = None
    best_source: Path | None = None
    best_update = -1

    for path in episode_files(root, prefix):
        episodes = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(episodes, list):
            continue
        for episode in episodes:
            update = load_episode_update(episode)
            key = (abs(update - target_update), -update)
            if best_key is None or key < best_key:
                best_key = key
                best_episode = episode
                best_source = path
                best_update = update

    if best_episode is None or best_source is None:
        raise RuntimeError(f"No episode with policy_update found under {root}")

    out_file.parent.mkdir(parents=True, exist_ok=True)
    torch.save([best_episode], out_file)
    print(
        f"[select episode] target_update={target_update} "
        f"selected_update={best_update} source={rel_cli_path(best_source)} "
        f"out={rel_cli_path(out_file)}",
        flush=True,
    )
    return best_update, out_file


def latest_checkpoint(policy_dir: Path, total_updates: int) -> Path:
    expected = policy_dir / f"ppo_ant_update_{total_updates:04d}.pt"
    if expected.is_file():
        return expected
    candidates = sorted(policy_dir.glob("ppo_ant_update_*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"No PPO checkpoints found under {policy_dir}")
    return candidates[0]


def westworld_checkpoint(exp_name: str) -> Path:
    ckpt_dir = REPO_ROOT / "CTFM" / exp_name / "checkpoints"
    last = ckpt_dir / "last.ckpt"
    if last.is_file():
        return last
    candidates = sorted(ckpt_dir.glob("**/*.ckpt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"No WestWorld checkpoints found under {ckpt_dir}")
    return candidates[0]


def ensure_scratch_train_target(exp_name: str) -> None:
    ckpt_dir = REPO_ROOT / "CTFM" / exp_name / "checkpoints"
    existing = sorted(ckpt_dir.glob("**/*.ckpt")) if ckpt_dir.exists() else []
    if existing:
        examples = ", ".join(rel_cli_path(p) for p in existing[:3])
        raise RuntimeError(
            "scratch training requested, but checkpoint files already exist in "
            f"{rel_cli_path(ckpt_dir)}: {examples}"
        )


def maybe_append(flag: str, value, cmd: list[str]) -> None:
    if value is not None:
        cmd.extend([flag, str(value)])


def add_common_render_args(args: argparse.Namespace, cmd: list[str]) -> None:
    cmd.extend(["--width", str(args.width), "--height", str(args.height)])
    cmd.extend(["--fps", str(args.fps), "--stride", str(args.render_stride)])
    if args.max_frames is not None:
        cmd.extend(["--max-frames", str(args.max_frames)])
    if args.camera is not None:
        cmd.extend(["--camera", args.camera])
    if args.allow_mujoco_py_fallback and cmd[1].endswith("render_ant_episode.py"):
        cmd.append("--allow-mujoco-py-fallback")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full Ant PPO and WestWorld pipeline.")
    parser.add_argument("--run-name", default=None, help="Run suffix used for fresh output directories.")
    parser.add_argument("--config-name", default="config_ant_running")
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--device", default=None, help="Forwarded to PPO collection and rendering scripts.")

    parser.add_argument("--total-updates", type=int, default=1_000_000)
    parser.add_argument("--collect-interval", type=int, default=5)
    parser.add_argument("--episodes-per-snapshot", type=int, default=20)
    parser.add_argument("--prefix", default="ant_running_ppo")
    parser.add_argument("--ppo-out-dir", type=Path, default=None)
    parser.add_argument("--policy-dir", type=Path, default=None)
    parser.add_argument("--rollout-steps", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--deterministic-collection", action="store_true")
    parser.add_argument("--no-joint-angle-clamp", action="store_true")

    parser.add_argument("--train-exp-name", default=None)
    parser.add_argument("--h5-dir", type=Path, default=None)
    parser.add_argument("--wandb-mode", default="offline")
    parser.add_argument("--train-extra-override", action="append", default=[], help="Extra Hydra override for train.py.")

    parser.add_argument("--render-out-dir", type=Path, default=None)
    parser.add_argument("--comparison-out", type=Path, default=None)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--render-stride", type=int, default=2)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--camera", default="track", help="Use empty string for MuJoCo default camera.")
    parser.add_argument("--allow-mujoco-py-fallback", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Print resolved settings without running stages.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.total_updates <= 0:
        print("--total-updates must be positive for the comparison stage.", file=sys.stderr)
        return 2

    run_name = args.run_name or datetime.now().strftime("pipeline-%Y%m%d-%H%M%S")
    ppo_out_dir = (args.ppo_out_dir or DEFAULT_PPO_ROOT / run_name).resolve()
    policy_dir = (args.policy_dir or REPO_ROOT / "Ant" / "ppo_checkpoints" / run_name).resolve()
    h5_dir = (args.h5_dir or REPO_ROOT / f"dataset_h5_ant_running_ppo_{run_name}").resolve()
    train_exp_name = args.train_exp_name or f"Ant-Running-WestWorld-{run_name}"
    render_out_dir = (args.render_out_dir or REPO_ROOT / "outputs" / run_name).resolve()
    comparison_out = (args.comparison_out or REPO_ROOT / "Ant" / "renders" / f"{run_name}_westworld_vs_gt.mp4").resolve()
    stats_path = ppo_out_dir / f"minmax_{args.prefix}.pt"

    print("[resolved paths]")
    for label, path in [
        ("ppo_out_dir", ppo_out_dir),
        ("policy_dir", policy_dir),
        ("h5_dir", h5_dir),
        ("train_exp_name", Path(train_exp_name)),
        ("render_out_dir", render_out_dir),
        ("comparison_out", comparison_out),
        ("stats_path", stats_path),
    ]:
        print(f"  {label}: {path if label == 'train_exp_name' else rel_cli_path(path)}")

    if args.dry_run:
        return 0

    def stage1(stage: str) -> None:
        cmd = [
            sys.executable,
            "Ant/ppo_collect_ant_data.py",
            "--total-updates",
            str(args.total_updates),
            "--collect-interval",
            str(args.collect_interval),
            "--episodes-per-snapshot",
            str(args.episodes_per_snapshot),
            "--prefix",
            args.prefix,
            "--out-dir",
            rel_cli_path(ppo_out_dir),
            "--policy-dir",
            rel_cli_path(policy_dir),
            "--seed",
            str(args.seed),
        ]
        maybe_append("--device", args.device, cmd)
        maybe_append("--rollout-steps", args.rollout_steps, cmd)
        maybe_append("--max-steps", args.max_steps, cmd)
        if args.deterministic_collection:
            cmd.append("--deterministic-collection")
        if args.no_joint_angle_clamp:
            cmd.append("--no-joint-angle-clamp")
        run_command(stage, cmd)
        if not stats_path.is_file():
            raise FileNotFoundError(f"Expected stats file was not created: {rel_cli_path(stats_path)}")

    def stage2(stage: str) -> None:
        targets = [
            ("10pct", int(round(args.total_updates * 0.10))),
            ("50pct", int(round(args.total_updates * 0.50))),
            ("100pct", args.total_updates),
        ]
        selected_dir = render_out_dir / "selected_episodes"
        for label, target_update in targets:
            selected_update, selected_file = select_episode_for_update(
                ppo_out_dir,
                args.prefix,
                target_update,
                selected_dir / f"episode_{label}_target_{target_update}.pt",
            )
            out_file = render_out_dir / f"ant_episode_{label}_update_{selected_update}.mp4"
            cmd = [
                sys.executable,
                "Ant/render_ant_episode.py",
                "--episodes",
                rel_cli_path(selected_file),
                "--episode-index",
                "0",
                "--out",
                rel_cli_path(out_file),
            ]
            add_common_render_args(args, cmd)
            run_command(f"{stage} ({label})", cmd)

    def stage3(stage: str) -> None:
        ensure_scratch_train_target(train_exp_name)
        cmd = [
            sys.executable,
            "train.py",
            "--config-name",
            args.config_name,
            "ckpt_path=null",
            f"exp_name={train_exp_name}",
            f"data.data_dir={rel_cli_path(ppo_out_dir)}",
            f"data.h5_dir={rel_cli_path(h5_dir)}",
            f"data.test_h5_dir={rel_cli_path(h5_dir)}",
        ]
        if args.wandb_mode is not None:
            cmd.append(f"wandb_mode={args.wandb_mode}")
        cmd.extend(args.train_extra_override)
        run_command(stage, cmd)
        westworld_checkpoint(train_exp_name)

    def stage4(stage: str) -> None:
        ww_ckpt = westworld_checkpoint(train_exp_name)
        ppo_ckpt = latest_checkpoint(policy_dir, args.total_updates)
        cmd = [
            sys.executable,
            "Ant/render_westworld_prediction.py",
            "--config-name",
            args.config_name,
            "--ckpt",
            rel_cli_path(ww_ckpt),
            "--ppo-ckpt",
            rel_cli_path(ppo_ckpt),
            "--stats",
            rel_cli_path(stats_path),
            "--out",
            rel_cli_path(comparison_out),
        ]
        maybe_append("--device", args.device, cmd)
        add_common_render_args(args, cmd)
        run_command(stage, cmd)

    stages: Iterable[tuple[int, str, object]] = [
        (1, "PPO data collection", stage1),
        (2, "Render 10/50/100 percent PPO trajectories", stage2),
        (3, "Train WestWorld from scratch", stage3),
        (4, "Render PPO vs WestWorld rollout comparison", stage4),
    ]

    try:
        for number, name, fn in stages:
            run_stage(number, name, fn)
    except StageError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print("\n[pipeline done]")
    print(f"ppo data: {rel_cli_path(ppo_out_dir)}")
    print(f"ppo renders: {rel_cli_path(render_out_dir)}")
    print(f"westworld checkpoint: {rel_cli_path(westworld_checkpoint(train_exp_name))}")
    print(f"comparison render: {rel_cli_path(comparison_out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
