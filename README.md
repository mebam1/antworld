<div align="center">

# WestWorld

**WestWorld: A Knowledge-Encoded Scalable Trajectory World Model for Diverse Robotic Systems**

**ICML 2026 Spotlight (Top 2.2%)**

[Paper](https://arxiv.org/abs/2603.14392) | [Project Page](https://westworldrobot.github.io/) | [Pretraining Dataset](https://huggingface.co/datasets/ywang077/Trajectory_world_model_dataset/) | [Checkpoints](https://huggingface.co/ywang077/WestWorld)

</div>

WestWorld is a scalable trajectory world model for diverse robotic systems. This repository contains training, evaluation, and robot-structure preprocessing code used in the paper.

## Installation

Recommended setup:

```bash
bash setup_env.bash /path/to/external/files westworld
conda activate westworld
```

The setup script installs the main dependencies, including MuJoCo 2.1, PyTorch 2.11.0 with CUDA 12.8 wheels, `mujoco-py`, `d4rl`, `lightning`, `wandb`, and the local `mjrl` / `mjmpc` packages.

The first argument controls where non-environment files such as MuJoCo, caches, build temp files, and runtime config directories are stored.

## Docker

Build the CUDA 12.8 development image:

```bash
DOCKER_BUILDKIT=1 COMPOSE_DOCKER_CLI_BUILD=1 docker compose build
```

After the first successful build, ordinary source-code changes should reuse the
expensive dependency layers. Use `--no-cache` only when CUDA, PyTorch, Conda, or
`setup_env.bash` dependency versions change:

```bash
docker compose build
```

Start an interactive shell:

```bash
docker compose run --rm westworld
```

Run training inside the container:

```bash
docker compose run --rm westworld python train.py
```

The compose file bind-mounts the whole repository at `/workspace/WestWorld`.
Host code changes are visible inside the container immediately, and files written
by the container under directories such as `outputs/`, `figure/`, `nohup/`, and
`wandb/` appear on the host immediately. Long-running Python processes still need
to be restarted to load changed Python source.

GPU execution requires Docker with NVIDIA Container Toolkit. By default the
container can see all GPUs. To restrict GPUs, set `NVIDIA_VISIBLE_DEVICES` before
running compose, for example:

```bash
NVIDIA_VISIBLE_DEVICES=0 docker compose run --rm westworld python train.py
```

PowerShell:

```powershell
$env:NVIDIA_VISIBLE_DEVICES="0"; docker compose run --rm westworld python train.py
```

## Quick Start

Train with the default config:

```bash
./run.sh
```

or

```bash
python train.py
```

`run.sh` launches `nohup python train.py` and writes logs to `nohup/`.

## Configuration

Main experiment settings are defined in [`configs/config.yaml`](configs/config.yaml).

- `configs/data/`: dataset-specific configuration
- `configs/method/`: model-specific configuration

Use `configs/config.yaml` to choose the active `data` and `method` config. For detailed hyperparameters, see the corresponding YAML file under `configs/method/`, such as:

- `configs/method/WestWorld.yaml`
- `configs/method/Trajworld.yaml`
- `configs/method/TDM.yaml`
- `configs/method/MLPEnsemble.yaml`

## Evaluation

To evaluate a pretrained model:

1. Put the checkpoint in `pre_trained/`.
2. Edit [`configs/config.yaml`](configs/config.yaml) and set `ckpt_path`, for example:

```yaml
ckpt_path: './pre_trained/westworld.ckpt'
```

3. Run the evaluation script that matches the model:

```bash
python evaluation_westworld.py
python evaluation_trajworld.py
python evaluation_TDM.py
python evaluation_MLPEnsemble.py
```

## MPPI Control

Part of the MPPI control experiments in the paper are provided in `MPPI/`, including Hopper and Walker2d with both ground truth dynamics and learned world models.

See [`MPPI/README.md`](MPPI/README.md) for details.


## Adding New Robots

The robot structure files used for the UniTraj and OpenX components in the paper are already processed. To add a new robot:

1. Place the robot XML (MJCF) file in `robotics_structure_xml/`.
2. Run:

```bash
python utils/preprocess_robotics_xml.py \
  --xml_dir robotics_structure_xml \
  --out_yaml robotics_structure_xml/robotics_structure_summary.yaml
```

3. Update `robotics_structure_xml/general_task_specific.yaml` with the task definition, including the observation and action body nodes for the new robot.

## Citation

If you find this repository useful, please cite:

```bibtex
@inproceedings{wang2026westworld,
  title={WestWorld: A Knowledge-Encoded Scalable Trajectory World Model for Diverse Robotic Systems},
  author={Wang, Yuchen and Kong, Jiangtao and Wei, Sizhe and Li, Xiaochang and Lin, Haohong and Zhao, Hongjue and Zhou, Tianyi and Gan, Lu and Shao, Huajie},
  booktitle={Forty-third International Conference on Machine Learning},
  year={2026},
  url={https://openreview.net/forum?id=ncRRCG4BfP}
}
```

## Acknowledgements

This mppi control part builds on:

- [mjmpc](https://github.com/google-deepmind/mujoco_mpc)
- [mjrl](https://github.com/aravindr93/mjrl)
- [Whole-Body MPPI](https://github.com/jrapudg/RTWholeBodyMPPI)
