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

The setup script installs the main dependencies, including MuJoCo 2.1, PyTorch 2.4.1, `mujoco-py`, `d4rl`, `lightning`, `wandb`, and the local `mjrl` / `mjmpc` packages.

The first argument controls where non-environment files such as MuJoCo, caches, build temp files, and runtime config directories are stored.

## Docker

Build the CUDA 11.8 container:

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

The compose file mounts `Trajworld_data/`, `dataset_h5/`, `dataset_h5_ant_running/`, `pre_trained/`, `outputs/`, `figure/`, and `wandb/` from the host, so datasets and checkpoints are not baked into the image. GPU execution requires Docker with NVIDIA Container Toolkit.

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
