# Ant Running Data Generation

Generate Ant Running episodes for task `131`.

## Random / Smooth Baselines

```bash
python Ant/generate_ant_data.py --episodes 1000 --policy smooth_random
```

Default output:

```text
Trajworld_data/UniTraj_pt/ant_running_pt/ant_running/
  episodes_ant_running_chunk1_E1000.pt
  minmax_ant_running.pt
```

Each saved episode is a plain dictionary:

```python
{
    "obs": torch.FloatTensor,      # [T, 29]
    "action": torch.FloatTensor,   # [T, 8]
    "reward": torch.FloatTensor,   # [T]
    "task": torch.LongTensor,      # [T], filled with 131
}
```

The default script saves min-max normalized values in `[0, 1]`, matching the
current WestWorld training path. Use `--no-normalize` only if another pipeline
will normalize the episodes later.

Useful options:

```bash
python Ant/generate_ant_data.py \
  --episodes 5000 \
  --max-steps 500 \
  --policy sinusoidal \
  --chunk-size 1000 \
  --seed 43
```

After generation, point `configs/data/robotics.yaml` at a fresh H5 directory
before training so the new PT episodes are converted:

```yaml
h5_dir: ./dataset_h5_ant
test_h5_dir: ./dataset_h5_ant
```

This repository also includes an Ant-only Hydra config:

```bash
python train.py --config-name config_ant_running
```

For scratch training instead of finetuning from `pre_trained/westworld.ckpt`:

```bash
python train.py --config-name config_ant_running ckpt_path=null
```

## PPO Policy Data

To collect data from imperfect PPO policies during training:

```bash
python Ant/ppo_collect_ant_data.py \
  --total-updates 30 \
  --collect-interval 5 \
  --episodes-per-snapshot 20 \
  --prefix ant_running_ppo
```

The script trains a small PPO policy and periodically snapshots the current
policy. The saved dataset therefore contains episodes from early, middle, and
later imperfect policies instead of only a final converged controller.

Outputs use the same WestWorld training keys:

```python
{
    "obs": torch.FloatTensor,      # [T, 29]
    "action": torch.FloatTensor,   # [T, 8]
    "reward": torch.FloatTensor,   # [T]
    "task": torch.LongTensor,      # [T], filled with 131
}
```

PPO-collected episodes also include render-only fields:

```python
{
    "qpos": torch.FloatTensor,          # [T, nq]
    "qvel": torch.FloatTensor,          # [T, nv]
    "policy_update": torch.LongTensor,  # [T]
}
```

By default PPO data is saved separately under:

```text
Trajworld_data/UniTraj_pt/ant_running_pt/ant_running_ppo/
```

Those extra keys are ignored by the H5 converter, so they do not affect
WestWorld training. To train only on PPO-collected data, use a separate H5 cache:

```bash
rm -rf dataset_h5_ant_running_ppo
python train.py --config-name config_ant_running \
  data.data_dir=./Trajworld_data/UniTraj_pt/ant_running_pt/ant_running_ppo \
  data.h5_dir=./dataset_h5_ant_running_ppo \
  data.test_h5_dir=./dataset_h5_ant_running_ppo
```

## Full PPO to WestWorld Pipeline

To run PPO data collection, render representative PPO rollouts, train
WestWorld from scratch, and render a final PPO-vs-WestWorld comparison in one
sequence:

```bash
python Ant/run_ant_westworld_pipeline.py \
  --total-updates 1000000 \
  --collect-interval 5 \
  --episodes-per-snapshot 20 \
  --prefix ant_running_ppo
```

The script runs these stages:

```text
1. Collect PPO snapshots with Ant/ppo_collect_ant_data.py
2. Render PPO trajectories near 10%, 50%, and 100% of total updates
3. Train WestWorld from scratch with train.py --config-name config_ant_running
4. Render the final PPO rollout against the trained WestWorld prediction
```

By default, each run gets fresh output directories using a timestamped run name:

```text
Trajworld_data/UniTraj_pt/ant_running_pt/<run-name>/   # PPO episodes and minmax
Ant/ppo_checkpoints/<run-name>/                        # PPO checkpoints
dataset_h5_ant_running_ppo_<run-name>/                 # H5 cache
CTFM/Ant-Running-WestWorld-<run-name>/                 # WestWorld checkpoints
outputs/<run-name>/                                    # 10/50/100% PPO videos
Ant/renders/<run-name>_westworld_vs_gt.mp4             # final comparison video
```

Use `--run-name` to make those paths stable:

```bash
python Ant/run_ant_westworld_pipeline.py \
  --run-name ant-ppo-million-v1 \
  --total-updates 1000000 \
  --collect-interval 5 \
  --episodes-per-snapshot 20
```

For a quick path check without starting training:

```bash
python Ant/run_ant_westworld_pipeline.py --dry-run --run-name test-run
```

The pipeline prevents accidental resume for the WestWorld scratch stage by
using a fresh `exp_name` and H5 directory. If checkpoint files already exist for
the selected `--train-exp-name`, the script fails before training.

If any command fails, the error message is prefixed with the failing stage, for
example:

```text
[Stage 3: Train WestWorld from scratch] failed: ...
```

Long PPO collection can produce a large dataset. `ppo_collect_ant_data.py`
flushes collected snapshots to temporary raw chunks during collection, then
writes normalized final chunks after `minmax_*.pt` is computed. Temporary raw
chunks are deleted by default; pass `--keep-raw-chunks` directly to
`ppo_collect_ant_data.py` when debugging that lower-level script.

## PPO in WestWorld Simulation

To train a new PPO policy from scratch using WestWorld as the transition
simulator:

```bash
python Ant/ppo_train_westworld_env.py \
  --ckpt ./CTFM/Ant-Running-WestWorld/checkpoints/last.ckpt \
  --stats Trajworld_data/UniTraj_pt/ant_running_pt/ant_running_ppo/minmax_ant_running_ppo.pt \
  --total-updates 100 \
  --rollout-steps 1024
```

This script uses the real MuJoCo Ant environment only for reset states. After
reset, PPO actions are applied to the frozen WestWorld model:

```text
current observation + PPO action -> WestWorld -> next observation
```

The current WestWorld Mamba kernels require a CUDA-enabled environment. The
normalization statistics passed with `--stats` should come from the dataset used
to train the selected WestWorld checkpoint.

Rewards and termination are computed from the WestWorld-predicted next
observation. Policy checkpoints are written to:

```text
Ant/ppo_westworld_checkpoints/
```

## 3D Rendering

Render a PPO-collected episode to an MP4:

```bash
python Ant/render_ant_episode.py \
  --episodes Trajworld_data/UniTraj_pt/ant_running_pt/ant_running_ppo \
  --episode-index 0 \
  --out Ant/renders/ant_episode.mp4 \
  --width 640 \
  --height 480
```

On headless servers, set MuJoCo rendering variables if needed:

```bash
export MUJOCO_GL=egl
python Ant/render_ant_episode.py --episode-index 0
```

If EGL is unavailable in the container, try software rendering:

```bash
MUJOCO_GL=osmesa python Ant/render_ant_episode.py --episode-index 0
```

Render the PPO ground-truth trajectory and a WestWorld predicted trajectory side
by side:

```bash
python Ant/render_westworld_prediction.py \
  --ckpt ./CTFM/Ant-Running-WestWorld/checkpoints/last.ckpt \
  --ppo-ckpt Ant/ppo_westworld_checkpoints/ppo_westworld_ant_update_0100.pt \
  --stats Trajworld_data/UniTraj_pt/ant_running_pt/ant_running_ppo/minmax_ant_running_ppo.pt \
  --out Ant/renders/westworld_vs_gt.mp4 \
  --width 640 \
  --height 480
```

With `--ppo-ckpt`, the script first runs that policy in the real MuJoCo Ant
environment. This fresh rollout becomes the GT trajectory and its action sequence
is passed to WestWorld for the side-by-side prediction. The rollout is
deterministic by default; pass `--ppo-stochastic` to sample policy actions.

The previous `--episodes ... --episode-index ...` input remains available when
rendering an already collected PPO episode.

WestWorld predicts the Ant observation channels, not root x/y position. The
renderer reconstructs predicted root x/y by integrating predicted linear
velocity from the GT prefix boundary. For a shape-only comparison at the GT
position, pass `--xy-mode gt`.
