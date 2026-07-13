# Ant Running Data Generation

Generate Ant Running episodes for task `131`.

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
