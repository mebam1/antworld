# Docker

이 프로젝트는 `docker compose`로 WestWorld 컨테이너를 실행합니다. 현재 작업 디렉터리는 컨테이너 안의 `/workspace/WestWorld`에 bind mount되므로, 컨테이너에서 생성한 데이터, checkpoint, render 결과는 호스트 프로젝트 폴더에 바로 반영됩니다.

## Build

```powershell
docker compose build
```

## Ant PPO 데이터 수집

Ant PPO 데이터는 PPO 학습 중간 snapshot을 섞지 않습니다. PPO를 `--total-updates`까지 끝까지 학습한 뒤, 최종 PPO policy로 새 episode를 수집합니다.

PowerShell:

```powershell
docker compose run --rm westworld python Ant/ppo_collect_ant_data.py `
  --total-updates 30 `
  --collect-episodes 1000 `
  --prefix ant_running_ppo
```

bash:

```bash
docker compose run --rm westworld python Ant/ppo_collect_ant_data.py \
  --total-updates 30 \
  --collect-episodes 1000 \
  --prefix ant_running_ppo
```

기존 PPO checkpoint에서 이어서 학습하려면 `--resume-ppo-ckpt`를 추가합니다. `--total-updates`는 추가 update 수가 아니라 최종 목표 update 번호입니다. 예를 들어 `ppo_ant_update_0030.pt`에서 아래 명령을 실행하면 31부터 1000까지 이어서 학습한 뒤 episode 10개를 수집합니다.

```bash
docker compose run --rm westworld python Ant/ppo_collect_ant_data.py \
  --resume-ppo-ckpt Ant/ppo_checkpoints/ppo_ant_update_0030.pt \
  --total-updates 1000 \
  --collect-episodes 10 \
  --prefix ant_running_ppo
```

참고:
- `--episodes-per-snapshot`은 호환용 alias로 남아 있지만, 새 코드에서는 최종 policy 수집 episode 수로만 쓰입니다.
- `--collect-interval`은 더 이상 데이터 수집 주기를 제어하지 않습니다.
- Ant observation은 이제 quaternion이 아니라 6D rotation representation을 포함하므로 `obs` 차원은 `31`입니다.

## Ant PPO 데이터로 학습

31D Ant 데이터로 학습할 때는 기존 29D H5 cache를 재사용하면 안 됩니다. 새 `h5_dir`를 쓰거나 기존 cache를 지우고 다시 생성하세요.

PowerShell:

```powershell
docker compose run --rm westworld python train.py --config-name config_ant_running `
  data.data_dir=./Trajworld_data/UniTraj_pt/ant_running_pt/ant_running_ppo `
  data.h5_dir=./dataset_h5_ant_running_ppo_31d `
  data.test_h5_dir=./dataset_h5_ant_running_ppo_31d
```

bash:

```bash
docker compose run --rm westworld python train.py --config-name config_ant_running \
  data.data_dir=./Trajworld_data/UniTraj_pt/ant_running_pt/ant_running_ppo \
  data.h5_dir=./dataset_h5_ant_running_ppo_31d \
  data.test_h5_dir=./dataset_h5_ant_running_ppo_31d
```

## 수집 데이터 렌더링

수집된 PPO episode를 그대로 MP4로 렌더링하려면 `Ant/render_collected_ant_data.py`를 사용합니다. `ppo_collect_ant_data.py`로 만든 데이터는 `qpos/qvel`을 포함하므로 실제 MuJoCo state로 렌더링됩니다.

먼저 수집된 episode 파일과 global index 범위를 확인할 수 있습니다.

PowerShell:

```powershell
docker compose run --rm westworld python Ant/render_collected_ant_data.py `
  --episodes Trajworld_data/UniTraj_pt/ant_running_pt/ant_running_ppo `
  --list
```

bash:

```bash
docker compose run --rm westworld python Ant/render_collected_ant_data.py \
  --episodes Trajworld_data/UniTraj_pt/ant_running_pt/ant_running_ppo \
  --list
```

episode 하나를 MP4로 저장합니다.

PowerShell:

```powershell
docker compose run --rm westworld python Ant/render_collected_ant_data.py `
  --episodes Trajworld_data/UniTraj_pt/ant_running_pt/ant_running_ppo `
  --episode-index 0 `
  --out Ant/renders/collected_ant_episode.mp4 `
  --width 640 `
  --height 480
```

bash:

```bash
docker compose run --rm westworld python Ant/render_collected_ant_data.py \
  --episodes Trajworld_data/UniTraj_pt/ant_running_pt/ant_running_ppo \
  --episode-index 0 \
  --out Ant/renders/collected_ant_episode.mp4 \
  --width 640 \
  --height 480
```

짧은 구간만 빠르게 확인하려면 `--start-step`, `--num-steps`, `--stride`를 조정합니다.

```bash
docker compose run --rm westworld python Ant/render_collected_ant_data.py \
  --episodes Trajworld_data/UniTraj_pt/ant_running_pt/ant_running_ppo \
  --episode-index 0 \
  --start-step 0 \
  --num-steps 200 \
  --stride 2 \
  --out Ant/renders/collected_ant_episode_short.mp4
```

## WestWorld 예측 렌더링

PPO checkpoint로 MuJoCo GT rollout을 새로 만들고, 같은 action sequence를 WestWorld에 넣어 예측 trajectory를 렌더링합니다. 예측된 Ant 6D rotation은 렌더링 전에 quaternion으로 변환됩니다.

PowerShell:

```powershell
docker compose run --rm westworld python Ant/render_westworld_prediction.py `
  --ckpt ./CTFM/Ant-Running-WestWorld/checkpoints/last.ckpt `
  --ppo-ckpt Ant/ppo_checkpoints/ppo_ant_update_0030.pt `
  --stats Trajworld_data/UniTraj_pt/ant_running_pt/ant_running_ppo/minmax_ant_running_ppo.pt `
  --out Ant/renders/westworld_vs_gt.mp4 `
  --width 640 `
  --height 480
```

bash:

```bash
docker compose run --rm westworld python Ant/render_westworld_prediction.py \
  --ckpt ./CTFM/Ant-Running-WestWorld/checkpoints/last.ckpt \
  --ppo-ckpt Ant/ppo_checkpoints/ppo_ant_update_0030.pt \
  --stats Trajworld_data/UniTraj_pt/ant_running_pt/ant_running_ppo/minmax_ant_running_ppo.pt \
  --out Ant/renders/westworld_vs_gt.mp4 \
  --width 640 \
  --height 480
```

## Closed-Loop Evaluation

Closed-loop 평가는 다음 설정을 따릅니다.

- 각 episode를 길이 150 segment로 분할
- 첫 50 step을 history input으로 사용
- 다음 100 step을 autoregressive rollout
- GT future trajectory와 비교해 MAE/MSE 출력

기본은 저장된 PPO episode 전체를 평가하고, 선택한 segment 하나를 side-by-side MP4로 렌더링합니다.

PowerShell:

```powershell
docker compose run --rm westworld python Ant/render_westworld_closed_loop_prediction.py `
  --ckpt ./CTFM/Ant-Running-WestWorld/checkpoints/last.ckpt `
  --episodes Trajworld_data/UniTraj_pt/ant_running_pt/ant_running_ppo `
  --stats Trajworld_data/UniTraj_pt/ant_running_pt/ant_running_ppo/minmax_ant_running_ppo.pt `
  --out Ant/renders/westworld_closed_loop_eval.mp4 `
  --width 640 `
  --height 480
```

bash:

```bash
docker compose run --rm westworld python Ant/render_westworld_closed_loop_prediction.py \
  --ckpt ./CTFM/Ant-Running-WestWorld/checkpoints/last.ckpt \
  --episodes Trajworld_data/UniTraj_pt/ant_running_pt/ant_running_ppo \
  --stats Trajworld_data/UniTraj_pt/ant_running_pt/ant_running_ppo/minmax_ant_running_ppo.pt \
  --out Ant/renders/westworld_closed_loop_eval.mp4 \
  --width 640 \
  --height 480
```

MAE/MSE만 빠르게 확인하고 렌더링을 생략하려면 `--no-render`를 추가합니다.

```bash
docker compose run --rm westworld python Ant/render_westworld_closed_loop_prediction.py \
  --ckpt ./CTFM/Ant-Running-WestWorld/checkpoints/last.ckpt \
  --episodes Trajworld_data/UniTraj_pt/ant_running_pt/ant_running_ppo \
  --stats Trajworld_data/UniTraj_pt/ant_running_pt/ant_running_ppo/minmax_ant_running_ppo.pt \
  --no-render
```

저장된 episode 대신 PPO checkpoint로 fresh MuJoCo rollout을 만든 뒤 평가하려면 `--gt-source ppo`를 사용합니다.

```bash
docker compose run --rm westworld python Ant/render_westworld_closed_loop_prediction.py \
  --gt-source ppo \
  --ppo-ckpt Ant/ppo_checkpoints/ppo_ant_update_0030.pt \
  --ckpt ./CTFM/Ant-Running-WestWorld/checkpoints/last.ckpt \
  --episodes Trajworld_data/UniTraj_pt/ant_running_pt/ant_running_ppo \
  --stats Trajworld_data/UniTraj_pt/ant_running_pt/ant_running_ppo/minmax_ant_running_ppo.pt \
  --out Ant/renders/westworld_closed_loop_eval_fresh_ppo.mp4
```

## Full Pipeline

PPO 최종 policy 데이터 수집, 최종 PPO rollout 렌더링, WestWorld scratch 학습, 최종 비교 렌더링을 순서대로 실행합니다.

```bash
docker compose run --rm westworld python Ant/run_ant_westworld_pipeline.py \
  --run-name ant-final-ppo-v1 \
  --total-updates 30 \
  --collect-episodes 1000 \
  --prefix ant_running_ppo
```

## GPU 선택

기본 설정은 사용 가능한 GPU를 컨테이너에 노출합니다. 특정 GPU만 쓰려면 실행 전에 `NVIDIA_VISIBLE_DEVICES`를 지정합니다.

PowerShell:

```powershell
$env:NVIDIA_VISIBLE_DEVICES="0"; docker compose run --rm westworld python Ant/ppo_collect_ant_data.py `
  --total-updates 30 `
  --collect-episodes 1000 `
  --prefix ant_running_ppo
```

bash:

```bash
NVIDIA_VISIBLE_DEVICES=0 docker compose run --rm westworld python Ant/ppo_collect_ant_data.py \
  --total-updates 30 \
  --collect-episodes 1000 \
  --prefix ant_running_ppo
```
