# Docker

이 프로젝트는 `docker compose`로 개발용 컨테이너를 실행합니다. 현재 작업 디렉터리 전체가 컨테이너의 `/workspace/WestWorld`에 bind mount되므로, 컨테이너에서 생성한 결과 파일은 호스트 프로젝트 폴더에 바로 반영됩니다.

## Build

```powershell
docker compose build
```

## Ant PPO 데이터 수집 실행

PowerShell:

```powershell
docker compose run --rm westworld python Ant/ppo_collect_ant_data.py `
  --total-updates 30 `
  --collect-interval 5 `
  --episodes-per-snapshot 20 `
  --prefix ant_running_ppo
```

bash:

```bash
docker compose run --rm westworld python Ant/ppo_collect_ant_data.py \
  --total-updates 30 \
  --collect-interval 5 \
  --episodes-per-snapshot 20 \
  --prefix ant_running_ppo
```

## Ant PPO 데이터로 학습 실행

PowerShell:

```powershell
docker compose run --rm westworld python train.py --config-name config_ant_running `
  data.data_dir=./Trajworld_data/UniTraj_pt/ant_running_pt/ant_running_ppo `
  data.h5_dir=./dataset_h5_ant_running_ppo `
  data.test_h5_dir=./dataset_h5_ant_running_ppo
```

bash:

```bash
docker compose run --rm westworld python train.py --config-name config_ant_running \
  data.data_dir=./Trajworld_data/UniTraj_pt/ant_running_pt/ant_running_ppo \
  data.h5_dir=./dataset_h5_ant_running_ppo \
  data.test_h5_dir=./dataset_h5_ant_running_ppo
```

## WestWorld 예측 렌더링 실행

PowerShell:

```powershell
docker compose run --rm westworld python Ant/render_westworld_prediction.py `
  --ckpt ./CTFM/Ant-Running-WestWorld/checkpoints/last.ckpt `
  --ppo-ckpt Ant/ppo_westworld_checkpoints/ppo_westworld_ant_update_0100.pt `
  --stats Trajworld_data/UniTraj_pt/ant_running_pt/ant_running_ppo/minmax_ant_running_ppo.pt `
  --out Ant/renders/westworld_vs_gt.mp4 `
  --width 640 `
  --height 480
```

bash:

```bash
docker compose run --rm westworld python Ant/render_westworld_prediction.py \
  --ckpt ./CTFM/Ant-Running-WestWorld/checkpoints/last.ckpt \
  --ppo-ckpt Ant/ppo_westworld_checkpoints/ppo_westworld_ant_update_0100.pt \
  --stats Trajworld_data/UniTraj_pt/ant_running_pt/ant_running_ppo/minmax_ant_running_ppo.pt \
  --out Ant/renders/westworld_vs_gt.mp4 \
  --width 640 \
  --height 480
```

Closed-loop WestWorld PPO rollout rendering:

PowerShell:

```powershell
docker compose run --rm westworld python Ant/render_westworld_closed_loop_prediction.py `
  --ckpt ./CTFM/Ant-Running-WestWorld/checkpoints/last.ckpt `
  --ppo-ckpt Ant/ppo_westworld_checkpoints/ppo_westworld_ant_update_0100.pt `
  --stats Trajworld_data/UniTraj_pt/ant_running_pt/ant_running_ppo/minmax_ant_running_ppo.pt `
  --out Ant/renders/westworld_closed_loop_vs_gt.mp4 `
  --width 640 `
  --height 480
```

bash:

```bash
docker compose run --rm westworld python Ant/render_westworld_closed_loop_prediction.py \
  --ckpt ./CTFM/Ant-Running-WestWorld/checkpoints/last.ckpt \
  --ppo-ckpt Ant/ppo_westworld_checkpoints/ppo_westworld_ant_update_0100.pt \
  --stats Trajworld_data/UniTraj_pt/ant_running_pt/ant_running_ppo/minmax_ant_running_ppo.pt \
  --out Ant/renders/westworld_closed_loop_vs_gt.mp4 \
  --width 640 \
  --height 480
```

## GPU 선택

전체 GPU를 쓰는 기본 설정 그대로 실행하면 위 명령을 쓰면 됩니다. 특정 GPU만 쓰려면 실행 전에 `NVIDIA_VISIBLE_DEVICES`를 지정합니다.

PowerShell:

```powershell
$env:NVIDIA_VISIBLE_DEVICES="0"; docker compose run --rm westworld python Ant/ppo_collect_ant_data.py `
  --total-updates 30 `
  --collect-interval 5 `
  --episodes-per-snapshot 20 `
  --prefix ant_running_ppo
```

bash:

```bash
NVIDIA_VISIBLE_DEVICES=0 docker compose run --rm westworld python Ant/ppo_collect_ant_data.py \
  --total-updates 30 \
  --collect-interval 5 \
  --episodes-per-snapshot 20 \
  --prefix ant_running_ppo
```
