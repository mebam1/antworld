#!/usr/bin/env bash
set -euo pipefail

# --- usage ---
usage() {
  cat <<'EOF'
Usage: setup_env.sh INSTALL_ROOT [ENV_NAME]
  INSTALL_ROOT  Base directory for external files created by this setup.
                MuJoCo, package caches, build caches, temp files, and runtime
                cache/config defaults are placed under this directory.
  ENV_NAME      Conda env name (default: westworld)

Env vars:
  USE_UV        Set to 1 to prefer "uv pip" over pip if available
  NVIDIA_LIB_DIR Override NVIDIA driver library path added to LD_LIBRARY_PATH
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ $# -lt 1 ]]; then
  usage >&2
  exit 2
fi

# === config ===
INSTALL_ROOT_INPUT="$1"
ENV_NAME="${2:-westworld}"          # env name
PYTHON_VERSION="3.10"
TORCH_INDEX_URL="https://download.pytorch.org/whl/cu128"
USE_UV="${USE_UV:-0}"          # set to 1 to prefer uv pip if available

mkdir -p "$INSTALL_ROOT_INPUT"
INSTALL_ROOT="$(cd "$INSTALL_ROOT_INPUT" && pwd)"

# Keep external setup files out of $HOME and under INSTALL_ROOT.
MUJOCO_ROOT="$INSTALL_ROOT/.mujoco"
MUJOCO_DIR="$MUJOCO_ROOT/mujoco210"
DOWNLOAD_DIR="$INSTALL_ROOT/downloads"
TMP_ROOT="$INSTALL_ROOT/tmp"
CACHE_ROOT="$INSTALL_ROOT/cache"
CONFIG_ROOT="$INSTALL_ROOT/config"

mkdir -p "$MUJOCO_ROOT" "$DOWNLOAD_DIR" "$TMP_ROOT" "$CACHE_ROOT" "$CONFIG_ROOT"

export CONDA_PKGS_DIRS="$INSTALL_ROOT/conda_pkgs"
export PIP_CACHE_DIR="$CACHE_ROOT/pip"
export UV_CACHE_DIR="$CACHE_ROOT/uv"
export XDG_CACHE_HOME="$CACHE_ROOT/xdg"
export XDG_CONFIG_HOME="$CONFIG_ROOT/xdg"
export MPLCONFIGDIR="$CONFIG_ROOT/matplotlib"
export TORCH_EXTENSIONS_DIR="$CACHE_ROOT/torch_extensions"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0}"
export TMPDIR="$TMP_ROOT"
export TEMP="$TMP_ROOT"
export TMP="$TMP_ROOT"
export WANDB_DIR="$INSTALL_ROOT/wandb"
mkdir -p \
  "$CONDA_PKGS_DIRS" \
  "$PIP_CACHE_DIR" \
  "$UV_CACHE_DIR" \
  "$XDG_CACHE_HOME" \
  "$XDG_CONFIG_HOME" \
  "$MPLCONFIGDIR" \
  "$TORCH_EXTENSIONS_DIR" \
  "$WANDB_DIR"

echo ">>> Using conda env: $ENV_NAME (python=$PYTHON_VERSION)"
echo ">>> External install root: $INSTALL_ROOT"
echo ">>> MuJoCo root: $MUJOCO_ROOT"
echo ">>> Package/cache root: $CACHE_ROOT"

eval "$(conda shell.bash hook)"

# 1) create and activate conda env
if conda env list | grep -q " $ENV_NAME "; then
  echo ">>> Conda env '$ENV_NAME' already exists, skipping create"
else
  conda create -n "$ENV_NAME" python="$PYTHON_VERSION" -y
fi

conda activate "$ENV_NAME"

# 1.1) write activate.d / deactivate.d hooks for MuJoCo paths
ACTIVATE_D="$CONDA_PREFIX/etc/conda/activate.d"
DEACTIVATE_D="$CONDA_PREFIX/etc/conda/deactivate.d"
mkdir -p "$ACTIVATE_D" "$DEACTIVATE_D"

INSTALL_ROOT_Q="$(printf '%q' "$INSTALL_ROOT")"
MUJOCO_DIR_Q="$(printf '%q' "$MUJOCO_DIR")"
CACHE_ROOT_Q="$(printf '%q' "$CACHE_ROOT")"
CONFIG_ROOT_Q="$(printf '%q' "$CONFIG_ROOT")"
TMP_ROOT_Q="$(printf '%q' "$TMP_ROOT")"

cat > "$ACTIVATE_D/westworld_paths.sh" <<EOF
# Automatically set WestWorld external paths when activating this env.
export WESTWORLD_INSTALL_ROOT=$INSTALL_ROOT_Q
export MUJOCO_PY_MUJOCO_PATH=$MUJOCO_DIR_Q

export _WW_HAD_LD_LIBRARY_PATH="\${LD_LIBRARY_PATH+x}"
export _WW_OLD_LD_LIBRARY_PATH="\${LD_LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="\${LD_LIBRARY_PATH:-}:\$MUJOCO_PY_MUJOCO_PATH/bin"

export _WW_HAD_XDG_CACHE_HOME="\${XDG_CACHE_HOME+x}"
export _WW_OLD_XDG_CACHE_HOME="\${XDG_CACHE_HOME:-}"
export XDG_CACHE_HOME=$CACHE_ROOT_Q/xdg

export _WW_HAD_CONDA_PKGS_DIRS="\${CONDA_PKGS_DIRS+x}"
export _WW_OLD_CONDA_PKGS_DIRS="\${CONDA_PKGS_DIRS:-}"
export CONDA_PKGS_DIRS=$INSTALL_ROOT_Q/conda_pkgs

export _WW_HAD_PIP_CACHE_DIR="\${PIP_CACHE_DIR+x}"
export _WW_OLD_PIP_CACHE_DIR="\${PIP_CACHE_DIR:-}"
export PIP_CACHE_DIR=$CACHE_ROOT_Q/pip

export _WW_HAD_UV_CACHE_DIR="\${UV_CACHE_DIR+x}"
export _WW_OLD_UV_CACHE_DIR="\${UV_CACHE_DIR:-}"
export UV_CACHE_DIR=$CACHE_ROOT_Q/uv

export _WW_HAD_XDG_CONFIG_HOME="\${XDG_CONFIG_HOME+x}"
export _WW_OLD_XDG_CONFIG_HOME="\${XDG_CONFIG_HOME:-}"
export XDG_CONFIG_HOME=$CONFIG_ROOT_Q/xdg

export _WW_HAD_MPLCONFIGDIR="\${MPLCONFIGDIR+x}"
export _WW_OLD_MPLCONFIGDIR="\${MPLCONFIGDIR:-}"
export MPLCONFIGDIR=$CONFIG_ROOT_Q/matplotlib

export _WW_HAD_TORCH_EXTENSIONS_DIR="\${TORCH_EXTENSIONS_DIR+x}"
export _WW_OLD_TORCH_EXTENSIONS_DIR="\${TORCH_EXTENSIONS_DIR:-}"
export TORCH_EXTENSIONS_DIR=$CACHE_ROOT_Q/torch_extensions

export _WW_HAD_TORCH_CUDA_ARCH_LIST="\${TORCH_CUDA_ARCH_LIST+x}"
export _WW_OLD_TORCH_CUDA_ARCH_LIST="\${TORCH_CUDA_ARCH_LIST:-}"
export TORCH_CUDA_ARCH_LIST="\${TORCH_CUDA_ARCH_LIST:-12.0}"

export _WW_HAD_WANDB_DIR="\${WANDB_DIR+x}"
export _WW_OLD_WANDB_DIR="\${WANDB_DIR:-}"
export WANDB_DIR=$INSTALL_ROOT_Q/wandb

export _WW_HAD_TMPDIR="\${TMPDIR+x}"
export _WW_OLD_TMPDIR="\${TMPDIR:-}"
export TMPDIR=$TMP_ROOT_Q

export _WW_HAD_TEMP="\${TEMP+x}"
export _WW_OLD_TEMP="\${TEMP:-}"
export TEMP=$TMP_ROOT_Q

export _WW_HAD_TMP="\${TMP+x}"
export _WW_OLD_TMP="\${TMP:-}"
export TMP=$TMP_ROOT_Q

# Add NVIDIA driver path for libcuda.so (override with NVIDIA_LIB_DIR if needed)
NVIDIA_LIB_DIR="\${NVIDIA_LIB_DIR:-}"
if [ -z "\$NVIDIA_LIB_DIR" ]; then
  for cand in /usr/lib/nvidia /usr/lib/x86_64-linux-gnu /usr/local/nvidia/lib64; do
    if [ -d "\$cand" ]; then
      NVIDIA_LIB_DIR="\$cand"
      break
    fi
  done
fi
if [ -n "\$NVIDIA_LIB_DIR" ] && [ -d "\$NVIDIA_LIB_DIR" ]; then
  export LD_LIBRARY_PATH="\${LD_LIBRARY_PATH:-}:\$NVIDIA_LIB_DIR"
fi
EOF

cat > "$DEACTIVATE_D/westworld_paths.sh" <<'EOF'
# Clean up WestWorld external paths when deactivating this env.
unset MUJOCO_PY_MUJOCO_PATH
unset WESTWORLD_INSTALL_ROOT

if [ "${_WW_HAD_LD_LIBRARY_PATH:-}" = "x" ]; then
  export LD_LIBRARY_PATH="$_WW_OLD_LD_LIBRARY_PATH"
else
  unset LD_LIBRARY_PATH
fi
unset _WW_HAD_LD_LIBRARY_PATH _WW_OLD_LD_LIBRARY_PATH

if [ "${_WW_HAD_XDG_CACHE_HOME:-}" = "x" ]; then
  export XDG_CACHE_HOME="$_WW_OLD_XDG_CACHE_HOME"
else
  unset XDG_CACHE_HOME
fi
unset _WW_HAD_XDG_CACHE_HOME _WW_OLD_XDG_CACHE_HOME

if [ "${_WW_HAD_CONDA_PKGS_DIRS:-}" = "x" ]; then
  export CONDA_PKGS_DIRS="$_WW_OLD_CONDA_PKGS_DIRS"
else
  unset CONDA_PKGS_DIRS
fi
unset _WW_HAD_CONDA_PKGS_DIRS _WW_OLD_CONDA_PKGS_DIRS

if [ "${_WW_HAD_PIP_CACHE_DIR:-}" = "x" ]; then
  export PIP_CACHE_DIR="$_WW_OLD_PIP_CACHE_DIR"
else
  unset PIP_CACHE_DIR
fi
unset _WW_HAD_PIP_CACHE_DIR _WW_OLD_PIP_CACHE_DIR

if [ "${_WW_HAD_UV_CACHE_DIR:-}" = "x" ]; then
  export UV_CACHE_DIR="$_WW_OLD_UV_CACHE_DIR"
else
  unset UV_CACHE_DIR
fi
unset _WW_HAD_UV_CACHE_DIR _WW_OLD_UV_CACHE_DIR

if [ "${_WW_HAD_XDG_CONFIG_HOME:-}" = "x" ]; then
  export XDG_CONFIG_HOME="$_WW_OLD_XDG_CONFIG_HOME"
else
  unset XDG_CONFIG_HOME
fi
unset _WW_HAD_XDG_CONFIG_HOME _WW_OLD_XDG_CONFIG_HOME

if [ "${_WW_HAD_MPLCONFIGDIR:-}" = "x" ]; then
  export MPLCONFIGDIR="$_WW_OLD_MPLCONFIGDIR"
else
  unset MPLCONFIGDIR
fi
unset _WW_HAD_MPLCONFIGDIR _WW_OLD_MPLCONFIGDIR

if [ "${_WW_HAD_TORCH_EXTENSIONS_DIR:-}" = "x" ]; then
  export TORCH_EXTENSIONS_DIR="$_WW_OLD_TORCH_EXTENSIONS_DIR"
else
  unset TORCH_EXTENSIONS_DIR
fi
unset _WW_HAD_TORCH_EXTENSIONS_DIR _WW_OLD_TORCH_EXTENSIONS_DIR

if [ "${_WW_HAD_TORCH_CUDA_ARCH_LIST:-}" = "x" ]; then
  export TORCH_CUDA_ARCH_LIST="$_WW_OLD_TORCH_CUDA_ARCH_LIST"
else
  unset TORCH_CUDA_ARCH_LIST
fi
unset _WW_HAD_TORCH_CUDA_ARCH_LIST _WW_OLD_TORCH_CUDA_ARCH_LIST

if [ "${_WW_HAD_WANDB_DIR:-}" = "x" ]; then
  export WANDB_DIR="$_WW_OLD_WANDB_DIR"
else
  unset WANDB_DIR
fi
unset _WW_HAD_WANDB_DIR _WW_OLD_WANDB_DIR

if [ "${_WW_HAD_TMPDIR:-}" = "x" ]; then
  export TMPDIR="$_WW_OLD_TMPDIR"
else
  unset TMPDIR
fi
unset _WW_HAD_TMPDIR _WW_OLD_TMPDIR

if [ "${_WW_HAD_TEMP:-}" = "x" ]; then
  export TEMP="$_WW_OLD_TEMP"
else
  unset TEMP
fi
unset _WW_HAD_TEMP _WW_OLD_TEMP

if [ "${_WW_HAD_TMP:-}" = "x" ]; then
  export TMP="$_WW_OLD_TMP"
else
  unset TMP
fi
unset _WW_HAD_TMP _WW_OLD_TMP
EOF

# 1.2) pick installer for Python packages (pip or uv pip)
USE_UV_PIP=0
if [ "$USE_UV" = "1" ]; then
  if command -v uv >/dev/null 2>&1; then
    USE_UV_PIP=1
    echo ">>> Using uv pip for Python packages"
  else
    pip install uv
    USE_UV_PIP=1
    echo ">>> Installed uv; using uv pip for Python packages"
  fi
else
  echo ">>> Using pip for Python packages"
fi

pip_install() {
  if [ "$USE_UV_PIP" = "1" ]; then
    uv pip install "$@"
  else
    pip install "$@"
  fi
}

# 2) use conda install system-like deps
echo ">>> Installing system-like deps via conda-forge"
conda install -y -c conda-forge glfw glew patchelf ffmpeg

# 2.1) help compilers find conda headers/libs (e.g., GL/glew.h for mujoco-py)
export CPATH="$CONDA_PREFIX/include${CPATH:+:$CPATH}"
export LIBRARY_PATH="$CONDA_PREFIX/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"

# 3) check MuJoCo 2.1
if [ ! -d "$MUJOCO_DIR" ]; then
  echo ">>> MuJoCo 2.1 not found at $MUJOCO_DIR, downloading..."
  mkdir -p "$MUJOCO_ROOT" "$DOWNLOAD_DIR"

  ARCHIVE="$DOWNLOAD_DIR/mujoco210-linux-x86_64.tar.gz"
  wget -q -O "$ARCHIVE" https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz
  tar -xzf "$ARCHIVE" -C "$MUJOCO_ROOT"
  rm "$ARCHIVE"
else
  echo ">>> Found MuJoCo at $MUJOCO_DIR"
fi

export MUJOCO_PY_MUJOCO_PATH="$MUJOCO_DIR"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:$MUJOCO_PY_MUJOCO_PATH/bin"

# 4) Gym / mujoco-py / d4rl 
echo ">>> Installing gym, mujoco-py and friends"
pip_install "gym==0.23.1"
pip_install "mujoco-py>=2.1,<2.2"   
pip_install "Cython<3" "importlib-metadata<5.0" six "imageio[ffmpeg]" d4rl tensordict matplotlib

# 5) install CUDA 12.8 and PyTorch 2.11.0
echo ">>> Installing PyTorch 2.11.0 + cu128"
pip_install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 \
  --index-url "$TORCH_INDEX_URL"   

# 6) Lightning + experiment logging
echo ">>> Installing Lightning and wandb"
pip_install lightning wandb

# 7) Config utils (hydra + omegaconf)
echo ">>> Installing config utils (hydra-core, omegaconf)"
pip_install hydra-core omegaconf

# 8) mamba-ssm (needs CUDA toolkit/nvcc matching the PyTorch CUDA build)
echo ">>> Installing mamba-ssm 2.3.2.post1"
pip_install ninja packaging wheel setuptools
pip_install "mamba-ssm==2.3.2.post1" --no-build-isolation

# 9) install local mjrl & mjmpc
echo ">>> Installing local packages mjrl & mjmpc"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR/mjrl"
pip_install -e .
cd "$ROOT_DIR/mjmpc"
pip_install -e .
cd "$ROOT_DIR"

# 10) quick self-check
echo ">>> Running quick import self-check"
python - <<'PY'
import gym, torch
import matplotlib
import mujoco_py
import wandb
import mjrl, mjmpc
from mamba_ssm import Mamba

print("gym      :", gym.__version__)
print("matplotlib:", matplotlib.__version__)
print("mujoco_py:", mujoco_py.__version__)
print("torch    :", torch.__version__, "  cuda:", torch.version.cuda, "  is_available:", torch.cuda.is_available())
print("wandb OK :", wandb.__version__)
print("mjrl OK  :", mjrl.__file__)
print("mjmpc OK :", mjmpc.__file__)
print("mamba_ssm OK, example model:")
m = Mamba(d_model=16, d_state=16, d_conv=4, expand=2)
print("  Mamba params:", sum(p.numel() for p in m.parameters()))
PY

echo ">>> Done. To use the environment later, run:"
echo "    conda activate $ENV_NAME"
