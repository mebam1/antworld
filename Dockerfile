# syntax=docker/dockerfile:1.7
FROM nvidia/cuda:12.8.0-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG CONDA_DIR=/opt/conda
ARG ENV_NAME=westworld
ARG WESTWORLD_EXTERNAL_ROOT=/opt/westworld-external

ENV CONDA_DIR=${CONDA_DIR} \
    ENV_NAME=${ENV_NAME} \
    WESTWORLD_EXTERNAL_ROOT=${WESTWORLD_EXTERNAL_ROOT} \
    PATH=${CONDA_DIR}/envs/${ENV_NAME}/bin:${CONDA_DIR}/bin:${PATH} \
    MUJOCO_PY_MUJOCO_PATH=${WESTWORLD_EXTERNAL_ROOT}/.mujoco/mujoco210 \
    PYTORCH_NVIDIA_LIB_ROOT=${CONDA_DIR}/envs/${ENV_NAME}/lib/python3.10/site-packages/nvidia \
    LD_LIBRARY_PATH=${CONDA_DIR}/envs/${ENV_NAME}/lib/python3.10/site-packages/nvidia/nccl/lib:${CONDA_DIR}/envs/${ENV_NAME}/lib/python3.10/site-packages/nvidia/cudnn/lib:${CONDA_DIR}/envs/${ENV_NAME}/lib/python3.10/site-packages/nvidia/cublas/lib:${WESTWORLD_EXTERNAL_ROOT}/.mujoco/mujoco210/bin:${LD_LIBRARY_PATH} \
    XDG_CACHE_HOME=${WESTWORLD_EXTERNAL_ROOT}/cache/xdg \
    XDG_CONFIG_HOME=${WESTWORLD_EXTERNAL_ROOT}/config/xdg \
    MPLCONFIGDIR=${WESTWORLD_EXTERNAL_ROOT}/config/matplotlib \
    TORCH_EXTENSIONS_DIR=${WESTWORLD_EXTERNAL_ROOT}/cache/torch_extensions \
    PIP_CACHE_DIR=${WESTWORLD_EXTERNAL_ROOT}/cache/pip \
    UV_CACHE_DIR=${WESTWORLD_EXTERNAL_ROOT}/cache/uv \
    CONDA_PKGS_DIRS=${WESTWORLD_EXTERNAL_ROOT}/conda_pkgs \
    TMPDIR=${WESTWORLD_EXTERNAL_ROOT}/tmp \
    WANDB_DIR=${WESTWORLD_EXTERNAL_ROOT}/wandb \
    CUDA_HOME=/usr/local/cuda \
    FORCE_CUDA=1 \
    TORCH_CUDA_ARCH_LIST=12.0 \
    NVIDIA_VISIBLE_DEVICES=1,2 \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics

SHELL ["/bin/bash", "-lc"]

RUN mkdir -p \
    "${WESTWORLD_EXTERNAL_ROOT}/tmp" \
    "${WESTWORLD_EXTERNAL_ROOT}/cache/pip" \
    "${WESTWORLD_EXTERNAL_ROOT}/cache/uv" \
    "${WESTWORLD_EXTERNAL_ROOT}/cache/xdg" \
    "${WESTWORLD_EXTERNAL_ROOT}/config/xdg" \
    "${WESTWORLD_EXTERNAL_ROOT}/config/matplotlib" \
    "${WESTWORLD_EXTERNAL_ROOT}/cache/torch_extensions" \
    "${WESTWORLD_EXTERNAL_ROOT}/conda_pkgs" \
    "${WESTWORLD_EXTERNAL_ROOT}/wandb"

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    bash \
    bzip2 \
    ca-certificates \
    curl \
    git \
    libegl1 \
    libgl1 \
    libgl1-mesa-dev \
    libglfw3 \
    libglfw3-dev \
    libglew-dev \
    libosmesa6-dev \
    libx11-6 \
    libxext6 \
    libxrender1 \
    make \
    patchelf \
    unzip \
    wget \
    && rm -rf /var/lib/apt/lists/*

RUN --mount=type=cache,target=/opt/westworld-external/cache/curl,sharing=locked \
    if [ ! -s /opt/westworld-external/cache/curl/miniforge.sh ]; then \
      curl -fsSL -o /opt/westworld-external/cache/curl/miniforge.sh https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh; \
    fi \
    && bash /opt/westworld-external/cache/curl/miniforge.sh -b -p "${CONDA_DIR}" \
    && conda config --system --set auto_update_conda false \
    && (conda config --system --remove-key channels || true) \
    && conda config --system --add channels conda-forge \
    && conda config --system --set channel_priority flexible

WORKDIR /workspace/WestWorld

# Keep the expensive dependency layer independent from the main project source.
# Changes outside setup_env/mjrl/mjmpc will reuse this layer.
COPY setup_env.bash setup_env.sh ./
COPY mjrl ./mjrl
COPY mjmpc ./mjmpc

RUN --mount=type=cache,target=/opt/westworld-external/conda_pkgs,sharing=locked \
    --mount=type=cache,target=/opt/westworld-external/cache/pip,sharing=locked \
    --mount=type=cache,target=/opt/westworld-external/cache/uv,sharing=locked \
    --mount=type=cache,target=/opt/westworld-external/cache/torch_extensions,sharing=locked \
    --mount=type=cache,target=/opt/westworld-external/tmp,sharing=locked \
    bash setup_env.bash "${WESTWORLD_EXTERNAL_ROOT}" "${ENV_NAME}"

COPY . .

RUN mkdir -p outputs nohup figure pre_trained Trajworld_data dataset_h5 dataset_h5_ant_running wandb

CMD ["bash"]
