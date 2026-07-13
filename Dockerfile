FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG CONDA_DIR=/opt/conda
ARG ENV_NAME=westworld
ARG WESTWORLD_EXTERNAL_ROOT=/opt/westworld-external

ENV CONDA_DIR=${CONDA_DIR} \
    ENV_NAME=${ENV_NAME} \
    WESTWORLD_EXTERNAL_ROOT=${WESTWORLD_EXTERNAL_ROOT} \
    PATH=${CONDA_DIR}/envs/${ENV_NAME}/bin:${CONDA_DIR}/bin:${PATH} \
    MUJOCO_PY_MUJOCO_PATH=${WESTWORLD_EXTERNAL_ROOT}/.mujoco/mujoco210 \
    LD_LIBRARY_PATH=${WESTWORLD_EXTERNAL_ROOT}/.mujoco/mujoco210/bin:/usr/local/cuda/lib64:${LD_LIBRARY_PATH} \
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
    NVIDIA_VISIBLE_DEVICES=1 \
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

RUN apt-get update && apt-get install -y --no-install-recommends \
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

RUN curl -fsSL -o /tmp/miniforge.sh https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh \
    && bash /tmp/miniforge.sh -b -p "${CONDA_DIR}" \
    && rm /tmp/miniforge.sh \
    && conda config --system --set auto_update_conda false \
    && (conda config --system --remove-key channels || true) \
    && conda config --system --add channels conda-forge \
    && conda config --system --set channel_priority flexible

WORKDIR /workspace/WestWorld
COPY . .

RUN bash setup_env.bash "${WESTWORLD_EXTERNAL_ROOT}" "${ENV_NAME}" \
    && conda clean -afy \
    && find "${WESTWORLD_EXTERNAL_ROOT}" -type f -name '*.tar.bz2' -delete \
    && find "${WESTWORLD_EXTERNAL_ROOT}" -type f -name '*.conda' -delete

RUN mkdir -p outputs nohup figure pre_trained Trajworld_data dataset_h5 dataset_h5_ant_running wandb

CMD ["bash"]
