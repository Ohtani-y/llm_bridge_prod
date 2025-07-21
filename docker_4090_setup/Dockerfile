# NVIDIA RTX 4090 x4 対応 LLM学習環境
FROM nvidia/cuda:12.4-devel-ubuntu22.04

# 基本パッケージのインストール
RUN apt-get update && apt-get install -y \
    wget \
    curl \
    git \
    git-lfs \
    build-essential \
    cmake \
    ninja-build \
    python3-dev \
    python3-pip \
    openssh-server \
    openssh-client \
    vim \
    htop \
    tmux \
    && rm -rf /var/lib/apt/lists/*

# Miniconda のインストール
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-py311_24.7.1-0-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh

ENV PATH="/opt/conda/bin:$PATH"

# Conda環境の設定
RUN conda init bash && \
    conda config --set auto_activate_base false

# 作業ディレクトリの設定
WORKDIR /workspace

# Python環境の作成
RUN conda create -n llm_env python=3.11 -y

# Conda環境をアクティベート
SHELL ["conda", "run", "-n", "llm_env", "/bin/bash", "-c"]

# CUDA Toolkit と cuDNN のインストール (RTX 4090対応)
RUN conda install cuda-toolkit=12.4.1 -c nvidia/label/cuda-12.4.1 -y && \
    conda install -c conda-forge cudnn -y && \
    conda install gcc_linux-64 gxx_linux-64 -y

# 基本Pythonパッケージのインストール
RUN pip install --upgrade pip wheel cmake ninja

# Git LFS の設定
RUN git lfs install

# 環境変数の設定 (RTX 4090最適化)
ENV CONDA_PATH="/opt/conda/envs/llm_env"
ENV CUDA_HOME="/opt/conda/envs/llm_env"
ENV CUDNN_PATH="/opt/conda/envs/llm_env/lib"
ENV LD_LIBRARY_PATH="/usr/lib64:/usr/lib:${CONDA_PATH}/lib:${CONDA_PATH}/lib/python3.11/site-packages/torch/lib:${LD_LIBRARY_PATH}"

# RTX 4090 最適化設定
ENV NCCL_SOCKET_IFNAME=eth0
ENV NVTE_FUSED_ATTN=0
ENV NVTE_DEBUG=1
ENV NVTE_DEBUG_LEVEL=0
ENV CUDA_DEVICE_MAX_CONNECTIONS=1
ENV GPU_MAX_HW_QUEUES=2
ENV TORCH_NCCL_HIGH_PRIORITY=1

# メモリ最適化 (RTX 4090 24GB対応)
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
ENV CUDA_LAUNCH_BLOCKING=0

# Verl フレームワークのインストール
RUN mkdir -p /workspace/deps && cd /workspace/deps && \
    git clone https://github.com/volcengine/verl.git && \
    cd verl && \
    USE_MEGATRON=1 bash scripts/install_vllm_sglang_mcore.sh && \
    pip install --no-deps -e .

# 追加依存関係のインストール
RUN pip install --no-cache-dir \
    six \
    regex \
    numpy==1.26.4 \
    deepspeed \
    wandb \
    huggingface_hub \
    tensorboard \
    mpi4py \
    sentencepiece \
    nltk \
    ninja \
    packaging \
    wheel \
    transformers \
    accelerate \
    safetensors \
    einops \
    peft \
    datasets \
    trl \
    matplotlib \
    sortedcontainers \
    brotli \
    zstandard \
    cryptography \
    colorama \
    audioread \
    soupsieve \
    defusedxml \
    babel \
    codetiming \
    zarr \
    tensorstore \
    pybind11 \
    scikit-learn \
    nest-asyncio \
    httpcore \
    pytest \
    pylatexenc \
    tensordict==0.9.1 \
    pyzmq==27.0 \
    ipython

# Ray のインストール
RUN pip install -U "ray[data,train,tune,serve]"

# Protobuf のアップグレード
RUN pip install --upgrade protobuf

# NVIDIA Apex のインストール (RTX 4090最適化)
RUN cd /workspace/deps && \
    git clone https://github.com/NVIDIA/apex && \
    cd apex && \
    pip cache purge && \
    python setup.py install \
        --cpp_ext --cuda_ext \
        --distributed_adam \
        --deprecated_fused_adam \
        --xentropy \
        --fast_multihead_attn

# Flash Attention 2 のインストール (RTX 4090対応)
RUN ulimit -v unlimited && \
    MAX_JOBS=4 pip install flash-attn==2.6.3 --no-build-isolation

# TransformerEngine のインストール
RUN cd /workspace/deps && \
    git clone https://github.com/NVIDIA/TransformerEngine && \
    cd TransformerEngine && \
    git submodule update --init --recursive && \
    git checkout release_v2.4 && \
    MAX_JOBS=4 NVTE_FRAMEWORK=pytorch pip install --no-cache-dir .

# SSH設定 (マルチノード対応)
RUN mkdir -p /root/.ssh && \
    ssh-keygen -t rsa -b 4096 -f /root/.ssh/id_rsa -N "" && \
    cat /root/.ssh/id_rsa.pub >> /root/.ssh/authorized_keys && \
    chmod 600 /root/.ssh/authorized_keys && \
    chmod 700 /root/.ssh

# SSH設定ファイル
RUN echo "Host *" >> /root/.ssh/config && \
    echo "    StrictHostKeyChecking no" >> /root/.ssh/config && \
    echo "    UserKnownHostsFile=/dev/null" >> /root/.ssh/config

# SSHサービスの設定
RUN echo "PermitRootLogin yes" >> /etc/ssh/sshd_config && \
    echo "PasswordAuthentication no" >> /etc/ssh/sshd_config && \
    echo "PubkeyAuthentication yes" >> /etc/ssh/sshd_config

# 起動スクリプトの作成
RUN echo '#!/bin/bash' > /workspace/start.sh && \
    echo 'service ssh start' >> /workspace/start.sh && \
    echo 'conda activate llm_env' >> /workspace/start.sh && \
    echo 'exec "$@"' >> /workspace/start.sh && \
    chmod +x /workspace/start.sh

# デフォルトのConda環境をアクティベート
RUN echo "conda activate llm_env" >> /root/.bashrc

# ポート開放
EXPOSE 22 6379 8265 37171 37172 37173

# エントリーポイント
ENTRYPOINT ["/workspace/start.sh"]
CMD ["/bin/bash"]
