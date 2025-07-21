#!/bin/bash

set -e

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

error_exit() {
    log "ERROR: $1"
    exit 1
}

check_command() {
    if ! command -v "$1" &> /dev/null; then
        error_exit "$1 is not available"
    fi
}

step_0_1_preparation() {
    log "Step 0-1: Python仮想環境作成前における下準備"
    
    cd ~/
    mkdir -p ~/conda_env
    
    cp ~/.bashrc ~/.bashrc.backup || log "Warning: Could not backup .bashrc"
    
    module reset
    module load nccl/2.22.3
    module load hpcx/2.18.1-gcc-cuda12/hpcx-mt
    module load miniconda/24.7.1-py311
    
    source /home/appli/miniconda3/24.7.1-py311/etc/profile.d/conda.sh
    
    which conda && echo "====" && conda --version || error_exit "conda is not available"
    
    log "Step 0-1 completed successfully"
}

step_0_2_conda_env_creation() {
    log "Step 0-2: conda環境生成"
    
    export CONDA_PATH="~/conda_env"
    echo "CONDA_PATH: $CONDA_PATH"
    
    conda create --prefix $CONDA_PATH python=3.11 -y || error_exit "Failed to create conda environment"
    
    LD_LIB_APPEND="/usr/lib64:/usr/lib:$CONDA_PATH/lib:$CONDA_PATH/lib/python3.11/site-packages/torch/lib:\$LD_LIBRARY_PATH"
    echo "LD_LIB_APPEND: $LD_LIB_APPEND"
    
    mkdir -p $CONDA_PATH/etc/conda/activate.d
    cat > $CONDA_PATH/etc/conda/activate.d/edit_environment_variable.sh << EOF
export ORIGINAL_LD_LIBRARY_PATH=$LD_LIBRARY_PATH
export ORIGINAL_CUDNN_PATH=$CUDNN_PATH
export ORIGINAL_CUDA_HOME=$CUDA_HOME
export ORIGINAL_CONDA_PATH=$CONDA_PATH
export LD_LIBRARY_PATH=$LD_LIB_APPEND
export CUDNN_PATH=$CONDA_PATH/lib
export CUDA_HOME=$CONDA_PATH/
export CONDA_PATH=$CONDA_PATH/
EOF
    chmod +x $CONDA_PATH/etc/conda/activate.d/edit_environment_variable.sh
    
    mkdir -p $CONDA_PATH/etc/conda/deactivate.d
    cat > $CONDA_PATH/etc/conda/deactivate.d/rollback_environment_variable.sh << EOF
export LD_LIBRARY_PATH=\$ORIGINAL_LD_LIBRARY_PATH
export LD_CUDNN_PATH=\$ORIGINAL_CUDNN_PATH
export LD_CUDA_HOME=\$ORIGINAL_CUDA_HOME
export CONDA_PATH=\$ORIGINAL_CONDA_PATH
unset ORIGINAL_LD_LIBRARY_PATH
unset ORIGINAL_CUDNN_PATH
unset ORIGINAL_CUDA_HOME
unset ORIGINAL_CONDA_PATH
EOF
    chmod +x $CONDA_PATH/etc/conda/deactivate.d/rollback_environment_variable.sh
    
    source ~/.bashrc
    source /home/appli/miniconda3/24.7.1-py311/etc/profile.d/conda.sh
    
    conda init
    conda config --set auto_activate_base false
    
    conda deactivate 2>/dev/null || true
    conda deactivate 2>/dev/null || true
    
    conda activate $CONDA_PATH || error_exit "Failed to activate conda environment"
    
    log "Step 0-2 completed successfully"
}

step_0_3_package_installation() {
    log "Step 0-3: パッケージ等のインストール"
    
    conda install cuda-toolkit=12.4.1 -c nvidia/label/cuda-12.4.1 -y || error_exit "Failed to install CUDA toolkit"
    conda install -c conda-forge cudnn -y || error_exit "Failed to install cuDNN"
    conda install gcc_linux-64 gxx_linux-64 -y || error_exit "Failed to install GCC"
    
    pip install --upgrade pip || error_exit "Failed to upgrade pip"
    pip install --upgrade wheel cmake ninja || error_exit "Failed to install build tools"
    
    conda install git -y || error_exit "Failed to install git"
    conda install anaconda::git-lfs -y || error_exit "Failed to install git-lfs"
    
    git lfs install || error_exit "Failed to initialize git-lfs"
    
    log "Step 0-3 completed successfully"
}

step_0_4_clone_repository() {
    log "Step 0-4: このgitレポジトリのクローン"
    
    cd ~/
    
    if [ ! -d "llm_bridge_prod" ]; then
        git clone https://github.com/matsuolab/llm_bridge_prod.git || error_exit "Failed to clone repository"
    else
        log "Repository already exists, skipping clone"
    fi
    
    cd ~/llm_bridge_prod/train
    ls -lh
    cd ../
    
    log "Step 0-4 completed successfully"
}

step_0_5_environment_check() {
    log "Step 0-5: conda環境プリント確認"
    
    conda deactivate 2>/dev/null || true
    conda activate $CONDA_PATH || error_exit "Failed to activate conda environment"
    
    conda env list
    echo "--- CONDA_PREFIX: ---"
    echo "CONDA_PREFIX: $CONDA_PREFIX"
    echo "--- pip, python パスはCONDA_PREFIXで始まる ---"
    echo "pip: $(which pip)"
    echo "python: $(which python)"
    echo "--- 環境変数 ---"
    printenv | grep CUDA || true
    printenv | grep CUDNN || true
    printenv | grep LD_LIB || true
    
    log "Step 0-5 completed successfully"
}

step_0_6_verl_installation() {
    log "Step 0-6: Verlのインストール"
    
    cd ~/
    mkdir -p deps
    cd ~/deps
    
    if [ ! -d "verl" ]; then
        git clone https://github.com/volcengine/verl.git || error_exit "Failed to clone verl repository"
    else
        log "Verl repository already exists, updating..."
        cd verl && git pull && cd ..
    fi
    
    cd verl
    USE_MEGATRON=1 bash scripts/install_vllm_sglang_mcore.sh || error_exit "Failed to install vllm_sglang_mcore"
    pip install --no-deps -e . || error_exit "Failed to install verl"
    
    pip install --no-cache-dir six regex numpy==1.26.4 deepspeed wandb huggingface_hub tensorboard mpi4py sentencepiece nltk ninja packaging wheel transformers accelerate safetensors einops peft datasets trl matplotlib sortedcontainers brotli zstandard cryptography colorama audioread soupsieve defusedxml babel codetiming zarr tensorstore pybind11 scikit-learn nest-asyncio httpcore pytest pylatexenc tensordict pyzmq==27.0 tensordict==0.9.1 ipython || error_exit "Failed to install required packages"
    
    pip install -U "ray[data,train,tune,serve]" || error_exit "Failed to install Ray"
    pip install --upgrade protobuf || error_exit "Failed to upgrade protobuf"
    
    cd ../
    
    log "Step 0-6 completed successfully"
}

step_0_7_apex_installation() {
    log "Step 0-7: apexのインストール"
    
    cd ~/deps
    
    if [ ! -d "apex" ]; then
        git clone https://github.com/NVIDIA/apex || error_exit "Failed to clone apex repository"
    else
        log "Apex repository already exists, updating..."
        cd apex && git pull && cd ..
    fi
    
    cd apex
    pip cache purge
    
    python setup.py install \
           --cpp_ext --cuda_ext \
           --distributed_adam \
           --deprecated_fused_adam \
           --xentropy \
           --fast_multihead_attn || error_exit "Failed to install apex"
    
    cd ../
    
    log "Step 0-7 completed successfully"
}

step_0_8_flash_attention_installation() {
    log "Step 0-8: Flash Attention 2のインストール"
    
    ulimit -v unlimited
    MAX_JOBS=64 pip install flash-attn==2.6.3 --no-build-isolation || error_exit "Failed to install Flash Attention 2"
    
    log "Step 0-8 completed successfully"
}

step_0_9_transformer_engine_installation() {
    log "Step 0-9: TransformerEngineのインストール"
    
    cd ~/deps
    
    if [ ! -d "TransformerEngine" ]; then
        git clone https://github.com/NVIDIA/TransformerEngine || error_exit "Failed to clone TransformerEngine repository"
    else
        log "TransformerEngine repository already exists, updating..."
        cd TransformerEngine && git pull && cd ..
    fi
    
    cd TransformerEngine
    git submodule update --init --recursive
    git checkout release_v2.4
    NMAX_JOBS=64 VTE_FRAMEWORK=pytorch pip install --no-cache-dir . || error_exit "Failed to install TransformerEngine"
    
    cd ../
    
    log "Step 0-9 completed successfully"
}

step_0_10_installation_check() {
    log "Step 0-10: インストール状況のチェック"
    
    python - <<'PY'
import importlib, apex, torch, sys

for mod in (
    "apex.transformer",
    "apex.normalization.fused_layer_norm",
    "apex.contrib.optimizers.distributed_fused_adam",
    "flash_attn",
    "verl.trainer",
    "ray",
    "transformer_engine",
):
    print("✅" if importlib.util.find_spec(mod) else "❌", mod)

try:
    import flash_attn
    flash_ver = getattr(flash_attn, "__version__", "unknown")
except ImportError:
    flash_ver = "not installed"

try:
    from verl.trainer import main_ppo as _main_ppo
    main_ppo_flag = "✅ main_ppo in verl.trainer"
except ImportError:
    main_ppo_flag = "❌ main_ppo in verl.trainer"
print(main_ppo_flag)

try:
    import ray
    ray_ver = getattr(ray, "__version__", "unknown")
except ImportError:
    ray_ver = "not installed"

try:
    import transformer_engine
    te_ver = getattr(transformer_engine, "__version__", "unknown")
except ImportError:
    te_ver = "not installed"

print("Flash-Attention ver.:", flash_ver, end=" | ")
print("Ray ver.:", ray_ver, end=" | ")
print("TransformerEngine ver.:", te_ver, end=" | ")
print("Apex ver.:", getattr(apex, "__version__", "unknown"),
      "| Torch CUDA:", torch.version.cuda,
      "| Python:", sys.version.split()[0])
PY
    
    if [ $? -eq 0 ]; then
        log "Step 0-10 completed successfully"
    else
        error_exit "Installation check failed"
    fi
}

run_all_steps() {
    log "Starting automated conda environment installation"
    
    step_0_1_preparation
    step_0_2_conda_env_creation
    step_0_3_package_installation
    step_0_4_clone_repository
    step_0_5_environment_check
    step_0_6_verl_installation
    step_0_7_apex_installation
    step_0_8_flash_attention_installation
    step_0_9_transformer_engine_installation
    step_0_10_installation_check
    
    log "All installation steps completed successfully!"
}

show_usage() {
    echo "Usage: $0 [option]"
    echo "Options:"
    echo "  all                    Run all installation steps"
    echo "  step_0_1              Run Step 0-1: Preparation"
    echo "  step_0_2              Run Step 0-2: Conda environment creation"
    echo "  step_0_3              Run Step 0-3: Package installation"
    echo "  step_0_4              Run Step 0-4: Repository clone"
    echo "  step_0_5              Run Step 0-5: Environment check"
    echo "  step_0_6              Run Step 0-6: Verl installation"
    echo "  step_0_7              Run Step 0-7: Apex installation"
    echo "  step_0_8              Run Step 0-8: Flash Attention installation"
    echo "  step_0_9              Run Step 0-9: TransformerEngine installation"
    echo "  step_0_10             Run Step 0-10: Installation check"
    echo "  help                  Show this help message"
}

case "${1:-all}" in
    all)
        run_all_steps
        ;;
    step_0_1)
        step_0_1_preparation
        ;;
    step_0_2)
        step_0_2_conda_env_creation
        ;;
    step_0_3)
        step_0_3_package_installation
        ;;
    step_0_4)
        step_0_4_clone_repository
        ;;
    step_0_5)
        step_0_5_environment_check
        ;;
    step_0_6)
        step_0_6_verl_installation
        ;;
    step_0_7)
        step_0_7_apex_installation
        ;;
    step_0_8)
        step_0_8_flash_attention_installation
        ;;
    step_0_9)
        step_0_9_transformer_engine_installation
        ;;
    step_0_10)
        step_0_10_installation_check
        ;;
    help|--help|-h)
        show_usage
        ;;
    *)
        echo "Unknown option: $1"
        show_usage
        exit 1
        ;;
esac