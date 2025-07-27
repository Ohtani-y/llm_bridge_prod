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

# ステップ前提条件チェック関数（CONDA_PREFIX活用版）
check_step_prerequisites() {
    local step_num=$1
    local expected_env_path="$HOME/conda_env"
    log "Checking prerequisites for step $step_num..."
    
    case $step_num in
        "0_1")
            # Step 0-1は前提条件なし
            return 0
            ;;
        "0_2")
            # Step 0-2の前提: Step 0-1完了（moduleロード済み、conda利用可能）
            if ! command -v conda &> /dev/null; then
                error_exit "Step 0-2 requires conda to be available (run step_0_1 first)"
            fi
            ;;
        "0_3")
            # Step 0-3の前提: conda環境が存在し、activate済み
            if [ ! -d "$expected_env_path" ]; then
                error_exit "Step 0-3 requires conda environment at $expected_env_path (run step_0_2 first)"
            fi
            
            # conda環境がactivateされているかチェック
            if [ -z "$CONDA_PREFIX" ] || [ "$CONDA_PREFIX" != "$expected_env_path" ]; then
                log "Activating conda environment..."
                source /home/appli/miniconda3/24.7.1-py311/etc/profile.d/conda.sh 2>/dev/null || true
                conda activate "$expected_env_path" || error_exit "Failed to activate conda environment"
            fi
            ;;
        "0_4"|"0_5"|"0_6"|"0_7"|"0_8"|"0_9"|"0_10")
            # Step 0-4以降の前提: conda環境がactivate済み、基本パッケージインストール済み
            if [ ! -d "$expected_env_path" ]; then
                error_exit "Step $step_num requires conda environment at $expected_env_path (run previous steps first)"
            fi
            
            # conda環境がactivateされているかチェック
            if [ -z "$CONDA_PREFIX" ] || [ "$CONDA_PREFIX" != "$expected_env_path" ]; then
                log "Activating conda environment..."
                source /home/appli/miniconda3/24.7.1-py311/etc/profile.d/conda.sh 2>/dev/null || true
                conda activate "$expected_env_path" || error_exit "Failed to activate conda environment"
            fi
            
            # 基本的なツールの存在確認
            if [ "$step_num" != "0_4" ] && [ "$step_num" != "0_5" ]; then
                if ! command -v git &> /dev/null; then
                    error_exit "Step $step_num requires git to be installed (run step_0_3 first)"
                fi
            fi
            ;;
        *)
            error_exit "Unknown step number: $step_num"
            ;;
    esac
    
    # 最終的な環境確認
    if [ "$step_num" != "0_1" ] && [ "$step_num" != "0_2" ]; then
        log "Current conda environment: $CONDA_PREFIX"
    fi
}

step_0_1_preparation() {
    check_step_prerequisites "0_1"
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
    check_step_prerequisites "0_2"
    log "Step 0-2: conda環境生成"
    
    local env_path="$HOME/conda_env"
    echo "Creating conda environment at: $env_path"
    
    conda create --prefix "$env_path" python=3.11 -y || error_exit "Failed to create conda environment"
    
    # activate.dスクリプトで環境変数を設定（CONDA_PREFIXを活用）
    mkdir -p "$env_path/etc/conda/activate.d"
    cat > "$env_path/etc/conda/activate.d/env_setup.sh" << 'EOF'
#!/bin/bash
# 元の環境変数をバックアップ
export ORIGINAL_LD_LIBRARY_PATH="$LD_LIBRARY_PATH"
export ORIGINAL_CUDNN_PATH="$CUDNN_PATH"
export ORIGINAL_CUDA_HOME="$CUDA_HOME"

# 新しい環境変数を設定（CONDA_PREFIXを活用）
export LD_LIBRARY_PATH="/usr/lib64:/usr/lib:$CONDA_PREFIX/lib:$CONDA_PREFIX/lib/python3.11/site-packages/torch/lib:$LD_LIBRARY_PATH"
export CUDNN_PATH="$CONDA_PREFIX/lib"
export CUDA_HOME="$CONDA_PREFIX"
EOF
    chmod +x "$env_path/etc/conda/activate.d/env_setup.sh"
    
    # deactivate.dスクリプトで環境変数を復元
    mkdir -p "$env_path/etc/conda/deactivate.d"
    cat > "$env_path/etc/conda/deactivate.d/env_cleanup.sh" << 'EOF'
#!/bin/bash
# 環境変数を復元
export LD_LIBRARY_PATH="$ORIGINAL_LD_LIBRARY_PATH"
export CUDNN_PATH="$ORIGINAL_CUDNN_PATH"
export CUDA_HOME="$ORIGINAL_CUDA_HOME"

# バックアップ変数をクリア
unset ORIGINAL_LD_LIBRARY_PATH
unset ORIGINAL_CUDNN_PATH
unset ORIGINAL_CUDA_HOME
EOF
    chmod +x "$env_path/etc/conda/deactivate.d/env_cleanup.sh"
    
    # conda初期化と設定
    source ~/.bashrc
    source /home/appli/miniconda3/24.7.1-py311/etc/profile.d/conda.sh
    
    conda init
    conda config --set auto_activate_base false
    
    # 既存の環境をdeactivate
    conda deactivate 2>/dev/null || true
    conda deactivate 2>/dev/null || true
    
    # 新しい環境をactivate
    conda activate "$env_path" || error_exit "Failed to activate conda environment"
    
    log "Conda environment created and activated: $CONDA_PREFIX"
    log "Step 0-2 completed successfully"
}

step_0_3_package_installation() {
    check_step_prerequisites "0_3"
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
    check_step_prerequisites "0_4"
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
    check_step_prerequisites "0_5"
    log "Step 0-5: conda環境プリント確認"
    
    # 環境情報の表示
    conda env list
    echo "=== 現在の環境情報 ==="
    echo "CONDA_PREFIX: $CONDA_PREFIX"
    echo "CONDA_DEFAULT_ENV: $CONDA_DEFAULT_ENV"
    echo "=== Python/pip パス ==="
    echo "python: $(which python)"
    echo "pip: $(which pip)"
    echo "=== CUDA/CUDNN 環境変数 ==="
    echo "CUDA_HOME: $CUDA_HOME"
    echo "CUDNN_PATH: $CUDNN_PATH"
    echo "=== ライブラリパス ==="
    echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
    
    # 環境の整合性チェック
    if [[ "$(which python)" == "$CONDA_PREFIX"* ]]; then
        log "✅ Python path is correctly using conda environment"
    else
        log "⚠️  Warning: Python path may not be using conda environment"
    fi
    
    if [[ "$(which pip)" == "$CONDA_PREFIX"* ]]; then
        log "✅ Pip path is correctly using conda environment"
    else
        log "⚠️  Warning: Pip path may not be using conda environment"
    fi
    
    log "Step 0-5 completed successfully"
}

step_0_6_verl_installation() {
    check_step_prerequisites "0_6"
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
    check_step_prerequisites "0_7"
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
    check_step_prerequisites "0_8"
    log "Step 0-8: Flash Attention 2のインストール"
    
    # メモリ制限を解除
    ulimit -v unlimited
    
    log "Attempting to install Flash Attention 2.6.3..."
    log "Note: This step may take 30-60 minutes due to compilation"
    log "Recommendation: Run this step in tmux/screen to prevent SSH disconnection"
    
    # プリビルド版が利用可能かチェック
    log "Checking for pre-built wheels..."
    pip index versions flash-attn 2>/dev/null | head -10 || true
    
    # インストール実行（タイムアウト対策付き）
    log "Starting Flash Attention 2 installation (this may take a while)..."
    
    # 環境情報をログ出力
    log "CUDA version: $(nvcc --version 2>/dev/null | grep 'release' || echo 'nvcc not found')"
    log "Python version: $(python --version)"
    log "PyTorch CUDA: $(python -c 'import torch; print(torch.version.cuda)' 2>/dev/null || echo 'unknown')"
    
    # インストール実行（進捗表示付き）
    if MAX_JOBS=32 pip install flash-attn==2.6.3 --no-build-isolation --verbose; then
        log "✅ Flash Attention 2 installation completed successfully"
    else
        log "❌ Flash Attention 2 installation failed"
        log "Troubleshooting suggestions:"
        log "1. Check available disk space: df -h"
        log "2. Check memory usage: free -h"
        log "3. Try with fewer jobs: MAX_JOBS=16"
        log "4. Consider using tmux/screen for long-running builds"
        log "5. Check CUDA compatibility with PyTorch version"
        error_exit "Failed to install Flash Attention 2"
    fi
    
    # インストール確認
    log "Verifying Flash Attention 2 installation..."
    python -c "import flash_attn; print(f'Flash Attention version: {flash_attn.__version__}')" || {
        log "❌ Flash Attention 2 import test failed"
        error_exit "Flash Attention 2 installation verification failed"
    }
    
    log "Step 0-8 completed successfully"
}

step_0_9_transformer_engine_installation() {
    check_step_prerequisites "0_9"
    log "Step 0-9: TransformerEngineのインストール"
    
    cd ~/deps || error_exit "Failed to change to ~/deps directory"
    
    # TransformerEngineリポジトリの取得/更新
    if [ ! -d "TransformerEngine" ]; then
        log "Cloning TransformerEngine repository..."
        git clone https://github.com/NVIDIA/TransformerEngine || error_exit "Failed to clone TransformerEngine repository"
    else
        log "TransformerEngine repository already exists, updating..."
        cd TransformerEngine
        git fetch --all || log "Warning: Failed to fetch latest changes"
        git pull || log "Warning: Failed to pull latest changes"
        cd ..
    fi
    
    cd TransformerEngine || error_exit "Failed to enter TransformerEngine directory"
    
    # サブモジュールの更新とブランチ切り替え
    log "Updating submodules..."
    git submodule update --init --recursive || error_exit "Failed to update submodules"
    
    log "Checking out release_v2.4..."
    git checkout release_v2.4 || error_exit "Failed to checkout release_v2.4"
    
    # 環境情報の表示
    log "Environment information for TransformerEngine build:"
    log "CUDA version: $(nvcc --version 2>/dev/null | grep 'release' || echo 'nvcc not found')"
    log "Python version: $(python --version)"
    log "PyTorch version: $(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'unknown')"
    log "CUDA_HOME: $CUDA_HOME"
    log "CUDNN_PATH: $CUDNN_PATH"
    
    # インストール実行
    log "Starting TransformerEngine installation (this may take 15-30 minutes)..."
    log "Note: Consider using tmux/screen for long-running builds"
    
    # オリジナルのビルド設定を使用
    if NMAX_JOBS=64 VTE_FRAMEWORK=pytorch pip install --no-cache-dir .; then
        log "✅ TransformerEngine installation completed successfully"
    else
        log "❌ TransformerEngine installation failed"
        log "Troubleshooting suggestions:"
        log "1. Check CUDA toolkit installation: nvcc --version"
        log "2. Check available disk space: df -h"
        log "3. Check memory usage: free -h"
        log "4. Verify PyTorch CUDA compatibility"
        log "5. Try with fewer jobs: MAX_JOBS=16"
        error_exit "Failed to install TransformerEngine"
    fi
    
    # インストール確認
    log "Verifying TransformerEngine installation..."
    python -c "import transformer_engine; print(f'TransformerEngine version: {transformer_engine.__version__}')" || {
        log "❌ TransformerEngine import test failed"
        error_exit "TransformerEngine installation verification failed"
    }
    
    cd ../ || log "Warning: Failed to return to parent directory"
    
    log "Step 0-9 completed successfully"
}

step_0_10_installation_check() {
    check_step_prerequisites "0_10"
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
    echo "=== LLM Bridge Production Environment Setup Script ==="
    echo "Usage: $0 [option]"
    echo ""
    echo "Options:"
    echo "  all                    Run all installation steps (recommended for first-time setup)"
    echo "  step_0_1              Run Step 0-1: Preparation (module loading, conda setup)"
    echo "  step_0_2              Run Step 0-2: Conda environment creation"
    echo "  step_0_3              Run Step 0-3: Package installation (CUDA, cuDNN, GCC, git)"
    echo "  step_0_4              Run Step 0-4: Repository clone"
    echo "  step_0_5              Run Step 0-5: Environment check and verification"
    echo "  step_0_6              Run Step 0-6: Verl installation (vLLM, Ray)"
    echo "  step_0_7              Run Step 0-7: NVIDIA Apex installation"
    echo "  step_0_8              Run Step 0-8: Flash Attention 2 installation (⚠️  Long build time)"
    echo "  step_0_9              Run Step 0-9: TransformerEngine installation (⚠️  Long build time)"
    echo "  step_0_10             Run Step 0-10: Final installation verification"
    echo "  help                  Show this help message"
    echo ""
    echo "⚠️  Important Notes:"
    echo "  • Steps 0-8 and 0-9 may take 30-60 minutes each due to compilation"
    echo "  • For long-running steps, use tmux/screen to prevent SSH disconnection:"
    echo "    tmux new-session -d -s install './install_conda_env.sh step_0_8'"
    echo "    tmux attach-session -t install"
    echo "  • Each step assumes previous steps have been completed successfully"
    echo "  • Individual steps will check prerequisites and fail if not met"
    echo ""
    echo "📋 Typical Usage:"
    echo "  First-time setup:     $0 all"
    echo "  Resume from step 8:   $0 step_0_8"
    echo "  Verify installation:  $0 step_0_10"
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