#!/bin/bash

set -e

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

error_exit() {
    log "ERROR: $1"
    exit 1
}

check_file() {
    if [ ! -f "$1" ]; then
        error_exit "Required file not found: $1"
    fi
}

check_directory() {
    if [ ! -d "$1" ]; then
        error_exit "Required directory not found: $1"
    fi
}

check_env_var() {
    if [ -z "${!1}" ]; then
        error_exit "Environment variable $1 is not set. Please check your .env file."
    fi
}

load_env_file() {
    local env_file="${1:-.env}"
    if [ -f "$env_file" ]; then
        log "Loading environment variables from $env_file"
        export $(grep -v '^#' "$env_file" | xargs)
    else
        log "WARNING: Environment file $env_file not found"
        log "Please create .env file with required variables (run: $0 create_env_template)"
    fi
}

setup_authentication() {
    log "Setting up authentication..."
    
    # Check required environment variables
    check_env_var "HF_TOKEN"
    check_env_var "WANDB_API_KEY"
    
    # Hugging Face authentication
    log "Setting up Hugging Face authentication..."
    echo "$HF_TOKEN" | huggingface-cli login --token --add-to-git-credential || error_exit "Hugging Face authentication failed"
    log "Hugging Face authentication successful"
    
    # WANDB authentication
    log "Setting up WANDB authentication..."
    wandb login "$WANDB_API_KEY" || error_exit "WANDB authentication failed"
    log "WANDB authentication successful"
}

generate_config_files() {
    log "Generating configuration files with environment variables..."
    
    # Check required SLURM environment variables
    local partition="${SLURM_PARTITION:-P12}"
    local nodes="${SLURM_NODES:-1}"
    local gpus="${SLURM_GPUS_PER_NODE:-3}"
    
    # Check required WANDB environment variables
    local wandb_entity="${WANDB_ENTITY:-P12_TEAM}"
    local wandb_project="${WANDB_PROJECT_NAME:-P12_verl_test}"
    
    log "Using SLURM settings: partition=$partition, nodes=$nodes, gpus=$gpus"
    log "Using WANDB settings: entity=$wandb_entity, project=$wandb_project"
    
    # Update SFT SBATCH script
    local sft_script="$HOME/llm_bridge_prod/train/scripts/mutinode_sft/_sft_llama.sh"
    if [ -f "$sft_script" ]; then
        cp "$sft_script" "${sft_script}.backup"
        sed -i "s/#SBATCH -p YOU_TEAM/#SBATCH -p $partition/" "$sft_script"
        sed -i "s/--nodes=2/--nodes=$nodes/" "$sft_script"
        sed -i "s/--gpus-per-node=8/--gpus-per-node=$gpus/" "$sft_script"
        log "Updated SFT SBATCH script: $sft_script"
    fi
    
    # Update PPO SBATCH script
    local ray_script="$HOME/llm_bridge_prod/train/scripts/mutinode_ppo/ray_cluster.sh"
    if [ -f "$ray_script" ]; then
        cp "$ray_script" "${ray_script}.backup"
        sed -i "s/#SBATCH -p YOU_TEAM/#SBATCH -p $partition/" "$ray_script"
        sed -i "s/--nodes=2/--nodes=$nodes/" "$ray_script"
        sed -i "s/--gpus-per-node=8/--gpus-per-node=$gpus/" "$ray_script"
        log "Updated Ray cluster SBATCH script: $ray_script"
    fi
    
    # Update SFT training script
    local sft_config="$HOME/llm_bridge_prod/train/scripts/mutinode_sft/sft_llama.sh"
    if [ -f "$sft_config" ]; then
        cp "$sft_config" "${sft_config}.backup"
        sed -i "s/YOU_TEAM_ENTITY_NAME/$wandb_entity/" "$sft_config"
        sed -i "s/competition_verl_test/$wandb_project/" "$sft_config"
        log "Updated SFT training config: $sft_config"
    fi
    
    # Update PPO training script
    local ppo_config="$HOME/llm_bridge_prod/train/scripts/mutinode_ppo/launch_training.py"
    if [ -f "$ppo_config" ]; then
        cp "$ppo_config" "${ppo_config}.backup"
        sed -i "s/NNODES = 2/NNODES = $nodes/" "$ppo_config"
        sed -i "s/GPUS_PER_NODE = 8/GPUS_PER_NODE = $gpus/" "$ppo_config"
        sed -i "s/YOU_TEAM_ENTITY_NAME/$wandb_entity/" "$ppo_config"
        sed -i "s/competition_verl_test/$wandb_project/" "$ppo_config"
        log "Updated PPO training config: $ppo_config"
    fi
    
    log "Configuration files updated successfully"
}

download_dataset() {
    log "Downloading GSM8K dataset..."
    
    local data_dir="$HOME/data/gsm8k"
    
    if [ -f "$data_dir/train.parquet" ] && [ -f "$data_dir/test.parquet" ]; then
        log "GSM8K dataset already exists at $data_dir"
        return 0
    fi
    
    mkdir -p "$data_dir"
    
    python3 -c "
import os
from datasets import load_dataset
import pandas as pd

print('Loading GSM8K dataset...')
dataset = load_dataset('gsm8k', 'main')

print('Converting to parquet format...')
train_df = dataset['train'].to_pandas()
test_df = dataset['test'].to_pandas()

print('Saving files...')
train_df.to_parquet('$data_dir/train.parquet')
test_df.to_parquet('$data_dir/test.parquet')

print('GSM8K dataset download completed')
" || error_exit "Failed to download GSM8K dataset"
    
    log "GSM8K dataset downloaded successfully to $data_dir"
}

download_model() {
    log "Downloading Llama-3.2-1B-Instruct model..."
    
    local model_dir="$HOME/model/Llama-3.2-1B-Instruct"
    
    if [ -d "$model_dir" ] && [ -f "$model_dir/config.json" ]; then
        log "Llama model already exists at $model_dir"
        return 0
    fi
    
    mkdir -p "$HOME/model"
    
    # Use git clone with authentication
    git clone https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct "$model_dir" || error_exit "Failed to download Llama model"
    
    log "Llama model downloaded successfully to $model_dir"
}

create_env_template() {
    log "Creating .env template file..."
    
    cat > .env.template << 'EOF'
# Authentication tokens
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
WANDB_API_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# WANDB Configuration
WANDB_ENTITY=P12_TEAM
WANDB_PROJECT_NAME=P12_verl_test
WANDB_RUN_NAME_SFT=P12_llama3.2_SFT
WANDB_RUN_NAME_PPO=P12_llama3.2_PPO

# SLURM Configuration
SLURM_PARTITION=P12
SLURM_NODES=1
SLURM_GPUS_PER_NODE=3
SLURM_CPUS_PER_TASK=240
SLURM_TIME=30:30:00

# Model and Data Paths
MODEL_PATH=$HOME/model/Llama-3.2-1B-Instruct
DATA_PATH=$HOME/data/gsm8k
CHECKPOINT_PATH=$HOME/training
EOF
    
    log "Environment template created: .env.template"
    log "Please copy to .env and fill in your actual tokens:"
    log "  cp .env.template .env"
    log "  nano .env"
}

step_2_0_python_env_activation() {
    log "Step 2-0: Python仮想環境の起動と認証設定"
    
    module reset
    module load nccl/2.22.3
    module load hpcx/2.18.1-gcc-cuda12/hpcx-mt
    module load miniconda/24.7.1-py311
    
    source /home/appli/miniconda3/24.7.1-py311/etc/profile.d/conda.sh
    
    which conda && echo "====" && conda --version || error_exit "conda is not available"
    
    export CONDA_PATH="~/conda_env"
    
    source ~/.bashrc
    conda init
    conda config --set auto_activate_base false
    
    conda deactivate 2>/dev/null || true
    conda deactivate 2>/dev/null || true
    
    conda activate $CONDA_PATH || error_exit "Failed to activate conda environment"
    
    # Setup authentication with environment variables
    setup_authentication
    
    log "Step 2-0 completed successfully"
}

step_2_1_download_data_model() {
    log "Step 2-1: データとモデルの自動ダウンロード"
    
    # Download dataset
    download_dataset
    
    # Download model
    download_model
    
    # Verify downloads
    local model_path="$HOME/model/Llama-3.2-1B-Instruct"
    local data_path="$HOME/data/gsm8k"
    
    if [ ! -d "$model_path" ] || [ ! -f "$model_path/config.json" ]; then
        error_exit "Llama model not found or incomplete at $model_path"
    fi
    
    if [ ! -f "$data_path/train.parquet" ] || [ ! -f "$data_path/test.parquet" ]; then
        error_exit "GSM8K dataset not found or incomplete at $data_path"
    fi
    
    log "Model and dataset verified successfully"
    log "Step 2-1 completed successfully"
}

step_2_2_multinode_sft_setup() {
    log "Step 2-2: マルチノードファインチューニングのセットアップ"
    
    mkdir -p ~/training/multinode/sft
    mkdir -p ~/training/multinode/sft/logs
    mkdir -p ~/training/multinode/sft/checkpoints
    
    log "Directories created for multinode SFT"
    
    local sft_script="$HOME/llm_bridge_prod/train/scripts/mutinode_sft/_sft_llama.sh"
    local sft_config="$HOME/llm_bridge_prod/train/scripts/mutinode_sft/sft_llama.sh"
    
    if [ ! -f "$sft_script" ]; then
        log "WARNING: SFT script not found at $sft_script"
        log "Configuration files should have been updated automatically"
    else
        log "SFT script found and configured at $sft_script"
    fi
    
    if [ ! -f "$sft_config" ]; then
        log "WARNING: SFT config not found at $sft_config"
        log "Configuration files should have been updated automatically"
    else
        log "SFT config found and configured at $sft_config"
    fi
    
    log "Step 2-2 completed successfully"
}

step_2_2_submit_sft_job() {
    log "Step 2-2: マルチノードSFTジョブの実行"
    
    local sft_script="$HOME/llm_bridge_prod/train/scripts/mutinode_sft/_sft_llama.sh"
    
    check_file "$sft_script"
    
    log "Submitting SFT job..."
    sbatch "$sft_script" || error_exit "Failed to submit SFT job"
    
    log "SFT job submitted successfully"
    log "Monitor progress with: tail -f ~/training/multinode/sft/logs/training_sft-*.out"
    log "Check model at: $HOME/training/multinode/sft/checkpoints/global_step_58"
    
    log "Step 2-2 SFT job submission completed successfully"
}

step_3_0_ray_cluster_setup() {
    log "Step 3-0: Ray clusterのセットアップ"
    
    mkdir -p ~/training/multinode/ppo
    mkdir -p ~/training/multinode/ppo/ray_cluster/logs
    mkdir -p ~/training/multinode/ppo/checkpoints
    
    log "Directories created for multinode PPO"
    
    local ray_script="$HOME/llm_bridge_prod/train/scripts/mutinode_ppo/ray_cluster.sh"
    
    if [ ! -f "$ray_script" ]; then
        log "WARNING: Ray cluster script not found at $ray_script"
        log "Configuration files should have been updated automatically"
    else
        log "Ray cluster script found and configured at $ray_script"
    fi
    
    log "Step 3-0 completed successfully"
}

step_3_0_start_ray_cluster() {
    log "Step 3-0: Ray clusterの起動"
    
    local ray_script="$HOME/llm_bridge_prod/train/scripts/mutinode_ppo/ray_cluster.sh"
    
    check_file "$ray_script"
    
    log "Starting Ray cluster..."
    sbatch "$ray_script" || error_exit "Failed to start Ray cluster"
    
    log "Ray cluster started successfully"
    log "Monitor cluster status with: cat ~/training/multinode/ppo/ray_cluster/logs/slurm-*.out"
    log "Look for Head IP in the logs (e.g., Head IP → 192.168.11.94:37173)"
    
    log "Step 3-0 Ray cluster startup completed successfully"
}

step_3_1_ppo_job_setup() {
    log "Step 3-1: PPOジョブのセットアップ"
    
    local job_submit="$HOME/llm_bridge_prod/train/scripts/mutinode_ppo/job_submit.sh"
    local launch_training="$HOME/llm_bridge_prod/train/scripts/mutinode_ppo/launch_training.py"
    
    if [ ! -f "$job_submit" ]; then
        log "WARNING: Job submit script not found at $job_submit"
        log "Configuration files should have been updated automatically"
    else
        log "Job submit script found and configured at $job_submit"
    fi
    
    if [ ! -f "$launch_training" ]; then
        log "WARNING: Launch training script not found at $launch_training"
        log "Configuration files should have been updated automatically"
    else
        log "Launch training script found and configured at $launch_training"
    fi
    
    log "IMPORTANT: Before running PPO, you need to:"
    log "1. Check Ray cluster logs for Head IP address"
    log "2. Update HEAD_IP in $job_submit (if not automatically updated)"
    
    log "Step 3-1 completed successfully"
}

step_3_1_run_ppo_training() {
    log "Step 3-1: PPOトレーニングの実行"
    
    local job_submit="$HOME/llm_bridge_prod/train/scripts/mutinode_ppo/job_submit.sh"
    
    check_file "$job_submit"
    
    log "IMPORTANT: This step requires manual intervention:"
    log "1. SSH to the head node (check Ray cluster logs for node name)"
    log "2. Activate conda environment on the head node"
    log "3. Check ray status with 'ray status'"
    log "4. Run the job submit script"
    log ""
    log "Manual commands to run on head node:"
    log "  source /etc/profile.d/modules.sh"
    log "  module reset"
    log "  module load hpcx/2.18.1-gcc-cuda12/hpcx-mt"
    log "  module load miniconda/24.7.1-py311"
    log "  source /home/appli/miniconda3/24.7.1-py311/etc/profile.d/conda.sh"
    log "  conda init"
    log "  conda config --set auto_activate_base false"
    log "  source ~/.bashrc"
    log "  export CONDA_PATH=\"~/conda_env\""
    log "  conda activate \$CONDA_PATH"
    log "  ray status"
    log "  bash $job_submit"
    log ""
    log "Monitor training with: ray job logs --follow raysubmit_XXX"
    log "Check model at: $HOME/training/multinode/ppo/checkpoints/global_step_435"
    
    log "Step 3-1 PPO training setup completed"
}

step_3_2_stop_ray_cluster() {
    log "Step 3-2: Ray clusterの停止"
    
    log "IMPORTANT: This step requires manual intervention:"
    log "1. SSH to the head node (same as used for PPO training)"
    log "2. Stop Ray cluster and clean up processes"
    log "3. Cancel SLURM job"
    log ""
    log "Manual commands to run on head node:"
    log "  ssh <HEAD_NODE_NAME>"
    log "  ray stop --force"
    log "  pkill -f ray"
    log "  scancel <SLURM_JOB_ID>"
    log ""
    log "Replace <HEAD_NODE_NAME> with the actual node name from Ray cluster logs"
    log "Replace <SLURM_JOB_ID> with the actual job ID from Step 3-0"
    
    log "Step 3-2 completed successfully"
}

step_3_3_checkpoint_conversion() {
    log "Step 3-3: チェックポイントの変換"
    
    log "IMPORTANT: This step should be performed on a single compute node (NOT login node)"
    log "Please refer to Step 1-5 in the single node documentation"
    log "The checkpoint conversion process is the same as single node"
    
    log "Step 3-3 completed successfully"
}

step_3_4_model_upload() {
    log "Step 3-4: モデルのアップロード"
    
    log "IMPORTANT: This step should be performed on a single compute node (NOT login node)"
    log "Please refer to Step 1-6 in the single node documentation"
    log "The model upload process is the same as single node"
    
    log "Step 3-4 completed successfully"
}

run_sft_workflow() {
    log "Starting SFT workflow"
    
    # Load environment variables
    load_env_file
    
    # Generate configuration files
    generate_config_files
    
    step_2_0_python_env_activation
    step_2_1_download_data_model
    step_2_2_multinode_sft_setup
    
    log "SFT workflow setup completed"
    log "To submit SFT job, run: $0 submit_sft"
}

run_ppo_workflow() {
    log "Starting PPO workflow"
    
    # Load environment variables
    load_env_file
    
    # Generate configuration files
    generate_config_files
    
    step_3_0_ray_cluster_setup
    
    log "PPO workflow setup completed"
    log "To start Ray cluster, run: $0 start_ray"
    log "After Ray cluster is running, run: $0 setup_ppo"
}

run_full_workflow() {
    log "Starting full multinode SFT+PPO workflow"
    
    # Load environment variables once
    load_env_file
    
    # Generate configuration files once
    generate_config_files
    
    # Run SFT workflow (without reloading env)
    step_2_0_python_env_activation
    step_2_1_download_data_model
    step_2_2_multinode_sft_setup
    
    # Run PPO workflow (without reloading env)
    step_3_0_ray_cluster_setup
    
    log "Full workflow setup completed"
    log "To continue: $0 submit_sft, then $0 start_ray, then $0 run_ppo"
}

show_usage() {
    echo "Usage: $0 [option]"
    echo "Multinode SFT and PPO Training Automation Script (Fully Automated)"
    echo ""
    echo "Setup:"
    echo "  create_env_template   Create .env template file"
    echo "  setup_auth           Setup HF and WANDB authentication only"
    echo "  generate_configs     Generate configuration files from environment variables"
    echo ""
    echo "Download:"
    echo "  download_data        Download GSM8K dataset only"
    echo "  download_model       Download Llama model only"
    echo "  download_all         Download both dataset and model"
    echo ""
    echo "Workflow Options:"
    echo "  all                  Setup both SFT and PPO workflows (fully automated)"
    echo "  sft                  Setup SFT workflow only (fully automated)"
    echo "  ppo                  Setup PPO workflow only (fully automated)"
    echo ""
    echo "Individual Steps:"
    echo "  step_2_0             Activate Python environment and setup authentication"
    echo "  step_2_1             Download data and model automatically"
    echo "  step_2_2_setup       Setup SFT directories and check scripts"
    echo "  submit_sft           Submit SFT job to SLURM"
    echo "  step_3_0_setup       Setup Ray cluster directories"
    echo "  start_ray            Start Ray cluster"
    echo "  setup_ppo            Setup PPO job configuration"
    echo "  run_ppo              Run PPO training (requires manual intervention)"
    echo "  stop_ray             Stop Ray cluster (requires manual intervention)"
    echo "  step_3_3             Checkpoint conversion instructions"
    echo "  step_3_4             Model upload instructions"
    echo ""
    echo "Utility:"
    echo "  help                 Show this help message"
    echo ""
    echo "Prerequisites:"
    echo "  1. Create .env file: $0 create_env_template"
    echo "  2. Fill in your tokens in .env file"
    echo "  3. Run conda environment setup: ./install_conda_env.sh all"
    echo ""
    echo "Note: Most steps are now fully automated using environment variables."
    echo "Some distributed training steps still require manual intervention."
}

case "${1:-all}" in
    all)
        run_full_workflow
        ;;
    sft)
        run_sft_workflow
        ;;
    ppo)
        run_ppo_workflow
        ;;
    create_env_template)
        create_env_template
        ;;
    setup_auth)
        load_env_file
        setup_authentication
        ;;
    generate_configs)
        load_env_file
        generate_config_files
        ;;
    download_data)
        load_env_file
        step_2_0_python_env_activation
        download_dataset
        ;;
    download_model)
        load_env_file
        step_2_0_python_env_activation
        download_model
        ;;
    download_all)
        load_env_file
        step_2_0_python_env_activation
        download_dataset
        download_model
        ;;
    step_2_0)
        load_env_file
        step_2_0_python_env_activation
        ;;
    step_2_1)
        load_env_file
        step_2_0_python_env_activation
        step_2_1_download_data_model
        ;;
    step_2_2_setup)
        load_env_file
        step_2_2_multinode_sft_setup
        ;;
    submit_sft)
        load_env_file
        step_2_2_submit_sft_job
        ;;
    step_3_0_setup)
        load_env_file
        step_3_0_ray_cluster_setup
        ;;
    start_ray)
        load_env_file
        step_3_0_start_ray_cluster
        ;;
    setup_ppo)
        load_env_file
        step_3_1_ppo_job_setup
        ;;
    run_ppo)
        load_env_file
        step_3_1_run_ppo_training
        ;;
    stop_ray)
        step_3_2_stop_ray_cluster
        ;;
    step_3_3)
        step_3_3_checkpoint_conversion
        ;;
    step_3_4)
        step_3_4_model_upload
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