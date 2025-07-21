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

step_2_0_python_env_activation() {
    log "Step 2-0: Python仮想環境の起動"
    
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
    
    log "Step 2-0 completed successfully"
}

step_2_1_download_data_model() {
    log "Step 2-1: gsm8kデータとLlamaモデルの確認"
    
    local model_path="$HOME/model/Llama-3.2-1B-Instruct"
    local data_path="$HOME/data/gsm8k"
    
    if [ ! -d "$model_path" ]; then
        log "WARNING: Llama model not found at $model_path"
        log "Please download the model manually to this location"
    else
        log "Llama model found at $model_path"
    fi
    
    if [ ! -d "$data_path" ]; then
        log "WARNING: GSM8K dataset not found at $data_path"
        log "Please download the dataset manually to this location"
    else
        log "GSM8K dataset found at $data_path"
    fi
    
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
        log "Please ensure the script exists and configure the SBATCH parameters:"
        log "  - #SBATCH -p YOU_TEAM"
        log "  - #SBATCH --nodelist=osk-gpu[XX-XX]"
        log "  - #SBATCH --nodes=2"
    else
        log "SFT script found at $sft_script"
    fi
    
    if [ ! -f "$sft_config" ]; then
        log "WARNING: SFT config not found at $sft_config"
        log "Please ensure the config exists and update WANDB settings"
    else
        log "SFT config found at $sft_config"
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
        log "Please ensure the script exists and configure the SBATCH parameters:"
        log "  - #SBATCH -p YOU_TEAM"
        log "  - #SBATCH --nodelist=osk-gpu[XX-XX]"
        log "  - #SBATCH --nodes=2"
    else
        log "Ray cluster script found at $ray_script"
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
        log "Please ensure the script exists and update HEAD_IP"
    else
        log "Job submit script found at $job_submit"
    fi
    
    if [ ! -f "$launch_training" ]; then
        log "WARNING: Launch training script not found at $launch_training"
        log "Please ensure the script exists and configure training parameters"
    else
        log "Launch training script found at $launch_training"
    fi
    
    log "IMPORTANT: Before running PPO, you need to:"
    log "1. Check Ray cluster logs for Head IP address"
    log "2. Update HEAD_IP in $job_submit"
    log "3. Update WANDB_ENTITY and other parameters in $launch_training"
    
    log "Step 3-1 completed successfully"
}

step_3_1_run_ppo_training() {
    log "Step 3-1: PPOトレーニングの実行"
    
    local job_submit="$HOME/llm_bridge_prod/train/scripts/mutinode_ppo/job_submit.sh"
    
    check_file "$job_submit"
    
    log "IMPORTANT: This step requires manual intervention:"
    log "1. SSH to the head node (e.g., ssh osk-gpu94)"
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
    log "1. SSH to the head node (e.g., ssh osk-gpu94)"
    log "2. Stop Ray cluster and clean up processes"
    log "3. Cancel SLURM job"
    log ""
    log "Manual commands to run on head node:"
    log "  ssh osk-gpu94"
    log "  ray stop --force"
    log "  pkill -f ray"
    log "  scancel <SLURM_JOB_ID>"
    log ""
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
    
    step_2_0_python_env_activation
    step_2_1_download_data_model
    step_2_2_multinode_sft_setup
    
    log "SFT workflow setup completed"
    log "To submit SFT job, run: $0 submit_sft"
}

run_ppo_workflow() {
    log "Starting PPO workflow"
    
    step_3_0_ray_cluster_setup
    
    log "PPO workflow setup completed"
    log "To start Ray cluster, run: $0 start_ray"
    log "After Ray cluster is running, run: $0 setup_ppo"
}

run_full_workflow() {
    log "Starting full multinode SFT+PPO workflow"
    
    run_sft_workflow
    run_ppo_workflow
    
    log "Full workflow setup completed"
    log "Please follow the individual steps to complete the training"
}

show_usage() {
    echo "Usage: $0 [option]"
    echo "Multinode SFT and PPO Training Automation Script"
    echo ""
    echo "Workflow Options:"
    echo "  all                    Setup both SFT and PPO workflows"
    echo "  sft                    Setup SFT workflow only"
    echo "  ppo                    Setup PPO workflow only"
    echo ""
    echo "Individual Steps:"
    echo "  step_2_0              Activate Python environment"
    echo "  step_2_1              Check data and model availability"
    echo "  step_2_2_setup        Setup SFT directories and check scripts"
    echo "  submit_sft            Submit SFT job to SLURM"
    echo "  step_3_0_setup        Setup Ray cluster directories"
    echo "  start_ray             Start Ray cluster"
    echo "  setup_ppo             Setup PPO job configuration"
    echo "  run_ppo               Run PPO training (requires manual intervention)"
    echo "  stop_ray              Stop Ray cluster (requires manual intervention)"
    echo "  step_3_3              Checkpoint conversion instructions"
    echo "  step_3_4              Model upload instructions"
    echo ""
    echo "Utility:"
    echo "  help                  Show this help message"
    echo ""
    echo "Note: This script automates setup and job submission."
    echo "Some steps require manual intervention due to the distributed nature of the training."
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
    step_2_0)
        step_2_0_python_env_activation
        ;;
    step_2_1)
        step_2_1_download_data_model
        ;;
    step_2_2_setup)
        step_2_2_multinode_sft_setup
        ;;
    submit_sft)
        step_2_2_submit_sft_job
        ;;
    step_3_0_setup)
        step_3_0_ray_cluster_setup
        ;;
    start_ray)
        step_3_0_start_ray_cluster
        ;;
    setup_ppo)
        step_3_1_ppo_job_setup
        ;;
    run_ppo)
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