#!/bin/bash


set -e

echo "=== RTX 4090 x4 LLM自動学習開始 ==="

if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

if [ -z "$WANDB_API_KEY" ] || [ -z "$HUGGINGFACE_API_KEY" ]; then
    echo "❌ 必要なAPIキーが設定されていません。.envファイルを確認してください。"
    exit 1
fi

echo "🚀 Dockerコンテナ起動中..."
docker-compose up -d

echo "⏳ コンテナの起動を待機中..."
sleep 30

echo "🔍 マスターノードの準備確認中..."
docker-compose exec llm-master bash -c "conda activate llm_env && python -c 'import torch; print(f\"PyTorch: {torch.__version__}\"); print(f\"CUDA available: {torch.cuda.is_available()}\"); print(f\"GPU count: {torch.cuda.device_count()}\")'"

echo "📥 データセットダウンロード中..."
docker-compose exec llm-master bash -c "
    conda activate llm_env &&
    cd /workspace &&
    mkdir -p data/gsm8k &&
    python -c \"
from datasets import load_dataset
import pandas as pd

print('GSM8Kデータセットをダウンロード中...')
train_dataset = load_dataset('gsm8k', 'main', split='train')
test_dataset = load_dataset('gsm8k', 'main', split='test')

train_df = pd.DataFrame(train_dataset)
test_df = pd.DataFrame(test_dataset)

train_df.to_parquet('/workspace/data/gsm8k/train.parquet')
test_df.to_parquet('/workspace/data/gsm8k/test.parquet')

print('データセット保存完了')
print(f'学習データ: {len(train_df)}件')
print(f'テストデータ: {len(test_df)}件')
\"
"

echo "📥 ベースモデルダウンロード中..."
docker-compose exec llm-master bash -c "
    conda activate llm_env &&
    cd /workspace &&
    mkdir -p models &&
    python -c \"
from transformers import AutoTokenizer, AutoModelForCausalLM
import os

model_name = 'meta-llama/Llama-3.2-1B-Instruct'
save_path = '/workspace/models/Llama-3.2-1B-Instruct'

print(f'モデル {model_name} をダウンロード中...')

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

tokenizer.save_pretrained(save_path)
model.save_pretrained(save_path)

print(f'モデル保存完了: {save_path}')
\"
"

echo "🎯 SFT学習開始..."
docker-compose exec llm-master bash -c "
    conda activate llm_env &&
    cd /workspace/llm_bridge_prod &&
    export MASTER_ADDR=llm-master &&
    export MASTER_PORT=37171 &&
    export NODE_RANK=0 &&
    export NNODES=2 &&
    export GPUS_PER_NODE=2 &&
    export NCCL_SOCKET_IFNAME=eth0 &&
    export WANDB_ENTITY=${WANDB_ENTITY} &&
    export WANDB_PROJECT_NAME=${WANDB_PROJECT_NAME} &&
    export WANDB_RUN_NAME=${WANDB_RUN_NAME}_sft &&
    
    torchrun --rdzv_backend c10d \
             --rdzv_endpoint \${MASTER_ADDR}:\${MASTER_PORT} \
             --nnodes \${NNODES} --nproc_per_node \${GPUS_PER_NODE} \
             --node_rank \${NODE_RANK} \
             -m verl.trainer.fsdp_sft_trainer \
             data.train_files=/workspace/data/gsm8k/train.parquet \
             data.val_files=/workspace/data/gsm8k/test.parquet \
             data.prompt_key=question \
             data.response_key=answer \
             data.micro_batch_size_per_gpu=4 \
             model.partial_pretrain=/workspace/models/Llama-3.2-1B-Instruct \
             trainer.project_name=gsm8k-sft-4090 \
             trainer.experiment_name=/workspace/checkpoints/sft_llama_4090 \
             trainer.total_epochs=2 \
             trainer.save_freq=500 \
             trainer.eval_freq=100 \
             trainer.log_freq=10
" &

echo "🔗 ワーカーノード学習参加..."
sleep 10
docker-compose exec llm-worker bash -c "
    conda activate llm_env &&
    cd /workspace/llm_bridge_prod &&
    export MASTER_ADDR=llm-master &&
    export MASTER_PORT=37171 &&
    export NODE_RANK=1 &&
    export NNODES=2 &&
    export GPUS_PER_NODE=2 &&
    export NCCL_SOCKET_IFNAME=eth0 &&
    export WANDB_ENTITY=${WANDB_ENTITY} &&
    export WANDB_PROJECT_NAME=${WANDB_PROJECT_NAME} &&
    export WANDB_RUN_NAME=${WANDB_RUN_NAME}_sft &&
    
    torchrun --rdzv_backend c10d \
             --rdzv_endpoint \${MASTER_ADDR}:\${MASTER_PORT} \
             --nnodes \${NNODES} --nproc_per_node \${GPUS_PER_NODE} \
             --node_rank \${NODE_RANK} \
             -m verl.trainer.fsdp_sft_trainer \
             data.train_files=/workspace/data/gsm8k/train.parquet \
             data.val_files=/workspace/data/gsm8k/test.parquet \
             data.prompt_key=question \
             data.response_key=answer \
             data.micro_batch_size_per_gpu=4 \
             model.partial_pretrain=/workspace/models/Llama-3.2-1B-Instruct \
             trainer.project_name=gsm8k-sft-4090 \
             trainer.experiment_name=/workspace/checkpoints/sft_llama_4090 \
             trainer.total_epochs=2 \
             trainer.save_freq=500 \
             trainer.eval_freq=100 \
             trainer.log_freq=10
" &

echo "🎉 学習開始完了！"
echo ""
echo "📊 学習状況の確認方法:"
echo "1. WandB: https://wandb.ai/${WANDB_ENTITY}/${WANDB_PROJECT_NAME}"
echo "2. ログ確認: docker-compose logs -f llm-master"
echo "3. GPU使用率: docker-compose exec llm-master nvidia-smi"
echo ""
echo "⏹️  学習停止: ./scripts/stop_training.sh"

wait
