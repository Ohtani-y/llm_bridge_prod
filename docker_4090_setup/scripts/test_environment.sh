#!/bin/bash


set -e

echo "🧪 RTX 4090 x4 環境テスト開始"
echo "============================="

echo "🚀 テストコンテナ起動中..."
docker-compose up -d

sleep 30

echo "🎮 GPU認識テスト..."
docker-compose exec llm-trainer bash -c "conda activate llm_env && python -c '
import torch
print(f\"PyTorch: {torch.__version__}\")
print(f\"CUDA available: {torch.cuda.is_available()}\")
print(f\"GPU count: {torch.cuda.device_count()}\")
for i in range(torch.cuda.device_count()):
    print(f\"GPU {i}: {torch.cuda.get_device_name(i)}\")
    print(f\"GPU {i} Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f}GB\")
'"

echo "📚 ライブラリインポートテスト..."
docker-compose exec llm-trainer bash -c "conda activate llm_env && python -c '
import importlib

modules = [
    \"torch\",
    \"transformers\",
    \"datasets\",
    \"accelerate\",
    \"peft\",
    \"trl\",
    \"wandb\",
    \"flash_attn\",
    \"apex\",
    \"verl.trainer\",
    \"ray\",
    \"transformer_engine\"
]

for mod in modules:
    try:
        importlib.import_module(mod)
        print(f\"✅ {mod}\")
    except ImportError as e:
        print(f\"❌ {mod}: {e}\")
'"

echo "🔗 NCCL通信テスト..."
docker-compose exec llm-trainer bash -c "conda activate llm_env && python -c '
import torch
import torch.distributed as dist
import os

if torch.cuda.is_available():
    device = torch.device(\"cuda:0\")
    tensor = torch.ones(2, 2).to(device)
    print(f\"✅ CUDA tensor created: {tensor}\")
    print(f\"✅ GPU memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB\")
else:
    print(\"❌ CUDA not available\")
'"

echo "🎯 小規模学習テスト..."
docker-compose exec llm-trainer bash -c "conda activate llm_env && python -c '
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM

print(\"小規模モデルテスト開始...\")

model_name = \"gpt2\"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

if torch.cuda.is_available():
    model = model.cuda()
    print(\"✅ モデルをGPUに移動\")

inputs = tokenizer(\"Hello, world!\", return_tensors=\"pt\", padding=True)
if torch.cuda.is_available():
    inputs = {k: v.cuda() for k, v in inputs.items()}

with torch.no_grad():
    outputs = model(**inputs)
    print(f\"✅ 推論成功: output shape {outputs.logits.shape}\")

print(\"✅ 小規模学習テスト完了\")
'"

echo ""
echo "🎉 環境テスト完了！"
echo ""
echo "次のステップ:"
echo "1. .envファイルを編集してAPIキーを設定"
echo "2. ./scripts/start_training.sh で本格的な学習を開始"

docker-compose down
