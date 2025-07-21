#!/bin/bash


set -e

echo "=== RTX 4090 x4 LLM学習環境セットアップ開始 ==="

mkdir -p data models checkpoints logs

if [ ! -f .env ]; then
    echo "⚠️  .envファイルが見つかりません。.env.exampleをコピーして設定してください。"
    cp .env.example .env
    echo "📝 .envファイルを編集してAPIキーを設定してください："
    echo "   - WANDB_API_KEY"
    echo "   - HUGGINGFACE_API_KEY"
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Composeがインストールされていません。"
    echo "インストール方法: https://docs.docker.com/compose/install/"
    exit 1
fi

if ! docker run --rm --gpus all nvidia/cuda:12.4-base-ubuntu22.04 nvidia-smi &> /dev/null; then
    echo "❌ NVIDIA Container Toolkitが正しく設定されていません。"
    echo "インストール方法: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
    exit 1
fi

echo "🔍 GPU情報確認中..."
nvidia-smi

GPU_COUNT=$(nvidia-smi --query-gpu=count --format=csv,noheader,nounits | head -1)
if [ "$GPU_COUNT" -lt 4 ]; then
    echo "⚠️  RTX 4090が4台未満です。現在: ${GPU_COUNT}台"
    echo "このスクリプトは4台のGPU用に最適化されています。"
fi

echo "✅ 環境セットアップ完了"
echo ""
echo "次のステップ:"
echo "1. .envファイルを編集してAPIキーを設定"
echo "2. ./scripts/start_training.sh を実行して学習開始"
