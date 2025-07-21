#!/bin/bash


echo "⏹️  RTX 4090 x4 LLM学習停止中..."

echo "🛑 学習プロセス停止中..."
docker-compose exec llm-trainer pkill -f torchrun 2>/dev/null || echo "学習プロセスが見つかりません"

echo "🐳 コンテナ停止中..."
docker-compose down

echo "📊 GPU使用状況確認:"
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits

echo "✅ 学習環境停止完了"
echo ""
echo "次回の学習開始: ./scripts/start_training.sh"
