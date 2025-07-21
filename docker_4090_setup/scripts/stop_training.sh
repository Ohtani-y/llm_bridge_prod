#!/bin/bash


echo "🛑 学習停止中..."

docker-compose exec llm-master pkill -f torchrun || true
docker-compose exec llm-worker pkill -f torchrun || true

docker-compose down

echo "✅ 学習停止完了"
