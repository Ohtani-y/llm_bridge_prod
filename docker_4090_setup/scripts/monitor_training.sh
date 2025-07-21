#!/bin/bash


echo "📊 RTX 4090 x4 学習監視ダッシュボード"
echo "=================================="

while true; do
    clear
    echo "📊 RTX 4090 x4 学習監視ダッシュボード"
    echo "=================================="
    echo "⏰ 時刻: $(date)"
    echo ""
    
    echo "🎮 GPU使用状況:"
    docker-compose exec llm-master nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits 2>/dev/null || echo "マスターノード: 停止中"
    docker-compose exec llm-worker nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits 2>/dev/null || echo "ワーカーノード: 停止中"
    echo ""
    
    echo "🐳 コンテナ状況:"
    docker-compose ps
    echo ""
    
    echo "🎯 学習プロセス:"
    MASTER_PROCESS=$(docker-compose exec llm-master pgrep -f torchrun 2>/dev/null | wc -l)
    WORKER_PROCESS=$(docker-compose exec llm-worker pgrep -f torchrun 2>/dev/null | wc -l)
    echo "マスターノード学習プロセス: ${MASTER_PROCESS}"
    echo "ワーカーノード学習プロセス: ${WORKER_PROCESS}"
    echo ""
    
    echo "📝 最新ログ (マスターノード):"
    docker-compose logs --tail=5 llm-master 2>/dev/null | tail -5
    echo ""
    
    echo "🔄 5秒後に更新... (Ctrl+Cで終了)"
    sleep 5
done
