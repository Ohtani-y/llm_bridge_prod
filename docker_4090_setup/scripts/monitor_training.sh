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
    docker-compose exec llm-trainer nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits 2>/dev/null || echo "学習ノード: 停止中"
    echo ""
    
    echo "🐳 コンテナ状況:"
    docker-compose ps
    echo ""
    
    echo "🎯 学習プロセス:"
    TRAINING_PROCESS=$(docker-compose exec llm-trainer pgrep -f torchrun 2>/dev/null | wc -l)
    echo "学習プロセス数: ${TRAINING_PROCESS}"
    echo ""
    
    echo "📝 最新ログ:"
    docker-compose logs --tail=5 llm-trainer 2>/dev/null | tail -5
    echo ""
    
    echo "🔄 5秒後に更新... (Ctrl+Cで終了)"
    sleep 5
done
