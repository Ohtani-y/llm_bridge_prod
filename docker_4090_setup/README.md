# RTX 4090 x4 LLM学習環境

このディレクトリには、NVIDIA RTX 4090 x4 Ubuntu環境でLLM学習を実行するためのDocker設定とスクリプトが含まれています。

## 🎯 特徴

- **RTX 4090 x4対応**: 4台のRTX 4090 GPUを効率的に活用
- **自動学習パイプライン**: データダウンロードから学習まで完全自動化
- **単一ノード分散学習**: 1台のPC内4GPU並列学習
- **メモリ最適化**: 24GB VRAM制限に対応した設定
- **監視機能**: リアルタイム学習状況監視

## 📋 前提条件

### ハードウェア要件
- NVIDIA RTX 4090 x4
- 64GB以上のシステムメモリ推奨
- 500GB以上の空きストレージ

### ソフトウェア要件
- Ubuntu 22.04 LTS
- Docker Engine 20.10+
- Docker Compose 2.0+
- NVIDIA Container Toolkit
- NVIDIA Driver 525.60.11+

## 🚀 クイックスタート

### 1. 環境セットアップ
```bash
cd docker_4090_setup
chmod +x scripts/*.sh
./scripts/setup_environment.sh
```

### 2. 設定ファイル編集
```bash
cp .env.example .env
# .envファイルを編集してAPIキーを設定
nano .env
```

必要な設定:
- `WANDB_API_KEY`: WandB APIキー
- `HUGGINGFACE_API_KEY`: Hugging Face APIキー

### 3. 環境テスト
```bash
./scripts/test_environment.sh
```

### 4. 学習開始
```bash
./scripts/start_training.sh
```

### 5. 学習監視
```bash
# 別ターミナルで実行
./scripts/monitor_training.sh
```

### 6. 学習停止
```bash
./scripts/stop_training.sh
```

## 📁 ディレクトリ構造

```
docker_4090_setup/
├── Dockerfile              # RTX 4090最適化Dockerイメージ
├── docker-compose.yml      # 単一ノード4GPU構成定義
├── .env.example            # 環境変数テンプレート
├── README.md               # このファイル
├── scripts/
│   ├── setup_environment.sh    # 初期環境セットアップ
│   ├── start_training.sh       # 自動学習開始
│   ├── stop_training.sh        # 学習停止
│   ├── monitor_training.sh     # 学習監視
│   └── test_environment.sh     # 環境テスト
├── data/                   # データセット保存先
├── models/                 # モデル保存先
├── checkpoints/            # チェックポイント保存先
└── logs/                   # ログファイル保存先
```

## ⚙️ RTX 4090最適化設定

### メモリ最適化
- `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512`
- `micro_batch_size_per_gpu=4` (24GB VRAM対応)
- Gradient Checkpointing有効化

### 通信最適化
- NCCL設定最適化
- InfiniBand無効化（PCIe環境対応）
- 適切なソケットインターフェース設定

### 並列化設定
- 単一ノード x 4GPU構成
- FSDP (Fully Sharded Data Parallel)
- Flash Attention 2対応

## 📊 学習設定

### デフォルト設定
- **モデル**: Llama-3.2-1B-Instruct
- **データセット**: GSM8K (数学問題)
- **学習方法**: SFT (Supervised Fine-Tuning)
- **バッチサイズ**: 4 per GPU
- **エポック数**: 2
- **学習率**: 1e-6

### カスタマイズ
`.env`ファイルで以下の設定を変更可能:
- `BATCH_SIZE`: バッチサイズ
- `LEARNING_RATE`: 学習率
- `MAX_EPOCHS`: エポック数
- `MODEL_NAME`: ベースモデル名

## 🔍 監視とログ

### WandB監視
学習進捗はWandBで確認できます:
```
https://wandb.ai/{WANDB_ENTITY}/{WANDB_PROJECT_NAME}
```

### リアルタイム監視
```bash
./scripts/monitor_training.sh
```

### ログ確認
```bash
# 全ログ
docker-compose logs

# 学習ノードログ
docker-compose logs llm-trainer
```

## 🛠️ トラブルシューティング

### GPU認識されない
```bash
# NVIDIA Container Toolkit確認
docker run --rm --gpus all nvidia/cuda:12.4-base-ubuntu22.04 nvidia-smi

# ドライバー確認
nvidia-smi
```

### メモリ不足エラー
`.env`ファイルで`BATCH_SIZE`を小さくしてください:
```bash
BATCH_SIZE=2  # デフォルト: 4
```

### 通信エラー
ファイアウォール設定を確認:
```bash
# 必要ポートの開放
sudo ufw allow 29500  # PyTorch Distributed
sudo ufw allow 8265   # Ray Dashboard
sudo ufw allow 6006   # TensorBoard
```

### コンテナ起動失敗
```bash
# コンテナ状態確認
docker-compose ps

# ログ確認
docker-compose logs

# 強制再起動
docker-compose down
docker-compose up -d --force-recreate
```

## 🏗️ アーキテクチャ

```
┌─────────────────────────────────────────────────────────────┐
│                    Docker Host (Ubuntu)                     │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────────┐ │
│  │              llm-trainer                                │ │
│  │           (単一ノード4GPU)                               │ │
│  │                                                         │ │
│  │  ┌───────────┐ ┌───────────┐ ┌───────────┐ ┌───────────┐ │ │
│  │  │ RTX 4090  │ │ RTX 4090  │ │ RTX 4090  │ │ RTX 4090  │ │ │
│  │  │  GPU:0    │ │  GPU:1    │ │  GPU:2    │ │  GPU:3    │ │ │
│  │  └───────────┘ └───────────┘ └───────────┘ └───────────┘ │ │
│  │                                                         │ │
│  │              PyTorch DDP + NCCL通信                     │ │
│  └─────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### 単一ノード4GPU構成の特徴
- **単一コンテナ**: 4つのGPU（0,1,2,3）を1つのコンテナで管理
- **PyTorch DDP**: DistributedDataParallelによる効率的な並列学習
- **NCCL通信**: GPU間での高速グラデーション同期
- **メモリ最適化**: RTX 4090の24GB制限に対応した設定

## 📈 性能最適化

### RTX 4090向け最適化済み項目
- ✅ Flash Attention 2
- ✅ Gradient Checkpointing
- ✅ Mixed Precision (FP16)
- ✅ FSDP Sharding
- ✅ Optimized NCCL Settings
- ✅ Memory-efficient Attention

### 追加最適化オプション
1. **DeepSpeed統合** (大規模モデル用)
2. **Gradient Accumulation** (実効バッチサイズ増加)
3. **Dynamic Loss Scaling** (数値安定性向上)

## 🔄 学習パイプライン

1. **環境初期化**: Docker環境構築
2. **データ準備**: GSM8Kダウンロード・前処理
3. **モデル準備**: Llama-3.2-1B-Instructダウンロード
4. **分散学習**: 単一ノード4GPU並列学習
5. **チェックポイント**: 定期的な学習状態保存
6. **評価**: 検証データでの性能評価
7. **モデル保存**: 最終モデルの保存

## 📝 ライセンス

このDocker環境は、元のllm_bridge_prodリポジトリのライセンスに従います。

## 🤝 サポート

問題が発生した場合:
1. `./scripts/test_environment.sh`で環境確認
2. ログファイルの確認
3. GPU使用率・メモリ使用量の確認
4. 必要に応じて設定調整

---

**注意**: この環境は単一PC内のRTX 4090 x4専用に最適化されています。他のGPU構成や複数サーバー環境では設定の調整が必要な場合があります。

## 🔧 単一ノード構成の詳細

### GPU割り当て
- **NVIDIA_VISIBLE_DEVICES**: `0,1,2,3` (全4GPU)
- **CUDA_VISIBLE_DEVICES**: `0,1,2,3`
- **torchrun設定**: `--standalone --nnodes=1 --nproc_per_node=4`

### NCCL通信設定
- **MASTER_ADDR**: `localhost` (単一ノード内通信)
- **NCCL_SOCKET_IFNAME**: `lo` (ローカルループバック)
- **NCCL_P2P_DISABLE**: `0` (GPU間P2P通信有効)
- **NCCL_IB_DISABLE**: `1` (InfiniBand無効、PCIe使用)

### メモリ最適化
- **micro_batch_size_per_gpu**: `3` (RTX 4090 24GB対応)
- **PYTORCH_CUDA_ALLOC_CONF**: `max_split_size_mb:512`
- **Gradient Checkpointing**: 有効
