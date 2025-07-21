# multinode_sft_ppo.sh スクリプト機能説明

## 概要
`multinode_sft_ppo.sh` は、`README_multi_node_SFT_PPO.md` のマルチノード分散学習手順を自動化したbashスクリプトです。

## 主な特徴

### 1. 分散学習対応
- **SFT**: マルチノードファインチューニング
- **PPO**: マルチノード強化学習
- **SLURM**: ジョブスケジューラー連携
- **Ray Cluster**: 分散処理基盤

### 2. ワークフロー管理
3つの主要ワークフロー:
- `sft`: SFTのみ実行
- `ppo`: PPOのみ実行  
- `all`: SFT→PPO順次実行

### 3. 手動介入ポイントの明確化
分散学習特有の手動操作を明示:
- SBATCH設定の更新
- Ray clusterのIPアドレス設定
- ノード間SSH接続

## 詳細機能

### **Step 2: マルチノードSFT**

#### Step 2-0: 環境起動 (`step_2_0_python_env_activation`)
```bash
# モジュール環境設定
module reset
module load nccl/2.22.3
module load hpcx/2.18.1-gcc-cuda12/hpcx-mt
module load miniconda/24.7.1-py311

# conda環境活性化
export CONDA_PATH="~/conda_env"
conda activate $CONDA_PATH
```

#### Step 2-1: データ確認 (`step_2_1_download_data_model`)
- Llama-3.2-1B-Instructモデル存在確認
- GSM8Kデータセット存在確認
- 警告メッセージによる手動ダウンロード指示

#### Step 2-2: SFT実行 (`step_2_2_multinode_sft_setup`, `step_2_2_submit_sft_job`)
```bash
# ディレクトリ構造作成
~/training/multinode/sft/
├── logs/           # SLURMログ
└── checkpoints/    # 学習済みモデル

# SLURMジョブ投入
sbatch ~/llm_bridge_prod/train/scripts/mutinode_sft/_sft_llama.sh
```

### **Step 3: マルチノードPPO**

#### Step 3-0: Ray Cluster (`step_3_0_ray_cluster_setup`, `step_3_0_start_ray_cluster`)
```bash
# ディレクトリ構造作成
~/training/multinode/ppo/
├── ray_cluster/logs/  # Ray clusterログ
└── checkpoints/       # PPOモデル

# Ray cluster起動
sbatch ~/llm_bridge_prod/train/scripts/mutinode_ppo/ray_cluster.sh
```

#### Step 3-1: PPO実行 (`step_3_1_ppo_job_setup`, `step_3_1_run_ppo_training`)
手動操作が必要:
```bash
# ヘッドノードへSSH接続
ssh osk-gpu94

# Ray状態確認
ray status

# PPOジョブ投入
bash ~/llm_bridge_prod/train/scripts/mutinode_ppo/job_submit.sh
```

#### Step 3-2: クリーンアップ (`step_3_2_stop_ray_cluster`)
```bash
# Ray cluster停止
ray stop --force
pkill -f ray
scancel <JOB_ID>
```

## 使用方法

### **基本ワークフロー**
```bash
# 全体セットアップ
./multinode_sft_ppo.sh all

# SFTのみ
./multinode_sft_ppo.sh sft

# PPOのみ  
./multinode_sft_ppo.sh ppo
```

### **個別ステップ実行**
```bash
# 環境準備
./multinode_sft_ppo.sh step_2_0

# SFTジョブ投入
./multinode_sft_ppo.sh submit_sft

# Ray cluster起動
./multinode_sft_ppo.sh start_ray

# PPO設定
./multinode_sft_ppo.sh setup_ppo
```

## 手動設定が必要な項目

### **1. SBATCH設定**
ファイル: `scripts/mutinode_sft/_sft_llama.sh`, `scripts/mutinode_ppo/ray_cluster.sh`
```bash
#SBATCH -p YOU_TEAM              # → 実際のpartition名
#SBATCH --nodelist=osk-gpu[94-95] # → 使用ノード
#SBATCH --nodes=2                 # → ノード数
```

### **2. WANDB設定**
ファイル: `scripts/mutinode_sft/sft_llama.sh`, `scripts/mutinode_ppo/launch_training.py`
```bash
WANDB_ENTITY="YOU_TEAM_ENTITY_NAME"  # → 実際の組織名
```

### **3. Ray Cluster IP**
ファイル: `scripts/mutinode_ppo/job_submit.sh`
```bash
HEAD_IP="192.168.11.94:37173"  # → Ray clusterログから取得
```

## READMEとの相違点

### ✅ 完全自動化された部分
- ディレクトリ作成
- 環境変数設定
- conda環境起動
- SLURMジョブ投入

### ⚠️ 手動操作が残る部分
- SBATCH設定ファイルの編集
- Ray clusterのIPアドレス設定
- ヘッドノードでのSSH操作
- チェックポイント変換 (Step 3-3)
- モデルアップロード (Step 3-4)

### 🚀 改善点
- エラーハンドリング強化
- ログ機能追加
- ファイル存在確認
- 分かりやすい手動操作指示

## 注意事項

### **実行環境**
- ログインノードでの実行推奨
- 計算ノードはSLURMで自動割り当て
- チェックポイント変換は計算ノードで実行

### **リソース管理**
- Ray clusterの適切な停止が必須
- 未停止のクラスターはリソース占有継続
- SLURMジョブIDの管理が重要

### **データ準備**
- モデルとデータセットの事前ダウンロード必須
- パス設定の確認が重要

## トラブルシューティング

### **よくある問題**
1. **Ray cluster接続失敗**: IPアドレス設定を確認
2. **SLURM権限エラー**: partition設定を確認  
3. **モデル未発見**: データパス設定を確認
4. **conda環境エラー**: 事前に`install_conda_env.sh`実行

### **ログ確認方法**
```bash
# SFTログ
tail -f ~/training/multinode/sft/logs/training_sft-*.out

# Ray clusterログ  
cat ~/training/multinode/ppo/ray_cluster/logs/slurm-*.out

# PPOトレーニングログ
ray job logs --follow raysubmit_XXX
```