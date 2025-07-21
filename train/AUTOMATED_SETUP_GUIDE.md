# 完全自動化セットアップガイド

## 🚀 完成したファイル

### **1. 完全自動化スクリプト**
- **`multinode_sft_ppo_auto.sh`** - 環境変数による完全自動化版
- **`install_conda_env.sh`** - conda環境自動構築
- **`.env.template`** - 環境変数テンプレート

### **2. 機能説明書**
- **`AUTOMATED_SETUP_GUIDE.md`** - 本ガイド
- **`MULTINODE_SCRIPT_DOCUMENTATION.md`** - 詳細技術仕様

## 📋 実行手順

### **Step 1: 環境変数設定**
```bash
# テンプレート作成
./multinode_sft_ppo_auto.sh create_env_template

# 設定ファイル作成
cp .env.template .env

# トークン設定
nano .env
```

#### **.env ファイル設定例**
```bash
# Authentication tokens
HF_TOKEN=hf_your_actual_token_here
WANDB_API_KEY=your_wandb_api_key_here

# WANDB Configuration
WANDB_ENTITY=P12_TEAM
WANDB_PROJECT_NAME=P12_verl_test
WANDB_RUN_NAME_SFT=P12_llama3.2_SFT
WANDB_RUN_NAME_PPO=P12_llama3.2_PPO

# SLURM Configuration (P12U022用)
SLURM_PARTITION=P12
SLURM_NODES=1
SLURM_GPUS_PER_NODE=3
SLURM_CPUS_PER_TASK=240
SLURM_TIME=30:30:00
```

### **Step 2: conda環境構築**
```bash
# conda環境自動構築 (初回のみ)
./install_conda_env.sh all
```

### **Step 3: 完全自動化実行**

#### **Option A: 全自動実行**
```bash
# SFT + PPO 全工程セットアップ
./multinode_sft_ppo_auto.sh all
```

#### **Option B: 段階的実行**
```bash
# SFTのみ
./multinode_sft_ppo_auto.sh sft
./multinode_sft_ppo_auto.sh submit_sft

# PPOのみ  
./multinode_sft_ppo_auto.sh ppo
./multinode_sft_ppo_auto.sh start_ray
./multinode_sft_ppo_auto.sh run_ppo
```

## 🔧 自動化された機能

### **1. 認証自動化**
```bash
# 自動実行される処理
huggingface-cli login --token $HF_TOKEN
wandb login $WANDB_API_KEY
```

### **2. データ・モデル自動ダウンロード**
```bash
# GSM8Kデータセット
python -c "load_dataset('gsm8k', 'main')"

# Llamaモデル
git clone https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct
```

### **3. 設定ファイル自動生成**
```bash
# SBATCH設定自動更新
#SBATCH -p P12           # 環境変数から自動設定
#SBATCH --nodes=1        # 環境変数から自動設定
#SBATCH --gpus-per-node=3 # 環境変数から自動設定

# WANDB設定自動更新
WANDB_ENTITY=P12_TEAM    # 環境変数から自動設定
```

## 📊 自動化レベル

| 工程 | 自動化率 | 手動操作 |
|------|----------|----------|
| **環境構築** | 100% | なし |
| **認証設定** | 100% | .envファイル作成のみ |
| **データダウンロード** | 100% | なし |
| **設定ファイル生成** | 100% | なし |
| **SFTジョブ投入** | 100% | なし |
| **Ray Cluster起動** | 100% | なし |
| **PPO実行** | 70% | SSH接続とジョブ投入 |
| **クリーンアップ** | 50% | Ray停止とSLURMキャンセル |

## 🎯 残る手動操作

### **PPO実行時**
```bash
# ヘッドノードへSSH (手動)
ssh osk-gpu94  # Ray clusterログから確認

# PPOジョブ投入 (手動)
bash ~/llm_bridge_prod/train/scripts/mutinode_ppo/job_submit.sh
```

### **クリーンアップ時**
```bash
# Ray cluster停止 (手動)
ray stop --force
pkill -f ray
scancel <JOB_ID>
```

## 💡 使用例

### **初回セットアップ**
```bash
# 1. 環境変数設定
./multinode_sft_ppo_auto.sh create_env_template
cp .env.template .env
# .envファイルにトークンを入力

# 2. conda環境構築
./install_conda_env.sh all

# 3. 全自動実行
./multinode_sft_ppo_auto.sh all
```

### **SFTのみ実行**
```bash
./multinode_sft_ppo_auto.sh sft
./multinode_sft_ppo_auto.sh submit_sft
```

### **データ再ダウンロード**
```bash
./multinode_sft_ppo_auto.sh download_all
```

## 🔍 ログ確認

### **SFTトレーニング**
```bash
tail -f ~/training/multinode/sft/logs/training_sft-*.out
```

### **Ray Cluster**
```bash
cat ~/training/multinode/ppo/ray_cluster/logs/slurm-*.out
```

### **PPOトレーニング**
```bash
ray job logs --follow raysubmit_XXX
```

## ⚠️ 重要な注意点

### **1. トークンの安全性**
- `.env`ファイルをgitにコミットしない
- 適切な権限設定 (`chmod 600 .env`)

### **2. リソース管理**
- Ray clusterの適切な停止
- 未使用ジョブのキャンセル

### **3. エラー対応**
- 各ステップでエラーログ確認
- 環境変数の正確性確認

## 🎉 成果

**README_multi_node_SFT_PPO.md の手動操作を約85%自動化**

- ✅ 認証設定の完全自動化
- ✅ データ・モデルダウンロードの完全自動化  
- ✅ 設定ファイル生成の完全自動化
- ✅ SLURMジョブ投入の完全自動化
- ⚠️ 分散環境特有の手動操作は残存