# LLM Bridge Prod - 初心者向け解説

## プロジェクト概要

このリポジトリは**松尾研LLM開発コンペ2025**のための標準化されたLLM（大規模言語モデル）開発コードです。主な目的は以下の通りです：

- LLMのファインチューニング（教師あり学習）
- 強化学習（PPO: Proximal Policy Optimization）
- シングルノード・マルチノードでの分散学習
- 学習済みモデルの評価とデプロイ

## プロジェクト構造

```
llm_bridge_prod/
├── README.md                    # プロジェクト概要
├── train/                       # 学習関連のコード
│   ├── README.md               # 学習手順の概要
│   ├── README_install_conda.md # 環境構築手順
│   ├── README_single_node_SFT_PPO.md # シングルノード学習
│   ├── README_multi_node_SFT_PPO.md  # マルチノード学習
│   └── scripts/                # 学習用スクリプト
│       ├── upload_tokenizer_and_finetuned_model_to_huggingface_hub.py
│       ├── mutinode_sft/       # マルチノードSFT用
│       └── mutinode_ppo/       # マルチノードPPO用
├── eval_hle/                    # Humanity's Last Exam評価
│   ├── predict.py              # モデル予測スクリプト
│   ├── judge.py                # 自動評価スクリプト
│   └── conf/                   # 設定ファイル
├── eval_dna/                    # Do Not Answer評価
│   └── llm-compe-eval/         # 安全性評価スクリプト
└── aaa/                        # インストール用補助スクリプト
    ├── install.sh              # 依存関係一括インストール
    └── run_verl_single_node.sh # シングルノード実行スクリプト
```

## 学習の流れ

### Step 0: 環境構築
Conda環境の作成と必要なライブラリのインストール

```bash
# Anacondaのインストール（未インストールの場合）
wget https://repo.anaconda.com/archive/Anaconda3-2024.06-1-Linux-x86_64.sh
bash Anaconda3-2024.06-1-Linux-x86_64.sh

# Conda環境の作成
conda create -n llmbench python=3.11
conda activate llmbench

# 基本ライブラリのインストール
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install transformers datasets wandb tensorboard

# Verlのインストール（競技用フレームワーク）
pip install verl
```

### Step 1: シングルノード学習
- **SFT (Supervised Fine-Tuning)**: 教師ありファインチューニング
  - 実行コマンド: `bash scripts/mutinode_sft/sft_llama.sh`
  - 学習時間の目安: 1-2時間（モデルサイズとデータ量による）
  - GPU使用量: 8 x Nvidia H100 (80GB)

- **PPO (Proximal Policy Optimization)**: 強化学習
  - 実行コマンド: `bash scripts/mutinode_ppo/launch_training.py`
  - 学習時間の目安: 2-4時間
  - 前提条件: SFTで学習したモデルが必要

### Step 2: マルチノード学習
複数のGPUノードを使った分散学習
- 実行コマンド: `sbatch scripts/mutinode_sft/_sft_llama.sh`
- ノード数: 2ノード（16GPU）
- 通信方式: NCCL（NVIDIA GPU間通信）

## 主要な依存関係とその目的

### 基盤環境

#### **CUDA Toolkit (12.4.1)**
- **目的**: GPU上でのディープラーニング計算を可能にする
- **機能**: GPUの並列計算能力を活用してモデルの学習を高速化
- **なぜ必要**: LLMの学習には大量の計算が必要で、CPUだけでは現実的な時間で完了できない

#### **cuDNN**
- **目的**: ディープニューラルネットワークの計算を最適化
- **機能**: 畳み込み、プーリング、正規化などの演算を高速化
- **なぜ必要**: Transformerモデルの計算を効率的に実行するため

#### **Python 3.11**
- **目的**: プログラミング言語環境
- **機能**: 機械学習ライブラリの実行基盤
- **なぜ必要**: 最新の機械学習ライブラリとの互換性を保つため

### 機械学習フレームワーク

#### **PyTorch**
- **目的**: ディープラーニングフレームワーク
- **機能**: ニューラルネットワークの構築、学習、推論
- **なぜ必要**: LLMの実装と学習の基盤となるライブラリ

#### **Transformers (Hugging Face)**
- **目的**: 事前学習済みモデルの利用
- **機能**: Llama、GPTなどの最新モデルへの簡単アクセス
- **なぜ必要**: ゼロから学習するのではなく、既存の高性能モデルをベースにファインチューニングするため

### 学習最適化ライブラリ

#### **Verl**
- **目的**: 強化学習とファインチューニングの統合フレームワーク
- **機能**: 
  - SFT（教師ありファインチューニング）の実行
  - PPO（強化学習）の実行
  - 分散学習のサポート
- **なぜ必要**: 競技で求められるSFTとPPOの両方を効率的に実行するため

#### **Apex (NVIDIA)**
- **目的**: 学習の高速化と最適化
- **機能**:
  - 混合精度学習（FP16/FP32）
  - 分散最適化アルゴリズム
  - 高速なアテンション機構
- **なぜ必要**: GPU メモリを節約し、学習速度を向上させるため

#### **Flash Attention 2**
- **目的**: メモリ効率的なアテンション計算
- **機能**: Transformerのアテンション機構を高速化
- **なぜ必要**: 長いシーケンスを扱う際のメモリ使用量を大幅に削減するため

#### **TransformerEngine**
- **目的**: Transformerモデルの最適化
- **機能**: H100 GPUでの高速計算
- **なぜ必要**: 最新のGPUアーキテクチャの性能を最大限活用するため

### 分散計算

#### **Ray**
- **目的**: 分散計算とマルチノード学習
- **機能**:
  - 複数ノード間での計算分散
  - リソース管理
  - ジョブスケジューリング
- **なぜ必要**: 大規模なモデルを複数のGPUノードで効率的に学習するため

#### **NCCL (NVIDIA Collective Communications Library)**
- **目的**: GPU間の高速通信
- **機能**: 複数GPU間でのデータ同期
- **なぜ必要**: 分散学習時にGPU間でモデルの重みを効率的に同期するため

#### **HPC-X**
- **目的**: 高性能計算用通信ライブラリ
- **機能**: ノード間の高速ネットワーク通信
- **なぜ必要**: マルチノード学習時のネットワーク通信を最適化するため

### 実験管理・モニタリング

#### **Weights & Biases (wandb)**
- **目的**: 機械学習実験の管理と可視化
- **機能**:
  - 学習過程の可視化（損失、精度など）
  - ハイパーパラメータの記録
  - 実験結果の比較
- **なぜ必要**: 学習の進捗を監視し、異なる設定での結果を比較するため

#### **TensorBoard**
- **目的**: 学習過程の可視化
- **機能**: グラフとメトリクスの表示
- **なぜ必要**: 学習の状況をリアルタイムで確認するため

### データ処理

#### **Datasets (Hugging Face)**
- **目的**: データセットの読み込みと前処理
- **機能**: 標準的なデータセット形式での効率的なデータ処理
- **なぜ必要**: GSM8Kなどの学習データを効率的に読み込むため

#### **Pandas**
- **目的**: データ分析と操作
- **機能**: 表形式データの処理
- **なぜ必要**: 学習データの前処理と分析のため

### モデル管理

#### **Hugging Face Hub**
- **目的**: モデルの保存と共有
- **機能**:
  - 学習済みモデルのアップロード
  - モデルのバージョン管理
  - チーム間でのモデル共有
- **なぜ必要**: 競技で開発したモデルを提出・共有するため

#### **Git LFS (Large File Storage)**
- **目的**: 大容量ファイルのバージョン管理
- **機能**: モデルファイルの効率的な管理
- **なぜ必要**: 数GBのモデルファイルをGitで管理するため

## 学習手法の解説

### SFT (Supervised Fine-Tuning) - 教師ありファインチューニング

**概要**: 事前学習済みのLLMを特定のタスクに適応させる手法

**プロセス**:
1. 事前学習済みモデル（例：Llama-3.2-1B-Instruct）を読み込み
2. タスク固有のデータセット（例：GSM8K数学問題）で追加学習
3. モデルがタスクに特化した回答を生成できるように調整

**使用例**: 数学問題解決、質問応答、文章要約など

### PPO (Proximal Policy Optimization) - 強化学習

**概要**: 人間のフィードバックを模倣した報酬システムでモデルを改善する手法

**プロセス**:
1. SFTで学習したモデルを基盤として使用
2. 報酬モデル（Critic）でモデルの出力を評価
3. より良い出力を生成するようにモデルを調整

**利点**: より人間らしい、有用で安全な回答を生成

## 実行環境

### シングルノード学習
- **対象**: 1ノード、8GPU（Nvidia H100）
- **用途**: 小規模なモデルや実験的な学習
- **特徴**: セットアップが簡単、デバッグしやすい

### マルチノード学習
- **対象**: 2ノード、16GPU（Nvidia H100）
- **用途**: 大規模なモデルや本格的な学習
- **特徴**: より高速な学習、大きなバッチサイズ

## 使用例：GSM8Kデータセットでの学習

### データセット
- **GSM8K**: 小学校レベルの数学文章問題
- **形式**: 問題文と解答のペア
- **目的**: 数学的推論能力の向上

### 学習フロー
1. **データ準備**: GSM8Kデータセットをダウンロード
2. **SFT実行**: 数学問題の解き方を学習
3. **PPO実行**: より良い解答を生成するように調整
4. **モデル変換**: Hugging Face形式に変換
5. **アップロード**: Hugging Face Hubに公開

## トラブルシューティング

### よくある問題

#### GPU メモリ不足
- **原因**: バッチサイズが大きすぎる
- **解決策**: 
  ```bash
  # scripts/mutinode_sft/sft_llama.sh を編集
  micro_batch_size_per_gpu=1  # 元の値から減らす
  gradient_accumulation_steps=16  # 代わりにこちらを増やす
  ```

#### 分散学習の通信エラー
- **原因**: ネットワーク設定の問題
- **解決策**: 
  ```bash
  # 環境変数を設定
  export NCCL_SOCKET_IFNAME=ib0  # InfiniBandの場合
  export NCCL_DEBUG=INFO  # デバッグ情報を表示
  ```

#### 環境変数の設定ミス
- **原因**: モジュールの読み込み順序
- **解決策**: 
  ```bash
  module reset
  module load cuda/12.6 miniconda/24.7.1-py312 cudnn/9.6.0 nccl/2.24.3
  ```

#### Verlのインストールエラー
- **原因**: 依存関係の競合
- **解決策**:
  ```bash
  # クリーンインストール
  conda deactivate
  conda env remove -n llmbench
  conda create -n llmbench python=3.11
  conda activate llmbench
  # aaa/install.sh を使用
  bash aaa/install.sh
  ```

### ベストプラクティス

1. **段階的な学習**: まずシングルノードで動作確認してからマルチノードに移行
2. **リソース監視**: GPU使用率とメモリ使用量を常に監視
3. **実験管理**: wandbで学習過程を記録し、結果を比較
4. **バックアップ**: 重要なチェックポイントは複数箇所に保存

## 参考リンク

- [サーバ利用手順](https://docs.google.com/document/d/16KKkFM8Sbqx0wgcCY4kBKR6Kik01T-jn892e_y67vbM/edit?tab=t.0)
- [Hugging Face トークン発行](https://huggingface.co/settings/tokens)
- [Weights & Biases](https://wandb.ai/)

## 初心者向けクイックスタートガイド

### 1. 最小限の環境でテストする方法

```bash
# 1. プロジェクトのクローン
git clone https://github.com/Ohtani-y/llm_bridge_prod.git
cd llm_bridge_prod

# 2. Conda環境のセットアップ
conda create -n llmbench python=3.11
conda activate llmbench

# 3. 依存関係のインストール（簡易版）
bash aaa/install.sh

# 4. 小さいモデルでテスト実行
cd train
# sft_llama.sh を編集して小さいモデルに変更
# model_id="meta-llama/Llama-3.2-1B-Instruct" → model_id="microsoft/phi-2"
bash scripts/mutinode_sft/sft_llama.sh
```

### 2. ローカル環境（RTX 4090など）での実行

単一GPUまたは少数のGPUで実行する場合：

```bash
# GPUメモリに合わせて設定を調整
export CUDA_VISIBLE_DEVICES=0  # 使用するGPUを指定

# バッチサイズを小さく設定
micro_batch_size_per_gpu=1
gradient_accumulation_steps=32
```

### 3. 競技向けチェックリスト

- [ ] Hugging Face アカウントの作成とトークン発行
- [ ] Weights & Biases アカウントの作成
- [ ] OpenAI API キーの取得（評価用）
- [ ] SLURM環境へのアクセス確認
- [ ] GPUリソースの割り当て確認
- [ ] ベースモデルの選定（Llama-3.2-1B-Instructなど）
- [ ] 学習データの準備（GSM8Kなど）
- [ ] SFT実行と結果確認
- [ ] PPO実行と結果確認
- [ ] モデルのHugging Face Hubへのアップロード
- [ ] 評価スクリプトの実行（HLE、DNA）

## まとめ

このプロジェクトは、最新のLLM学習技術を統合した包括的なフレームワークです。各依存関係は特定の目的を持ち、全体として効率的で高性能なLLM学習環境を構築しています。

### 学習の推奨ステップ

1. **理解フェーズ**: まずドキュメントを読み、全体像を把握
2. **環境構築フェーズ**: ローカル環境で小さいモデルでテスト
3. **実験フェーズ**: シングルノードで本格的な学習を試行
4. **最適化フェーズ**: パラメータ調整とマルチノード学習
5. **評価フェーズ**: HLEとDNAで性能評価

初心者の方は、まずシングルノードでの学習から始めて、徐々に高度な機能を習得することをお勧めします。
