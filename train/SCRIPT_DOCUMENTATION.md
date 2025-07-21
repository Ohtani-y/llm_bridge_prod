# install_conda_env.sh スクリプト機能説明

## 概要
`install_conda_env.sh` は、`README_install_conda.md` の手動インストール手順を自動化したbashスクリプトです。

## 主な機能

### 1. エラーハンドリング
- `set -e`: エラー発生時の自動終了
- `error_exit()`: エラーメッセージ表示と終了
- `check_command()`: コマンド存在確認

### 2. ログ機能
- `log()`: タイムスタンプ付きログメッセージ
- 各ステップの開始・完了ログ

### 3. ステップ分割実行
全10ステップを個別実行可能:

#### Step 0-1: 下準備 (`step_0_1_preparation`)
- ディレクトリ作成
- .bashrcバックアップ
- モジュール環境設定
- conda初期化確認

#### Step 0-2: conda環境生成 (`step_0_2_conda_env_creation`)
- conda環境作成
- 環境変数設定スクリプト生成
- activate/deactivate設定

#### Step 0-3: パッケージインストール (`step_0_3_package_installation`)
- CUDA toolkit
- cuDNN
- GCC
- pip, wheel, cmake, ninja
- git, git-lfs

#### Step 0-4: リポジトリクローン (`step_0_4_clone_repository`)
- llm_bridge_prod クローン
- 既存チェック機能付き

#### Step 0-5: 環境確認 (`step_0_5_environment_check`)
- conda環境一覧表示
- パス確認
- 環境変数確認

#### Step 0-6: Verlインストール (`step_0_6_verl_installation`)
- verlリポジトリクローン
- vllm_sglang_mcore インストール
- 依存パッケージインストール

#### Step 0-7: Apexインストール (`step_0_7_apex_installation`)
- apexリポジトリクローン
- カスタムオプション付きインストール

#### Step 0-8: Flash Attentionインストール (`step_0_8_flash_attention_installation`)
- ulimit設定
- flash-attn==2.6.3 インストール

#### Step 0-9: TransformerEngineインストール (`step_0_9_transformer_engine_installation`)
- TransformerEngineクローン
- サブモジュール更新
- release_v2.4 チェックアウト

#### Step 0-10: インストール確認 (`step_0_10_installation_check`)
- 各モジュールのインポートテスト
- バージョン情報表示

## 使用方法

```bash
# 全ステップ実行
./install_conda_env.sh all

# 個別ステップ実行
./install_conda_env.sh step_0_1
./install_conda_env.sh step_0_2
# ... step_0_10まで

# ヘルプ表示
./install_conda_env.sh help
```

## 改善点

### 元のREADMEとの相違点
1. **エラーハンドリング**: 元のREADMEにはないエラー処理を追加
2. **冗長性排除**: 既存ディレクトリ・リポジトリのチェック機能
3. **ログ機能**: インストール進捗の可視化
4. **分割実行**: 任意のステップからの再開可能

### 追加された安全機能
- コマンド実行前の存在確認
- バックアップファイル作成時の警告処理
- conda deactivateの失敗を許容（2>/dev/null || true）
- 既存リポジトリの更新機能

## 実行前提条件
- 計算ノードでの実行（ログインノードでの実行禁止）
- 適切なpartition設定
- SSH key設定（GitHub接続用）

## 注意事項
- 各ステップは時間がかかる場合があります
- インターネット接続が必要です
- 十分なディスク容量を確保してください