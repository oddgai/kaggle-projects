# MLflow設定

## 設定ファイルのセットアップ

1. サンプル設定ファイルをコピー
```bash
cp config/mlflow_config.example.json config/mlflow_config.json
```

2. `config/mlflow_config.json`を編集して環境に合わせて設定

### 設定項目

#### Databricks環境
- `tracking_uri`: 通常は`"databricks"`
- `experiment_path`: Databricks上の実験パス
  - 例：`"/Shared/data_science/your_username/project-name"`

#### ローカル環境  
- `tracking_uri`: ローカルMLflowのURI
  - 例：`"file:///tmp/mlruns"`
- `experiment_name`: ローカル実験名

#### その他
- `default_environment`: デフォルトで使用する環境（`"databricks"`または`"local"`）

### セキュリティ

⚠️ **重要**: `config/mlflow_config.json`には機密情報（社内パス等）が含まれるため：
- このファイルは`.gitignore`で除外されています
- Gitにコミットしないでください
- チーム内で共有する際は別途セキュアな方法で共有してください

### トラブルシューティング

#### Databricks接続エラー
- Databricks CLIの設定を確認
- ワークスペースへのアクセス権限を確認
- `experiment_path`の権限を確認

#### ローカル接続エラー  
- MLflowのインストールを確認：`pip install mlflow`
- ディレクトリの書き込み権限を確認