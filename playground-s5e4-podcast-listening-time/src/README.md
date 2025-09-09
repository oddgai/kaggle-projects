# Experiment Utilities

## submit_experiment.py

汎用実験提出スクリプト - 任意の実験をKaggleに提出して詳細分析を実行

### 使用方法

```bash
# 基本的な使用法
uv run python src/submit_experiment.py exp007 results/exp007/submission.csv 12.5

# 予想スコア付きで提出
uv run python src/submit_experiment.py exp007 results/exp007/submission.csv 12.5 --predicted-public 12.7

# アプローチ説明付きで提出  
uv run python src/submit_experiment.py exp007 results/exp007/submission.csv 12.5 \
  --approach "Advanced ensemble with domain knowledge" \
  --predicted-public 12.7
```

### 機能

- Kaggleへの自動提出
- 予想スコアとの比較分析
- CV-Kaggle Gap分析
- 過去実験との自動比較
- 目標達成度チェック
- 結果の自動保存（results/{experiment_id}/kaggle_result.json）

## update_mlflow.py

汎用MLflow更新スクリプト - 任意の実験のKaggle結果をMLflowに記録

### 使用方法

```bash
# 基本的な使用法（自動でJSONファイル読み込み）
uv run python src/update_mlflow.py exp007

# カスタム分析設定付き
uv run python src/update_mlflow.py exp007 --analysis-config analysis_config.json

# 直接スコア指定
uv run python src/update_mlflow.py exp007 --public-score 12.65 --cv-rmse 12.5 --predicted-public 12.7

# 外部JSONファイル指定
uv run python src/update_mlflow.py exp007 --kaggle-results custom_results.json
```

### 機能

- MLflow run自動検索・更新
- CV-Kaggle Gap自動分析
- 予想精度評価
- 全実験との自動比較
- オーバーフィッティング検出
- カスタム分析設定サポート
- 目標達成度チェック

### カスタム分析設定例

```json
{
  "custom_metrics": {
    "ensemble_weight": 0.7,
    "feature_importance_score": 85.2
  },
  "custom_tags": {
    "strategy_type": "ensemble",
    "feature_selection": "recursive",
    "experiment_status": "SUCCESS"
  },
  "lessons_learned": [
    "Ensemble improved stability",
    "Feature selection reduced overfitting",
    "Domain knowledge features effective"
  ],
  "next_improvements": [
    "1. Try stacking ensemble",
    "2. Add more domain features", 
    "3. Optimize hyperparameters"
  ],
  "strategy_evaluation": {
    "effectiveness": "HIGH",
    "overfitting_control": "GOOD",
    "prediction_accuracy": "EXCELLENT"
  }
}
```

## 統合ワークフロー例

```bash
# 1. 実験実行
uv run python experiments/exp007/run_exp007.py

# 2. Kaggle提出
uv run python src/submit_experiment.py exp007 results/exp007/submission.csv 12.5 \
  --predicted-public 12.7 --approach "Advanced ensemble"

# 3. MLflow更新（自動でJSONファイルを使用）
uv run python src/update_mlflow.py exp007

# または、カスタム分析付きで
uv run python src/update_mlflow.py exp007 --analysis-config exp007_analysis.json
```