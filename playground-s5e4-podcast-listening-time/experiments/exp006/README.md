# 実験006: "Less is More" - シンプルで堅牢なアプローチ

## 背景

exp005でCV RMSE 10.2268を達成したものの、Kaggle Public Score 13.10978と大幅な乖離（Gap: 2.88）が発生。これは典型的なオーバーフィッティングであり、以下が原因と分析した。

- 過度な外れ値除去（28%のデータ削減）
- 複雑すぎる特徴量エンジニアリング（20+個の新規特徴量）
- ターゲットエンコーディングでのリーク
- 不十分な検証（3-fold CV）

## 実験方針："Less is More"

複雑さを削減し、シンプルで堅牢なモデルを構築する。CV性能より汎化性能を重視。

### 主要改善点

1. **保守的な外れ値処理**
   - exp005: IQR factor 1.5（28%削減）→ exp006: 上下1%のみ除去（2%削減）
   - データを大切に保持

2. **特徴量エンジニアリングの簡素化**
   - exp005: 53個の特徴量 → exp006: 20個程度に厳選
   - 重要度の高い特徴量のみ使用

3. **ターゲットエンコーディングの改善**
   - Holdout方式での厳密な検証
   - より保守的なスムージング

4. **検証の強化**
   - exp005: 3-fold CV → exp006: 5-fold CV
   - より信頼性の高い性能評価

5. **正則化の強化**
   - 全モデルの正則化パラメータを増強
   - オーバーフィッティング抑制

## モデル構成

### アンサンブル
- **LightGBM**: 正則化強化版
- **XGBoost**: Optuna最適化 + 正則化強化版
- **等重みアンサンブル**: CatBoostは除外して2モデルで安定化

### 使用特徴量（厳選20個）

#### 基本特徴量（7個）
- Episode_Length_minutes
- Host_Popularity_percentage  
- Guest_Popularity_percentage
- Number_of_Ads
- Episode_Sentiment（数値変換）
- Publication_Day（数値変換）
- Publication_Time（数値変換）

#### 実績のある拡張特徴量（10個）
- Episode_Length_squared, Episode_Length_log
- Host_Guest_Popularity_Sum, Host_Guest_Popularity_Ratio
- Ad_Density, Ads_per_Hour
- Has_Guest, Has_Ads
- Is_Weekend, Is_Prime_Time

#### ターゲットエンコーディング（3個）
- Genre_target_encoded
- Podcast_Name_target_encoded（慎重に実装）
- Episode_Title_target_encoded（慎重に実装）

## 期待する結果

### 成功指標
| 指標 | 目標値 | 理由 |
|------|--------|------|
| **CV-Kaggle Gap** | < 0.3 | オーバーフィッティング解消 |
| **Public Score** | < 12.5 | exp003/004を上回る |
| **CV標準偏差** | < 0.05 | 安定した性能 |

### スコア予想
| 指標 | exp005実績 | exp006目標 |
|------|------------|-------------|
| **CV RMSE** | 10.2268 | 12.2-12.4 |
| **Public Score** | 13.10978 | **12.2-12.4** |
| **Private Score** | 13.00859 | **12.1-12.3** |
| **CV-Kaggle Gap** | 2.8830 | **< 0.3** |

## 技術的詳細

### 外れ値処理
```python
# 上下1%のみ除去（保守的）
def remove_outliers_conservative(df, target_col):
    lower = df[target_col].quantile(0.01)
    upper = df[target_col].quantile(0.99)
    return df[(df[target_col] >= lower) & (df[target_col] <= upper)]
```

### ターゲットエンコーディング
```python
# より厳密なHoldout実装
from sklearn.model_selection import KFold
def safe_target_encoding(X_train, y_train, X_test, cat_cols, alpha=20):
    # KFold with holdout validation
    # より保守的なスムージング
```

### 正則化設定
```python
# LightGBM強化
lgb_params['reg_alpha'] = 2.0
lgb_params['reg_lambda'] = 2.0

# XGBoost強化  
xgb_params['reg_alpha'] = 3.0
xgb_params['reg_lambda'] = 3.0
```

## 実験結果の評価基準

1. **CV-Kaggle Gapが0.3以下**: オーバーフィッティング解消
2. **Public Score < 12.5**: 実用性確認
3. **複数回実行で安定**: 再現性確認
4. **特徴量重要度が妥当**: 解釈可能性確認

## exp005からの学習事項

### 成功した技術（継承）
- データ型の適切な変換
- アンサンブル手法の基本構造
- MLflow実験管理

### 失敗した技術（修正）
- 過度な外れ値除去
- 複雑すぎる特徴量エンジニアリング
- 不十分なターゲットエンコーディング検証
- 浅いクロスバリデーション

## ファイル構成

- `README.md`: この説明ファイル
- `run_exp006.py`: 実験実行スクリプト
- `log_to_mlflow.py`: MLflow記録スクリプト
- `../../results/exp006/`: 実験結果保存ディレクトリ