# 実験005: Advanced Feature Engineering + CatBoost

## 目的

トップスコア（11.44833）との差（1.41081）を縮めるため、競技上位解の手法を参考に高度な特徴量エンジニアリングとCatBoostモデルを導入し、大幅な性能向上を図る。

## 実験設定

### 主要改善点

1. **高度な特徴量エンジニアリング（20+個の新規特徴量）**
   - 相互作用特徴量: Length_Host_Interaction, Genre_Host_Interaction
   - 時間ベース特徴量: Is_Weekend, Hour_of_Day
   - グループ統計: Genre_Mean_Length, Host_Avg_Popularity
   - 比率・割合特徴量: Popularity_Density, Listen_Rate_Ratio
   - 高度なターゲットエンコーディング

2. **CatBoostモデルの追加**
   - カテゴリカル変数の優れた処理能力
   - 自動特徴量選択とオーバーフィッティング抑制

3. **データ品質改善**
   - IQRによる外れ値除去
   - StandardScalerによる正規化
   - カテゴリカル変数のクリッピング

4. **改良アンサンブル**
   - LightGBM + XGBoost + CatBoost
   - 3モデルの重み最適化

### モデル構成

1. **LightGBM**: exp001の最適化版
2. **XGBoost**: exp003のOptuna最適化版
3. **CatBoost**: 新規追加（カテゴリカル変数特化）

### 特徴量（40+個予定）

#### 基本特徴量（10個）
- Podcast_Name, Episode_Title, Genre
- Publication_Day, Publication_Time
- Episode_Sentiment, Episode_Length_minutes
- Host_Popularity_percentage, Guest_Popularity_percentage
- Number_of_Ads

#### exp004までの拡張特徴量（13個）
- Ad_Density, Host_Guest_Popularity_Diff, Has_Guest
- Episode_Length_squared, Episode_Length_log
- Host_Guest_Popularity_Sum, Host_Guest_Popularity_Ratio
- Ads_per_Hour, Has_Ads, Episode_Length_Category
- Podcast_Name_target_encoded, Episode_Title_target_encoded, Genre_target_encoded

#### exp005新規特徴量（20+個）
**相互作用特徴量:**
- Length_Host_Interaction = Episode_Length * Host_Popularity
- Length_Guest_Interaction = Episode_Length * Guest_Popularity
- Genre_Host_Interaction = Genre_encoded * Host_Popularity
- Genre_Guest_Interaction = Genre_encoded * Guest_Popularity
- Ads_Host_Interaction = Number_of_Ads * Host_Popularity

**時間ベース特徴量:**
- Is_Weekend = Publication_Day in [5, 6]
- Is_Weekday = Publication_Day in [0, 1, 2, 3, 4]
- Hour_of_Day = Publication_Time hour
- Is_Prime_Time = Hour_of_Day in [7-9, 17-19]
- Is_Morning = Hour_of_Day in [6-12]
- Is_Evening = Hour_of_Day in [18-22]

**比率・密度特徴量:**
- Popularity_Density = (Host_Pop + Guest_Pop) / Episode_Length
- Ad_Effectiveness = Number_of_Ads * Host_Popularity / Episode_Length
- Listen_Potential = (Host_Pop + Guest_Pop) * Episode_Sentiment / 100
- Content_Quality_Score = Episode_Sentiment * (Host_Pop + Guest_Pop) / 200

**グループ統計特徴量:**
- Genre_Mean_Length = Genre別平均Episode_Length
- Genre_Mean_Host_Pop = Genre別平均Host_Popularity
- Podcast_Mean_Length = Podcast別平均Episode_Length
- Day_Mean_Length = Publication_Day別平均Episode_Length
- Hour_Mean_Popularity = Hour別平均Host_Popularity

**カテゴリ派生特徴量:**
- Host_Popularity_Category = pd.cut(Host_Popularity, bins=5)
- Guest_Popularity_Category = pd.cut(Guest_Popularity, bins=5)
- Combined_Popularity_Category = (Host + Guest) カテゴリ化

### 検証方法

- **Cross Validation**: 5-fold
- **Train/Validation split**: 80%/20%
- **評価指標**: RMSE（主指標）、MAE、R²

### 目標性能

- **目標CV RMSE**: 12.5以下（現在12.7938から0.29以上の改善）
- **期待Public Score**: 12.7程度
- **期待Private Score**: 12.6程度

### 前実験との比較基準

- **exp001 CV RMSE**: 13.0023
- **exp002 CV RMSE**: 12.8926  
- **exp003 CV RMSE**: 12.7938 (現在のベスト)
- **exp004 CV RMSE**: 12.8203
- **exp005 目標**: 12.5以下

### 期待される効果

1. **特徴量エンジニアリング**: 0.2-0.3のRMSE改善
2. **CatBoostモデル**: カテゴリカル変数処理の向上
3. **外れ値除去**: データ品質向上による安定化
4. **3モデルアンサンブル**: 汎化性能向上

## ファイル構成

- `README.md`: この説明ファイル
- `exp005.ipynb`: 実験用Jupyter Notebook
- `run_exp005.py`: 実験実行スクリプト
- `log_to_mlflow.py`: MLflow記録スクリプト
- `../../results/exp005/`: 実験結果保存ディレクトリ
  - `experiment_results.json`: 実験結果詳細
  - `submission.csv`: Kaggle提出用ファイル
  - `model_lgb.pkl`: LightGBMモデル
  - `model_xgb.pkl`: XGBoostモデル  
  - `model_cat.pkl`: CatBoostモデル