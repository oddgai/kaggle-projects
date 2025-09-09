# Playground Series - Season 5, Episode 4

ポッドキャスト聴取時間予測

## コンペ概要

### 目標

ポッドキャストエピソードのメタデータ（ジャンル、ゲストの人気度、公開時刻など）から、各エピソードが聴取される時間（分単位）を予測する回帰問題です。

### 評価指標

- **主要評価指標**
  - RMSE (Root Mean Squared Error)

- **参考指標**
  - MAE (Mean Absolute Error)
  - R² (決定係数)

### 特徴量

予測に使用可能な主要な特徴量

- Episode_Length_minutes: エピソードの長さ（分）
- Podcast_Name: ポッドキャスト名
- Episode_Title: エピソードタイトル
- Host_Popularity_%: ホストの人気度（パーセンテージ）
- Guest_Popularity_%: ゲストの人気度（パーセンテージ）
- Publication_Day: 公開曜日
- Publication_Time: 公開時刻
- Number_of_Ads: 広告の数
- Episode_Sentiment: エピソードのセンチメント
- Genre: ジャンル

### 目標変数

- Listening_Time_minutes: 聴取時間（分）

## 実験管理

すべての実験はMLflowで管理しています：

**🔗 [MLflow実験トラッキング](https://dbc-55810bf1-184f.cloud.databricks.com/ml/experiments/2509878923275965)**

### 現在の最高スコア
- 🏆 **exp003**: Public 12.94051 (CV RMSE: 12.7938)
- 🥈 **exp004**: Public 12.96794 (CV RMSE: 12.8203)

## プロジェクト構造

```
playground-s5e4-podcast-listening-time/
├── README.md
├── data/
│   ├── README.md
│   ├── train.csv
│   ├── test.csv
│   └── sample_submission.csv
├── experiments/
│   ├── exp000/  # EDA
│   ├── exp001/  # ベースライン
│   └── ...
├── results/
│   ├── exp001/
│   │   ├── submission.csv
│   │   └── model.pkl
│   └── ...
└── tmp/
```

## 重要な日程

- 開始日: 2025年4月
- 終了日: 2025年4月末

## リンク

- [コンペティションページ](https://www.kaggle.com/competitions/playground-series-s5e4)