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

## 実験リスト

| 実験ID | 日付 | モデル | CV Score | Public Score | 概要 |
|--------|------|--------|----------|--------------|------|
| exp000 | 2025-09-02 | EDA | - | - | 探索的データ分析 ✅ |
| exp001 | 2025-09-02 | LightGBM | 13.0023 | **13.13295** | ベースラインモデル ✅ |

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