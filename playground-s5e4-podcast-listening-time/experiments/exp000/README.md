# 実験000: 探索的データ分析 (EDA)

## 目的

データの理解と特徴量エンジニアリングのアイデア発掘

## 分析内容

- データの基本統計量
- 欠損値の確認
- 目標変数の分布
- 特徴量間の相関
- カテゴリカル変数の分布
- 外れ値の検出

## 主要な発見

### データの基本情報
- 訓練データ: 750,000行 × 12列
- テストデータ: 250,000行 × 11列
- 目標変数: Listening_Time_minutes（0-120分の範囲）

### 欠損値の状況
- Guest_Popularity_percentage: 19.5%の欠損値
- Episode_Length_minutes: 11.6%の欠損値
- Number_of_Ads: 0.001%の欠損値

### カテゴリカル変数の分析
- Genre: 10種類（Sports、Technology、True Crimeが多い）
- Publication_Day: 7曜日でほぼ均等分布
- Publication_Time: 4時間帯でほぼ均等分布
- Episode_Sentiment: 3種類でほぼ均等分布
- Podcast_Name: 48種類
- Episode_Title: 100種類

### 数値変数の分析
- Episode_Length_minutes: 平均64.5分、範囲0-325分
- Host_Popularity_percentage: 平均59.9%、範囲1.3-119.5%
- Guest_Popularity_percentage: 平均52.2%、範囲0-119.9%
- Number_of_Ads: 平均1.35個、範囲0-104個

## 特徴量エンジニアリングのアイデア

### 1. 欠損値処理
- Guest_Popularity_percentage: ゲストなしエピソードの可能性→0や特別値で補完
- Episode_Length_minutes: ジャンルやポッドキャスト別の中央値で補完

### 2. 新規特徴量の作成
- ホストとゲストの人気度の差分・比率
- エピソード長と聴取時間の比率（完聴率の推定）
- 時間帯×曜日の組み合わせ特徴量
- 広告密度（広告数/エピソード長）

### 3. カテゴリカル変数のエンコーディング
- Genre: ワンホットエンコーディングまたはターゲットエンコーディング
- Publication_Time/Day: 順序エンコーディングの検討
- Podcast_Name/Episode_Title: 頻度エンコーディングまたはターゲットエンコーディング

### 4. 数値変数の変換
- 歪んだ分布の対数変換
- 外れ値の処理
- 標準化/正規化

## ノートブック

- `exp000.ipynb`: EDA用ノートブック