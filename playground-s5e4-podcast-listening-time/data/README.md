# データ概要

## ファイル一覧

- `train.csv`: 訓練データ
- `test.csv`: テストデータ
- `sample_submission.csv`: 提出用サンプルファイル

## データ項目

### 入力特徴量

| カラム名 | 説明 | データ型 |
|---------|------|----------|
| Episode_Length_minutes | エピソードの長さ（分） | float |
| Podcast_Name | ポッドキャスト名 | string |
| Episode_Title | エピソードタイトル | string |
| Host_Popularity_% | ホストの人気度（%） | float |
| Guest_Popularity_% | ゲストの人気度（%） | float |
| Publication_Day | 公開曜日 | string |
| Publication_Time | 公開時刻 | string |
| Number_of_Ads | 広告の数 | int |
| Episode_Sentiment | エピソードのセンチメント | float |
| Genre | ジャンル | string |

### 目標変数

| カラム名 | 説明 | データ型 |
|---------|------|----------|
| Listening_Time_minutes | 聴取時間（分） | float |

## データの特徴

- タスクタイプ: 回帰
- 評価指標: RMSE (Root Mean Squared Error)
- データセットの特徴: ポッドキャストのメタデータから聴取時間を予測

## データ取得方法

```bash
# Kaggle CLIを使用してデータをダウンロード
kaggle competitions download -c playground-series-s5e4
unzip playground-series-s5e4.zip -d data/
```

## 注意事項

- テストデータには目標変数（Listening_Time_minutes）が含まれていません
- 提出ファイルはsample_submission.csvと同じフォーマットで作成する必要があります