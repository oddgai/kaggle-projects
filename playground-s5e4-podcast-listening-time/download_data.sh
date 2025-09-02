#!/bin/bash

# Kaggle Playground Series S5E4のデータをダウンロード

echo "Kaggle Playground Series S5E4のデータをダウンロードします..."

# Kaggle APIの認証確認
if [ ! -f ~/.kaggle/kaggle.json ]; then
    echo "エラー: ~/.kaggle/kaggle.json が見つかりません"
    echo "Kaggle APIトークンを設定してください"
    echo "https://www.kaggle.com/docs/api#authentication"
    exit 1
fi

# データディレクトリに移動
cd data

# データをダウンロード
echo "データをダウンロード中..."
kaggle competitions download -c playground-series-s5e4

# 解凍
echo "データを解凍中..."
unzip -o playground-series-s5e4.zip

# 不要なzipファイルを削除
rm playground-series-s5e4.zip

echo "データのダウンロードが完了しました"
echo "ファイル一覧:"
ls -la *.csv