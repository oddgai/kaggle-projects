# 実験001: ベースラインモデル

## 目的

シンプルなベースラインモデルの構築

## 実験設定

### モデル

- **アルゴリズム**: LightGBM
- **タスク**: 回帰問題（RMSE最適化）

### 特徴量

**基本特徴量（10個）**
- カテゴリカル変数: Podcast_Name, Episode_Title, Genre, Publication_Day, Publication_Time, Episode_Sentiment
- 数値変数: Episode_Length_minutes, Host_Popularity_percentage, Guest_Popularity_percentage, Number_of_Ads

**新規作成特徴量（3個）**
- Ad_Density: 広告数/エピソード長
- Host_Guest_Popularity_Diff: ホストとゲストの人気度差
- Has_Guest: ゲスト存在フラグ

### ハイパーパラメータ

```python
lgb_params = {
    'objective': 'regression',
    'metric': 'rmse', 
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1,
    'random_state': 42
}
```

### 検証方法

- 5-fold Cross Validation
- Train/Validation split: 80%/20%

## 結果

### スコア

- **CV Score (RMSE)**: 13.0023 ± 0.0216
- **Public Score (RMSE)**: 13.13295
- **Private Score (RMSE)**: 13.02087
- **検証データRMSE**: 12.9741
- **訓練データRMSE**: 12.6652
- **R²スコア**: 0.7712

### 特徴量重要度トップ5

1. **Episode_Length_minutes**: 2,569,071,625（エピソード長が最重要）
2. **Ad_Density**: 159,828,568（新規特徴量が2位！）
3. **Number_of_Ads**: 46,218,796（広告数）
4. **Host_Popularity_percentage**: 22,167,891（ホスト人気度）
5. **Guest_Popularity_percentage**: 11,931,373（ゲスト人気度）

## 学習内容

### 重要な発見

1. **エピソード長が最重要**: 聴取時間を予測する上で最も強力な特徴量
2. **広告密度の効果**: 新規作成したAd_Densityが2番目に重要な特徴量
3. **人気度の影響**: ホストとゲストの人気度も重要な予測要因
4. **安定したCV性能**: 5-fold CVで標準偏差0.02と安定した結果

### 前処理の効果

- 欠損値補完: 中央値による補完が効果的
- ラベルエンコーディング: カテゴリカル変数の適切な数値化
- 新規特徴量: 特にAd_Densityが高い予測力を示した

## 改善案

### 特徴量エンジニアリング
1. **時系列特徴量**: 公開時間帯×曜日の組み合わせ
2. **ターゲットエンコーディング**: カテゴリカル変数の高度なエンコーディング
3. **相互作用特徴量**: 人気度×ジャンルなどの組み合わせ

### モデル改善
1. **ハイパーパラメータ調整**: Optuna等による最適化
2. **アンサンブル**: XGBoost、CatBoost等との組み合わせ
3. **正則化強化**: 過学習抑制のためのパラメータ調整

### 次の実験方向
1. より高度な特徴量エンジニアリング（exp002）
2. ハイパーパラメータ最適化（exp003）
3. アンサンブルモデル（exp004）

## ファイル

- `exp001.ipynb`: 実験用ノートブック
- `train.py`: モデル訓練スクリプト
- `config.yaml`: 設定ファイル