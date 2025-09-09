# Improvement Plan - exp005: トップスコア達成への道筋

## 現状分析

**現在のベストスコア:**
- exp003 CV RMSE: 12.7938, Public: 12.94051, Private: 12.85914  
- exp004 CV RMSE: 12.8203, Public: 12.96794, Private: 12.87623
- **トップスコアとの差:** 1.41081 (目標: 11.44833)

## 研究から得られた上位手法

### 1. 高度な特徴量エンジニアリング
**成功事例で使われている特徴量:**
- **相互作用特徴量**: Length_Host_Interaction, Genre_Host_Interaction
- **時間ベース特徴量**: Is_Weekend, Hour_of_Day, Season_of_Year  
- **グループ統計**: Genre_Mean_Length, Host_Avg_Popularity, Guest_Avg_Rating
- **比率・割合特徴量**: Listen_Rate_Ratio, Popularity_Density
- **エンコーディング改良**: より高度なターゲットエンコーディング

### 2. 新しいモデル追加
**上位解で使われているモデル:**
- **CatBoost**: カテゴリカル変数の扱いが優秀
- **H2O AutoML**: 自動最適化で高性能
- **Neural Networks**: 非線形パターンの捕捉

### 3. アンサンブル手法の改良
**現在**: 固定50:50重み → **改良**: 動的重み最適化
- Optuna使用での重み最適化
- スタッキング（メタモデル）
- ブレンディング手法

### 4. データ前処理の強化
**上位解のポイント:**
- IQRによる外れ値除去（現在未実装）
- StandardScalerでの正規化
- カテゴリカル変数のクリッピング

## 具体的な改善計画

### exp005: Advanced Feature Engineering + CatBoost
**目標RMSE:** 12.5以下
**主な改良点:**
1. 20個以上の新規特徴量作成
2. CatBoostモデルの追加
3. より高度なターゲットエンコーディング
4. IQRによる外れ値除去

### exp006: H2O AutoML Implementation
**目標RMSE:** 12.3以下
**主な改良点:**
1. H2O AutoMLによる自動最適化
2. 複数モデルの自動アンサンブル
3. 自動特徴量選択

### exp007: Neural Network + Advanced Ensemble
**目標RMSE:** 12.0以下
**主な改良点:**
1. Deep Learningモデル追加
2. 5モデルアンサンブル（LGB, XGB, CatBoost, H2O, NN）
3. Optunaでの重み最適化
4. スタッキング手法

### exp008: Ultimate Ensemble
**目標RMSE:** 11.8以下（トップスコア接近）
**主な改良点:**
1. 全モデルの最適版をアンサンブル
2. 2段階スタッキング
3. 最高レベル特徴量エンジニアリング

## 実装優先順位

### Phase 1: Feature Engineering強化 (exp005)
```python
# 新規特徴量リスト（20個以上）
- Length_Host_Interaction = Episode_Length * Host_Popularity
- Genre_Host_Interaction = Genre_encoded * Host_Popularity  
- Is_Weekend = Publication_Day in [5,6]
- Hour_of_Day = Publication_Time hour
- Popularity_Density = (Host + Guest) / Episode_Length
- Listen_Rate_Ratio = Expected_Listen / Episode_Length
- Host_Guest_Synergy = Host_Pop * Guest_Pop * Has_Guest
- Genre_Length_Category = pd.cut(Episode_Length, bins=genre_stats)
- Ad_Effectiveness = Ads * Host_Popularity / Length
- Prime_Time_Flag = Hour_of_Day in [7-9, 17-19]
```

### Phase 2: CatBoost追加 (exp005続き)
```python
# CatBoostの設定
cat_features = ['Podcast_Name', 'Episode_Title', 'Genre', 
               'Publication_Day', 'Episode_Length_Category']
catboost_params = {
    'iterations': 2000,
    'learning_rate': 0.05,
    'depth': 8,
    'cat_features': cat_features,
    'eval_metric': 'RMSE',
    'random_seed': 42
}
```

### Phase 3: H2O AutoML (exp006)
```python
# H2O AutoMLの設定  
h2o_automl = H2OAutoML(max_models=20, seed=42, 
                       max_runtime_secs=3600)
```

### Phase 4: Advanced Ensemble (exp007-008)
```python
# 5モデルアンサンブル + 重み最適化
models = [lgb_model, xgb_model, cat_model, h2o_model, nn_model]
# Optunaで重み最適化
# スタッキングでメタモデル
```

## 期待効果

- **exp005**: CV RMSE 12.5 → Public 12.7程度
- **exp006**: CV RMSE 12.3 → Public 12.5程度  
- **exp007**: CV RMSE 12.0 → Public 12.2程度
- **exp008**: CV RMSE 11.8 → Public 12.0程度（目標達成）

**トップスコアとの差を1.4 → 0.6程度まで縮小可能**

## 次のステップ

1. **exp005の実装開始**: Advanced Feature Engineering + CatBoost
2. 各実験でMLflow記録とGit管理の継続
3. 段階的にトップスコアに接近