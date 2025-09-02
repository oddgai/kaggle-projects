#!/usr/bin/env python3
"""
実験004: 高速アンサンブルモデル (LightGBM + XGBoost)
固定重み(50:50)で迅速に結果を取得
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
import pickle
import json

from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import lightgbm as lgb
import xgboost as xgb

warnings.filterwarnings('ignore')
np.random.seed(42)

print("=== 実験004開始（高速アンサンブル: LightGBM 50% + XGBoost 50%）===")

# データの読み込み
data_dir = Path('../../data')

train_df = pd.read_csv(data_dir / 'train.csv')
test_df = pd.read_csv(data_dir / 'test.csv')
sample_submission = pd.read_csv(data_dir / 'sample_submission.csv')

print(f"訓練データ: {train_df.shape}")
print(f"テストデータ: {test_df.shape}")

target_col = 'Listening_Time_minutes'

def enhanced_preprocess_data(df, is_train=True, target_col='Listening_Time_minutes', target_encoder=None):
    """データの前処理を行う関数"""
    df_processed = df.copy()
    
    ids = None
    if 'id' in df_processed.columns:
        ids = df_processed['id']
        df_processed = df_processed.drop('id', axis=1)
    
    if is_train and target_col in df_processed.columns:
        y = df_processed[target_col]
        df_processed = df_processed.drop(target_col, axis=1)
    else:
        y = None
    
    categorical_cols = df_processed.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = df_processed.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # 欠損値補完
    for col in numerical_cols:
        if df_processed[col].isnull().sum() > 0:
            median_value = df_processed[col].median()
            df_processed[col] = df_processed[col].fillna(median_value)
    
    df_processed['Guest_Popularity_percentage'] = df_processed['Guest_Popularity_percentage'].fillna(0)
    
    # 新規特徴量
    df_processed['Ad_Density'] = df_processed['Number_of_Ads'] / (df_processed['Episode_Length_minutes'] + 1e-8)
    df_processed['Host_Guest_Popularity_Diff'] = (df_processed['Host_Popularity_percentage'] - 
                                                   df_processed['Guest_Popularity_percentage'])
    df_processed['Has_Guest'] = (df_processed['Guest_Popularity_percentage'] > 0).astype(int)
    df_processed['Episode_Length_squared'] = df_processed['Episode_Length_minutes'] ** 2
    df_processed['Episode_Length_log'] = np.log(df_processed['Episode_Length_minutes'] + 1)
    df_processed['Host_Guest_Popularity_Sum'] = df_processed['Host_Popularity_percentage'] + df_processed['Guest_Popularity_percentage']
    df_processed['Host_Guest_Popularity_Ratio'] = df_processed['Host_Popularity_percentage'] / (df_processed['Guest_Popularity_percentage'] + 1)
    df_processed['Ads_per_Hour'] = df_processed['Number_of_Ads'] / ((df_processed['Episode_Length_minutes'] / 60) + 1e-8)
    df_processed['Has_Ads'] = (df_processed['Number_of_Ads'] > 0).astype(int)
    df_processed['Episode_Length_Category'] = pd.cut(df_processed['Episode_Length_minutes'], 
                                                     bins=[0, 30, 60, 90, float('inf')], 
                                                     labels=['short', 'medium', 'long', 'very_long'])
    
    # カテゴリカル変数の前処理
    le_dict = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df_processed[col] = df_processed[col].fillna('Missing')
        df_processed[col] = le.fit_transform(df_processed[col])
        le_dict[col] = le
    
    le_length_cat = LabelEncoder()
    df_processed['Episode_Length_Category'] = le_length_cat.fit_transform(df_processed['Episode_Length_Category'])
    le_dict['Episode_Length_Category'] = le_length_cat
    
    # ターゲットエンコーディング（訓練データの場合のみ）
    if is_train and y is not None and target_encoder is None:
        target_encoding_cols = ['Podcast_Name', 'Episode_Title', 'Genre']
        target_encoder = {}
        
        for col in target_encoding_cols:
            if col in categorical_cols:
                global_mean = y.mean()
                category_means = df_processed.groupby(col)[col].apply(lambda x: len(x))
                category_targets = pd.DataFrame({'original_col': df_processed[col], 'target': y})
                category_target_mean = category_targets.groupby('original_col')['target'].mean()
                
                alpha = 10
                smoothed_means = (category_target_mean * category_means + global_mean * alpha) / (category_means + alpha)
                
                target_encoder[col] = smoothed_means
                df_processed[f'{col}_target_encoded'] = df_processed[col].map(target_encoder[col]).fillna(global_mean)
    
    elif target_encoder is not None:
        for col, encoder in target_encoder.items():
            if col in df_processed.columns:
                global_mean = encoder.mean()
                df_processed[f'{col}_target_encoded'] = df_processed[col].map(encoder).fillna(global_mean)
    
    return df_processed, y, ids, le_dict, target_encoder if is_train else None

# データの前処理実行
print("\n=== データの前処理 ===")
X_train_processed, y_train, train_ids, le_dict_train, target_encoder = enhanced_preprocess_data(train_df, is_train=True)
X_test_processed, _, test_ids, _, _ = enhanced_preprocess_data(test_df, is_train=False, target_encoder=target_encoder)

print(f"処理後の訓練データ形状: {X_train_processed.shape}")
print(f"処理後のテストデータ形状: {X_test_processed.shape}")

# 固定重み設定
lgb_weight = 0.5
xgb_weight = 0.5

print(f"\n=== 固定重み設定 ===")
print(f"LightGBM重み: {lgb_weight}")
print(f"XGBoost重み: {xgb_weight}")

# 5-Fold Cross Validation
print("\n=== 5-Fold Cross Validation ===")
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []

for fold, (train_idx, val_idx) in enumerate(kfold.split(X_train_processed)):
    print(f"Fold {fold + 1}/5")
    
    X_fold_train, X_fold_val = X_train_processed.iloc[train_idx], X_train_processed.iloc[val_idx]
    y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
    
    # LightGBMモデル訓練
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
    
    train_dataset = lgb.Dataset(X_fold_train, label=y_fold_train)
    val_dataset = lgb.Dataset(X_fold_val, label=y_fold_val, reference=train_dataset)
    
    lgb_model = lgb.train(
        lgb_params,
        train_dataset,
        valid_sets=[val_dataset],
        num_boost_round=1000,
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)]
    )
    
    lgb_pred = lgb_model.predict(X_fold_val, num_iteration=lgb_model.best_iteration)
    
    # XGBoostモデル訓練
    xgb_params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'booster': 'gbtree',
        'max_depth': 10,
        'learning_rate': 0.035487574582325,
        'subsample': 0.7162926496452663,
        'colsample_bytree': 0.9761481011191779,
        'colsample_bylevel': 0.8369241306283279,
        'min_child_weight': 6,
        'reg_alpha': 1.1603476776684138,
        'reg_lambda': 1.145770443800777,
        'random_state': 42,
        'n_jobs': -1,
        'verbosity': 0
    }
    
    train_data = xgb.DMatrix(X_fold_train, label=y_fold_train)
    val_data = xgb.DMatrix(X_fold_val, label=y_fold_val)
    
    xgb_model = xgb.train(
        xgb_params,
        train_data,
        evals=[(val_data, 'eval')],
        num_boost_round=1419,
        early_stopping_rounds=100,
        verbose_eval=0
    )
    
    xgb_pred = xgb_model.predict(val_data, iteration_range=(0, xgb_model.best_iteration))
    
    # アンサンブル予測
    ensemble_pred = lgb_weight * lgb_pred + xgb_weight * xgb_pred
    fold_rmse = np.sqrt(mean_squared_error(y_fold_val, ensemble_pred))
    cv_scores.append(fold_rmse)
    
    print(f"Fold {fold + 1} RMSE: {fold_rmse:.4f}")

cv_mean = np.mean(cv_scores)
cv_std = np.std(cv_scores)

print(f"\n=== Cross Validation結果 ===")
print(f"平均RMSE: {cv_mean:.4f} ± {cv_std:.4f}")
print(f"各FoldのRMSE: {[f'{score:.4f}' for score in cv_scores]}")

# 最終モデル訓練
print("\n=== 最終モデル訓練 ===")
X_train, X_val, y_train_split, y_val = train_test_split(
    X_train_processed, y_train, test_size=0.2, random_state=42
)

# LightGBMモデル
print("LightGBMモデル訓練中...")
train_dataset = lgb.Dataset(X_train, label=y_train_split)
val_dataset = lgb.Dataset(X_val, label=y_val, reference=train_dataset)

lgb_final_model = lgb.train(
    lgb_params,
    train_dataset,
    valid_sets=[train_dataset, val_dataset],
    num_boost_round=1000,
    callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)]
)

# XGBoostモデル
print("XGBoostモデル訓練中...")
train_data = xgb.DMatrix(X_train, label=y_train_split)
val_data = xgb.DMatrix(X_val, label=y_val)

xgb_final_model = xgb.train(
    xgb_params,
    train_data,
    evals=[(train_data, 'train'), (val_data, 'eval')],
    num_boost_round=1419,
    early_stopping_rounds=100,
    verbose_eval=0
)

# 予測と評価
lgb_train_pred = lgb_final_model.predict(X_train, num_iteration=lgb_final_model.best_iteration)
lgb_val_pred = lgb_final_model.predict(X_val, num_iteration=lgb_final_model.best_iteration)

xgb_train_pred = xgb_final_model.predict(xgb.DMatrix(X_train), iteration_range=(0, xgb_final_model.best_iteration))
xgb_val_pred = xgb_final_model.predict(xgb.DMatrix(X_val), iteration_range=(0, xgb_final_model.best_iteration))

ensemble_train_pred = lgb_weight * lgb_train_pred + xgb_weight * xgb_train_pred
ensemble_val_pred = lgb_weight * lgb_val_pred + xgb_weight * xgb_val_pred

# 評価指標計算
ensemble_train_rmse = np.sqrt(mean_squared_error(y_train_split, ensemble_train_pred))
ensemble_train_mae = mean_absolute_error(y_train_split, ensemble_train_pred)
ensemble_train_r2 = r2_score(y_train_split, ensemble_train_pred)

ensemble_val_rmse = np.sqrt(mean_squared_error(y_val, ensemble_val_pred))
ensemble_val_mae = mean_absolute_error(y_val, ensemble_val_pred)
ensemble_val_r2 = r2_score(y_val, ensemble_val_pred)

print(f"\n=== アンサンブル性能 ===")
print(f"訓練RMSE: {ensemble_train_rmse:.4f}")
print(f"検証RMSE: {ensemble_val_rmse:.4f}")
print(f"検証MAE: {ensemble_val_mae:.4f}")
print(f"検証R²: {ensemble_val_r2:.4f}")

# テストデータでの予測
print("\n=== テストデータでの予測 ===")
lgb_test_pred = lgb_final_model.predict(X_test_processed, num_iteration=lgb_final_model.best_iteration)
xgb_test_pred = xgb_final_model.predict(xgb.DMatrix(X_test_processed), iteration_range=(0, xgb_final_model.best_iteration))

ensemble_test_pred = lgb_weight * lgb_test_pred + xgb_weight * xgb_test_pred

# 提出ファイル作成
submission_df = pd.DataFrame({
    'id': test_ids,
    target_col: ensemble_test_pred
})

print(f"提出ファイルの統計:")
print(submission_df[target_col].describe())

# ファイル保存
results_dir = Path('../../results/exp004')
results_dir.mkdir(parents=True, exist_ok=True)

submission_path = results_dir / 'submission.csv'
submission_df.to_csv(submission_path, index=False)

# モデル保存
with open(results_dir / 'lgb_model.pkl', 'wb') as f:
    pickle.dump(lgb_final_model, f)
with open(results_dir / 'xgb_model.pkl', 'wb') as f:
    pickle.dump(xgb_final_model, f)

# 実験結果記録
experiment_results = {
    'experiment_id': 'exp004',
    'model_type': 'Ensemble_LightGBM_XGBoost_Fast',
    'ensemble_method': 'fixed_weights',
    'lgb_weight': float(lgb_weight),
    'xgb_weight': float(xgb_weight),
    'features': list(X_train_processed.columns),
    'num_features': len(X_train_processed.columns),
    'ensemble_train_rmse': float(ensemble_train_rmse),
    'ensemble_train_mae': float(ensemble_train_mae),
    'ensemble_train_r2': float(ensemble_train_r2),
    'ensemble_val_rmse': float(ensemble_val_rmse),
    'ensemble_val_mae': float(ensemble_val_mae),
    'ensemble_val_r2': float(ensemble_val_r2),
    'cv_scores': [float(score) for score in cv_scores],
    'cv_rmse_mean': float(cv_mean),
    'cv_rmse_std': float(cv_std),
    'public_score': None,
    'private_score': None,
    'train_size': len(X_train_processed),
    'test_size': len(X_test_processed),
    'target_variable': target_col,
    'cv_folds': 5,
    'validation_split': 0.2,
    'random_state': 42,
    'preprocessing': {
        'missing_value_strategy': 'median_imputation',
        'categorical_encoding': 'label_encoding_and_target_encoding',
        'feature_engineering': 'enhanced_features,target_encoding,polynomial_features'
    }
}

# 実験結果をJSONファイルに保存
with open(results_dir / 'experiment_results.json', 'w', encoding='utf-8') as f:
    json.dump(experiment_results, f, indent=2, ensure_ascii=False)

print(f"\n=== 実験結果サマリー ===")
print(f"モデル: {experiment_results['model_type']}")
print(f"アンサンブル検証RMSE: {experiment_results['ensemble_val_rmse']:.4f}")
print(f"アンサンブルCV平均RMSE: {experiment_results['cv_rmse_mean']:.4f} ± {experiment_results['cv_rmse_std']:.4f}")

print(f"\n全実験との比較:")
print(f"exp001 (LightGBM) CV RMSE: 13.0023")
print(f"exp002 (XGBoost) CV RMSE: 12.8926")
print(f"exp003 (XGBoost最適) CV RMSE: 12.7938")
print(f"exp004 (アンサンブル) CV RMSE: {experiment_results['cv_rmse_mean']:.4f}")

improvement_from_best = 12.7938 - experiment_results['cv_rmse_mean']
improvement_from_001 = 13.0023 - experiment_results['cv_rmse_mean']
print(f"exp003からの改善度: {improvement_from_best:.4f}")
print(f"exp001からの改善度: {improvement_from_001:.4f}")

# 目標達成確認
if experiment_results['cv_rmse_mean'] < 12.0:
    print(f"\n🎉 目標達成！ CV RMSE {experiment_results['cv_rmse_mean']:.4f} < 12.0")
    print("アンサンブルモデルで目標スコア12未満を達成しました！")
else:
    print(f"\n⚠️ 目標未達成。CV RMSE {experiment_results['cv_rmse_mean']:.4f} >= 12.0")
    print(f"目標まで残り: {experiment_results['cv_rmse_mean'] - 12.0:.4f}")

print("\n=== 実験004完了 ===")