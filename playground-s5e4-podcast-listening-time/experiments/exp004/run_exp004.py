#!/usr/bin/env python3
"""
実験004: アンサンブルモデル (LightGBM + XGBoost)
Playground Series S5E4: ポッドキャスト聴取時間予測
exp001のLightGBMとexp003の最適化XGBoostを組み合わせたアンサンブルで目標12未満を達成
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
import pickle
import json
from itertools import product

from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import lightgbm as lgb
import xgboost as xgb

warnings.filterwarnings('ignore')

# 表示設定
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)

# スタイル設定
sns.set_style('whitegrid')
plt.style.use('default')

# シード値の設定
np.random.seed(42)

print("=== 実験004開始（LightGBM + XGBoost アンサンブル）===")

# データの読み込み
data_dir = Path('../../data')

train_df = pd.read_csv(data_dir / 'train.csv')
test_df = pd.read_csv(data_dir / 'test.csv')
sample_submission = pd.read_csv(data_dir / 'sample_submission.csv')

print(f"訓練データ: {train_df.shape}")
print(f"テストデータ: {test_df.shape}")

target_col = 'Listening_Time_minutes'

def enhanced_preprocess_data(df, is_train=True, target_col='Listening_Time_minutes', target_encoder=None):
    """
    拡張されたデータの前処理を行う関数（exp002/003と同じ）
    """
    # データのコピー
    df_processed = df.copy()
    
    # IDカラムを保存
    ids = None
    if 'id' in df_processed.columns:
        ids = df_processed['id']
        df_processed = df_processed.drop('id', axis=1)
    
    # 目標変数の分離（訓練データの場合）
    if is_train and target_col in df_processed.columns:
        y = df_processed[target_col]
        df_processed = df_processed.drop(target_col, axis=1)
    else:
        y = None
    
    # カテゴリカル変数と数値変数を分離
    categorical_cols = df_processed.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = df_processed.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # 数値変数の欠損値補完
    for col in numerical_cols:
        if df_processed[col].isnull().sum() > 0:
            median_value = df_processed[col].median()
            df_processed[col] = df_processed[col].fillna(median_value)
    
    # Guest_Popularity_percentageの欠損値を0で補完
    df_processed['Guest_Popularity_percentage'] = df_processed['Guest_Popularity_percentage'].fillna(0)
    
    # 新規特徴量の作成
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
    
    # Episode_Length_Categoryのエンコーディング
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
print("\\n=== データの前処理 ===")
X_train_processed, y_train, train_ids, le_dict_train, target_encoder = enhanced_preprocess_data(train_df, is_train=True)
X_test_processed, _, test_ids, _, _ = enhanced_preprocess_data(test_df, is_train=False, target_encoder=target_encoder)

print(f"処理後の訓練データ形状: {X_train_processed.shape}")
print(f"処理後のテストデータ形状: {X_test_processed.shape}")

def train_lightgbm_model(X_train, y_train, X_val, y_val):
    """
    LightGBMモデルを訓練（exp001と同じ設定）
    """
    # exp001と同じパラメータ設定
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
    
    # データセット作成
    train_dataset = lgb.Dataset(X_train, label=y_train)
    val_dataset = lgb.Dataset(X_val, label=y_val, reference=train_dataset)
    
    # モデル訓練
    model = lgb.train(
        lgb_params,
        train_dataset,
        valid_sets=[train_dataset, val_dataset],
        num_boost_round=1000,
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)]
    )
    
    return model

def train_xgboost_model(X_train, y_train, X_val, y_val):
    """
    XGBoostモデルを訓練（exp003の最適化パラメータ使用）
    """
    # exp003の最適化されたパラメータ
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
    
    # データセット作成
    train_data = xgb.DMatrix(X_train, label=y_train)
    val_data = xgb.DMatrix(X_val, label=y_val)
    
    # モデル訓練
    model = xgb.train(
        xgb_params,
        train_data,
        evals=[(train_data, 'train'), (val_data, 'eval')],
        num_boost_round=1419,  # exp003の最適値
        early_stopping_rounds=100,
        verbose_eval=0
    )
    
    return model

def find_optimal_weights(X_train_processed, y_train):
    """
    Cross Validationで最適なアンサンブル重みを探索
    """
    print("\\n=== アンサンブル重み最適化 ===")
    
    # 重み候補
    weight_candidates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    best_weight = 0.5
    best_score = float('inf')
    
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    
    for lgb_weight in weight_candidates:
        xgb_weight = 1.0 - lgb_weight
        fold_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(kfold.split(X_train_processed)):
            # データ分割
            X_fold_train, X_fold_val = X_train_processed.iloc[train_idx], X_train_processed.iloc[val_idx]
            y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            
            # LightGBMモデル訓練
            lgb_model = train_lightgbm_model(X_fold_train, y_fold_train, X_fold_val, y_fold_val)
            lgb_pred = lgb_model.predict(X_fold_val, num_iteration=lgb_model.best_iteration)
            
            # XGBoostモデル訓練
            xgb_model = train_xgboost_model(X_fold_train, y_fold_train, X_fold_val, y_fold_val)
            xgb_pred = xgb_model.predict(xgb.DMatrix(X_fold_val), iteration_range=(0, xgb_model.best_iteration))
            
            # アンサンブル予測
            ensemble_pred = lgb_weight * lgb_pred + xgb_weight * xgb_pred
            
            # RMSE計算
            fold_rmse = np.sqrt(mean_squared_error(y_fold_val, ensemble_pred))
            fold_scores.append(fold_rmse)
        
        avg_score = np.mean(fold_scores)
        print(f"LightGBM重み: {lgb_weight:.1f}, XGBoost重み: {xgb_weight:.1f}, CV RMSE: {avg_score:.4f}")
        
        if avg_score < best_score:
            best_score = avg_score
            best_weight = lgb_weight
    
    return best_weight, 1.0 - best_weight, best_score

# 最適重み探索
best_lgb_weight, best_xgb_weight, best_ensemble_score = find_optimal_weights(X_train_processed, y_train)

print(f"\\n=== 最適重み結果 ===")
print(f"最適LightGBM重み: {best_lgb_weight:.1f}")
print(f"最適XGBoost重み: {best_xgb_weight:.1f}")
print(f"最適アンサンブルCV RMSE: {best_ensemble_score:.4f}")

# 最終アンサンブルモデル訓練
print("\\n=== 最終アンサンブルモデル訓練 ===")

# 訓練・検証データの分割
X_train, X_val, y_train_split, y_val = train_test_split(
    X_train_processed, y_train, 
    test_size=0.2, 
    random_state=42
)

# 個別モデル訓練
print("LightGBMモデル訓練中...")
lgb_final_model = train_lightgbm_model(X_train, y_train_split, X_val, y_val)

print("XGBoostモデル訓練中...")
xgb_final_model = train_xgboost_model(X_train, y_train_split, X_val, y_val)

# 個別モデルの予測
lgb_train_pred = lgb_final_model.predict(X_train, num_iteration=lgb_final_model.best_iteration)
lgb_val_pred = lgb_final_model.predict(X_val, num_iteration=lgb_final_model.best_iteration)

xgb_train_pred = xgb_final_model.predict(xgb.DMatrix(X_train), iteration_range=(0, xgb_final_model.best_iteration))
xgb_val_pred = xgb_final_model.predict(xgb.DMatrix(X_val), iteration_range=(0, xgb_final_model.best_iteration))

# アンサンブル予測
ensemble_train_pred = best_lgb_weight * lgb_train_pred + best_xgb_weight * xgb_train_pred
ensemble_val_pred = best_lgb_weight * lgb_val_pred + best_xgb_weight * xgb_val_pred

# 評価
def calculate_metrics(y_true, y_pred, dataset_name):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    print(f"\\n=== {dataset_name}の評価結果 ===")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R²: {r2:.4f}")
    
    return rmse, mae, r2

# 個別モデル評価
print("\\n=== 個別モデル性能 ===")
lgb_train_rmse, _, _ = calculate_metrics(y_train_split, lgb_train_pred, "LightGBM訓練")
lgb_val_rmse, _, _ = calculate_metrics(y_val, lgb_val_pred, "LightGBM検証")

xgb_train_rmse, _, _ = calculate_metrics(y_train_split, xgb_train_pred, "XGBoost訓練")
xgb_val_rmse, _, _ = calculate_metrics(y_val, xgb_val_pred, "XGBoost検証")

# アンサンブル評価
print("\\n=== アンサンブルモデル性能 ===")
ensemble_train_rmse, ensemble_train_mae, ensemble_train_r2 = calculate_metrics(y_train_split, ensemble_train_pred, "アンサンブル訓練")
ensemble_val_rmse, ensemble_val_mae, ensemble_val_r2 = calculate_metrics(y_val, ensemble_val_pred, "アンサンブル検証")

# 最終Cross Validation
print("\\n=== 最終5-Fold Cross Validation ===")
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
final_cv_scores = []

for fold, (train_idx, val_idx) in enumerate(kfold.split(X_train_processed)):
    print(f"Fold {fold + 1}/5")
    
    X_fold_train, X_fold_val = X_train_processed.iloc[train_idx], X_train_processed.iloc[val_idx]
    y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
    
    # LightGBMモデル
    fold_lgb_model = train_lightgbm_model(X_fold_train, y_fold_train, X_fold_val, y_fold_val)
    fold_lgb_pred = fold_lgb_model.predict(X_fold_val, num_iteration=fold_lgb_model.best_iteration)
    
    # XGBoostモデル
    fold_xgb_model = train_xgboost_model(X_fold_train, y_fold_train, X_fold_val, y_fold_val)
    fold_xgb_pred = fold_xgb_model.predict(xgb.DMatrix(X_fold_val), iteration_range=(0, fold_xgb_model.best_iteration))
    
    # アンサンブル予測
    fold_ensemble_pred = best_lgb_weight * fold_lgb_pred + best_xgb_weight * fold_xgb_pred
    fold_rmse = np.sqrt(mean_squared_error(y_fold_val, fold_ensemble_pred))
    final_cv_scores.append(fold_rmse)
    
    print(f"Fold {fold + 1} RMSE: {fold_rmse:.4f}")

cv_mean = np.mean(final_cv_scores)
cv_std = np.std(final_cv_scores)

print(f"\\n=== 最終Cross Validation結果 ===")
print(f"平均RMSE: {cv_mean:.4f} ± {cv_std:.4f}")
print(f"各FoldのRMSE: {[f'{score:.4f}' for score in final_cv_scores]}")

# テストデータでの予測
print("\\n=== テストデータでの予測 ===")
lgb_test_pred = lgb_final_model.predict(X_test_processed, num_iteration=lgb_final_model.best_iteration)
xgb_test_pred = xgb_final_model.predict(xgb.DMatrix(X_test_processed), iteration_range=(0, xgb_final_model.best_iteration))

ensemble_test_pred = best_lgb_weight * lgb_test_pred + best_xgb_weight * xgb_test_pred

# 提出ファイルの作成
submission_df = pd.DataFrame({
    'id': test_ids,
    target_col: ensemble_test_pred
})

print(f"提出ファイルの統計: {submission_df[target_col].describe()}")

# ファイル保存
results_dir = Path('../../results/exp004')
results_dir.mkdir(parents=True, exist_ok=True)

submission_path = results_dir / 'submission.csv'
submission_df.to_csv(submission_path, index=False)
print(f"提出ファイルを保存: {submission_path}")

# モデルの保存
lgb_model_path = results_dir / 'lgb_model.pkl'
xgb_model_path = results_dir / 'xgb_model.pkl'

with open(lgb_model_path, 'wb') as f:
    pickle.dump(lgb_final_model, f)
with open(xgb_model_path, 'wb') as f:
    pickle.dump(xgb_final_model, f)

print(f"LightGBMモデルを保存: {lgb_model_path}")
print(f"XGBoostモデルを保存: {xgb_model_path}")

# 実験結果の記録
experiment_results = {
    'experiment_id': 'exp004',
    'model_type': 'Ensemble_LightGBM_XGBoost',
    'ensemble_method': 'weighted_average',
    'lgb_weight': float(best_lgb_weight),
    'xgb_weight': float(best_xgb_weight),
    
    # 個別モデル設定
    'lgb_params': {
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
    },
    'xgb_params': {
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
        'n_jobs': -1
    },
    
    # 特徴量情報
    'features': list(X_train_processed.columns),
    'num_features': len(X_train_processed.columns),
    
    # 評価指標
    'ensemble_train_rmse': float(ensemble_train_rmse),
    'ensemble_train_mae': float(ensemble_train_mae),
    'ensemble_train_r2': float(ensemble_train_r2),
    'ensemble_val_rmse': float(ensemble_val_rmse),
    'ensemble_val_mae': float(ensemble_val_mae),
    'ensemble_val_r2': float(ensemble_val_r2),
    
    # 個別モデル性能
    'lgb_val_rmse': float(lgb_val_rmse),
    'xgb_val_rmse': float(xgb_val_rmse),
    
    # Cross Validation結果
    'cv_scores': [float(score) for score in final_cv_scores],
    'cv_rmse_mean': float(cv_mean),
    'cv_rmse_std': float(cv_std),
    
    # Kaggleスコア
    'public_score': None,
    'private_score': None,
    
    # データ情報
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
results_json_path = results_dir / 'experiment_results.json'
with open(results_json_path, 'w', encoding='utf-8') as f:
    json.dump(experiment_results, f, indent=2, ensure_ascii=False)
print(f"実験結果を保存: {results_json_path}")

print("\\n=== 実験結果サマリー ===")
print(f"モデル: {experiment_results['model_type']}")
print(f"LightGBM重み: {best_lgb_weight:.1f}")
print(f"XGBoost重み: {best_xgb_weight:.1f}")
print(f"アンサンブル検証RMSE: {experiment_results['ensemble_val_rmse']:.4f}")
print(f"アンサンブルCV平均RMSE: {experiment_results['cv_rmse_mean']:.4f} ± {experiment_results['cv_rmse_std']:.4f}")

print(f"\\n全実験との比較:")
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
    print(f"\\n🎉 目標達成！ CV RMSE {experiment_results['cv_rmse_mean']:.4f} < 12.0")
    print("アンサンブルモデルで目標スコア12未満を達成しました！")
else:
    print(f"\\n⚠️  目標未達成。CV RMSE {experiment_results['cv_rmse_mean']:.4f} >= 12.0")
    print(f"目標まで残り: {experiment_results['cv_rmse_mean'] - 12.0:.4f}")
    print("さらなる改善が必要です。")

print("\\n=== 実験004完了 ===")