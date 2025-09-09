#!/usr/bin/env python3
"""
exp006: "Less is More" - シンプルで堅牢なアプローチ
exp005の過学習を解消し、CV-Kaggle乖離を最小化
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import lightgbm as lgb
import xgboost as xgb
import pickle
import json
import warnings
from datetime import datetime
from pathlib import Path

warnings.filterwarnings('ignore')

def remove_outliers_conservative(df, target_col='Listening_Time_minutes'):
    """
    保守的な外れ値除去: 上下1%のみ除去
    """
    print("=== 保守的外れ値除去（上下1%） ===")
    
    original_size = len(df)
    lower = df[target_col].quantile(0.01)
    upper = df[target_col].quantile(0.99)
    
    df_clean = df[(df[target_col] >= lower) & (df[target_col] <= upper)].copy()
    removed = original_size - len(df_clean)
    removal_rate = removed / original_size * 100
    
    print(f"  下位1%閾値: {lower:.2f}")
    print(f"  上位1%閾値: {upper:.2f}")
    print(f"  除去件数: {removed}個 ({removal_rate:.1f}%)")
    print(f"  残存件数: {len(df_clean)}個")
    
    return df_clean

def create_simplified_features(df):
    """
    exp006: シンプルで堅牢な特徴量エンジニアリング（20個に厳選）
    """
    df_new = df.copy()
    
    print("=== シンプル特徴量エンジニアリング ===")
    
    # データ型の適切な変換
    print("  データ型変換...")
    
    # Episode_Sentimentを数値に変換
    sentiment_map = {'Positive': 1, 'Neutral': 0, 'Negative': -1}
    df_new['Episode_Sentiment'] = df_new['Episode_Sentiment'].map(sentiment_map).fillna(0)
    
    # Publication_Dayを数値に変換
    day_map = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 
               'Friday': 4, 'Saturday': 5, 'Sunday': 6}
    df_new['Publication_Day'] = df_new['Publication_Day'].map(day_map).fillna(0)
    
    # Publication_Timeを数値に変換
    time_map = {'Morning': 8, 'Afternoon': 14, 'Evening': 19, 'Night': 23}
    df_new['Publication_Time'] = df_new['Publication_Time'].map(time_map).fillna(12)
    
    # 基本数値特徴量の欠損値補完
    numeric_cols = ['Episode_Length_minutes', 'Host_Popularity_percentage', 
                   'Guest_Popularity_percentage', 'Number_of_Ads']
    for col in numeric_cols:
        if col in df_new.columns:
            df_new[col] = df_new[col].fillna(df_new[col].median())
    
    # ゲスト人気度の欠損値は0に設定
    if 'Guest_Popularity_percentage' in df_new.columns:
        df_new['Guest_Popularity_percentage'] = df_new['Guest_Popularity_percentage'].fillna(0)
    
    # 実績のある拡張特徴量のみ作成（10個）
    print("  実績ある拡張特徴量作成...")
    
    # 数学的変換（2個）
    df_new['Episode_Length_squared'] = df_new['Episode_Length_minutes'] ** 2
    df_new['Episode_Length_log'] = np.log1p(df_new['Episode_Length_minutes'])
    
    # 人気度関連（2個）
    df_new['Host_Guest_Popularity_Sum'] = (df_new['Host_Popularity_percentage'] + 
                                          df_new['Guest_Popularity_percentage'])
    df_new['Host_Guest_Popularity_Ratio'] = (df_new['Host_Popularity_percentage'] / 
                                            (df_new['Guest_Popularity_percentage'] + 1))
    
    # 広告関連（2個）
    df_new['Ad_Density'] = df_new['Number_of_Ads'] / (df_new['Episode_Length_minutes'] + 1)
    df_new['Ads_per_Hour'] = df_new['Number_of_Ads'] / (df_new['Episode_Length_minutes'] / 60 + 1)
    
    # フラグ特徴量（2個）
    df_new['Has_Guest'] = (df_new['Guest_Popularity_percentage'] > 0).astype(int)
    df_new['Has_Ads'] = (df_new['Number_of_Ads'] > 0).astype(int)
    
    # 時間特徴量（2個）
    df_new['Is_Weekend'] = df_new['Publication_Day'].isin([5, 6]).astype(int)
    # プライムタイム: Morning(8), Evening(19)
    df_new['Is_Prime_Time'] = df_new['Publication_Time'].isin([8, 19]).astype(int)
    
    print(f"  作成した特徴量数: {len(df_new.columns) - len(df.columns)}個")
    print(f"  総特徴量数: {len(df_new.columns)}個")
    
    return df_new

def safe_target_encoding(X_train, y_train, X_test, categorical_cols, alpha=20, random_state=42):
    """
    安全なターゲットエンコーディング（Holdout方式）
    """
    print("=== 安全なターゲットエンコーディング ===")
    
    X_train_encoded = X_train.copy()
    X_test_encoded = X_test.copy()
    
    # 全体平均
    global_mean = y_train.mean()
    
    for col in categorical_cols:
        # まず、ラベルエンコーディング済みの列名を確認
        encoded_col_name = f'{col}_encoded'
        if encoded_col_name in X_train.columns:
            print(f"  {col}のターゲットエンコーディング...")
            
            # 5-fold holdout でターゲットエンコーディング
            kf = KFold(n_splits=5, shuffle=True, random_state=random_state)
            encoded_values = np.zeros(len(X_train))
            
            # 訓練データのエンコーディング（Holdout方式）
            for train_idx, val_idx in kf.split(X_train):
                X_fold_train = X_train.iloc[train_idx]
                y_fold_train = y_train.iloc[train_idx]
                
                # Foldの統計計算（エンコード済みの列を使用）
                category_stats = pd.DataFrame({
                    encoded_col_name: X_fold_train[encoded_col_name],
                    'target': y_fold_train.values
                }).groupby(encoded_col_name)['target'].agg(['mean', 'count']).reset_index()
                
                # スムージング適用（より保守的）
                category_stats['smoothed_mean'] = (
                    (category_stats['count'] * category_stats['mean'] + alpha * global_mean) / 
                    (category_stats['count'] + alpha)
                )
                
                # Validation setにエンコーディング適用
                encoding_map = dict(zip(category_stats[encoded_col_name], category_stats['smoothed_mean']))
                encoded_values[val_idx] = X_train.iloc[val_idx][encoded_col_name].map(encoding_map).fillna(global_mean)
            
            X_train_encoded[f'{col}_target_encoded'] = encoded_values
            
            # テストデータのエンコーディング（全訓練データ使用）
            category_stats = pd.DataFrame({
                encoded_col_name: X_train[encoded_col_name],
                'target': y_train.values
            }).groupby(encoded_col_name)['target'].agg(['mean', 'count']).reset_index()
            
            category_stats['smoothed_mean'] = (
                (category_stats['count'] * category_stats['mean'] + alpha * global_mean) / 
                (category_stats['count'] + alpha)
            )
            
            encoding_map = dict(zip(category_stats[encoded_col_name], category_stats['smoothed_mean']))
            X_test_encoded[f'{col}_target_encoded'] = X_test[encoded_col_name].map(encoding_map).fillna(global_mean)
            
            print(f"    ユニークカテゴリ数: {len(encoding_map)}")
            print(f"    スムージング係数: {alpha}")
    
    return X_train_encoded, X_test_encoded

def prepare_data():
    """
    データ準備と前処理
    """
    print("=== データ読み込み ===")
    
    train_df = pd.read_csv('data/train.csv')
    test_df = pd.read_csv('data/test.csv')
    
    print(f"Train shape: {train_df.shape}")
    print(f"Test shape: {test_df.shape}")
    
    # 保守的外れ値除去（訓練データのみ）
    train_df = remove_outliers_conservative(train_df)
    print(f"外れ値除去後のTrain shape: {train_df.shape}")
    
    # シンプル特徴量エンジニアリング
    train_df = create_simplified_features(train_df)
    test_df = create_simplified_features(test_df)
    
    # カテゴリカル特徴量のラベルエンコーディング
    categorical_cols = ['Podcast_Name', 'Episode_Title', 'Genre']
    label_encoders = {}
    
    for col in categorical_cols:
        if col in train_df.columns:
            le = LabelEncoder()
            train_df[f'{col}_encoded'] = le.fit_transform(train_df[col].astype(str))
            test_df[f'{col}_encoded'] = le.transform(test_df[col].astype(str))
            label_encoders[col] = le
    
    # 特徴量選択（20個に厳選）
    base_features = [
        'Episode_Length_minutes', 'Host_Popularity_percentage', 'Guest_Popularity_percentage',
        'Number_of_Ads', 'Episode_Sentiment', 'Publication_Day', 'Publication_Time'
    ]
    
    extended_features = [
        'Episode_Length_squared', 'Episode_Length_log', 'Host_Guest_Popularity_Sum',
        'Host_Guest_Popularity_Ratio', 'Ad_Density', 'Ads_per_Hour', 
        'Has_Guest', 'Has_Ads', 'Is_Weekend', 'Is_Prime_Time'
    ]
    
    encoded_features = ['Podcast_Name_encoded', 'Episode_Title_encoded', 'Genre_encoded']
    
    # 基本特徴量セット
    feature_cols = base_features + extended_features + encoded_features
    
    X = train_df[feature_cols].fillna(0)
    y = train_df['Listening_Time_minutes']
    X_test = test_df[feature_cols].fillna(0)
    
    print(f"\n基本特徴量数: {len(feature_cols)}")
    
    # 安全なターゲットエンコーディング（全データで実行）
    target_encode_cols = ['Genre', 'Podcast_Name', 'Episode_Title']
    X_encoded, X_test_encoded = safe_target_encoding(X, y, X_test, target_encode_cols)
    
    # ターゲットエンコーディング特徴量を追加
    target_encoded_cols = [f'{col}_target_encoded' for col in target_encode_cols]
    final_feature_cols = feature_cols + target_encoded_cols
    
    # 最終的な特徴量セット作成
    X_final = X_encoded[final_feature_cols]
    X_test_final = X_test_encoded[final_feature_cols]
    
    print(f"最終特徴量数: {len(final_feature_cols)}")
    print("使用される特徴量:")
    for i, col in enumerate(final_feature_cols):
        if i % 4 == 0:
            print(f"  {col}")
        else:
            print(f", {col}", end="")
    print()
    
    return X_final, y, X_test_final, test_df['id'], final_feature_cols, label_encoders

def train_lightgbm_regularized(X_train, y_train, X_val, y_val):
    """
    正則化強化版LightGBM
    """
    print("=== LightGBM訓練（正則化強化） ===")
    
    # 正則化を強化したパラメータ
    lgb_params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.1,
        'feature_fraction': 0.8,  # 少し下げて汎化性向上
        'bagging_fraction': 0.8,  # 少し下げて汎化性向上
        'bagging_freq': 5,
        'reg_alpha': 2.0,         # 大幅強化
        'reg_lambda': 2.0,        # 大幅強化
        'min_child_samples': 20,  # 強化
        'verbose': -1,
        'random_state': 42
    }
    
    lgb_train = lgb.Dataset(X_train, label=y_train)
    lgb_val = lgb.Dataset(X_val, label=y_val, reference=lgb_train)
    
    model = lgb.train(
        lgb_params,
        lgb_train,
        num_boost_round=1000,
        valid_sets=[lgb_train, lgb_val],
        callbacks=[lgb.early_stopping(stopping_rounds=100), lgb.log_evaluation(100)]
    )
    
    return model

def train_xgboost_regularized(X_train, y_train, X_val, y_val):
    """
    正則化強化版XGBoost（exp003ベース）
    """
    print("=== XGBoost訓練（正則化強化） ===")
    
    # exp003の最適化パラメータ + 正則化強化
    xgb_params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'max_depth': 8,           # 少し浅くして過学習抑制
        'learning_rate': 0.05,    # 少し下げて安定化
        'subsample': 0.7,
        'colsample_bytree': 0.8,  # 少し下げて汎化性向上
        'min_child_weight': 8,    # 強化
        'reg_alpha': 3.0,         # 大幅強化
        'reg_lambda': 3.0,        # 大幅強化
        'random_state': 42
    }
    
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    
    model = xgb.train(
        xgb_params,
        dtrain,
        num_boost_round=1000,
        evals=[(dtrain, 'train'), (dval, 'val')],
        early_stopping_rounds=100,
        verbose_eval=100
    )
    
    return model

def run_experiment():
    """
    exp006メイン実行
    """
    print("=" * 60)
    print("実験006: Less is More - シンプルで堅牢なアプローチ")
    print("=" * 60)
    
    # データ準備
    X, y, X_test, test_ids, feature_cols, label_encoders = prepare_data()
    
    # データ分割
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"\nTrain: {X_train.shape}, Validation: {X_val.shape}")
    
    # 標準化（LightGBMとXGBoost用）
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), 
        columns=feature_cols, 
        index=X_train.index
    )
    X_val_scaled = pd.DataFrame(
        scaler.transform(X_val), 
        columns=feature_cols, 
        index=X_val.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), 
        columns=feature_cols
    )
    
    # モデル訓練（正則化強化版）
    lgb_model = train_lightgbm_regularized(X_train_scaled, y_train, X_val_scaled, y_val)
    xgb_model = train_xgboost_regularized(X_train_scaled, y_train, X_val_scaled, y_val)
    
    # 予測
    lgb_train_pred = lgb_model.predict(X_train_scaled, num_iteration=lgb_model.best_iteration)
    lgb_val_pred = lgb_model.predict(X_val_scaled, num_iteration=lgb_model.best_iteration)
    
    xgb_train_pred = xgb_model.predict(
        xgb.DMatrix(X_train_scaled), 
        iteration_range=(0, xgb_model.best_iteration)
    )
    xgb_val_pred = xgb_model.predict(
        xgb.DMatrix(X_val_scaled), 
        iteration_range=(0, xgb_model.best_iteration)
    )
    
    # アンサンブル（等重み）
    train_ensemble = (lgb_train_pred + xgb_train_pred) / 2
    val_ensemble = (lgb_val_pred + xgb_val_pred) / 2
    
    # 評価
    train_rmse = np.sqrt(mean_squared_error(y_train, train_ensemble))
    train_mae = mean_absolute_error(y_train, train_ensemble)
    train_r2 = r2_score(y_train, train_ensemble)
    
    val_rmse = np.sqrt(mean_squared_error(y_val, val_ensemble))
    val_mae = mean_absolute_error(y_val, val_ensemble)
    val_r2 = r2_score(y_val, val_ensemble)
    
    print(f"\n=== アンサンブル結果 ===")
    print(f"Train RMSE: {train_rmse:.6f}, MAE: {train_mae:.6f}, R²: {train_r2:.6f}")
    print(f"Val RMSE: {val_rmse:.6f}, MAE: {val_mae:.6f}, R²: {val_r2:.6f}")
    
    # 5-Fold Cross Validation
    print(f"\n=== 5-Fold Cross Validation ===")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        print(f"Fold {fold + 1}/5...")
        
        X_fold_train, X_fold_val = X.iloc[train_idx], X.iloc[val_idx]
        y_fold_train, y_fold_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # 標準化
        fold_scaler = StandardScaler()
        X_fold_train_scaled = pd.DataFrame(
            fold_scaler.fit_transform(X_fold_train),
            columns=feature_cols
        )
        X_fold_val_scaled = pd.DataFrame(
            fold_scaler.transform(X_fold_val),
            columns=feature_cols
        )
        
        # モデル訓練
        fold_lgb = train_lightgbm_regularized(X_fold_train_scaled, y_fold_train, X_fold_val_scaled, y_fold_val)
        fold_xgb = train_xgboost_regularized(X_fold_train_scaled, y_fold_train, X_fold_val_scaled, y_fold_val)
        
        # 予測とアンサンブル
        fold_lgb_pred = fold_lgb.predict(X_fold_val_scaled, num_iteration=fold_lgb.best_iteration)
        fold_xgb_pred = fold_xgb.predict(
            xgb.DMatrix(X_fold_val_scaled), 
            iteration_range=(0, fold_xgb.best_iteration)
        )
        
        fold_ensemble = (fold_lgb_pred + fold_xgb_pred) / 2
        fold_rmse = np.sqrt(mean_squared_error(y_fold_val, fold_ensemble))
        cv_scores.append(fold_rmse)
        
        print(f"  Fold {fold + 1} RMSE: {fold_rmse:.6f}")
    
    cv_mean = np.mean(cv_scores)
    cv_std = np.std(cv_scores)
    
    print(f"\nCV RMSE: {cv_mean:.6f} ± {cv_std:.6f}")
    print(f"CV Scores: {cv_scores}")
    
    # CV-Kaggle Gap予想
    estimated_public_score = cv_mean + 0.15  # 保守的な見積もり
    print(f"予想Public Score: {estimated_public_score:.6f}")
    
    # テストデータ予測
    lgb_test_pred = lgb_model.predict(X_test_scaled, num_iteration=lgb_model.best_iteration)
    xgb_test_pred = xgb_model.predict(
        xgb.DMatrix(X_test_scaled), 
        iteration_range=(0, xgb_model.best_iteration)
    )
    
    test_ensemble = (lgb_test_pred + xgb_test_pred) / 2
    
    # 結果保存
    results_dir = Path('results/exp006')
    results_dir.mkdir(exist_ok=True)
    
    # 提出ファイル
    submission = pd.DataFrame({
        'id': test_ids,
        'Listening_Time_minutes': test_ensemble
    })
    submission.to_csv(results_dir / 'submission.csv', index=False)
    
    # モデル保存
    with open(results_dir / 'model_lgb.pkl', 'wb') as f:
        pickle.dump(lgb_model, f)
    with open(results_dir / 'model_xgb.pkl', 'wb') as f:
        pickle.dump(xgb_model, f)
    with open(results_dir / 'scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    with open(results_dir / 'label_encoders.pkl', 'wb') as f:
        pickle.dump(label_encoders, f)
    
    # 実験結果保存
    results = {
        'experiment_id': 'exp006',
        'model_type': 'Less_is_More_LightGBM_XGBoost',
        'ensemble_method': 'equal_weights',
        'lgb_weight': 0.5,
        'xgb_weight': 0.5,
        'features': feature_cols,
        'num_features': len(feature_cols),
        'ensemble_train_rmse': train_rmse,
        'ensemble_train_mae': train_mae,
        'ensemble_train_r2': train_r2,
        'ensemble_val_rmse': val_rmse,
        'ensemble_val_mae': val_mae,
        'ensemble_val_r2': val_r2,
        'cv_scores': cv_scores,
        'cv_rmse_mean': cv_mean,
        'cv_rmse_std': cv_std,
        'estimated_public_score': estimated_public_score,
        'public_score': None,
        'private_score': None,
        'train_size': len(X_train),
        'test_size': len(X_test),
        'target_variable': 'Listening_Time_minutes',
        'cv_folds': 5,
        'validation_split': 0.2,
        'random_state': 42,
        'preprocessing': {
            'outlier_removal': 'conservative_1_percent',
            'missing_value_strategy': 'median_imputation',
            'categorical_encoding': 'label_encoding_and_safe_target_encoding',
            'feature_engineering': 'simplified_20_features',
            'normalization': 'StandardScaler',
            'regularization': 'enhanced'
        },
        'approach': 'Less_is_More',
        'overfitting_prevention': {
            'conservative_outlier_removal': True,
            'simplified_features': True,
            'safe_target_encoding': True,
            'enhanced_regularization': True,
            'robust_cv': True
        }
    }
    
    with open(results_dir / 'experiment_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n=== 実験完了 ===")
    print(f"Results saved to: {results_dir}")
    print(f"CV RMSE: {cv_mean:.6f} ± {cv_std:.6f}")
    print(f"予想Public Score: {estimated_public_score:.6f}")
    print(f"exp005からの方針転換: オーバーフィッティング抑制優先")
    print(f"目標CV-Kaggle Gap: < 0.3 (exp005は2.88)")
    
    return results

if __name__ == "__main__":
    results = run_experiment()