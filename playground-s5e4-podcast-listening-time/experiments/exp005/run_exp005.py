#!/usr/bin/env python3
"""
exp005: Advanced Feature Engineering + CatBoost
高度な特徴量エンジニアリング、CatBoost追加、外れ値除去によるさらなる性能向上
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
import pickle
import json
import warnings
from datetime import datetime
from pathlib import Path

warnings.filterwarnings('ignore')

def remove_outliers_iqr(df, columns, factor=1.5):
    """
    IQR手法による外れ値除去
    """
    df_clean = df.copy()
    
    for col in columns:
        if col in df_clean.columns and df_clean[col].dtype in ['int64', 'float64']:
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - factor * IQR
            upper_bound = Q3 + factor * IQR
            
            outliers_before = len(df_clean)
            df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
            outliers_after = len(df_clean)
            
            print(f"  {col}: {outliers_before - outliers_after}個の外れ値を除去")
    
    return df_clean

def create_advanced_features(df):
    """
    exp005: 高度な特徴量エンジニアリング（20+個の新規特徴量）
    """
    df_new = df.copy()
    
    print("=== 高度な特徴量エンジニアリング ===")
    
    # データ型の修正と前処理
    print("  データ型の修正...")
    
    # Episode_Sentimentを数値に変換
    sentiment_map = {'Positive': 1, 'Neutral': 0, 'Negative': -1}
    df_new['Episode_Sentiment'] = df_new['Episode_Sentiment'].map(sentiment_map).fillna(0)
    
    # Publication_Dayを数値に変換（曜日番号として）
    day_map = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 
               'Friday': 4, 'Saturday': 5, 'Sunday': 6}
    df_new['Publication_Day'] = df_new['Publication_Day'].map(day_map).fillna(0)
    
    # Publication_Timeを数値に変換（時間帯として）
    time_map = {'Morning': 8, 'Afternoon': 14, 'Evening': 19, 'Night': 23}
    df_new['Publication_Time'] = df_new['Publication_Time'].map(time_map).fillna(12)
    
    # 基本数値特徴量の欠損値補完
    numeric_cols = ['Episode_Length_minutes', 'Host_Popularity_percentage', 'Guest_Popularity_percentage', 'Number_of_Ads']
    for col in numeric_cols:
        if col in df_new.columns:
            df_new[col] = df_new[col].fillna(df_new[col].median())
    
    # ゲスト人気度の欠損値は0に設定（ゲストなしを意味）
    if 'Guest_Popularity_percentage' in df_new.columns:
        df_new['Guest_Popularity_percentage'] = df_new['Guest_Popularity_percentage'].fillna(0)
    
    # 既存の拡張特徴量（exp004まで）
    print("  既存拡張特徴量の作成...")
    df_new['Ad_Density'] = df_new['Number_of_Ads'] / (df_new['Episode_Length_minutes'] + 1)
    df_new['Host_Guest_Popularity_Diff'] = df_new['Host_Popularity_percentage'] - df_new['Guest_Popularity_percentage']
    df_new['Has_Guest'] = (df_new['Guest_Popularity_percentage'] > 0).astype(int)
    df_new['Episode_Length_squared'] = df_new['Episode_Length_minutes'] ** 2
    df_new['Episode_Length_log'] = np.log1p(df_new['Episode_Length_minutes'])
    df_new['Host_Guest_Popularity_Sum'] = df_new['Host_Popularity_percentage'] + df_new['Guest_Popularity_percentage']
    df_new['Host_Guest_Popularity_Ratio'] = df_new['Host_Popularity_percentage'] / (df_new['Guest_Popularity_percentage'] + 1)
    df_new['Ads_per_Hour'] = df_new['Number_of_Ads'] / (df_new['Episode_Length_minutes'] / 60 + 1)
    df_new['Has_Ads'] = (df_new['Number_of_Ads'] > 0).astype(int)
    
    # Episode_Length_Category
    df_new['Episode_Length_Category'] = pd.cut(df_new['Episode_Length_minutes'], 
                                              bins=[0, 30, 60, 120, float('inf')], 
                                              labels=['短時間', '中時間', '長時間', '超長時間'])
    
    # === exp005新規特徴量: 相互作用特徴量 ===
    print("  相互作用特徴量の作成...")
    df_new['Length_Host_Interaction'] = df_new['Episode_Length_minutes'] * df_new['Host_Popularity_percentage']
    df_new['Length_Guest_Interaction'] = df_new['Episode_Length_minutes'] * df_new['Guest_Popularity_percentage']
    df_new['Ads_Host_Interaction'] = df_new['Number_of_Ads'] * df_new['Host_Popularity_percentage']
    df_new['Ads_Guest_Interaction'] = df_new['Number_of_Ads'] * df_new['Guest_Popularity_percentage']
    df_new['Sentiment_Length_Interaction'] = df_new['Episode_Sentiment'] * df_new['Episode_Length_minutes']
    df_new['Sentiment_Host_Interaction'] = df_new['Episode_Sentiment'] * df_new['Host_Popularity_percentage']
    
    # === exp005新規特徴量: 時間ベース特徴量 ===
    print("  時間ベース特徴量の作成...")
    df_new['Is_Weekend'] = df_new['Publication_Day'].isin([5, 6]).astype(int)
    df_new['Is_Weekday'] = df_new['Publication_Day'].isin([0, 1, 2, 3, 4]).astype(int)
    
    # Publication_Timeから時間を抽出（24時間形式と仮定）
    df_new['Hour_of_Day'] = df_new['Publication_Time'] % 24
    df_new['Is_Prime_Time'] = df_new['Hour_of_Day'].isin([7, 8, 9, 17, 18, 19]).astype(int)
    df_new['Is_Morning'] = df_new['Hour_of_Day'].between(6, 12).astype(int)
    df_new['Is_Evening'] = df_new['Hour_of_Day'].between(18, 22).astype(int)
    df_new['Is_Late_Night'] = df_new['Hour_of_Day'].isin([23, 0, 1, 2, 3, 4, 5]).astype(int)
    
    # === exp005新規特徴量: 比率・密度特徴量 ===
    print("  比率・密度特徴量の作成...")
    df_new['Popularity_Density'] = (df_new['Host_Popularity_percentage'] + df_new['Guest_Popularity_percentage']) / (df_new['Episode_Length_minutes'] + 1)
    df_new['Ad_Effectiveness'] = df_new['Number_of_Ads'] * df_new['Host_Popularity_percentage'] / (df_new['Episode_Length_minutes'] + 1)
    df_new['Listen_Potential'] = (df_new['Host_Popularity_percentage'] + df_new['Guest_Popularity_percentage']) * df_new['Episode_Sentiment'] / 100
    df_new['Content_Quality_Score'] = df_new['Episode_Sentiment'] * (df_new['Host_Popularity_percentage'] + df_new['Guest_Popularity_percentage']) / 200
    
    # === exp005新規特徴量: グループ統計特徴量 ===
    print("  グループ統計特徴量の作成...")
    
    # Genre別統計
    genre_stats = df_new.groupby('Genre').agg({
        'Episode_Length_minutes': 'mean',
        'Host_Popularity_percentage': 'mean',
        'Guest_Popularity_percentage': 'mean',
        'Episode_Sentiment': 'mean'
    }).add_suffix('_Genre_Mean')
    df_new = df_new.merge(genre_stats, left_on='Genre', right_index=True, how='left')
    
    # Podcast別統計（上位頻出のみ）
    podcast_counts = df_new['Podcast_Name'].value_counts()
    frequent_podcasts = podcast_counts[podcast_counts >= 10].index
    podcast_stats = df_new[df_new['Podcast_Name'].isin(frequent_podcasts)].groupby('Podcast_Name').agg({
        'Episode_Length_minutes': 'mean',
        'Host_Popularity_percentage': 'mean'
    }).add_suffix('_Podcast_Mean')
    df_new = df_new.merge(podcast_stats, left_on='Podcast_Name', right_index=True, how='left')
    
    # Publication_Day別統計
    day_stats = df_new.groupby('Publication_Day').agg({
        'Episode_Length_minutes': 'mean',
        'Host_Popularity_percentage': 'mean'
    }).add_suffix('_Day_Mean')
    df_new = df_new.merge(day_stats, left_on='Publication_Day', right_index=True, how='left')
    
    # Hour別統計
    hour_stats = df_new.groupby('Hour_of_Day').agg({
        'Host_Popularity_percentage': 'mean',
        'Guest_Popularity_percentage': 'mean'
    }).add_suffix('_Hour_Mean')
    df_new = df_new.merge(hour_stats, left_on='Hour_of_Day', right_index=True, how='left')
    
    # === exp005新規特徴量: カテゴリ派生特徴量 ===
    print("  カテゴリ派生特徴量の作成...")
    df_new['Host_Popularity_Category'] = pd.cut(df_new['Host_Popularity_percentage'], 
                                               bins=5, labels=['Very_Low', 'Low', 'Medium', 'High', 'Very_High'])
    df_new['Guest_Popularity_Category'] = pd.cut(df_new['Guest_Popularity_percentage'], 
                                                bins=5, labels=['Very_Low', 'Low', 'Medium', 'High', 'Very_High'])
    df_new['Combined_Popularity_Category'] = pd.cut(df_new['Host_Guest_Popularity_Sum'], 
                                                   bins=5, labels=['Very_Low', 'Low', 'Medium', 'High', 'Very_High'])
    
    # === 統計情報の出力 ===
    print(f"  作成された特徴量数: {len(df_new.columns) - len(df.columns)}個")
    print(f"  総特徴量数: {len(df_new.columns)}個")
    
    return df_new

def enhanced_target_encoding(df_train, df_test, categorical_cols, target_col, alpha=10):
    """
    より高度なターゲットエンコーディング
    """
    print("=== 高度なターゲットエンコーディング ===")
    
    df_train_encoded = df_train.copy()
    df_test_encoded = df_test.copy()
    
    global_mean = df_train[target_col].mean()
    
    for col in categorical_cols:
        if col in df_train.columns:
            print(f"  {col}のターゲットエンコーディング...")
            
            # カテゴリごとの統計を計算
            category_stats = df_train.groupby(col)[target_col].agg(['mean', 'count'])
            
            # スムージング適用
            smoothed_mean = (category_stats['count'] * category_stats['mean'] + alpha * global_mean) / (category_stats['count'] + alpha)
            
            # エンコーディング適用
            encoding_map = smoothed_mean.to_dict()
            df_train_encoded[f'{col}_target_encoded'] = df_train[col].map(encoding_map).fillna(global_mean)
            df_test_encoded[f'{col}_target_encoded'] = df_test[col].map(encoding_map).fillna(global_mean)
            
            print(f"    ユニーク値数: {len(encoding_map)}")
    
    return df_train_encoded, df_test_encoded

def prepare_data():
    """
    データ読み込みと前処理
    """
    print("=== データ読み込み ===")
    
    train_df = pd.read_csv('data/train.csv')
    test_df = pd.read_csv('data/test.csv')
    
    print(f"Train shape: {train_df.shape}")
    print(f"Test shape: {test_df.shape}")
    
    # === 外れ値除去（訓練データのみ） ===
    print("\n=== 外れ値除去（IQR手法） ===")
    outlier_cols = ['Episode_Length_minutes', 'Host_Popularity_percentage', 
                   'Guest_Popularity_percentage', 'Number_of_Ads', 'Listening_Time_minutes']
    train_df = remove_outliers_iqr(train_df, outlier_cols)
    print(f"外れ値除去後のTrain shape: {train_df.shape}")
    
    # === 高度な特徴量エンジニアリング ===
    train_df = create_advanced_features(train_df)
    test_df = create_advanced_features(test_df)
    
    # === カテゴリカル変数のエンコーディング ===
    categorical_cols = ['Podcast_Name', 'Episode_Title', 'Genre', 'Episode_Length_Category',
                       'Host_Popularity_Category', 'Guest_Popularity_Category', 'Combined_Popularity_Category']
    
    label_encoders = {}
    for col in categorical_cols:
        if col in train_df.columns:
            le = LabelEncoder()
            train_df[f'{col}_encoded'] = le.fit_transform(train_df[col].astype(str))
            test_df[f'{col}_encoded'] = le.transform(test_df[col].astype(str))
            label_encoders[col] = le
    
    # === ターゲットエンコーディング ===
    target_encode_cols = ['Podcast_Name', 'Episode_Title', 'Genre']
    train_df, test_df = enhanced_target_encoding(train_df, test_df, target_encode_cols, 'Listening_Time_minutes')
    
    # === 特徴量選択 ===
    feature_cols = [col for col in train_df.columns if col not in ['id', 'Listening_Time_minutes'] 
                   and not col in categorical_cols]
    
    X = train_df[feature_cols].fillna(0)
    y = train_df['Listening_Time_minutes']
    X_test = test_df[feature_cols].fillna(0)
    
    print(f"\n最終特徴量数: {len(feature_cols)}")
    print(f"使用される特徴量:")
    for i, col in enumerate(feature_cols):
        if i % 5 == 0:
            print(f"  {col}")
        else:
            print(f", {col}", end="")
    print()
    
    return X, y, X_test, test_df['id'], feature_cols, label_encoders

def train_lightgbm_model(X_train, y_train, X_val, y_val):
    """
    LightGBMモデルの訓練（exp001最適化版）
    """
    print("=== LightGBMモデル訓練 ===")
    
    # exp001の最適パラメータ
    lgb_params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.1,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.9,
        'bagging_freq': 5,
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

def train_xgboost_model(X_train, y_train, X_val, y_val):
    """
    XGBoostモデルの訓練（exp003最適化版）
    """
    print("=== XGBoostモデル訓練 ===")
    
    # exp003のOptuna最適化パラメータ
    xgb_params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'max_depth': 10,
        'learning_rate': 0.035487574582325,
        'subsample': 0.7162926496452663,
        'colsample_bytree': 0.9761481011191779,
        'min_child_weight': 6,
        'reg_alpha': 1.1603476776684138,
        'reg_lambda': 1.145770443800777,
        'random_state': 42
    }
    
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    
    model = xgb.train(
        xgb_params,
        dtrain,
        num_boost_round=1419,
        evals=[(dtrain, 'train'), (dval, 'val')],
        early_stopping_rounds=100,
        verbose_eval=100
    )
    
    return model

def train_catboost_model(X_train, y_train, X_val, y_val, categorical_features=None):
    """
    CatBoostモデルの訓練（新規）
    """
    print("=== CatBoostモデル訓練 ===")
    
    model = CatBoostRegressor(
        iterations=1000,
        learning_rate=0.1,
        depth=6,
        loss_function='RMSE',
        eval_metric='RMSE',
        random_seed=42,
        cat_features=categorical_features,
        early_stopping_rounds=50,
        verbose=False  # ログを減らして高速化
    )
    
    model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        plot=False
    )
    
    return model

def run_experiment():
    """
    exp005メイン実行
    """
    print("=" * 60)
    print("実験005: Advanced Feature Engineering + CatBoost")
    print("=" * 60)
    
    # データ準備
    X, y, X_test, test_ids, feature_cols, label_encoders = prepare_data()
    
    # データ分割
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"\nTrain: {X_train.shape}, Validation: {X_val.shape}")
    
    # 標準化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=feature_cols, index=X_train.index)
    X_val_scaled = pd.DataFrame(X_val_scaled, columns=feature_cols, index=X_val.index)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=feature_cols)
    
    # モデル訓練
    lgb_model = train_lightgbm_model(X_train_scaled, y_train, X_val_scaled, y_val)
    xgb_model = train_xgboost_model(X_train_scaled, y_train, X_val_scaled, y_val)
    # CatBoostは標準化なしで訓練（カテゴリカル変数は自動処理）
    cat_model = train_catboost_model(X_train, y_train, X_val, y_val, None)
    
    # 予測
    lgb_train_pred = lgb_model.predict(X_train_scaled, num_iteration=lgb_model.best_iteration)
    lgb_val_pred = lgb_model.predict(X_val_scaled, num_iteration=lgb_model.best_iteration)
    
    xgb_train_pred = xgb_model.predict(xgb.DMatrix(X_train_scaled), iteration_range=(0, xgb_model.best_iteration))
    xgb_val_pred = xgb_model.predict(xgb.DMatrix(X_val_scaled), iteration_range=(0, xgb_model.best_iteration))
    
    cat_train_pred = cat_model.predict(X_train)
    cat_val_pred = cat_model.predict(X_val)
    
    # アンサンブル（等重み）
    train_ensemble = (lgb_train_pred + xgb_train_pred + cat_train_pred) / 3
    val_ensemble = (lgb_val_pred + xgb_val_pred + cat_val_pred) / 3
    
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
    
    # Cross Validation
    print(f"\n=== 3-Fold Cross Validation ===")
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    cv_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        print(f"Fold {fold + 1}/3...")
        
        X_fold_train, X_fold_val = X.iloc[train_idx], X.iloc[val_idx]
        y_fold_train, y_fold_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # 標準化
        fold_scaler = StandardScaler()
        X_fold_train_scaled = fold_scaler.fit_transform(X_fold_train)
        X_fold_val_scaled = fold_scaler.transform(X_fold_val)
        
        X_fold_train_scaled = pd.DataFrame(X_fold_train_scaled, columns=feature_cols)
        X_fold_val_scaled = pd.DataFrame(X_fold_val_scaled, columns=feature_cols)
        
        # モデル訓練
        fold_lgb = train_lightgbm_model(X_fold_train_scaled, y_fold_train, X_fold_val_scaled, y_fold_val)
        fold_xgb = train_xgboost_model(X_fold_train_scaled, y_fold_train, X_fold_val_scaled, y_fold_val)
        fold_cat = train_catboost_model(X_fold_train, y_fold_train, X_fold_val, y_fold_val, None)
        
        # 予測とアンサンブル
        fold_lgb_pred = fold_lgb.predict(X_fold_val_scaled, num_iteration=fold_lgb.best_iteration)
        fold_xgb_pred = fold_xgb.predict(xgb.DMatrix(X_fold_val_scaled), iteration_range=(0, fold_xgb.best_iteration))
        fold_cat_pred = fold_cat.predict(X_fold_val)
        
        fold_ensemble = (fold_lgb_pred + fold_xgb_pred + fold_cat_pred) / 3
        fold_rmse = np.sqrt(mean_squared_error(y_fold_val, fold_ensemble))
        cv_scores.append(fold_rmse)
        
        print(f"  Fold {fold + 1} RMSE: {fold_rmse:.6f}")
    
    cv_mean = np.mean(cv_scores)
    cv_std = np.std(cv_scores)
    
    print(f"\nCV RMSE: {cv_mean:.6f} ± {cv_std:.6f}")
    print(f"CV Scores: {cv_scores}")
    
    # テストデータ予測
    lgb_test_pred = lgb_model.predict(X_test_scaled, num_iteration=lgb_model.best_iteration)
    xgb_test_pred = xgb_model.predict(xgb.DMatrix(X_test_scaled), iteration_range=(0, xgb_model.best_iteration))
    cat_test_pred = cat_model.predict(X_test)
    
    test_ensemble = (lgb_test_pred + xgb_test_pred + cat_test_pred) / 3
    
    # 結果保存
    results_dir = Path('results/exp005')
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
    with open(results_dir / 'model_cat.pkl', 'wb') as f:
        pickle.dump(cat_model, f)
    with open(results_dir / 'scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    with open(results_dir / 'label_encoders.pkl', 'wb') as f:
        pickle.dump(label_encoders, f)
    
    # 実験結果保存
    results = {
        'experiment_id': 'exp005',
        'model_type': 'Advanced_Feature_Engineering_CatBoost_Ensemble',
        'ensemble_method': 'equal_weights',
        'lgb_weight': 1/3,
        'xgb_weight': 1/3,
        'cat_weight': 1/3,
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
        'public_score': None,
        'private_score': None,
        'train_size': len(X_train),
        'test_size': len(X_test),
        'target_variable': 'Listening_Time_minutes',
        'cv_folds': 3,
        'validation_split': 0.2,
        'random_state': 42,
        'preprocessing': {
            'outlier_removal': 'IQR_method',
            'missing_value_strategy': 'median_imputation',
            'categorical_encoding': 'label_encoding_and_target_encoding',
            'feature_engineering': 'advanced_20plus_features',
            'normalization': 'StandardScaler'
        },
        'improvement_over_exp003': 12.7938 - cv_mean if cv_mean < 12.7938 else None
    }
    
    with open(results_dir / 'experiment_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n=== 実験完了 ===")
    print(f"Results saved to: {results_dir}")
    print(f"CV RMSE: {cv_mean:.6f} (目標: 12.5以下)")
    print(f"exp003からの改善: {12.7938 - cv_mean:.6f}" if cv_mean < 12.7938 else f"exp003より劣化: {cv_mean - 12.7938:.6f}")
    print(f"目標達成: {'✓' if cv_mean <= 12.5 else '✗'}")
    
    return results

if __name__ == "__main__":
    results = run_experiment()