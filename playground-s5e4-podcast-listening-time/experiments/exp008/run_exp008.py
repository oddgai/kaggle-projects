#!/usr/bin/env python3
"""
exp008: Ultra-Optimized LightGBM with Optuna & Advanced Feature Engineering
Target: Public Score 11.70 (masaishi's approach inspired)

Key Features:
- Optuna hyperparameter optimization (30-85% RMSE improvement)
- LightGBM single model with DART boosting
- Advanced feature engineering (Ugly Ducklings technique)
- LightGBM-optimized preprocessing
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import optuna
import warnings
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

def load_data():
    """データ読み込み"""
    train = pd.read_csv('data/train.csv')
    test = pd.read_csv('data/test.csv')
    return train, test

def advanced_feature_engineering(df, is_train=True):
    """
    高度な特徴量エンジニアリング（2024-2025 Best Practices）
    """
    df = df.copy()
    
    # 1. 基本的な数値変換（LightGBM最適化）
    sentiment_map = {'Positive': 2, 'Neutral': 1, 'Negative': 0}
    df['Episode_Sentiment_num'] = df['Episode_Sentiment'].map(sentiment_map)
    
    # 2. 時間ベース特徴量
    # Publication_Day数値変換
    day_map = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 
               'Friday': 4, 'Saturday': 5, 'Sunday': 6}
    df['Publication_Day_num'] = df['Publication_Day'].map(day_map)
    df['Is_Weekend'] = df['Publication_Day_num'].isin([5, 6]).astype(int)
    
    # Publication_Time順序エンコーディング
    time_order = {'Morning': 0, 'Afternoon': 1, 'Evening': 2, 'Night': 3}
    df['Publication_Time_ord'] = df['Publication_Time'].map(time_order)
    
    # 3. 核心特徴量（masaishiアプローチ推定）
    df['Episode_Length_log'] = np.log1p(df['Episode_Length_minutes'])
    df['Host_Pop_log'] = np.log1p(df['Host_Popularity_percentage'])
    df['Guest_Pop_log'] = np.log1p(df['Guest_Popularity_percentage'])
    df['Ads_log'] = np.log1p(df['Number_of_Ads'])
    
    # 4. 密度・比率特徴量
    df['Ads_per_Minute'] = df['Number_of_Ads'] / (df['Episode_Length_minutes'] + 1)
    df['Host_Pop_per_Minute'] = df['Host_Popularity_percentage'] / (df['Episode_Length_minutes'] + 1)
    df['Guest_Pop_per_Minute'] = df['Guest_Popularity_percentage'] / (df['Episode_Length_minutes'] + 1)
    df['Pop_Balance'] = df['Host_Popularity_percentage'] / (df['Guest_Popularity_percentage'] + 1)
    df['Total_Pop'] = df['Host_Popularity_percentage'] + df['Guest_Popularity_percentage']
    df['Pop_Density'] = df['Total_Pop'] / (df['Episode_Length_minutes'] + 1)
    
    # 5. 相互作用特徴量（LightGBM重要）
    df['Length_Host_Int'] = df['Episode_Length_minutes'] * df['Host_Popularity_percentage']
    df['Length_Guest_Int'] = df['Episode_Length_minutes'] * df['Guest_Popularity_percentage']
    df['Pop_Ads_Int'] = df['Total_Pop'] * df['Number_of_Ads']
    df['Sentiment_Pop_Int'] = df['Episode_Sentiment_num'] * df['Total_Pop']
    df['Weekend_Pop_Int'] = df['Is_Weekend'] * df['Total_Pop']
    
    # 6. テキスト長特徴量
    df['Podcast_Name_len'] = df['Podcast_Name'].str.len()
    df['Episode_Title_len'] = df['Episode_Title'].str.len()
    df['Title_Name_ratio'] = df['Episode_Title_len'] / (df['Podcast_Name_len'] + 1)
    
    # 7. グループ統計量（Ugly Ducklings technique）
    if is_train:
        # ジャンルベース統計
        genre_stats = df.groupby('Genre').agg({
            'Episode_Length_minutes': ['mean', 'std', 'median'],
            'Host_Popularity_percentage': ['mean', 'std'],
            'Guest_Popularity_percentage': ['mean', 'std'],
            'Number_of_Ads': ['mean', 'std'],
            'Total_Pop': ['mean', 'std']
        }).round(6)
        genre_stats.columns = [f'Genre_{col[0]}_{col[1]}' for col in genre_stats.columns]
        
        # ポッドキャスト名統計
        podcast_stats = df.groupby('Podcast_Name').agg({
            'Episode_Length_minutes': ['mean', 'std'],
            'Host_Popularity_percentage': 'mean',
            'Number_of_Ads': 'mean'
        }).round(6)
        podcast_stats.columns = [f'Podcast_{col[0]}_{col[1]}' if isinstance(col, tuple) else f'Podcast_{col}' for col in podcast_stats.columns]
        
        # 保存
        genre_stats.to_csv('experiments/exp008/genre_stats.csv')
        podcast_stats.to_csv('experiments/exp008/podcast_stats.csv')
        
        # ターゲットエンコーディング
        if 'Listening_Time_minutes' in df.columns:
            genre_target = df.groupby('Genre')['Listening_Time_minutes'].agg(['mean', 'std']).round(6)
            genre_target.columns = ['Genre_target_mean', 'Genre_target_std']
            
            podcast_target = df.groupby('Podcast_Name')['Listening_Time_minutes'].agg(['mean', 'std']).round(6)
            podcast_target.columns = ['Podcast_target_mean', 'Podcast_target_std']
            
            genre_target.to_csv('experiments/exp008/genre_target.csv')
            podcast_target.to_csv('experiments/exp008/podcast_target.csv')
    else:
        # テストデータ用読み込み
        genre_stats = pd.read_csv('experiments/exp008/genre_stats.csv', index_col=0)
        podcast_stats = pd.read_csv('experiments/exp008/podcast_stats.csv', index_col=0)
        genre_target = pd.read_csv('experiments/exp008/genre_target.csv', index_col=0)
        podcast_target = pd.read_csv('experiments/exp008/podcast_target.csv', index_col=0)
    
    # マージ
    df = df.merge(genre_stats, left_on='Genre', right_index=True, how='left')
    df = df.merge(podcast_stats, left_on='Podcast_Name', right_index=True, how='left')
    df = df.merge(genre_target, left_on='Genre', right_index=True, how='left')
    df = df.merge(podcast_target, left_on='Podcast_Name', right_index=True, how='left')
    
    # 8. Ugly Ducklings特徴量
    df['Length_vs_Genre'] = abs(df['Episode_Length_minutes'] - df['Genre_Episode_Length_minutes_mean']) / (df['Genre_Episode_Length_minutes_std'] + 1e-6)
    df['Host_vs_Genre'] = abs(df['Host_Popularity_percentage'] - df['Genre_Host_Popularity_percentage_mean']) / (df['Genre_Host_Popularity_percentage_std'] + 1e-6)
    df['Guest_vs_Genre'] = abs(df['Guest_Popularity_percentage'] - df['Genre_Guest_Popularity_percentage_mean']) / (df['Genre_Guest_Popularity_percentage_std'] + 1e-6)
    
    # 9. 相対特徴量
    df['Length_Genre_ratio'] = df['Episode_Length_minutes'] / (df['Genre_Episode_Length_minutes_mean'] + 1)
    df['Pop_Genre_ratio'] = df['Total_Pop'] / (df['Genre_Total_Pop_mean'] + 1)
    
    return df

def prepare_data_for_lightgbm(df, categorical_cols, target_col=None):
    """LightGBM最適化データ準備"""
    df = df.copy()
    
    # すべての数値変換（LightGBM要求）
    for col in df.columns:
        if col not in ['id', target_col] and col in df.columns:
            if df[col].dtype == 'object':
                # 文字列型を数値型に変換
                if col in categorical_cols:
                    le = LabelEncoder()
                    df[col] = le.fit_transform(df[col].astype(str))
                else:
                    # その他のobject型も数値化
                    try:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    except:
                        le = LabelEncoder()
                        df[col] = le.fit_transform(df[col].astype(str))
    
    # 欠損値をLightGBM推奨値で埋める
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(-999)
    
    # カテゴリカル特徴量をLabel Encoding
    label_encoders = {}
    for col in categorical_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le
    
    if target_col and target_col in df.columns:
        X = df.drop(columns=[target_col, 'id'] + [c for c in ['Podcast_Name', 'Episode_Title'] if c in df.columns])
        y = df[target_col]
        return X, y, label_encoders
    else:
        X = df.drop(columns=['id'] + [c for c in ['Podcast_Name', 'Episode_Title'] if c in df.columns])
        return X, label_encoders

def optuna_objective(trial, X_train, y_train, categorical_features):
    """Optuna最適化目的関数"""
    
    # ハイパーパラメータ空間（2024-2025 Best Practices）
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': trial.suggest_categorical('boosting_type', ['gbdt', 'dart']),
        'num_leaves': trial.suggest_int('num_leaves', 20, 300),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 10, 100),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 10.0),
        'verbose': -1,
        'random_state': 42
    }
    
    # DART特有パラメータ
    if params['boosting_type'] == 'dart':
        params['drop_rate'] = trial.suggest_float('drop_rate', 0.05, 0.2)
        params['skip_drop'] = trial.suggest_float('skip_drop', 0.3, 0.7)
    
    # 3-fold CV for optimization (speed)
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    cv_scores = []
    
    for train_idx, val_idx in kf.split(X_train):
        X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        model = lgb.LGBMRegressor(n_estimators=1000, **params)
        model.fit(
            X_fold_train, y_fold_train,
            eval_set=[(X_fold_val, y_fold_val)],
            callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)],
            categorical_feature=categorical_features
        )
        
        pred = model.predict(X_fold_val)
        rmse = np.sqrt(mean_squared_error(y_fold_val, pred))
        cv_scores.append(rmse)
    
    return np.mean(cv_scores)

def main():
    print("=== exp008: Ultra-Optimized LightGBM ===")
    print("筋肉のようなOptuna最適化で11.70スコア狙うで〜💪")
    
    # データ読み込み
    train, test = load_data()
    print(f"Train: {train.shape}, Test: {test.shape}")
    
    # 高度特徴量エンジニアリング
    print("\n高度特徴量エンジニアリング中...")
    train_fe = advanced_feature_engineering(train, is_train=True)
    test_fe = advanced_feature_engineering(test, is_train=False)
    
    # データ準備
    categorical_cols = ['Genre', 'Publication_Time', 'Episode_Sentiment']
    X_train, y_train, encoders = prepare_data_for_lightgbm(
        train_fe, categorical_cols, 'Listening_Time_minutes'
    )
    X_test, _ = prepare_data_for_lightgbm(test_fe, categorical_cols)
    
    print(f"特徴量数: {len(X_train.columns)}")
    print(f"カテゴリカル特徴量: {categorical_cols}")
    
    # カテゴリカル特徴量インデックス
    categorical_feature_indices = [i for i, col in enumerate(X_train.columns) 
                                  if col in categorical_cols]
    
    print(f"\nOptuna最適化開始... (50 trials)")
    
    # Optuna最適化
    study = optuna.create_study(
        direction='minimize',
        sampler=optuna.samplers.TPESampler(seed=42)
    )
    
    study.optimize(
        lambda trial: optuna_objective(trial, X_train, y_train, categorical_feature_indices),
        n_trials=50
    )
    
    best_params = study.best_params
    print(f"Best RMSE: {study.best_value:.6f}")
    print(f"Best params: {best_params}")
    
    # 最終モデル学習（5-fold CV）
    print(f"\n最終モデル学習 (5-fold CV)")
    n_splits = 5
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    cv_scores = []
    models = []
    test_predictions = np.zeros(len(X_test))
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train), 1):
        print(f"\nFold {fold}/{n_splits}")
        
        X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        # 最適パラメータでモデル構築
        final_params = {
            'objective': 'regression',
            'metric': 'rmse',
            'verbose': -1,
            'random_state': 42,
            'n_estimators': 2000,
            **best_params
        }
        
        model = lgb.LGBMRegressor(**final_params)
        model.fit(
            X_fold_train, y_fold_train,
            eval_set=[(X_fold_val, y_fold_val)],
            callbacks=[lgb.early_stopping(150), lgb.log_evaluation(100)],
            categorical_feature=categorical_feature_indices
        )
        
        # Validation予測
        val_pred = model.predict(X_fold_val)
        rmse = np.sqrt(mean_squared_error(y_fold_val, val_pred))
        cv_scores.append(rmse)
        print(f"Fold {fold} RMSE: {rmse:.6f}")
        
        # テスト予測
        test_pred = model.predict(X_test)
        test_predictions += test_pred / n_splits
        
        models.append(model)
    
    # CV結果
    mean_cv = np.mean(cv_scores)
    std_cv = np.std(cv_scores)
    print(f"\n=== CV結果 ===")
    print(f"平均RMSE: {mean_cv:.6f} ± {std_cv:.6f}")
    print(f"個別スコア: {[f'{s:.6f}' for s in cv_scores]}")
    
    # 予想Public Score（控えめ）
    predicted_public = mean_cv + 0.05
    print(f"予想Public Score: {predicted_public:.4f}")
    
    # 提出ファイル作成
    submission = pd.DataFrame({
        'id': test['id'],
        'Listening_Time_minutes': test_predictions
    })
    submission.to_csv('results/exp008/submission.csv', index=False)
    print(f"\n✓ 提出ファイル保存: results/exp008/submission.csv")
    
    # 特徴量重要度保存
    feature_importance = models[0].feature_importances_
    importance_df = pd.DataFrame({
        'feature': X_train.columns,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)
    importance_df.to_csv('results/exp008/feature_importance.csv', index=False)
    
    print(f"\n🎯 TOP 10 重要特徴量:")
    for i, (feature, imp) in enumerate(zip(importance_df['feature'][:10], 
                                         importance_df['importance'][:10]), 1):
        print(f"{i:2d}. {feature}: {imp:.0f}")
    
    print(f"\n筋肉のような最適化完了！11.70スコア期待や〜💪")
    
    return mean_cv, std_cv, predicted_public, best_params

if __name__ == "__main__":
    cv_mean, cv_std, pred_public, best_params = main()