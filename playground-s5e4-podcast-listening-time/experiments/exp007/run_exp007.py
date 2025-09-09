#!/usr/bin/env python3
"""
exp007: Competition Insights Implementation
- 高度な特徴量エンジニアリング（Discussion insights）
- XGBoost + CatBoost + LightGBM アンサンブル
- グループベース統計量と相互作用特徴量
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
import warnings
warnings.filterwarnings('ignore')

def load_data():
    """データ読み込み"""
    train = pd.read_csv('data/train.csv')
    test = pd.read_csv('data/test.csv')
    return train, test

def advanced_feature_engineering(df, is_train=True):
    """
    Competition insightsに基づく高度な特徴量エンジニアリング
    """
    df = df.copy()
    
    # 1. 基本的な数値変換
    sentiment_map = {'Positive': 1, 'Neutral': 0, 'Negative': -1}
    df['Episode_Sentiment_num'] = df['Episode_Sentiment'].map(sentiment_map)
    
    # 2. 時間ベース特徴量
    df['Is_Weekend'] = df['Publication_Day'].isin([5, 6]).astype(int)
    
    # Publication_Time順序エンコーディング
    time_order = {'Morning': 0, 'Afternoon': 1, 'Evening': 2, 'Night': 3}
    df['Publication_Time_ord'] = df['Publication_Time'].map(time_order)
    
    # 3. 広告密度特徴量
    df['Ads_per_Minute'] = df['Number_of_Ads'] / (df['Episode_Length_minutes'] + 1)
    
    # 4. 人気度特徴量
    df['Host_Popularity_per_Minute'] = df['Host_Popularity_percentage'] / (df['Episode_Length_minutes'] + 1)
    df['Guest_Popularity_per_Minute'] = df['Guest_Popularity_percentage'] / (df['Episode_Length_minutes'] + 1)
    df['Host_Guest_Popularity_Ratio'] = df['Host_Popularity_percentage'] / (df['Guest_Popularity_percentage'] + 1)
    df['Total_Popularity'] = df['Host_Popularity_percentage'] + df['Guest_Popularity_percentage']
    df['Popularity_Density'] = df['Total_Popularity'] / (df['Episode_Length_minutes'] + 1)
    
    # 5. 相互作用特徴量
    df['Length_Host_Interaction'] = df['Episode_Length_minutes'] * df['Host_Popularity_percentage']
    df['Length_Guest_Interaction'] = df['Episode_Length_minutes'] * df['Guest_Popularity_percentage']
    df['Length_Ads_Interaction'] = df['Episode_Length_minutes'] * df['Number_of_Ads']
    df['Ads_per_Host_Popularity'] = df['Number_of_Ads'] / (df['Host_Popularity_percentage'] + 1)
    df['Ads_per_Guest_Popularity'] = df['Number_of_Ads'] / (df['Guest_Popularity_percentage'] + 1)
    
    # 6. テキスト特徴量
    df['Podcast_Name_Length'] = df['Podcast_Name'].str.len()
    df['Episode_Title_Length'] = df['Episode_Title'].str.len()
    df['Title_Name_Ratio'] = df['Episode_Title_Length'] / (df['Podcast_Name_Length'] + 1)
    
    # 7. グループベース統計量（訓練データから計算）
    if is_train:
        # ジャンル平均
        genre_stats = df.groupby('Genre').agg({
            'Episode_Length_minutes': ['mean', 'std'],
            'Host_Popularity_percentage': 'mean',
            'Guest_Popularity_percentage': 'mean',
            'Number_of_Ads': 'mean'
        }).round(2)
        genre_stats.columns = ['_'.join(col) for col in genre_stats.columns]
        
        # ポッドキャスト名統計
        podcast_stats = df.groupby('Podcast_Name').agg({
            'Episode_Length_minutes': ['mean', 'std'],
            'Host_Popularity_percentage': 'mean',
            'Number_of_Ads': 'mean'
        }).round(2)
        podcast_stats.columns = ['Podcast_' + '_'.join(col) for col in podcast_stats.columns]
        
        # 保存（テスト用）
        genre_stats.to_csv('experiments/exp007/genre_stats.csv')
        podcast_stats.to_csv('experiments/exp007/podcast_stats.csv')
    else:
        # テストデータの場合は読み込み
        genre_stats = pd.read_csv('experiments/exp007/genre_stats.csv', index_col=0)
        podcast_stats = pd.read_csv('experiments/exp007/podcast_stats.csv', index_col=0)
    
    # マージ
    df = df.merge(genre_stats, left_on='Genre', right_index=True, how='left')
    df = df.merge(podcast_stats, left_on='Podcast_Name', right_index=True, how='left')
    
    # 8. 相対特徴量（ジャンル平均との差）
    df['Length_vs_Genre_Mean'] = df['Episode_Length_minutes'] - df['Episode_Length_minutes_mean']
    df['Host_Pop_vs_Genre_Mean'] = df['Host_Popularity_percentage'] - df['Host_Popularity_percentage_mean']
    
    # 9. カテゴリカル特徴量のターゲットエンコーディング（簡易版）
    if is_train:
        target_mean = df['Listening_Time_minutes'].mean()
        
        genre_target = df.groupby('Genre')['Listening_Time_minutes'].mean()
        podcast_target = df.groupby('Podcast_Name')['Listening_Time_minutes'].mean()
        
        genre_target.to_csv('experiments/exp007/genre_target.csv')
        podcast_target.to_csv('experiments/exp007/podcast_target.csv')
        
        df['Genre_Target_Encoded'] = df['Genre'].map(genre_target).fillna(target_mean)
        df['Podcast_Target_Encoded'] = df['Podcast_Name'].map(podcast_target).fillna(target_mean)
    else:
        genre_target = pd.read_csv('experiments/exp007/genre_target.csv', index_col=0)['Listening_Time_minutes']
        podcast_target = pd.read_csv('experiments/exp007/podcast_target.csv', index_col=0)['Listening_Time_minutes']
        
        target_mean = genre_target.mean()
        df['Genre_Target_Encoded'] = df['Genre'].map(genre_target).fillna(target_mean)
        df['Podcast_Target_Encoded'] = df['Podcast_Name'].map(podcast_target).fillna(target_mean)
    
    # 10. 欠損値処理
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    
    return df

def remove_outliers_iqr(df, target_col='Listening_Time_minutes', threshold=1.5):
    """IQR法によるアウトライア除去"""
    if target_col not in df.columns:
        return df
    
    Q1 = df[target_col].quantile(0.25)
    Q3 = df[target_col].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR
    
    initial_size = len(df)
    df_clean = df[(df[target_col] >= lower_bound) & (df[target_col] <= upper_bound)].copy()
    removed = initial_size - len(df_clean)
    
    print(f"アウトライア除去: {removed}件 ({removed/initial_size*100:.1f}%)")
    return df_clean

def get_feature_columns(df):
    """モデル学習用の特徴量カラム取得"""
    exclude_cols = ['id', 'Listening_Time_minutes', 'Podcast_Name', 'Episode_Title', 
                   'Genre', 'Episode_Sentiment', 'Publication_Time', 'Publication_Day']
    
    # 数値型カラムのみを選択
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    feature_cols = [col for col in numeric_cols if col not in exclude_cols]
    
    print(f"除外カラム: {exclude_cols}")
    print(f"数値型カラム数: {len(numeric_cols)}")
    
    return feature_cols

def train_models(X_train, y_train, X_valid, y_valid):
    """3つのモデルを学習"""
    models = {}
    predictions = {}
    
    # 1. XGBoost（最高性能）
    xgb_params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'max_depth': 8,
        'learning_rate': 0.05,
        'n_estimators': 1000,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'n_jobs': -1
    }
    
    xgb_model = xgb.XGBRegressor(**xgb_params)
    xgb_model.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        verbose=False
    )
    models['xgboost'] = xgb_model
    predictions['xgboost'] = xgb_model.predict(X_valid)
    
    # 2. LightGBM
    lgb_params = {
        'objective': 'regression',
        'metric': 'rmse',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'n_estimators': 1000,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1
    }
    
    lgb_model = lgb.LGBMRegressor(**lgb_params)
    lgb_model.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
    )
    models['lightgbm'] = lgb_model
    predictions['lightgbm'] = lgb_model.predict(X_valid)
    
    # 3. CatBoost
    cat_params = {
        'iterations': 1000,
        'learning_rate': 0.05,
        'depth': 8,
        'loss_function': 'RMSE',
        'random_seed': 42,
        'verbose': False,
        'early_stopping_rounds': 50
    }
    
    cat_model = CatBoostRegressor(**cat_params)
    cat_model.fit(
        X_train, y_train,
        eval_set=(X_valid, y_valid),
        use_best_model=True
    )
    models['catboost'] = cat_model
    predictions['catboost'] = cat_model.predict(X_valid)
    
    return models, predictions

def find_best_weights(predictions, y_valid):
    """最適なアンサンブル重みを探索（グリッドサーチ）"""
    best_rmse = float('inf')
    best_weights = None
    
    # シンプルなグリッドサーチ
    model_names = list(predictions.keys())
    
    # 3モデルの重み組み合わせを試す
    for w1 in np.arange(0.1, 1.0, 0.1):
        for w2 in np.arange(0.1, 1.0-w1, 0.1):
            w3 = 1.0 - w1 - w2
            if w3 < 0.1:
                continue
                
            weights = [w1, w2, w3]
            pred = np.zeros_like(y_valid)
            for i, model_name in enumerate(model_names):
                pred += weights[i] * predictions[model_name]
            
            rmse = np.sqrt(mean_squared_error(y_valid, pred))
            if rmse < best_rmse:
                best_rmse = rmse
                best_weights = weights
    
    # デフォルトは等重み
    if best_weights is None:
        best_weights = [1/len(predictions)] * len(predictions)
    
    return np.array(best_weights)

def main():
    print("=== exp007: Competition Insights実装 ===")
    print("筋肉のように強力な特徴量エンジニアリングするで〜💪")
    
    # データ読み込み
    train, test = load_data()
    print(f"訓練データ: {train.shape}")
    print(f"テストデータ: {test.shape}")
    
    # 高度な特徴量エンジニアリング
    print("\n高度な特徴量エンジニアリング中...")
    train_fe = advanced_feature_engineering(train, is_train=True)
    test_fe = advanced_feature_engineering(test, is_train=False)
    
    # アウトライア除去（IQR法）
    if 'Listening_Time_minutes' in train_fe.columns:
        train_clean = remove_outliers_iqr(train_fe)
    else:
        train_clean = train_fe
    
    # 特徴量選択
    feature_cols = get_feature_columns(train_clean)
    print(f"\n使用特徴量数: {len(feature_cols)}")
    
    X = train_clean[feature_cols].values
    y = train_clean['Listening_Time_minutes'].values
    X_test = test_fe[feature_cols].values
    
    # スケーリング
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_test = scaler.transform(X_test)
    
    # 5-fold CV
    n_splits = 5
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    cv_scores = []
    all_models = []
    
    print(f"\n{n_splits}-fold Cross Validation開始...")
    for fold, (train_idx, valid_idx) in enumerate(kf.split(X), 1):
        print(f"\nFold {fold}/{n_splits}")
        
        X_train, X_valid = X[train_idx], X[valid_idx]
        y_train, y_valid = y[train_idx], y[valid_idx]
        
        # モデル学習
        models, predictions = train_models(X_train, y_train, X_valid, y_valid)
        
        # 最適重み探索
        best_weights = find_best_weights(predictions, y_valid)
        
        # アンサンブル予測
        ensemble_pred = np.zeros_like(y_valid)
        for i, model_name in enumerate(models.keys()):
            ensemble_pred += best_weights[i] * predictions[model_name]
            rmse = np.sqrt(np.mean((predictions[model_name] - y_valid) ** 2))
            print(f"  {model_name}: RMSE = {rmse:.6f} (weight: {best_weights[i]:.3f})")
        
        fold_rmse = np.sqrt(np.mean((ensemble_pred - y_valid) ** 2))
        cv_scores.append(fold_rmse)
        print(f"  Ensemble: RMSE = {fold_rmse:.6f}")
        
        all_models.append((models, best_weights))
    
    # CV結果
    mean_cv = np.mean(cv_scores)
    std_cv = np.std(cv_scores)
    print(f"\n=== CV結果 ===")
    print(f"平均RMSE: {mean_cv:.6f} ± {std_cv:.6f}")
    print(f"Fold scores: {[f'{s:.6f}' for s in cv_scores]}")
    
    # テストデータ予測（全フォルドのアンサンブル）
    print("\nテストデータ予測中...")
    test_preds = np.zeros((len(X_test), n_splits))
    
    for fold, (models, weights) in enumerate(all_models):
        fold_pred = np.zeros(len(X_test))
        for i, model_name in enumerate(models.keys()):
            fold_pred += weights[i] * models[model_name].predict(X_test)
        test_preds[:, fold] = fold_pred
    
    # 最終予測（全フォルドの平均）
    final_predictions = test_preds.mean(axis=1)
    
    # 提出ファイル作成
    submission = pd.DataFrame({
        'id': test['id'],
        'Listening_Time_minutes': final_predictions
    })
    submission.to_csv('results/exp007/submission.csv', index=False)
    print(f"\n提出ファイル保存: results/exp007/submission.csv")
    
    # 予想スコア
    predicted_public = mean_cv + 0.1  # 控えめな予想
    print(f"\n予想Public Score: {predicted_public:.4f}")
    print("筋肉のような強力モデル完成や〜！💪")
    
    return mean_cv, std_cv, predicted_public

if __name__ == "__main__":
    cv_mean, cv_std, predicted_public = main()