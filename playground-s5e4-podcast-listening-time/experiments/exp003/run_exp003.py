#!/usr/bin/env python3
"""
実験003: Optunaによるハイパーパラメータ最適化
Playground Series S5E4: ポッドキャスト聴取時間予測
exp002のXGBoostモデルをベースに、Optunaでハイパーパラメータを最適化
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
import pickle
import json
import optuna
from optuna.samplers import TPESampler

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.impute import SimpleImputer
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

print("=== 実験003開始（Optunaハイパーパラメータ最適化）===")

# データの読み込み
data_dir = Path('../../data')

train_df = pd.read_csv(data_dir / 'train.csv')
test_df = pd.read_csv(data_dir / 'test.csv')
sample_submission = pd.read_csv(data_dir / 'sample_submission.csv')

print(f"訓練データ: {train_df.shape}")
print(f"テストデータ: {test_df.shape}")

# 目標変数の確認
target_col = 'Listening_Time_minutes'

def enhanced_preprocess_data(df, is_train=True, target_col='Listening_Time_minutes', target_encoder=None):
    """
    拡張されたデータの前処理を行う関数（exp002と同じ）
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
    
    # Guest_Popularity_percentageの欠損値を0で補完（ゲストなしを意味）
    df_processed['Guest_Popularity_percentage'] = df_processed['Guest_Popularity_percentage'].fillna(0)
    
    # 基本的な新規特徴量の作成
    df_processed['Ad_Density'] = df_processed['Number_of_Ads'] / (df_processed['Episode_Length_minutes'] + 1e-8)
    df_processed['Host_Guest_Popularity_Diff'] = (df_processed['Host_Popularity_percentage'] - 
                                                   df_processed['Guest_Popularity_percentage'])
    df_processed['Has_Guest'] = (df_processed['Guest_Popularity_percentage'] > 0).astype(int)
    
    # 新しい特徴量の作成
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
                # 各カテゴリの平均値を計算（スムージング付き）
                global_mean = y.mean()
                category_means = df_processed.groupby(col)[col].apply(lambda x: len(x))
                category_targets = pd.DataFrame({'original_col': df_processed[col], 'target': y})
                category_target_mean = category_targets.groupby('original_col')['target'].mean()
                
                # スムージング（α=10）
                alpha = 10
                smoothed_means = (category_target_mean * category_means + global_mean * alpha) / (category_means + alpha)
                
                target_encoder[col] = smoothed_means
                df_processed[f'{col}_target_encoded'] = df_processed[col].map(target_encoder[col]).fillna(global_mean)
    
    elif target_encoder is not None:
        # テストデータの場合、保存されたエンコーダーを使用
        for col, encoder in target_encoder.items():
            if col in df_processed.columns:
                global_mean = encoder.mean()  # 訓練データの平均値
                df_processed[f'{col}_target_encoded'] = df_processed[col].map(encoder).fillna(global_mean)
    
    return df_processed, y, ids, le_dict, target_encoder if is_train else None

# データの前処理実行
print("\n=== データの前処理 ===")
X_train_processed, y_train, train_ids, le_dict_train, target_encoder = enhanced_preprocess_data(train_df, is_train=True)
X_test_processed, _, test_ids, _, _ = enhanced_preprocess_data(test_df, is_train=False, target_encoder=target_encoder)

print(f"処理後の訓練データ形状: {X_train_processed.shape}")
print(f"処理後のテストデータ形状: {X_test_processed.shape}")

# Optunaによるハイパーパラメータ最適化
def objective(trial):
    """
    Optuna最適化の目的関数
    """
    # ハイパーパラメータの探索範囲を定義
    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'booster': 'gbtree',
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.6, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 2.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 2.0),
        'random_state': 42,
        'n_jobs': -1,
        'verbosity': 0
    }
    
    # 5-fold Cross Validation
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = []
    
    for train_idx, val_idx in kfold.split(X_train_processed):
        # データ分割
        X_fold_train, X_fold_val = X_train_processed.iloc[train_idx], X_train_processed.iloc[val_idx]
        y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        # データセット作成
        fold_train_data = xgb.DMatrix(X_fold_train, label=y_fold_train)
        fold_val_data = xgb.DMatrix(X_fold_val, label=y_fold_val)
        
        # モデル訓練
        num_boost_round = trial.suggest_int('num_boost_round', 100, 3000)
        fold_model = xgb.train(
            params,
            fold_train_data,
            num_boost_round=num_boost_round,
            evals=[(fold_val_data, 'eval')],
            early_stopping_rounds=100,
            verbose_eval=0
        )
        
        # 予測と評価
        fold_pred = fold_model.predict(fold_val_data, iteration_range=(0, fold_model.best_iteration))
        fold_rmse = np.sqrt(mean_squared_error(y_fold_val, fold_pred))
        cv_scores.append(fold_rmse)
    
    return np.mean(cv_scores)

# Optuna最適化の実行
print("\n=== Optunaによるハイパーパラメータ最適化開始 ===")
print("試行回数: 100回")

study = optuna.create_study(direction='minimize', sampler=TPESampler(seed=42))
study.optimize(objective, n_trials=100, timeout=3600)  # 1時間のタイムアウト

print("\n=== 最適化結果 ===")
print(f"最適スコア (CV RMSE): {study.best_value:.4f}")
print(f"最適パラメータ: {study.best_params}")

# 最適パラメータでのモデル再訓練
print("\n=== 最適パラメータでの最終モデル訓練 ===")
best_params = study.best_params.copy()
num_boost_round = best_params.pop('num_boost_round')

# 訓練・検証データの分割
X_train, X_val, y_train_split, y_val = train_test_split(
    X_train_processed, y_train, 
    test_size=0.2, 
    random_state=42
)

# データセットの作成
train_data = xgb.DMatrix(X_train, label=y_train_split)
val_data = xgb.DMatrix(X_val, label=y_val)

# 最適化されたモデル訓練
best_params.update({
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'booster': 'gbtree',
    'random_state': 42,
    'n_jobs': -1
})

final_model = xgb.train(
    best_params,
    train_data,
    evals=[(train_data, 'train'), (val_data, 'eval')],
    num_boost_round=num_boost_round,
    early_stopping_rounds=100,
    verbose_eval=100
)

# 最終評価
y_train_pred = final_model.predict(xgb.DMatrix(X_train), iteration_range=(0, final_model.best_iteration))
y_val_pred = final_model.predict(xgb.DMatrix(X_val), iteration_range=(0, final_model.best_iteration))

train_rmse = np.sqrt(mean_squared_error(y_train_split, y_train_pred))
train_mae = mean_absolute_error(y_train_split, y_train_pred)
train_r2 = r2_score(y_train_split, y_train_pred)

val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
val_mae = mean_absolute_error(y_val, y_val_pred)
val_r2 = r2_score(y_val, y_val_pred)

print(f"\n=== 最終評価結果 ===")
print(f"訓練RMSE: {train_rmse:.4f}")
print(f"検証RMSE: {val_rmse:.4f}")
print(f"CV最適スコア: {study.best_value:.4f}")

# 最終Cross Validation（最適パラメータで）
print("\n=== 最終5-Fold Cross Validation ===")
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
final_cv_scores = []

for fold, (train_idx, val_idx) in enumerate(kfold.split(X_train_processed)):
    print(f"Fold {fold + 1}/5")
    
    # データ分割
    X_fold_train, X_fold_val = X_train_processed.iloc[train_idx], X_train_processed.iloc[val_idx]
    y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
    
    # データセット作成
    fold_train_data = xgb.DMatrix(X_fold_train, label=y_fold_train)
    fold_val_data = xgb.DMatrix(X_fold_val, label=y_fold_val)
    
    # モデル訓練
    fold_model = xgb.train(
        best_params,
        fold_train_data,
        evals=[(fold_train_data, 'train'), (fold_val_data, 'eval')],
        num_boost_round=num_boost_round,
        early_stopping_rounds=100,
        verbose_eval=0
    )
    
    # 予測と評価
    fold_pred = fold_model.predict(fold_val_data, iteration_range=(0, fold_model.best_iteration))
    fold_rmse = np.sqrt(mean_squared_error(y_fold_val, fold_pred))
    final_cv_scores.append(fold_rmse)
    
    print(f"Fold {fold + 1} RMSE: {fold_rmse:.4f}")

cv_mean = np.mean(final_cv_scores)
cv_std = np.std(final_cv_scores)

print(f"\n=== 最終Cross Validation結果 ===")
print(f"平均RMSE: {cv_mean:.4f} ± {cv_std:.4f}")
print(f"各FoldのRMSE: {[f'{score:.4f}' for score in final_cv_scores]}")

# 特徴量重要度の取得
feature_importance = final_model.get_score(importance_type='gain')
feature_names = X_train_processed.columns

importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': [feature_importance.get(f'f{i}', 0) for i in range(len(feature_names))]
}).sort_values('importance', ascending=False)

print("\n=== 特徴量重要度 Top 15 ===")
print(importance_df.head(15))

# 可視化
plt.figure(figsize=(12, 8))
top_features = importance_df.head(20)
sns.barplot(data=top_features, x='importance', y='feature')
plt.title('特徴量重要度 Top 20 (XGBoost最適化 - exp003)')
plt.xlabel('重要度')
plt.tight_layout()
plt.savefig('feature_importance_plot.png', dpi=80, bbox_inches='tight')
print("特徴量重要度グラフを保存")

# テストデータでの予測
test_data = xgb.DMatrix(X_test_processed)
test_predictions = final_model.predict(test_data, iteration_range=(0, final_model.best_iteration))

# 提出ファイルの作成
submission_df = pd.DataFrame({
    'id': test_ids,
    target_col: test_predictions
})

# ファイル保存
results_dir = Path('../../results/exp003')
results_dir.mkdir(parents=True, exist_ok=True)

submission_path = results_dir / 'submission.csv'
submission_df.to_csv(submission_path, index=False)
print(f"\n提出ファイルを保存: {submission_path}")

# モデルの保存
model_path = results_dir / 'model.pkl'
with open(model_path, 'wb') as f:
    pickle.dump(final_model, f)
print(f"モデルを保存: {model_path}")

# 実験結果の記録
experiment_results = {
    'experiment_id': 'exp003',
    'model_type': 'XGBoost_Optimized',
    'optimization_method': 'Optuna',
    'n_trials': study.n_trials,
    'model_params': best_params,
    'best_num_boost_round': num_boost_round,
    
    # 特徴量情報
    'features': list(X_train_processed.columns),
    'num_features': len(X_train_processed.columns),
    'new_features': ['Ad_Density', 'Host_Guest_Popularity_Diff', 'Has_Guest', 
                    'Episode_Length_squared', 'Episode_Length_log',
                    'Host_Guest_Popularity_Sum', 'Host_Guest_Popularity_Ratio',
                    'Ads_per_Hour', 'Has_Ads', 'Episode_Length_Category'],
    
    # 評価指標
    'train_rmse': float(train_rmse),
    'train_mae': float(train_mae),
    'train_r2': float(train_r2),
    'val_rmse': float(val_rmse),
    'val_mae': float(val_mae),
    'val_r2': float(val_r2),
    
    # Cross Validation結果
    'cv_scores': [float(score) for score in final_cv_scores],
    'cv_rmse_mean': float(cv_mean),
    'cv_rmse_std': float(cv_std),
    'optuna_best_score': float(study.best_value),
    
    # 特徴量重要度
    'feature_importance': {
        row['feature']: int(row['importance']) if row['importance'] > 0 else 0
        for _, row in importance_df.iterrows()
    },
    
    # Kaggleスコア（手動で更新）
    'public_score': None,
    'private_score': None,
    
    # データ情報
    'train_size': len(X_train_processed),
    'test_size': len(X_test_processed),
    'target_variable': target_col,
    
    # 実験設定
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

print("\n=== 実験結果サマリー ===")
print(f"モデル: {experiment_results['model_type']}")
print(f"最適化試行回数: {study.n_trials}")
print(f"特徴量数: {len(experiment_results['features'])}個")
print(f"検証RMSE: {experiment_results['val_rmse']:.4f}")
print(f"CV平均RMSE: {experiment_results['cv_rmse_mean']:.4f} ± {experiment_results['cv_rmse_std']:.4f}")
print(f"Optuna最適スコア: {experiment_results['optuna_best_score']:.4f}")

print(f"\n前実験との比較:")
print(f"exp001 CV RMSE: 13.0023")
print(f"exp002 CV RMSE: 12.8926")
print(f"exp003 CV RMSE: {experiment_results['cv_rmse_mean']:.4f}")
improvement_from_002 = 12.8926 - experiment_results['cv_rmse_mean']
improvement_from_001 = 13.0023 - experiment_results['cv_rmse_mean']
print(f"exp002からの改善度: {improvement_from_002:.4f}")
print(f"exp001からの改善度: {improvement_from_001:.4f}")

# 目標達成確認
if experiment_results['cv_rmse_mean'] < 12.0:
    print(f"\n🎉 目標達成！ CV RMSE {experiment_results['cv_rmse_mean']:.4f} < 12.0")
else:
    print(f"\n⚠️  まだ目標未達成。CV RMSE {experiment_results['cv_rmse_mean']:.4f} >= 12.0")
    print("exp004（アンサンブル）で更なる改善を目指します。")

print("\n=== 実験003完了 ===")