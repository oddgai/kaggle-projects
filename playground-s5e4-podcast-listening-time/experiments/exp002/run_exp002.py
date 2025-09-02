#!/usr/bin/env python3
"""
実験002: XGBoostモデル
Playground Series S5E4: ポッドキャスト聴取時間予測
exp001のベースラインを改善し、XGBoostモデルで性能向上を図る
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
import pickle
import json

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
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

print("=== 実験002開始 ===")
print("ライブラリのインポート完了")

# データの読み込み
data_dir = Path('../../data')

train_df = pd.read_csv(data_dir / 'train.csv')
test_df = pd.read_csv(data_dir / 'test.csv')
sample_submission = pd.read_csv(data_dir / 'sample_submission.csv')

print(f"訓練データ: {train_df.shape}")
print(f"テストデータ: {test_df.shape}")
print(f"提出サンプル: {sample_submission.shape}")

# 目標変数の確認
target_col = 'Listening_Time_minutes'
print(f"\n目標変数: {target_col}")
print(f"目標変数の統計: {train_df[target_col].describe()}")

def enhanced_preprocess_data(df, is_train=True, target_col='Listening_Time_minutes', target_encoder=None):
    """
    拡張されたデータの前処理を行う関数
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
    
    print(f"カテゴリカル変数: {len(categorical_cols)}個")
    print(f"数値変数: {len(numerical_cols)}個")
    
    # 数値変数の欠損値補完
    for col in numerical_cols:
        if df_processed[col].isnull().sum() > 0:
            median_value = df_processed[col].median()
            df_processed[col] = df_processed[col].fillna(median_value)
            print(f"{col}の欠損値を中央値{median_value:.2f}で補完")
    
    # Guest_Popularity_percentageの欠損値を0で補完（ゲストなしを意味）
    df_processed['Guest_Popularity_percentage'] = df_processed['Guest_Popularity_percentage'].fillna(0)
    
    # 基本的な新規特徴量の作成（exp001と同じ）
    df_processed['Ad_Density'] = df_processed['Number_of_Ads'] / (df_processed['Episode_Length_minutes'] + 1e-8)
    df_processed['Host_Guest_Popularity_Diff'] = (df_processed['Host_Popularity_percentage'] - 
                                                   df_processed['Guest_Popularity_percentage'])
    df_processed['Has_Guest'] = (df_processed['Guest_Popularity_percentage'] > 0).astype(int)
    
    # 新しい特徴量の作成
    # 時間系特徴量
    df_processed['Episode_Length_squared'] = df_processed['Episode_Length_minutes'] ** 2
    df_processed['Episode_Length_log'] = np.log(df_processed['Episode_Length_minutes'] + 1)
    
    # 人気度関連特徴量
    df_processed['Host_Guest_Popularity_Sum'] = df_processed['Host_Popularity_percentage'] + df_processed['Guest_Popularity_percentage']
    df_processed['Host_Guest_Popularity_Ratio'] = df_processed['Host_Popularity_percentage'] / (df_processed['Guest_Popularity_percentage'] + 1)
    
    # 広告関連特徴量
    df_processed['Ads_per_Hour'] = df_processed['Number_of_Ads'] / ((df_processed['Episode_Length_minutes'] / 60) + 1e-8)
    df_processed['Has_Ads'] = (df_processed['Number_of_Ads'] > 0).astype(int)
    
    # エピソード長カテゴリ
    df_processed['Episode_Length_Category'] = pd.cut(df_processed['Episode_Length_minutes'], 
                                                     bins=[0, 30, 60, 90, float('inf')], 
                                                     labels=['short', 'medium', 'long', 'very_long'])
    
    # カテゴリカル変数の前処理
    le_dict = {}
    for col in categorical_cols:
        le = LabelEncoder()
        # 欠損値がある場合は'Missing'で埋める
        df_processed[col] = df_processed[col].fillna('Missing')
        df_processed[col] = le.fit_transform(df_processed[col])
        le_dict[col] = le
        print(f"{col}をラベルエンコーディング: {len(le.classes_)}カテゴリ")
    
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
                print(f"{col}のターゲットエンコーディング完了")
    
    elif target_encoder is not None:
        # テストデータの場合、保存されたエンコーダーを使用
        for col, encoder in target_encoder.items():
            if col in df_processed.columns:
                global_mean = encoder.mean()  # 訓練データの平均値
                df_processed[f'{col}_target_encoded'] = df_processed[col].map(encoder).fillna(global_mean)
    
    return df_processed, y, ids, le_dict, target_encoder if is_train else None

# データの前処理実行
print("\n=== 訓練データの前処理 ===")
X_train_processed, y_train, train_ids, le_dict_train, target_encoder = enhanced_preprocess_data(train_df, is_train=True)

print("\n=== テストデータの前処理 ===")
X_test_processed, _, test_ids, _, _ = enhanced_preprocess_data(test_df, is_train=False, target_encoder=target_encoder)

print(f"\n処理後の訓練データ形状: {X_train_processed.shape}")
print(f"処理後のテストデータ形状: {X_test_processed.shape}")
print(f"特徴量一覧: {X_train_processed.columns.tolist()}")

# 訓練・検証データの分割
X_train, X_val, y_train_split, y_val = train_test_split(
    X_train_processed, y_train, 
    test_size=0.2, 
    random_state=42
)

print(f"\n訓練データ: {X_train.shape}")
print(f"検証データ: {X_val.shape}")

# XGBoostモデルの訓練
print("\n=== XGBoostモデルの訓練 ===")

# パラメータ設定
xgb_params = {
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'booster': 'gbtree',
    'max_depth': 7,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'colsample_bylevel': 0.8,
    'min_child_weight': 3,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'random_state': 42,
    'n_jobs': -1
}

# データセットの作成
train_data = xgb.DMatrix(X_train, label=y_train_split)
val_data = xgb.DMatrix(X_val, label=y_val)

# モデル訓練
xgb_model = xgb.train(
    xgb_params,
    train_data,
    evals=[(train_data, 'train'), (val_data, 'eval')],
    num_boost_round=2000,
    early_stopping_rounds=100,
    verbose_eval=100
)

print(f"最適なイテレーション数: {xgb_model.best_iteration}")

# 予測
y_train_pred = xgb_model.predict(xgb.DMatrix(X_train), iteration_range=(0, xgb_model.best_iteration))
y_val_pred = xgb_model.predict(xgb.DMatrix(X_val), iteration_range=(0, xgb_model.best_iteration))

# 評価指標の計算
def calculate_metrics(y_true, y_pred, dataset_name):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    print(f"\n=== {dataset_name}の評価結果 ===")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R²: {r2:.4f}")
    
    return rmse, mae, r2

# 訓練データでの評価
train_rmse, train_mae, train_r2 = calculate_metrics(y_train_split, y_train_pred, "訓練データ")

# 検証データでの評価
val_rmse, val_mae, val_r2 = calculate_metrics(y_val, y_val_pred, "検証データ")

# 5-fold Cross Validation
print("\n=== 5-Fold Cross Validation ===")

kfold = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []

for fold, (train_idx, val_idx) in enumerate(kfold.split(X_train_processed)):
    print(f"\nFold {fold + 1}/5")
    
    # データ分割
    X_fold_train, X_fold_val = X_train_processed.iloc[train_idx], X_train_processed.iloc[val_idx]
    y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
    
    # データセット作成
    fold_train_data = xgb.DMatrix(X_fold_train, label=y_fold_train)
    fold_val_data = xgb.DMatrix(X_fold_val, label=y_fold_val)
    
    # モデル訓練
    fold_model = xgb.train(
        xgb_params,
        fold_train_data,
        evals=[(fold_train_data, 'train'), (fold_val_data, 'eval')],
        num_boost_round=2000,
        early_stopping_rounds=100,
        verbose_eval=0
    )
    
    # 予測と評価
    fold_pred = fold_model.predict(fold_val_data, iteration_range=(0, fold_model.best_iteration))
    fold_rmse = np.sqrt(mean_squared_error(y_fold_val, fold_pred))
    cv_scores.append(fold_rmse)
    
    print(f"Fold {fold + 1} RMSE: {fold_rmse:.4f}")

print(f"\n=== Cross Validation結果 ===")
print(f"平均RMSE: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
print(f"各FoldのRMSE: {[f'{score:.4f}' for score in cv_scores]}")

# 特徴量重要度の取得
feature_importance = xgb_model.get_score(importance_type='gain')
feature_names = X_train_processed.columns

# 重要度をDataFrameにまとめる（全特徴量を含む）
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
plt.title('特徴量重要度 Top 20 (XGBoost)')
plt.xlabel('重要度')
plt.tight_layout()
plt.savefig('feature_importance_plot.png', dpi=80, bbox_inches='tight')
print("特徴量重要度グラフを保存: feature_importance_plot.png")

# テストデータでの予測
test_data = xgb.DMatrix(X_test_processed)
test_predictions = xgb_model.predict(test_data, iteration_range=(0, xgb_model.best_iteration))

# 提出ファイルの作成
submission_df = pd.DataFrame({
    'id': test_ids,
    target_col: test_predictions
})

print("\n=== 提出ファイルの統計 ===")
print(submission_df[target_col].describe())

# ファイル保存
results_dir = Path('../../results/exp002')
results_dir.mkdir(parents=True, exist_ok=True)

submission_path = results_dir / 'submission.csv'
submission_df.to_csv(submission_path, index=False)
print(f"\n提出ファイルを保存: {submission_path}")

# サンプル提出ファイルとの形式確認
print(f"\n=== フォーマット確認 ===")
print(f"サンプル提出ファイル形状: {sample_submission.shape}")
print(f"作成した提出ファイル形状: {submission_df.shape}")
print(f"カラム名一致: {list(sample_submission.columns) == list(submission_df.columns)}")

# モデルの保存
model_path = results_dir / 'model.pkl'
with open(model_path, 'wb') as f:
    pickle.dump(xgb_model, f)
print(f"モデルを保存: {model_path}")

# 実験結果の記録（MLflow用に詳細データを追加）
experiment_results = {
    'experiment_id': 'exp002',
    'model_type': 'XGBoost',
    'model_params': xgb_params,
    
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
    'cv_scores': [float(score) for score in cv_scores],
    'cv_rmse_mean': float(np.mean(cv_scores)),
    'cv_rmse_std': float(np.std(cv_scores)),
    
    # 特徴量重要度
    'feature_importance': {
        row['feature']: int(row['importance']) 
        for _, row in importance_df.iterrows()
    },
    
    # Kaggleスコア（手動で更新）
    'public_score': None,  # 提出後に更新
    'private_score': None,  # 提出後に更新
    
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
print(f"特徴量数: {len(experiment_results['features'])}個")
print(f"検証RMSE: {experiment_results['val_rmse']:.4f}")
print(f"CV平均RMSE: {experiment_results['cv_rmse_mean']:.4f} ± {experiment_results['cv_rmse_std']:.4f}")
print(f"\n重要な特徴量トップ5:")
for i, (feature, importance) in enumerate(list(experiment_results['feature_importance'].items())[:5]):
    print(f"  {i+1}. {feature}: {importance:,}")

print(f"\nexp001との比較:")
print(f"exp001 CV RMSE: 13.0023")
print(f"exp002 CV RMSE: {experiment_results['cv_rmse_mean']:.4f}")
improvement = 13.0023 - experiment_results['cv_rmse_mean']
print(f"改善度: {improvement:.4f} ({'改善' if improvement > 0 else '悪化'})")

print("\n=== 実験002完了 ===")