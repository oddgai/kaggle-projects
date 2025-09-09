#!/usr/bin/env python3
"""
exp004の実験結果をMLflowに記録するスクリプト
LightGBM + XGBoost アンサンブル結果
"""

import json
import mlflow
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from datetime import datetime
from pathlib import Path
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.mlflow_utils import setup_mlflow, get_experiment_url

def create_readme_html():
    """
    exp004のREADME.mdをHTMLに変換
    """
    html_content = '<html><body><h1>実験004: LightGBM + XGBoost アンサンブル</h1> <h2>目的</h2> <p>exp001のLightGBMとexp003の最適化XGBoostを組み合わせたアンサンブルモデルで目標12未満を達成する</p> <h2>実験設定</h2> <h3>モデル</h3> <ul><li><strong>アルゴリズム</strong>: アンサンブル（LightGBM + XGBoost）</li><li><strong>アンサンブル手法</strong>: 重み付き平均（固定重み）</li><li><strong>LightGBM重み</strong>: 0.5</li><li><strong>XGBoost重み</strong>: 0.5</li><li><strong>タスク</strong>: 回帰問題（RMSE最適化）</li></ul> <h3>個別モデル設定</h3> <h4>LightGBM（exp001設定）</h4> <table border="1" style="border-collapse: collapse; margin: 10px 0;"><tr><th style="padding: 8px; background-color: #f0f0f0;">Parameter</th><th style="padding: 8px; background-color: #f0f0f0;">Value</th></tr><tr><td style="padding: 8px;">objective</td><td style="padding: 8px;">regression</td></tr><tr><td style="padding: 8px;">metric</td><td style="padding: 8px;">rmse</td></tr><tr><td style="padding: 8px;">boosting_type</td><td style="padding: 8px;">gbdt</td></tr><tr><td style="padding: 8px;">num_leaves</td><td style="padding: 8px;">31</td></tr><tr><td style="padding: 8px;">learning_rate</td><td style="padding: 8px;">0.05</td></tr><tr><td style="padding: 8px;">feature_fraction</td><td style="padding: 8px;">0.9</td></tr><tr><td style="padding: 8px;">bagging_fraction</td><td style="padding: 8px;">0.8</td></tr><tr><td style="padding: 8px;">bagging_freq</td><td style="padding: 8px;">5</td></tr></table> <h4>XGBoost（exp003最適化設定）</h4> <table border="1" style="border-collapse: collapse; margin: 10px 0;"><tr><th style="padding: 8px; background-color: #f0f0f0;">Parameter</th><th style="padding: 8px; background-color: #f0f0f0;">Value</th></tr><tr><td style="padding: 8px;">objective</td><td style="padding: 8px;">reg:squarederror</td></tr><tr><td style="padding: 8px;">max_depth</td><td style="padding: 8px;">10</td></tr><tr><td style="padding: 8px;">learning_rate</td><td style="padding: 8px;">0.0355</td></tr><tr><td style="padding: 8px;">subsample</td><td style="padding: 8px;">0.7163</td></tr><tr><td style="padding: 8px;">colsample_bytree</td><td style="padding: 8px;">0.9761</td></tr><tr><td style="padding: 8px;">colsample_bylevel</td><td style="padding: 8px;">0.8369</td></tr><tr><td style="padding: 8px;">min_child_weight</td><td style="padding: 8px;">6</td></tr><tr><td style="padding: 8px;">reg_alpha</td><td style="padding: 8px;">1.1603</td></tr><tr><td style="padding: 8px;">reg_lambda</td><td style="padding: 8px;">1.1458</td></tr></table> <h3>検証方法</h3> <ul><li>5-fold Cross Validation</li><li>Train/Validation split: 80%/20%</li><li>固定重み（50:50）による高速化</li></ul> <h2>結果</h2> <h3>スコア</h3> <ul><li><strong>CV Score (RMSE)</strong>: 12.8203 ± 0.0188</li><li><strong>検証データRMSE</strong>: 12.8006</li><li><strong>訓練データRMSE</strong>: 11.0453</li><li><strong>R²スコア</strong>: 0.7773</li></ul> <h3>全実験との比較</h3> <ul><li><strong>exp001 (LightGBM) CV RMSE</strong>: 13.0023</li><li><strong>exp002 (XGBoost) CV RMSE</strong>: 12.8926</li><li><strong>exp003 (XGBoost最適) CV RMSE</strong>: 12.7938 ← ベスト</li><li><strong>exp004 (アンサンブル) CV RMSE</strong>: 12.8203</li><li><strong>exp001からの改善度</strong>: 0.1820</li><li><strong>exp003からの変化</strong>: -0.0265 (微小悪化)</li></ul> <h2>学習内容</h2> <h3>重要な発見</h3> <ul><li><strong>exp003の優秀さ</strong>: Optuna最適化XGBoostが既に非常に強力</li><li><strong>アンサンブルの限界</strong>: 50:50固定重みでは期待した改善なし</li><li><strong>個別モデル性能</strong>: 既に高度に最適化されたモデルのアンサンブルは困難</li><li><strong>安定性の向上</strong>: CV標準偏差0.019と安定した結果</li></ul> <h3>アンサンブル戦略の評価</h3> <ul><li>固定重み（50:50）は実装が簡単</li><li>重み最適化は計算コストが高い割に効果限定的</li><li>exp003単体の方が優秀という結果</li></ul> <h2>目標に対する評価</h2> <h3>達成状況</h3> <ul><li><strong>目標RMSE</strong>: 12未満</li><li><strong>現在のRMSE</strong>: 12.8203</li><li><strong>目標未達成</strong>: 残り0.8203差</li><li><strong>最終結論</strong>: exp003が最高性能を維持</li></ul> <h2>実験シリーズ総括</h2> <h3>全体成果</h3> <ul><li><strong>開始時</strong>: exp001 (13.0023)</li><li><strong>最高性能</strong>: exp003 (12.7938)</li><li><strong>総改善度</strong>: 0.2085 (大幅改善)</li><li><strong>技術習得</strong>: Optuna最適化、MLflow管理、アンサンブル手法</li></ul> <h3>推奨アプローチ</h3> <ul><li>exp003のOptuna最適化XGBoostを最終モデルとして採用</li><li>さらなる改善には特徴量エンジニアリングやデータ拡張が必要</li></ul> <h2>ファイル</h2> <ul><li><code>run_exp004_fast.py</code>: 高速アンサンブル実行スクリプト</li><li><code>log_to_mlflow.py</code>: MLflow記録スクリプト</li></ul></body></html>'
    
    with open("README.html", "w", encoding="utf-8") as f:
        f.write(html_content)
    
    return "README.html"

def log_experiment_to_mlflow():
    """
    exp004の実験結果をMLflowに記録
    """
    # MLflow設定（外部設定ファイル使用）
    experiment_path, environment = setup_mlflow()
    
    # 2. JSONファイルから実験結果を読み込み
    results_path = Path('../../results/exp004/experiment_results.json')
    with open(results_path, 'r', encoding='utf-8') as f:
        exp_results = json.load(f)
    
    # 3. Run名とDescription設定
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    run_name = f"exp004_{timestamp}"
    description = "LightGBM + XGBoost アンサンブルモデル（固定重み50:50）。CV RMSE 12.8203を達成。exp003（12.7938）を上回らず、単体最適化モデルの優秀さを確認。実験シリーズ完了。"
    
    # 4. MLflowに記録
    with mlflow.start_run(run_name=run_name, description=description):
        # アンサンブル関連パラメータの記録
        mlflow.log_param("ensemble_method", exp_results['ensemble_method'])
        mlflow.log_param("lgb_weight", exp_results['lgb_weight'])
        mlflow.log_param("xgb_weight", exp_results['xgb_weight'])
        
        # 特徴量情報の記録
        mlflow.log_param("num_features", exp_results['num_features'])
        mlflow.log_param("feature_engineering", exp_results['preprocessing']['feature_engineering'])
        mlflow.log_param("categorical_encoding", exp_results['preprocessing']['categorical_encoding'])
        
        # アンサンブル評価指標の記録
        mlflow.log_metric("ensemble_train_rmse", exp_results['ensemble_train_rmse'])
        mlflow.log_metric("ensemble_train_mae", exp_results['ensemble_train_mae'])
        mlflow.log_metric("ensemble_train_r2", exp_results['ensemble_train_r2'])
        mlflow.log_metric("ensemble_val_rmse", exp_results['ensemble_val_rmse'])
        mlflow.log_metric("ensemble_val_mae", exp_results['ensemble_val_mae'])
        mlflow.log_metric("ensemble_val_r2", exp_results['ensemble_val_r2'])
        mlflow.log_metric("cv_rmse_mean", exp_results['cv_rmse_mean'])
        mlflow.log_metric("cv_rmse_std", exp_results['cv_rmse_std'])
        
        # 全実験比較指標
        mlflow.log_metric("improvement_from_exp001", 13.0023 - exp_results['cv_rmse_mean'])
        mlflow.log_metric("change_from_exp003", exp_results['cv_rmse_mean'] - 12.7938)
        mlflow.log_metric("distance_to_target", exp_results['cv_rmse_mean'] - 12.0)
        mlflow.log_metric("is_best_model", 0)  # exp003が最高性能
        
        # 各Foldのスコア
        for i, score in enumerate(exp_results['cv_scores']):
            mlflow.log_metric(f"cv_fold_{i+1}_rmse", score)
        
        # 実験シリーズ完了マーク
        mlflow.log_metric("experiment_series_final", 1)
        mlflow.log_metric("total_experiments", 4)
        mlflow.log_metric("best_score_achieved", 12.7938)  # exp003
        
        # README.mdのHTML変換とアーティファクト記録
        readme_html_path = create_readme_html()
        mlflow.log_artifact(readme_html_path)
        
        # 実験結果JSONファイルをアーティファクトとして記録
        mlflow.log_artifact(str(results_path))
        
        print(f"MLflow記録完了: {run_name}")
        print(f"Run ID: {mlflow.active_run().info.run_id}")
        print("実験シリーズ完了マークを記録")

if __name__ == "__main__":
    log_experiment_to_mlflow()