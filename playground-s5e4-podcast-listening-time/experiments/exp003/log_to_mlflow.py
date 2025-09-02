#!/usr/bin/env python3
"""
exp003の実験結果をMLflowに記録するスクリプト
Optunaハイパーパラメータ最適化結果
"""

import json
import mlflow
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from datetime import datetime
from pathlib import Path

def create_readme_html():
    """
    exp003のREADME.mdをHTMLに変換
    """
    html_content = '<html><body><h1>実験003: Optuna ハイパーパラメータ最適化</h1> <h2>目的</h2> <p>exp002のXGBoostモデルに対してOptunaを使用したハイパーパラメータ最適化を行い、さらなる性能向上を図る</p> <h2>実験設定</h2> <h3>モデル</h3> <ul><li><strong>アルゴリズム</strong>: XGBoost (最適化版)</li><li><strong>最適化手法</strong>: Optuna TPE サンプラー</li><li><strong>試行回数</strong>: 30回</li><li><strong>タスク</strong>: 回帰問題（RMSE最適化）</li></ul> <h3>特徴量</h3> <p><strong>基本特徴量（10個）</strong></p> <ul><li>Podcast_Name</li><li>Episode_Title</li><li>Genre</li><li>Publication_Day</li><li>Publication_Time</li><li>Episode_Sentiment</li><li>Episode_Length_minutes</li><li>Host_Popularity_percentage</li><li>Guest_Popularity_percentage</li><li>Number_of_Ads</li></ul> <p><strong>新規作成特徴量（13個）</strong></p> <ul><li>Ad_Density: 広告数/エピソード長</li><li>Host_Guest_Popularity_Diff: ホストとゲストの人気度差</li><li>Has_Guest: ゲスト存在フラグ</li><li>Episode_Length_squared: エピソード長の2乗</li><li>Episode_Length_log: エピソード長の対数</li><li>Host_Guest_Popularity_Sum: ホストとゲストの人気度合計</li><li>Host_Guest_Popularity_Ratio: ホストとゲストの人気度比率</li><li>Ads_per_Hour: 1時間あたりの広告数</li><li>Has_Ads: 広告存在フラグ</li><li>Episode_Length_Category: エピソード長カテゴリ</li><li>Podcast_Name_target_encoded: ポッドキャスト名のターゲットエンコーディング</li><li>Episode_Title_target_encoded: エピソードタイトルのターゲットエンコーディング</li><li>Genre_target_encoded: ジャンルのターゲットエンコーディング</li></ul> <h3>最適化されたハイパーパラメータ</h3> <table border="1" style="border-collapse: collapse; margin: 10px 0;"><tr><th style="padding: 8px; background-color: #f0f0f0;">Parameter</th><th style="padding: 8px; background-color: #f0f0f0;">Value</th></tr><tr><td style="padding: 8px;">objective</td><td style="padding: 8px;">reg:squarederror</td></tr><tr><td style="padding: 8px;">eval_metric</td><td style="padding: 8px;">rmse</td></tr><tr><td style="padding: 8px;">max_depth</td><td style="padding: 8px;">10</td></tr><tr><td style="padding: 8px;">learning_rate</td><td style="padding: 8px;">0.0355</td></tr><tr><td style="padding: 8px;">subsample</td><td style="padding: 8px;">0.7163</td></tr><tr><td style="padding: 8px;">colsample_bytree</td><td style="padding: 8px;">0.9761</td></tr><tr><td style="padding: 8px;">colsample_bylevel</td><td style="padding: 8px;">0.8369</td></tr><tr><td style="padding: 8px;">min_child_weight</td><td style="padding: 8px;">6</td></tr><tr><td style="padding: 8px;">reg_alpha</td><td style="padding: 8px;">1.1603</td></tr><tr><td style="padding: 8px;">reg_lambda</td><td style="padding: 8px;">1.1458</td></tr><tr><td style="padding: 8px;">num_boost_round</td><td style="padding: 8px;">1419</td></tr></table> <h3>検証方法</h3> <ul><li>5-fold Cross Validation</li><li>Train/Validation split: 80%/20%</li><li>Optuna TPE サンプラー（シード値42）</li></ul> <h2>結果</h2> <h3>スコア</h3> <ul><li><strong>CV Score (RMSE)</strong>: 12.7938 ± 0.0176</li><li><strong>検証データRMSE</strong>: 12.7745</li><li><strong>訓練データRMSE</strong>: 9.6568</li><li><strong>Optuna最適スコア</strong>: 12.7938</li></ul> <h3>前実験との比較</h3> <ul><li><strong>exp001 CV RMSE</strong>: 13.0023</li><li><strong>exp002 CV RMSE</strong>: 12.8926</li><li><strong>exp003 CV RMSE</strong>: 12.7938</li><li><strong>exp002からの改善度</strong>: 0.0988 (改善)</li><li><strong>exp001からの改善度</strong>: 0.2085 (大幅改善)</li></ul> <h2>学習内容</h2> <h3>重要な発見</h3> <ul><li><strong>Optunaの威力</strong>: 自動ハイパーパラメータ最適化により大幅な性能向上</li><li><strong>深い木構造</strong>: max_depth=10が最適解</li><li><strong>適度な正則化</strong>: reg_alpha=1.16, reg_lambda=1.15のバランス</li><li><strong>高い特徴量選択率</strong>: colsample_bytree=0.98で多くの特徴量を活用</li><li><strong>安定したCV性能</strong>: 5-fold CVで標準偏差0.018と非常に安定</li></ul> <h3>最適化プロセスの効果</h3> <ul><li>30回の試行で効率的に最適解を発見</li><li>TPEサンプラーによる効果的なパラメータ空間探索</li><li>過学習を抑制しつつ性能向上を実現</li></ul> <h2>目標に対する評価</h2> <h3>達成状況</h3> <ul><li><strong>目標RMSE</strong>: 12未満</li><li><strong>現在のRMSE</strong>: 12.7938</li><li><strong>残り差分</strong>: 0.7938 (かなり接近)</li><li><strong>次のステップ</strong>: アンサンブルモデル（exp004）で最終目標達成を目指す</li></ul> <h2>改善案</h2> <h3>アンサンブル戦略</h3> <ul><li><strong>LightGBM + XGBoost</strong>: exp001とexp003の組み合わせ</li><li><strong>CatBoost追加</strong>: 第3のGBDTモデルとの組み合わせ</li><li><strong>重み付きアンサンブル</strong>: 各モデルの性能に応じた重み設定</li></ul> <h3>次の実験方向</h3> <ul><li>アンサンブルモデル（exp004）</li><li>より高度な特徴量エンジニアリング</li><li>スタッキングや2段階学習</li></ul> <h2>ファイル</h2> <ul><li><code>exp003.ipynb</code>: 実験用ノートブック</li><li><code>run_exp003.py</code>: Optuna最適化実行スクリプト</li><li><code>log_to_mlflow.py</code>: MLflow記録スクリプト</li></ul></body></html>'
    
    with open("README.html", "w", encoding="utf-8") as f:
        f.write(html_content)
    
    return "README.html"

def log_experiment_to_mlflow():
    """
    exp003の実験結果をMLflowに記録
    """
    # 1. Databricks MLflow設定
    try:
        mlflow.set_tracking_uri("databricks")
        print("Databricks MLflow接続成功")
    except Exception as e:
        print(f"Databricks MLflow接続エラー: {e}")
        print("ローカルMLflowに切り替え")
        mlflow.set_tracking_uri("file:///tmp/mlruns")
    
    # 実験パスの設定
    experiment_name = "/Shared/data_science/z_ogai/playground-s5e4-podcast-listening-time"
    try:
        mlflow.set_experiment(experiment_name)
        print(f"実験パス設定: {experiment_name}")
    except Exception as e:
        print(f"Databricks実験パス設定エラー: {e}")
        mlflow.set_experiment("playground-s5e4-podcast-listening-time")
        print("ローカル実験名に設定")
    
    # 2. JSONファイルから実験結果を読み込み
    results_path = Path('../../results/exp003/experiment_results.json')
    with open(results_path, 'r', encoding='utf-8') as f:
        exp_results = json.load(f)
    
    # 3. Run名とDescription設定
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    run_name = f"exp003_{timestamp}"
    description = "XGBoost Optunaハイパーパラメータ最適化モデル。30回の最適化でCV RMSE 12.7938を達成。exp002（12.8926）から0.0988改善し、目標12未満まで0.7938に迫る大幅な性能向上を実現。"
    
    # 4. MLflowに記録
    with mlflow.start_run(run_name=run_name, description=description):
        # 最適化関連パラメータの記録
        mlflow.log_param("optimization_method", exp_results['optimization_method'])
        mlflow.log_param("n_trials", exp_results['n_trials'])
        mlflow.log_param("best_num_boost_round", exp_results['best_num_boost_round'])
        
        # モデルパラメータの記録
        for param_name, param_value in exp_results['model_params'].items():
            mlflow.log_param(f"xgb_{param_name}", param_value)
        
        # 特徴量情報の記録
        mlflow.log_param("num_features", exp_results['num_features'])
        mlflow.log_param("feature_engineering", exp_results['preprocessing']['feature_engineering'])
        mlflow.log_param("categorical_encoding", exp_results['preprocessing']['categorical_encoding'])
        
        # 評価指標の記録
        mlflow.log_metric("train_rmse", exp_results['train_rmse'])
        mlflow.log_metric("val_rmse", exp_results['val_rmse'])
        mlflow.log_metric("cv_rmse_mean", exp_results['cv_rmse_mean'])
        mlflow.log_metric("cv_rmse_std", exp_results['cv_rmse_std'])
        mlflow.log_metric("optuna_best_score", exp_results['optuna_best_score'])
        
        # 改善度の記録
        mlflow.log_metric("improvement_from_exp002", 12.8926 - exp_results['cv_rmse_mean'])
        mlflow.log_metric("improvement_from_exp001", 13.0023 - exp_results['cv_rmse_mean'])
        mlflow.log_metric("distance_to_target", exp_results['cv_rmse_mean'] - 12.0)
        
        # 各Foldのスコア
        for i, score in enumerate(exp_results['cv_scores']):
            mlflow.log_metric(f"cv_fold_{i+1}_rmse", score)
        
        # README.mdのHTML変換とアーティファクト記録
        readme_html_path = create_readme_html()
        mlflow.log_artifact(readme_html_path)
        
        # 実験結果JSONファイルをアーティファクトとして記録
        mlflow.log_artifact(str(results_path))
        
        print(f"MLflow記録完了: {run_name}")
        print(f"Run ID: {mlflow.active_run().info.run_id}")

if __name__ == "__main__":
    log_experiment_to_mlflow()