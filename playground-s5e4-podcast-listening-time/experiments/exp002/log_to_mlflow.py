#!/usr/bin/env python3
"""
exp002の実験結果をMLflowに記録するスクリプト
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

def create_feature_importance_plot(feature_importance_dict):
    """
    特徴量重要度のプロットを作成
    """
    # 辞書をDataFrameに変換してソート
    importance_df = pd.DataFrame(
        list(feature_importance_dict.items()), 
        columns=['feature', 'importance']
    ).sort_values('importance', ascending=False)
    
    # 上位20個の特徴量のみ表示
    top_features = importance_df.head(20)
    
    plt.figure(figsize=(12, 8))
    sns.barplot(data=top_features, x='importance', y='feature')
    plt.title('特徴量重要度 Top 20 (XGBoost - exp002)')
    plt.xlabel('重要度')
    plt.ylabel('特徴量')
    plt.tight_layout()
    plt.savefig("feature_importance_plot.png", dpi=80, bbox_inches="tight")
    plt.close()
    
    return "feature_importance_plot.png"

def create_readme_html():
    """
    exp002のREADME.mdをHTMLに変換
    """
    html_content = '<html><body><h1>実験002: XGBoostモデル</h1> <h2>目的</h2> <p>exp001のLightGBMベースラインを改善し、XGBoostモデルと拡張された特徴量エンジニアリングで性能向上を図る</p> <h2>実験設定</h2> <h3>モデル</h3> <ul><li><strong>アルゴリズム</strong>: XGBoost</li><li><strong>タスク</strong>: 回帰問題（RMSE最適化）</li></ul> <h3>特徴量</h3> <p><strong>基本特徴量（10個）</strong></p> <ul><li>Podcast_Name</li><li>Episode_Title</li><li>Genre</li><li>Publication_Day</li><li>Publication_Time</li><li>Episode_Sentiment</li><li>Episode_Length_minutes</li><li>Host_Popularity_percentage</li><li>Guest_Popularity_percentage</li><li>Number_of_Ads</li></ul> <p><strong>新規作成特徴量（13個）</strong></p> <ul><li>Ad_Density: 広告数/エピソード長</li><li>Host_Guest_Popularity_Diff: ホストとゲストの人気度差</li><li>Has_Guest: ゲスト存在フラグ</li><li>Episode_Length_squared: エピソード長の2乗</li><li>Episode_Length_log: エピソード長の対数</li><li>Host_Guest_Popularity_Sum: ホストとゲストの人気度合計</li><li>Host_Guest_Popularity_Ratio: ホストとゲストの人気度比率</li><li>Ads_per_Hour: 1時間あたりの広告数</li><li>Has_Ads: 広告存在フラグ</li><li>Episode_Length_Category: エピソード長カテゴリ</li><li>Podcast_Name_target_encoded: ポッドキャスト名のターゲットエンコーディング</li><li>Episode_Title_target_encoded: エピソードタイトルのターゲットエンコーディング</li><li>Genre_target_encoded: ジャンルのターゲットエンコーディング</li></ul> <h3>ハイパーパラメータ</h3> <table border="1" style="border-collapse: collapse; margin: 10px 0;"><tr><th style="padding: 8px; background-color: #f0f0f0;">Parameter</th><th style="padding: 8px; background-color: #f0f0f0;">Value</th></tr><tr><td style="padding: 8px;">objective</td><td style="padding: 8px;">reg:squarederror</td></tr><tr><td style="padding: 8px;">eval_metric</td><td style="padding: 8px;">rmse</td></tr><tr><td style="padding: 8px;">booster</td><td style="padding: 8px;">gbtree</td></tr><tr><td style="padding: 8px;">max_depth</td><td style="padding: 8px;">7</td></tr><tr><td style="padding: 8px;">learning_rate</td><td style="padding: 8px;">0.05</td></tr><tr><td style="padding: 8px;">subsample</td><td style="padding: 8px;">0.8</td></tr><tr><td style="padding: 8px;">colsample_bytree</td><td style="padding: 8px;">0.8</td></tr><tr><td style="padding: 8px;">colsample_bylevel</td><td style="padding: 8px;">0.8</td></tr><tr><td style="padding: 8px;">min_child_weight</td><td style="padding: 8px;">3</td></tr><tr><td style="padding: 8px;">reg_alpha</td><td style="padding: 8px;">0.1</td></tr><tr><td style="padding: 8px;">reg_lambda</td><td style="padding: 8px;">1.0</td></tr></table> <h3>検証方法</h3> <ul><li>5-fold Cross Validation</li><li>Train/Validation split: 80%/20%</li></ul> <h2>結果</h2> <h3>スコア</h3> <ul><li><strong>CV Score (RMSE)</strong>: 12.8926 ± 0.0207</li><li><strong>検証データRMSE</strong>: 12.8754</li><li><strong>訓練データRMSE</strong>: 11.6658</li><li><strong>R²スコア</strong>: 0.7747</li></ul> <h3>exp001との比較</h3> <ul><li><strong>exp001 CV RMSE</strong>: 13.0023</li><li><strong>exp002 CV RMSE</strong>: 12.8926</li><li><strong>改善度</strong>: 0.1097 (改善)</li></ul> <h2>学習内容</h2> <h3>重要な発見</h3> <ul><li><strong>XGBoostの性能</strong>: LightGBMより若干の改善を達成</li><li><strong>ターゲットエンコーディング</strong>: カテゴリカル変数の高度なエンコーディングを実装</li><li><strong>特徴量拡張</strong>: 多項式特徴量、相互作用特徴量を追加</li><li><strong>安定したCV性能</strong>: 5-fold CVで標準偏差0.02と安定した結果</li></ul> <h3>前処理の効果</h3> <ul><li>ターゲットエンコーディング: スムージング（α=10）付きで実装</li><li>多項式特徴量: エピソード長の2乗、対数変換</li><li>相互作用特徴量: 人気度の合計、比率等</li></ul> <h2>改善案</h2> <h3>ハイパーパラメータ最適化</h3> <ul><li><strong>Optunaによる最適化</strong>: より細かいパラメータ調整</li><li><strong>学習率の調整</strong>: より細かい学習率スケジューリング</li><li><strong>正則化の強化</strong>: 過学習抑制のための調整</li></ul> <h3>アンサンブル</h3> <ul><li><strong>LightGBM + XGBoost</strong>: 2つのGBDTモデルのアンサンブル</li><li><strong>CatBoost追加</strong>: 第3のGBDTモデルとの組み合わせ</li></ul> <h3>次の実験方向</h3> <ul><li>ハイパーパラメータ最適化（exp003）</li><li>アンサンブルモデル（exp004）</li><li>より高度な特徴量エンジニアリング</li></ul> <h2>ファイル</h2> <ul><li><code>exp002.ipynb</code>: 実験用ノートブック</li><li><code>run_exp002.py</code>: 実験実行スクリプト</li><li><code>log_to_mlflow.py</code>: MLflow記録スクリプト</li></ul></body></html>'
    
    with open("README.html", "w", encoding="utf-8") as f:
        f.write(html_content)
    
    return "README.html"

def log_experiment_to_mlflow():
    """
    exp002の実験結果をMLflowに記録
    """
    # MLflow設定（外部設定ファイル使用）
    experiment_path, environment = setup_mlflow()
    
    # 2. JSONファイルから実験結果を読み込み
    results_path = Path('../../results/exp002/experiment_results.json')
    with open(results_path, 'r', encoding='utf-8') as f:
        exp_results = json.load(f)
    
    # 3. Run名とDescription設定
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    run_name = f"exp002_{timestamp}"
    description = "XGBoostベースラインモデル。拡張された特徴量エンジニアリング（ターゲットエンコーディング、多項式特徴量、相互作用特徴量）を実装。exp001（LightGBM）から派生し、12.8926 RMSEを達成。"
    
    # 4. MLflowに記録
    with mlflow.start_run(run_name=run_name, description=description):
        # パラメータの記録
        for param_name, param_value in exp_results['model_params'].items():
            mlflow.log_param(param_name, param_value)
        
        # 特徴量情報の記録
        mlflow.log_param("num_features", exp_results['num_features'])
        mlflow.log_param("feature_engineering", exp_results['preprocessing']['feature_engineering'])
        mlflow.log_param("categorical_encoding", exp_results['preprocessing']['categorical_encoding'])
        
        # 評価指標の記録
        mlflow.log_metric("train_rmse", exp_results['train_rmse'])
        mlflow.log_metric("train_mae", exp_results['train_mae'])
        mlflow.log_metric("train_r2", exp_results['train_r2'])
        mlflow.log_metric("val_rmse", exp_results['val_rmse'])
        mlflow.log_metric("val_mae", exp_results['val_mae'])
        mlflow.log_metric("val_r2", exp_results['val_r2'])
        mlflow.log_metric("cv_rmse_mean", exp_results['cv_rmse_mean'])
        mlflow.log_metric("cv_rmse_std", exp_results['cv_rmse_std'])
        
        # 各Foldのスコア
        for i, score in enumerate(exp_results['cv_scores']):
            mlflow.log_metric(f"cv_fold_{i+1}_rmse", score)
        
        # 特徴量重要度のプロット作成とアーティファクト記録
        # 注: 現在のfeature_importanceは全て0なので、プロットは作成しないか、注釈付きで作成
        print("注意: 特徴量重要度が全て0のため、プロットをスキップ")
        
        # README.mdのHTML変換とアーティファクト記録
        readme_html_path = create_readme_html()
        mlflow.log_artifact(readme_html_path)
        
        # 実験結果JSONファイルをアーティファクトとして記録
        mlflow.log_artifact(str(results_path))
        
        print(f"MLflow記録完了: {run_name}")
        print(f"Run ID: {mlflow.active_run().info.run_id}")

if __name__ == "__main__":
    log_experiment_to_mlflow()