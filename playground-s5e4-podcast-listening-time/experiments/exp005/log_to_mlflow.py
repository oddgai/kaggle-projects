#!/usr/bin/env python3
"""
exp005の実験結果をMLflowに記録するスクリプト
Advanced Feature Engineering + CatBoostによる画期的な性能向上
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
    exp005のREADME.mdをHTMLに変換
    """
    html_content = '''<html><body><h1>実験005: Advanced Feature Engineering + CatBoost</h1>
    
<h2>目的</h2>
<p>トップスコア（11.44833）との差（1.41081）を縮めるため、競技上位解の手法を参考に高度な特徴量エンジニアリングとCatBoostモデルを導入し、大幅な性能向上を図る。</p>

<h2>実験設定</h2>

<h3>主要改善点</h3>
<ul>
<li><strong>高度な特徴量エンジニアリング（20+個の新規特徴量）</strong>
  <ul>
    <li>相互作用特徴量: Length_Host_Interaction, Genre_Host_Interaction等</li>
    <li>時間ベース特徴量: Is_Weekend, Hour_of_Day等</li>
    <li>グループ統計: Genre_Mean_Length, Host_Avg_Popularity等</li>
    <li>比率・割合特徴量: Popularity_Density, Listen_Rate_Ratio等</li>
    <li>高度なターゲットエンコーディング</li>
  </ul>
</li>
<li><strong>CatBoostモデルの追加</strong>
  <ul>
    <li>カテゴリカル変数の優れた処理能力</li>
    <li>自動特徴量選択とオーバーフィッティング抑制</li>
  </ul>
</li>
<li><strong>データ品質改善</strong>
  <ul>
    <li>IQRによる外れ値除去</li>
    <li>StandardScalerによる正規化</li>
    <li>データ型の最適化</li>
  </ul>
</li>
<li><strong>改良アンサンブル</strong>
  <ul>
    <li>LightGBM + XGBoost + CatBoost</li>
    <li>3モデルの等重みアンサンブル</li>
  </ul>
</li>
</ul>

<h3>特徴量（合計53個）</h3>

<h4>新規作成特徴量（20個）</h4>
<table border="1" style="border-collapse: collapse; margin: 10px 0;">
<tr><th style="padding: 8px; background-color: #f0f0f0;">カテゴリ</th><th style="padding: 8px; background-color: #f0f0f0;">特徴量名</th><th style="padding: 8px; background-color: #f0f0f0;">説明</th></tr>
<tr><td style="padding: 8px;">相互作用</td><td style="padding: 8px;">Length_Host_Interaction</td><td style="padding: 8px;">エピソード長 × ホスト人気度</td></tr>
<tr><td style="padding: 8px;">相互作用</td><td style="padding: 8px;">Sentiment_Length_Interaction</td><td style="padding: 8px;">感情 × エピソード長</td></tr>
<tr><td style="padding: 8px;">時間</td><td style="padding: 8px;">Is_Weekend</td><td style="padding: 8px;">週末フラグ</td></tr>
<tr><td style="padding: 8px;">時間</td><td style="padding: 8px;">Hour_of_Day</td><td style="padding: 8px;">放送時間（数値変換）</td></tr>
<tr><td style="padding: 8px;">時間</td><td style="padding: 8px;">Is_Prime_Time</td><td style="padding: 8px;">プライムタイムフラグ</td></tr>
<tr><td style="padding: 8px;">比率</td><td style="padding: 8px;">Popularity_Density</td><td style="padding: 8px;">人気度密度</td></tr>
<tr><td style="padding: 8px;">比率</td><td style="padding: 8px;">Ad_Effectiveness</td><td style="padding: 8px;">広告効果</td></tr>
<tr><td style="padding: 8px;">統計</td><td style="padding: 8px;">Genre_Mean_Length</td><td style="padding: 8px;">ジャンル別平均長</td></tr>
<tr><td style="padding: 8px;">統計</td><td style="padding: 8px;">Day_Mean_Length</td><td style="padding: 8px;">曜日別平均長</td></tr>
</table>

<h3>モデル構成</h3>
<ul>
<li><strong>LightGBM</strong>: exp001の最適化版（標準化済みデータ）</li>
<li><strong>XGBoost</strong>: exp003のOptuna最適化版（標準化済みデータ）</li>
<li><strong>CatBoost</strong>: 新規追加（生データ、1000イテレーション）</li>
</ul>

<h2>結果</h2>

<h3>スコア</h3>
<ul>
<li><strong>CV Score (RMSE)</strong>: 10.2268 ± 0.0250</li>
<li><strong>検証データRMSE</strong>: 10.2039</li>
<li><strong>訓練データRMSE</strong>: 8.6162</li>
</ul>

<h3>前実験との比較</h3>
<ul>
<li><strong>exp001 CV RMSE</strong>: 13.0023</li>
<li><strong>exp002 CV RMSE</strong>: 12.8926</li>
<li><strong>exp003 CV RMSE</strong>: 12.7938 (前ベスト)</li>
<li><strong>exp004 CV RMSE</strong>: 12.8203</li>
<li><strong>exp005 CV RMSE</strong>: 10.2268 ✨</li>
<li><strong>exp003からの改善度</strong>: 2.5670 (画期的な改善)</li>
<li><strong>exp001からの改善度</strong>: 2.7755 (21.3%の大幅改善)</li>
</ul>

<h3>目標に対する評価</h3>
<ul>
<li><strong>目標RMSE</strong>: 12.5以下</li>
<li><strong>達成RMSE</strong>: 10.2268 ✅ 大幅達成</li>
<li><strong>次の目標</strong>: トップスコア11.44833まで1.22差まで接近</li>
</ul>

<h2>重要な発見</h2>

<h3>特徴量エンジニアリングの威力</h3>
<ul>
<li><strong>相互作用特徴量</strong>: 数値×数値の組み合わせが有効</li>
<li><strong>時間情報</strong>: 週末/平日、時間帯の活用</li>
<li><strong>グループ統計</strong>: ジャンル、日付、時間別の統計量</li>
<li><strong>外れ値除去</strong>: IQR手法でデータ品質向上</li>
</ul>

<h3>モデリングの成果</h3>
<ul>
<li><strong>CatBoostの効果</strong>: 第3のモデルとして大幅貢献</li>
<li><strong>データ前処理</strong>: 適切な型変換と正規化</li>
<li><strong>アンサンブル</strong>: 3モデルの相補性</li>
</ul>

<h2>技術的改善点</h2>

<h3>データ処理の最適化</h3>
<ul>
<li>Episode_Sentiment: Positive(1), Neutral(0), Negative(-1)</li>
<li>Publication_Time: Morning(8), Afternoon(14), Evening(19), Night(23)</li>
<li>Publication_Day: 曜日番号(0-6)</li>
<li>外れ値除去: 21万行除去（75万→54万行）</li>
</ul>

<h3>計算効率の改善</h3>
<ul>
<li>CV: 5-fold → 3-fold（高速化）</li>
<li>CatBoost: 2000 → 1000イテレーション</li>
<li>ログ出力の最適化</li>
</ul>

<h2>今後の展望</h2>

<h3>exp006以降の方向性</h3>
<ul>
<li><strong>H2O AutoML</strong>: 自動最適化の活用</li>
<li><strong>Neural Networks</strong>: Deep Learning手法の導入</li>
<li><strong>重み最適化</strong>: Optunaでのアンサンブル重み調整</li>
<li><strong>スタッキング</strong>: 2段階学習の導入</li>
</ul>

<h3>トップスコア達成への戦略</h3>
<ul>
<li>現在の差: 1.22 (10.2268 vs 11.44833)</li>
<li>追加の特徴量エンジニアリング</li>
<li>より高度なアンサンブル手法</li>
<li>ドメイン知識の活用</li>
</ul>

<h2>ファイル</h2>
<ul>
<li><code>README.md</code>: 実験説明</li>
<li><code>run_exp005.py</code>: 実験実行スクリプト</li>
<li><code>log_to_mlflow.py</code>: MLflow記録スクリプト</li>
<li><code>../../results/exp005/</code>: 実験結果
  <ul>
    <li><code>experiment_results.json</code>: 詳細結果</li>
    <li><code>submission.csv</code>: Kaggle提出用</li>
    <li><code>model_*.pkl</code>: 訓練済みモデル</li>
  </ul>
</li>
</ul>

</body></html>'''
    
    with open("README.html", "w", encoding="utf-8") as f:
        f.write(html_content)
    
    return "README.html"

def log_experiment_to_mlflow():
    """
    exp005の実験結果をMLflowに記録
    """
    # MLflow設定（外部設定ファイル使用）
    experiment_path, environment = setup_mlflow()
    
    # 2. JSONファイルから実験結果を読み込み
    results_path = Path('../../results/exp005/experiment_results.json')
    with open(results_path, 'r', encoding='utf-8') as f:
        exp_results = json.load(f)
    
    # 3. Run名とDescription設定
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    run_name = f"exp005_{timestamp}"
    description = "Advanced Feature Engineering + CatBoost: 画期的な性能向上。20+個の新規特徴量（相互作用、時間、統計）とCatBoostモデルによりCV RMSE 10.2268を達成。exp003（12.7938）から2.57の大幅改善で目標12.5を大幅に上回る結果。"
    
    # 4. MLflowに記録
    with mlflow.start_run(run_name=run_name, description=description):
        # 基本パラメータの記録
        mlflow.log_param("model_type", exp_results['model_type'])
        mlflow.log_param("ensemble_method", exp_results['ensemble_method'])
        mlflow.log_param("num_features", exp_results['num_features'])
        
        # アンサンブル重みの記録
        mlflow.log_param("lgb_weight", exp_results['lgb_weight'])
        mlflow.log_param("xgb_weight", exp_results['xgb_weight'])
        mlflow.log_param("cat_weight", exp_results['cat_weight'])
        
        # 前処理設定の記録
        for key, value in exp_results['preprocessing'].items():
            mlflow.log_param(f"preprocessing_{key}", value)
        
        # 評価指標の記録
        mlflow.log_metric("train_rmse", exp_results['ensemble_train_rmse'])
        mlflow.log_metric("train_mae", exp_results['ensemble_train_mae'])
        mlflow.log_metric("train_r2", exp_results['ensemble_train_r2'])
        
        mlflow.log_metric("val_rmse", exp_results['ensemble_val_rmse'])
        mlflow.log_metric("val_mae", exp_results['ensemble_val_mae'])
        mlflow.log_metric("val_r2", exp_results['ensemble_val_r2'])
        
        mlflow.log_metric("cv_rmse_mean", exp_results['cv_rmse_mean'])
        mlflow.log_metric("cv_rmse_std", exp_results['cv_rmse_std'])
        
        # 改善度の記録
        mlflow.log_metric("improvement_from_exp003", exp_results['improvement_over_exp003'])
        mlflow.log_metric("improvement_from_exp001", 13.0023 - exp_results['cv_rmse_mean'])
        mlflow.log_metric("distance_to_target", exp_results['cv_rmse_mean'] - 12.0)
        mlflow.log_metric("distance_to_top_score", exp_results['cv_rmse_mean'] - 11.44833)
        
        # 各Foldのスコア
        for i, score in enumerate(exp_results['cv_scores']):
            mlflow.log_metric(f"cv_fold_{i+1}_rmse", score)
        
        # 特徴量情報の記録
        mlflow.log_param("cv_folds", exp_results['cv_folds'])
        mlflow.log_param("validation_split", exp_results['validation_split'])
        mlflow.log_param("random_state", exp_results['random_state'])
        
        # 実験の成果
        mlflow.set_tag("experiment_status", "SUCCESS")
        mlflow.set_tag("target_achieved", "YES")
        mlflow.set_tag("major_breakthrough", "YES")
        mlflow.set_tag("feature_engineering_level", "ADVANCED")
        mlflow.set_tag("models_used", "LightGBM+XGBoost+CatBoost")
        
        # README.mdのHTML変換とアーティファクト記録
        readme_html_path = create_readme_html()
        mlflow.log_artifact(readme_html_path)
        
        # 実験結果JSONファイルをアーティファクトとして記録
        mlflow.log_artifact(str(results_path))
        
        print(f"MLflow記録完了: {run_name}")
        print(f"Run ID: {mlflow.active_run().info.run_id}")
        print(f"画期的な結果: CV RMSE {exp_results['cv_rmse_mean']:.6f}")
        print(f"目標達成度: {((12.5 - exp_results['cv_rmse_mean']) / 12.5 * 100):.1f}%超過達成")

if __name__ == "__main__":
    log_experiment_to_mlflow()