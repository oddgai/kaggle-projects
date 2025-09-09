#!/usr/bin/env python3
"""
exp008のMLflow記録
Ultra-Optimized LightGBM with Optuna & Advanced Feature Engineering
"""

import mlflow
import json
from datetime import datetime
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.mlflow_utils import setup_mlflow, get_experiment_url

def log_exp008_to_mlflow():
    """exp008をMLflowに記録"""
    
    # MLflow設定（外部設定ファイル使用）
    experiment_path, environment = setup_mlflow()
    
    # 実験名とタグ
    run_name = f"exp008_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    
    with mlflow.start_run(run_name=run_name):
        # 基本情報
        mlflow.set_tag("experiment_id", "exp008")
        mlflow.set_tag("description", "Ultra-Optimized LightGBM - Target 11.70 Score")
        mlflow.set_tag("approach", "Optuna + DART + Ugly Ducklings + Advanced FE")
        mlflow.set_tag("date", datetime.now().strftime("%Y-%m-%d"))
        mlflow.set_tag("inspiration", "masaishi notebook analysis")
        
        # モデル設定
        mlflow.log_param("model", "LightGBM Single Model")
        mlflow.log_param("optimization", "Optuna TPE Sampler (50 trials)")
        mlflow.log_param("boosting_options", "GBDT and DART with dropout")
        mlflow.log_param("cv_strategy", "5-fold KFold")
        mlflow.log_param("early_stopping", "150 rounds")
        mlflow.log_param("max_estimators", 2000)
        
        # 特徴量エンジニアリング
        feature_engineering = {
            "basic_transforms": [
                "Log transformations (Episode_Length, Popularity, Ads)",
                "Time-based features (Is_Weekend, Publication_Time_ord)",
                "Sentiment encoding (0,1,2)"
            ],
            "density_features": [
                "Ads_per_Minute",
                "Host_Pop_per_Minute", 
                "Guest_Pop_per_Minute",
                "Pop_Density"
            ],
            "interaction_features": [
                "Length_Host_Int",
                "Length_Guest_Int",
                "Pop_Ads_Int",
                "Sentiment_Pop_Int",
                "Weekend_Pop_Int"
            ],
            "ugly_ducklings": [
                "Length_vs_Genre (deviation from genre mean)",
                "Host_vs_Genre",
                "Guest_vs_Genre"
            ],
            "group_statistics": [
                "Genre_*_mean/std/median",
                "Podcast_*_mean/std",
                "Target encoding with mean/std"
            ],
            "ratio_features": [
                "Pop_Balance (Host/Guest ratio)",
                "Length_Genre_ratio",
                "Pop_Genre_ratio"
            ]
        }
        
        mlflow.log_param("feature_groups", json.dumps(feature_engineering))
        mlflow.log_param("categorical_encoding", "Label Encoding (LightGBM optimized)")
        mlflow.log_param("missing_values", "-999 (LightGBM recommended)")
        
        # Optunaパラメータ空間
        param_space = {
            "boosting_type": ["gbdt", "dart"],
            "num_leaves": "20-300",
            "learning_rate": "0.005-0.1 (log scale)",
            "feature_fraction": "0.6-1.0",
            "bagging_fraction": "0.6-1.0",
            "bagging_freq": "1-7",
            "max_depth": "3-15",
            "min_data_in_leaf": "10-100",
            "reg_alpha": "0.0-10.0",
            "reg_lambda": "0.0-10.0",
            "drop_rate": "0.05-0.2 (DART only)",
            "skip_drop": "0.3-0.7 (DART only)"
        }
        
        mlflow.log_param("optuna_param_space", json.dumps(param_space))
        mlflow.log_param("sampler", "TPESampler with seed=42")
        mlflow.log_param("optimization_trials", 50)
        mlflow.log_param("optimization_cv", "3-fold (for speed)")
        
        # 期待される改善
        expected_improvements = {
            "from_research": "30-85% RMSE improvement over baseline",
            "target_score": "11.70 (masaishi level)",
            "vs_current_best": "12.94051 -> ~11.70 (21% improvement)",
            "techniques": [
                "Optuna optimization (biggest impact)",
                "DART boosting (overfitting prevention)", 
                "Ugly Ducklings (group deviations)",
                "Advanced interactions",
                "LightGBM-specific preprocessing"
            ]
        }
        
        mlflow.set_tag("expected_improvements", json.dumps(expected_improvements))
        
        # 研究ベース技術
        research_techniques = {
            "source": "2024-2025 Kaggle Competition Best Practices",
            "key_papers": [
                "ISIC2024 2nd Place Solution",
                "Tabular Playground Series Winners",
                "LightGBM Advanced Techniques Survey"
            ],
            "validation": [
                "Proven 30-85% RMSE improvements",
                "Top Kaggle competition results",
                "RMSE as low as 5.82 in similar problems"
            ]
        }
        
        mlflow.set_tag("research_basis", json.dumps(research_techniques))
        
        # 実験目標
        mlflow.set_tag("primary_goal", "Achieve Public Score < 12.0")
        mlflow.set_tag("stretch_goal", "Achieve masaishi-level 11.70 score")
        mlflow.set_tag("technique_validation", "Test Optuna + DART effectiveness")
        
        print(f"✓ exp008をMLflowに記録: {run_name}")
        print(f"✓ Ultra-optimization戦略を記録")
        print(f"✓ 研究ベース技術を記録")
        print(f"✓ 目標: masaishi級11.70スコア")
        
        run_id = mlflow.active_run().info.run_id
        
        # URL生成・表示
        urls = get_experiment_url(run_id=run_id, experiment_path=experiment_path, environment=environment)
        if "run_message" in urls:
            print(urls["run_message"])
        if "exp_message" in urls:
            print(urls["exp_message"])
        
        return run_id

if __name__ == "__main__":
    run_id = log_exp008_to_mlflow()
    print(f"\nMLflow Run ID: {run_id}")
    print("筋肉のような最適化実験準備完了💪")