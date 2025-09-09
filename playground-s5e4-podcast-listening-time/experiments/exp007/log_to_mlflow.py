#!/usr/bin/env python3
"""
exp007のMLflow記録
"""

import mlflow
import pandas as pd
import json
from datetime import datetime
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.mlflow_utils import setup_mlflow, get_experiment_url

def log_exp007_to_mlflow():
    """exp007をMLflowに記録"""
    
    # MLflow設定（外部設定ファイル使用）
    experiment_path, environment = setup_mlflow()
    
    # 実験名とタグ
    run_name = f"exp007_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    
    with mlflow.start_run(run_name=run_name):
        # 基本情報
        mlflow.set_tag("experiment_id", "exp007")
        mlflow.set_tag("description", "Competition Insights Implementation")
        mlflow.set_tag("approach", "Advanced Feature Engineering + XGB/CatBoost/LightGBM Ensemble")
        mlflow.set_tag("date", datetime.now().strftime("%Y-%m-%d"))
        
        # モデル設定
        mlflow.log_param("models", "XGBoost + LightGBM + CatBoost")
        mlflow.log_param("ensemble_method", "Weighted Average (Optimized)")
        mlflow.log_param("cv_strategy", "5-fold KFold")
        mlflow.log_param("outlier_removal", "IQR method (threshold=1.5)")
        mlflow.log_param("scaling", "StandardScaler")
        
        # 特徴量エンジニアリング
        feature_engineering = {
            "time_based": ["Is_Weekend", "Publication_Time_ord"],
            "ad_features": ["Ads_per_Minute"],
            "popularity_features": [
                "Host_Popularity_per_Minute",
                "Guest_Popularity_per_Minute", 
                "Host_Guest_Popularity_Ratio",
                "Total_Popularity",
                "Popularity_Density"
            ],
            "interaction_features": [
                "Length_Host_Interaction",
                "Length_Guest_Interaction",
                "Length_Ads_Interaction",
                "Ads_per_Host_Popularity",
                "Ads_per_Guest_Popularity"
            ],
            "text_features": [
                "Podcast_Name_Length",
                "Episode_Title_Length",
                "Title_Name_Ratio"
            ],
            "group_statistics": [
                "Genre_Mean_Length",
                "Genre_Std_Length",
                "Podcast_Mean_Length",
                "Length_vs_Genre_Mean",
                "Host_Pop_vs_Genre_Mean"
            ],
            "target_encoding": [
                "Genre_Target_Encoded",
                "Podcast_Target_Encoded"
            ]
        }
        
        mlflow.log_param("feature_groups", json.dumps(feature_engineering))
        mlflow.log_param("total_features", "50+")
        
        # ハイパーパラメータ
        xgb_params = {
            "max_depth": 8,
            "learning_rate": 0.05,
            "n_estimators": 1000,
            "subsample": 0.8,
            "colsample_bytree": 0.8
        }
        
        lgb_params = {
            "num_leaves": 31,
            "learning_rate": 0.05,
            "n_estimators": 1000,
            "subsample": 0.8,
            "colsample_bytree": 0.8
        }
        
        cat_params = {
            "iterations": 1000,
            "learning_rate": 0.05,
            "depth": 8,
            "early_stopping_rounds": 50
        }
        
        mlflow.log_params({f"xgb_{k}": v for k, v in xgb_params.items()})
        mlflow.log_params({f"lgb_{k}": v for k, v in lgb_params.items()})
        mlflow.log_params({f"cat_{k}": v for k, v in cat_params.items()})
        
        # Competition Insights
        insights = {
            "data_insights": [
                "Missing data: Episode_Length 11.6%, Guest_Popularity 19.47%",
                "Strong correlation: Episode_Length vs Listening_Time",
                "Negative correlation: Number_of_Ads vs Listening_Time"
            ],
            "key_strategies": [
                "Advanced feature engineering most critical",
                "XGBoost performed best (CV RMSE: 12.93)",
                "Ensemble of gradient boosting models",
                "Group-based aggregations crucial"
            ],
            "improvements_from_discussion": [
                "Ads_per_Minute feature",
                "Popularity density features",
                "Genre and Podcast group statistics",
                "IQR outlier removal",
                "5-fold CV validation"
            ]
        }
        
        mlflow.set_tag("competition_insights", json.dumps(insights))
        
        # 期待される結果
        mlflow.set_tag("expected_improvement", "CV RMSE < 12.8 (beat exp003/004)")
        mlflow.set_tag("strategy_source", "Kaggle Discussion Analysis")
        
        print(f"✓ exp007をMLflowに記録: {run_name}")
        print(f"✓ Competition Insights戦略を記録")
        print(f"✓ 高度な特徴量エンジニアリング設定を記録")
        
        run_id = mlflow.active_run().info.run_id
        
        # URL生成・表示
        urls = get_experiment_url(run_id=run_id, experiment_path=experiment_path, environment=environment)
        if "run_message" in urls:
            print(urls["run_message"])
        if "exp_message" in urls:
            print(urls["exp_message"])
        
        return run_id

if __name__ == "__main__":
    run_id = log_exp007_to_mlflow()
    print(f"\nMLflow Run ID: {run_id}")