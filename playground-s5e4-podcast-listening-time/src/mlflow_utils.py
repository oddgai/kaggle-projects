#!/usr/bin/env python3
"""
MLflow設定管理用ユーティリティ
"""

import json
import mlflow
from pathlib import Path

def load_mlflow_config(config_path="config/mlflow_config.json"):
    """MLflow設定ファイルを読み込み"""
    config_file = Path(config_path)
    
    if not config_file.exists():
        raise FileNotFoundError(f"MLflow設定ファイルが見つかりません: {config_path}")
    
    with open(config_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def setup_mlflow(config_path="config/mlflow_config.json", environment=None):
    """
    MLflow接続を設定
    
    Args:
        config_path: 設定ファイルパス
        environment: 強制する環境 ("databricks" or "local")
    
    Returns:
        tuple: (experiment_name_or_path, environment_used)
    """
    config = load_mlflow_config(config_path)
    
    # 環境決定
    if environment is None:
        environment = config.get("default_environment", "databricks")
    
    env_config = config.get(environment, {})
    
    if not env_config:
        raise ValueError(f"不明な環境: {environment}")
    
    # Databricks接続試行
    if environment == "databricks":
        try:
            mlflow.set_tracking_uri(env_config["tracking_uri"])
            print("Databricks MLflow接続成功")
            
            experiment_path = env_config["experiment_path"]
            mlflow.set_experiment(experiment_path)
            print(f"実験パス設定: {experiment_path}")
            
            return experiment_path, "databricks"
            
        except Exception as e:
            print(f"Databricks MLflow接続エラー: {e}")
            print("ローカルMLflowに切り替え")
            environment = "local"
            env_config = config.get("local", {})
    
    # ローカル設定
    if environment == "local":
        mlflow.set_tracking_uri(env_config["tracking_uri"])
        print(f"ローカルMLflow設定: {env_config['tracking_uri']}")
        
        experiment_name = env_config["experiment_name"]
        mlflow.set_experiment(experiment_name)
        print(f"ローカル実験名設定: {experiment_name}")
        
        return experiment_name, "local"
    
    raise ValueError(f"MLflow設定に失敗: {environment}")

def get_experiment_url(run_id=None, experiment_path=None, environment="databricks"):
    """
    実験・RunのURLを生成
    
    Args:
        run_id: MLflow Run ID
        experiment_path: 実験パス
        environment: 環境
    
    Returns:
        dict: URLとメッセージ
    """
    if environment == "databricks" and experiment_path:
        # Databricks URL生成
        base_url = "https://dbc-55810bf1-184f.cloud.databricks.com"
        
        if run_id:
            # 実験IDを取得してRunURLを生成
            try:
                experiment = mlflow.get_experiment_by_name(experiment_path)
                run_url = f"{base_url}/ml/experiments/{experiment.experiment_id}/runs/{run_id}"
                exp_url = f"{base_url}/ml/experiments/{experiment.experiment_id}"
                
                return {
                    "run_url": run_url,
                    "experiment_url": exp_url,
                    "run_message": f"🏃 View run at: {run_url}",
                    "exp_message": f"🧪 View experiment at: {exp_url}"
                }
            except Exception as e:
                print(f"URL生成エラー: {e}")
                return {}
        else:
            try:
                experiment = mlflow.get_experiment_by_name(experiment_path)
                exp_url = f"{base_url}/ml/experiments/{experiment.experiment_id}"
                return {
                    "experiment_url": exp_url,
                    "exp_message": f"🧪 View experiment at: {exp_url}"
                }
            except Exception as e:
                print(f"URL生成エラー: {e}")
                return {}
    
    return {"message": "ローカル環境のためURLは生成されません"}

# 使用例
if __name__ == "__main__":
    # 設定テスト
    try:
        experiment_path, env = setup_mlflow()
        print(f"✓ MLflow設定完了: {env} - {experiment_path}")
        
        # URL生成テスト
        urls = get_experiment_url(experiment_path=experiment_path, environment=env)
        if "exp_message" in urls:
            print(urls["exp_message"])
        else:
            print(urls.get("message", "URLテスト完了"))
            
    except Exception as e:
        print(f"✗ MLflow設定エラー: {e}")