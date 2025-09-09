#!/usr/bin/env python3
"""
汎用MLflow更新スクリプト - 任意の実験のKaggle結果をMLflowに記録
"""

import mlflow
import json
import argparse
import sys
import os
from pathlib import Path
from typing import Dict, List, Optional, Any

# パッケージパスを追加（直接実行時の対応）
sys.path.append(os.path.dirname(__file__))
from mlflow_utils import setup_mlflow, get_experiment_url

class MLflowExperimentUpdater:
    def __init__(self):
        # MLflow設定（外部設定ファイル使用）
        self.experiment_name, self.environment = setup_mlflow()
        
        # 実験間比較用の基準データ（exp007追加）
        self.reference_scores = {
            'exp003': {'public': 12.94051, 'private': 12.85914, 'cv': 12.7938},
            'exp004': {'public': 12.96794, 'private': 12.87623, 'cv': 12.8203},
            'exp005': {'public': 13.10978, 'private': 13.00859, 'cv': 10.2268},
            'exp006': {'public': 13.13878, 'private': 13.03079, 'cv': 12.623664},
            'exp007': {'public': 13.01865, 'private': 12.90607, 'cv': 12.943302}
        }
        
        self.top_kaggle_score = 11.44833
    
    def find_experiment_run(self, experiment_id: str) -> Optional[Dict]:
        """指定実験のMLflow runを検索"""
        experiment = mlflow.get_experiment_by_name(self.experiment_name)
        all_runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["start_time DESC"]
        )
        
        for _, run in all_runs.iterrows():
            run_name = run.get('tags.mlflow.runName', '')
            if experiment_id in run_name:
                print(f"{experiment_id} run発見: {run_name} (ID: {run['run_id']})")
                return run.to_dict()
        
        print(f"{experiment_id}のrunが見つかりませんでした")
        return None
    
    def update_experiment_results(self, experiment_id: str, kaggle_results: Dict[str, Any], 
                                analysis_config: Optional[Dict] = None):
        """
        実験結果をMLflowに記録
        
        Args:
            experiment_id: 実験ID (例: "exp007")
            kaggle_results: Kaggle結果データ
            analysis_config: 追加分析設定
        """
        run_data = self.find_experiment_run(experiment_id)
        if not run_data:
            return False
        
        run_id = run_data['run_id']
        
        with mlflow.start_run(run_id=run_id):
            # 基本Kaggleスコア記録
            public_score = kaggle_results.get('public_score')
            private_score = kaggle_results.get('private_score')
            cv_rmse = kaggle_results.get('cv_rmse')
            
            if public_score:
                mlflow.log_metric("kaggle_public_score", public_score)
            if private_score:
                mlflow.log_metric("kaggle_private_score", private_score)
            
            # CV-Kaggle Gap分析
            if public_score and cv_rmse:
                gap = public_score - cv_rmse
                mlflow.log_metric("cv_kaggle_gap", gap)
                mlflow.log_metric("cv_to_public_ratio", public_score / cv_rmse)
                
                # オーバーフィッティング判定
                is_overfitting = gap > 0.5
                mlflow.set_tag("overfitting_detected", "YES" if is_overfitting else "NO")
            
            # 予想精度分析
            predicted_public = kaggle_results.get('predicted_public_score')
            if predicted_public and public_score:
                prediction_error = abs(public_score - predicted_public)
                mlflow.log_metric("prediction_error", prediction_error)
                
                accuracy_level = "HIGH" if prediction_error <= 0.2 else "MEDIUM" if prediction_error <= 0.5 else "LOW"
                mlflow.set_tag("prediction_accuracy", accuracy_level)
            
            # 実験間比較
            if public_score:
                self._log_experiment_comparisons(experiment_id, public_score)
            
            # 基本タグ設定
            mlflow.set_tag("kaggle_submitted", "true")
            if public_score:
                mlflow.set_tag("kaggle_public_score", str(public_score))
            if private_score:
                mlflow.set_tag("kaggle_private_score", str(private_score))
            
            # 追加分析（設定で指定）
            if analysis_config:
                self._apply_custom_analysis(analysis_config, kaggle_results)
            
            print(f"✓ {experiment_id} MLflow更新完了")
            if public_score:
                print(f"✓ Public Score: {public_score}")
            if private_score:
                print(f"✓ Private Score: {private_score}")
            if cv_rmse and public_score:
                print(f"✓ CV-Kaggle Gap: {public_score - cv_rmse:.6f}")
        
        return True
    
    def _log_experiment_comparisons(self, current_exp: str, public_score: float):
        """他実験との比較メトリクス記録"""
        for exp_id, scores in self.reference_scores.items():
            if exp_id != current_exp and 'public' in scores:
                diff = public_score - scores['public']
                mlflow.log_metric(f"vs_{exp_id}_public", diff)
        
        # 最高記録チェック
        all_public_scores = [scores.get('public', float('inf')) for scores in self.reference_scores.values()]
        is_best = public_score < min(all_public_scores)
        mlflow.set_tag("new_record", "YES" if is_best else "NO")
        
        # トップスコアとの差
        distance_to_top = public_score - self.top_kaggle_score
        mlflow.log_metric("distance_to_kaggle_top", distance_to_top)
        
        # 目標達成
        target_achieved = public_score < 12.0
        mlflow.set_tag("target_achieved", "YES" if target_achieved else "NO")
    
    def _apply_custom_analysis(self, config: Dict, results: Dict):
        """カスタム分析の適用"""
        # 追加メトリクス
        if 'custom_metrics' in config:
            for metric_name, value in config['custom_metrics'].items():
                mlflow.log_metric(metric_name, value)
        
        # 追加タグ
        if 'custom_tags' in config:
            for tag_name, value in config['custom_tags'].items():
                mlflow.set_tag(tag_name, str(value))
        
        # 学習事項
        if 'lessons_learned' in config:
            lessons = config['lessons_learned']
            if isinstance(lessons, list):
                mlflow.set_tag("lessons_learned", "; ".join(lessons))
            else:
                mlflow.set_tag("lessons_learned", str(lessons))
        
        # 改善提案
        if 'next_improvements' in config:
            improvements = config['next_improvements']
            if isinstance(improvements, list):
                mlflow.set_tag("next_improvements", "; ".join(improvements))
            else:
                mlflow.set_tag("next_improvements", str(improvements))
        
        # 戦略評価
        if 'strategy_evaluation' in config:
            eval_data = config['strategy_evaluation']
            for key, value in eval_data.items():
                mlflow.set_tag(f"strategy_{key}", str(value))

def load_kaggle_results(experiment_id: str) -> Optional[Dict]:
    """実験のKaggle結果JSONを読み込み"""
    # resultsディレクトリから読み込み
    result_file = Path(f"results/{experiment_id}/kaggle_result.json")
    if result_file.exists():
        with open(result_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    # 後方互換性のため、旧形式も試行
    legacy_file = Path(f"{experiment_id}_kaggle_result.json")
    if legacy_file.exists():
        with open(legacy_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    return None

def main():
    parser = argparse.ArgumentParser(description='実験結果をMLflowに記録')
    parser.add_argument('experiment_id', help='実験ID (例: exp007)')
    parser.add_argument('--kaggle-results', help='Kaggle結果JSONファイルパス (デフォルト: {experiment_id}_kaggle_result.json)')
    parser.add_argument('--analysis-config', help='追加分析設定JSONファイルパス')
    parser.add_argument('--public-score', type=float, help='Public Score (JSON未使用時)')
    parser.add_argument('--private-score', type=float, help='Private Score (JSON未使用時)')
    parser.add_argument('--cv-rmse', type=float, help='CV RMSE (JSON未使用時)')
    parser.add_argument('--predicted-public', type=float, help='予想Public Score (JSON未使用時)')
    
    args = parser.parse_args()
    
    updater = MLflowExperimentUpdater()
    
    # Kaggle結果の取得
    if args.kaggle_results:
        with open(args.kaggle_results, 'r', encoding='utf-8') as f:
            kaggle_results = json.load(f)
    elif args.public_score or args.cv_rmse:
        kaggle_results = {
            'public_score': args.public_score,
            'private_score': args.private_score,
            'cv_rmse': args.cv_rmse,
            'predicted_public_score': args.predicted_public
        }
    else:
        kaggle_results = load_kaggle_results(args.experiment_id)
        if not kaggle_results:
            print(f"Kaggle結果が見つかりません: {args.experiment_id}")
            print(f"期待される場所: results/{args.experiment_id}/kaggle_result.json")
            print("--public-score等でスコアを直接指定するか、JSONファイルを用意してください")
            return
    
    # 追加分析設定の取得
    analysis_config = None
    if args.analysis_config:
        with open(args.analysis_config, 'r', encoding='utf-8') as f:
            analysis_config = json.load(f)
    
    # MLflow更新実行
    success = updater.update_experiment_results(
        args.experiment_id, 
        kaggle_results, 
        analysis_config
    )
    
    if success:
        print(f"\n=== {args.experiment_id} MLflow更新完了 ===")
    else:
        print(f"\n=== {args.experiment_id} MLflow更新失敗 ===")

if __name__ == "__main__":
    main()