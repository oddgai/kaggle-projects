#!/usr/bin/env python3
"""
汎用実験提出スクリプト - 任意の実験をKaggleに提出して分析
"""

from kaggle.api.kaggle_api_extended import KaggleApi
import json
import time
import argparse
from pathlib import Path

class ExperimentSubmitter:
    def __init__(self, competition='playground-series-s5e4'):
        self.competition = competition
        self.api = KaggleApi()
        self.api.authenticate()
        
        # 過去の実験結果（更新が必要）
        self.past_experiments = {
            'exp003': {'public': 12.94051, 'cv_gap': 0.15},
            'exp004': {'public': 12.96794, 'cv_gap': 0.15}, 
            'exp005': {'public': 13.10978, 'cv_gap': 2.88},
            'exp006': {'public': 13.13878, 'cv_gap': 0.515}
        }
        
        self.top_score = 11.44833
        
    def submit_experiment(self, experiment_id, submission_file, cv_rmse, 
                         predicted_public=None, approach_description="", message_suffix=""):
        """
        実験を提出して詳細分析を実行
        """
        # 提出メッセージ作成
        if message_suffix:
            message = f"{experiment_id}: {approach_description} {message_suffix}"
        else:
            message = f"{experiment_id}: {approach_description} (CV RMSE: {cv_rmse})"
        
        print(f"=== {experiment_id}の提出 ===")
        print(f"ファイル: {submission_file}")
        print(f"CV RMSE: {cv_rmse}")
        if predicted_public:
            print(f"予想Public Score: {predicted_public}")
        
        # 提出実行
        self.api.competition_submit(
            file_name=submission_file,
            message=message,
            competition=self.competition
        )
        
        print("提出完了！スコア取得待機中...")
        time.sleep(30)
        
        # 結果取得
        submissions = self.api.competition_submissions(self.competition)
        
        if not submissions:
            print("提出結果を取得できませんでした")
            return None
            
        latest = submissions[0]
        actual_public = float(latest.public_score) if latest.public_score else None
        actual_private = float(latest.private_score) if latest.private_score else None
        
        print(f"\n=== 提出結果 ===")
        print(f"Public Score: {actual_public}")
        print(f"Private Score: {actual_private if actual_private else 'N/A'}")
        print(f"提出日時: {latest.date}")
        print(f"ステータス: {latest.status}")
        
        # 詳細分析
        if actual_public:
            self._analyze_results(experiment_id, cv_rmse, predicted_public, 
                                actual_public, approach_description)
        
        # 結果データ作成
        result = {
            'experiment_id': experiment_id,
            'public_score': actual_public,
            'private_score': actual_private,
            'submission_date': str(latest.date),
            'status': str(latest.status),
            'cv_rmse': cv_rmse,
            'predicted_public_score': predicted_public,
            'prediction_error': abs(actual_public - predicted_public) if predicted_public and actual_public else None,
            'actual_cv_kaggle_gap': actual_public - cv_rmse if actual_public else None,
            'approach': approach_description
        }
        
        # 結果保存（resultsディレクトリに保存）
        result_dir = Path(f"results/{experiment_id}")
        result_dir.mkdir(parents=True, exist_ok=True)
        result_file = result_dir / "kaggle_result.json"
        
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        print(f"\n結果を{result_file}に保存しました")
        return result
    
    def _analyze_results(self, experiment_id, cv_rmse, predicted_public, actual_public, approach):
        """
        結果の詳細分析
        """
        actual_gap = actual_public - cv_rmse
        
        print(f"\n=== {experiment_id}分析 ===")
        print(f"CV RMSE: {cv_rmse:.6f}")
        if predicted_public:
            prediction_error = abs(actual_public - predicted_public)
            print(f"予想Public Score: {predicted_public:.6f}")
            print(f"実際Public Score: {actual_public:.6f}")
            print(f"予想誤差: {prediction_error:.6f}")
        print(f"CV-Kaggle Gap: {actual_gap:.6f}")
        
        # 目標達成チェック
        if actual_gap <= 0.3:
            print("✅ Gap目標達成: < 0.3")
        else:
            print(f"❌ Gap目標未達: {actual_gap:.3f} >= 0.3")
            
        if predicted_public and abs(actual_public - predicted_public) <= 0.2:
            print("✅ 予想精度: 高精度")
        elif predicted_public:
            print("❌ 予想精度: 予想から外れ")
        
        # 過去実験との比較
        print(f"\n=== 過去実験比較 ===")
        for exp_id, data in self.past_experiments.items():
            if exp_id != experiment_id:
                diff = actual_public - data['public']
                symbol = "🔽" if diff < 0 else "🔺"
                print(f"{symbol} vs {exp_id}: {diff:+.5f}")
        
        # 記録更新チェック
        past_scores = [data['public'] for data in self.past_experiments.values()]
        if actual_public < min(past_scores):
            print("🎉 新記録達成！")
        
        # トップスコアとの比較
        distance_to_top = actual_public - self.top_score
        print(f"\nトップスコアとの差: {distance_to_top:.5f}")
        
        # パフォーマンス評価
        if actual_public < 12.0:
            print("🎯 目標達成: < 12.0")
        else:
            print(f"目標まであと: {actual_public - 12.0:.5f}")

def main():
    parser = argparse.ArgumentParser(description='実験をKaggleに提出')
    parser.add_argument('experiment_id', help='実験ID (例: exp007)')
    parser.add_argument('submission_file', help='提出ファイルパス')
    parser.add_argument('cv_rmse', type=float, help='CV RMSE')
    parser.add_argument('--predicted-public', type=float, help='予想Public Score')
    parser.add_argument('--approach', default='', help='アプローチ説明')
    parser.add_argument('--message-suffix', default='', help='提出メッセージ追加')
    
    args = parser.parse_args()
    
    submitter = ExperimentSubmitter()
    result = submitter.submit_experiment(
        args.experiment_id,
        args.submission_file, 
        args.cv_rmse,
        args.predicted_public,
        args.approach,
        args.message_suffix
    )
    
    if result and result['public_score']:
        print(f"\n=== {args.experiment_id}最終結果 ===")
        print(f"Public Score: {result['public_score']}")
        if result['prediction_error']:
            print(f"予想誤差: {result['prediction_error']:.6f}")
        if result['actual_cv_kaggle_gap']:
            print(f"CV-Kaggle Gap: {result['actual_cv_kaggle_gap']:.6f}")

if __name__ == "__main__":
    main()