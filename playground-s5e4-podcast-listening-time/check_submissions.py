#!/usr/bin/env python3
"""
Kaggleの提出履歴を確認してスコアを取得
"""

from kaggle.api.kaggle_api_extended import KaggleApi
import json
from datetime import datetime

def check_recent_submissions(competition='playground-series-s5e4', count=5):
    """
    最近の提出履歴を確認
    """
    # API初期化
    api = KaggleApi()
    api.authenticate()
    
    print(f"=== 最近の{count}件の提出履歴 ===\n")
    
    # 提出履歴を取得
    submissions = api.competition_submissions(competition)
    
    results = []
    for i, sub in enumerate(submissions[:count]):
        print(f"提出 {i+1}:")
        print(f"  日時: {sub.date}")
        print(f"  メッセージ: {sub.description}")
        print(f"  Public Score: {sub.public_score}")
        print(f"  Private Score: {sub.private_score if sub.private_score else 'N/A'}")
        print(f"  ステータス: {sub.status}")
        print()
        
        results.append({
            'date': str(sub.date),
            'description': sub.description,
            'public_score': float(sub.public_score) if sub.public_score else None,
            'private_score': float(sub.private_score) if sub.private_score else None,
            'status': sub.status
        })
    
    # 結果を保存
    with open('submission_history.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"履歴をsubmission_history.jsonに保存しました")
    
    # exp003の結果を探す
    exp003_score = None
    for sub in results:
        if 'exp003' in sub['description']:
            exp003_score = sub['public_score']
            print(f"\n✓ exp003のPublic Score: {exp003_score}")
            break
    
    return results

if __name__ == "__main__":
    check_recent_submissions()