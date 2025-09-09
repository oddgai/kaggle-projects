#!/usr/bin/env python3
"""
全てのlog_to_mlflow.pyファイルを一括でアップデート
"""

import os
import re
import glob
from pathlib import Path

def update_imports(content):
    """Importセクションを更新"""
    # 既存のimport mlflowの後に新しいimportを追加
    if "from src.mlflow_utils import setup_mlflow, get_experiment_url" in content:
        print("  - Import already updated")
        return content
    
    # importの追加
    new_imports = """import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.mlflow_utils import setup_mlflow, get_experiment_url"""
    
    # パターンを探してimportを追加
    patterns = [
        (r"(from pathlib import Path)", r"\1\n" + new_imports),
        (r"(from datetime import datetime)", r"\1\n" + new_imports),
        (r"(import mlflow.*?\n)", r"\1" + new_imports + "\n")
    ]
    
    for pattern, replacement in patterns:
        if re.search(pattern, content):
            content = re.sub(pattern, replacement, content)
            print("  - Import added")
            break
    else:
        print("  - Warning: Could not find place to add imports")
    
    return content

def update_mlflow_setup(content):
    """MLflow設定部分を更新"""
    
    # 既存のsetup_mlflow()が使用されているかチェック
    if "setup_mlflow()" in content:
        print("  - MLflow setup already updated")
        return content
    
    # 複数のパターンに対応したMLflow設定の置換
    old_patterns = [
        # パターン1: 詳細な設定
        r'''    # (?:1\. )?(?:Databricks )?MLflow設定.*?\n    try:\n        mlflow\.set_tracking_uri\("databricks"\).*?\n.*?print\("ローカルMLflowに切り替え"\)\n        mlflow\.set_tracking_uri\("file:///tmp/mlruns"\)\s*\n\s*# 実験パスの設定.*?\n.*?experiment_name = "[^"]*"\n    try:\n        mlflow\.set_experiment\(experiment_name\).*?\n.*?mlflow\.set_experiment\("[^"]*"\)\n        print\("ローカル実験名に設定"\)''',
        
        # パターン2: シンプルな設定
        r'''    try:\n        mlflow\.set_tracking_uri\("databricks"\).*?\n.*?mlflow\.set_tracking_uri\("file:///tmp/mlruns"\)\s*\n\s*experiment_name = "[^"]*"\n    try:\n        mlflow\.set_experiment\(experiment_name\).*?\n.*?mlflow\.set_experiment\("[^"]*"\).*?'''
    ]
    
    new_setup = """    # MLflow設定（外部設定ファイル使用）
    experiment_path, environment = setup_mlflow()"""
    
    for pattern in old_patterns:
        if re.search(pattern, content, re.DOTALL):
            content = re.sub(pattern, new_setup, content, flags=re.DOTALL)
            print("  - MLflow setup updated")
            return content
    
    # 単純なパターンも試す
    simple_patterns = [
        r'''    # Databricks MLflow設定.*?\n.*?mlflow\.set_experiment\("[^"]*"\)\n        print\("ローカル実験名に設定"\)''',
        r'''    try:\n        mlflow\.set_tracking_uri\("databricks"\).*?print\("ローカル実験名に設定"\)''',
    ]
    
    for pattern in simple_patterns:
        if re.search(pattern, content, re.DOTALL):
            content = re.sub(pattern, new_setup, content, flags=re.DOTALL)
            print("  - MLflow setup updated (simple pattern)")
            return content
    
    print("  - Warning: Could not find MLflow setup to replace")
    return content

def update_experiment_file(file_path):
    """単一の実験ファイルを更新"""
    print(f"\nUpdating: {file_path}")
    
    # exp006, exp007は既に更新済み
    if 'exp006' in str(file_path) or 'exp007' in str(file_path):
        print("  - Already updated, skipping")
        return
    
    try:
        # ファイル読み込み
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Import更新
        content = update_imports(content)
        
        # MLflow設定更新
        content = update_mlflow_setup(content)
        
        # 変更があった場合のみファイルを保存
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print("  ✓ File updated")
        else:
            print("  - No changes needed")
            
    except Exception as e:
        print(f"  ✗ Error updating {file_path}: {e}")

def main():
    """メイン実行"""
    print("=== MLflow設定一括更新 ===")
    
    # 全てのlog_to_mlflow.pyファイルを検索
    log_files = glob.glob("experiments/*/log_to_mlflow.py")
    
    print(f"Found {len(log_files)} files to update:")
    for file in log_files:
        print(f"  - {file}")
    
    print("\nStarting updates...")
    
    for file_path in log_files:
        update_experiment_file(file_path)
    
    print(f"\n=== 更新完了 ===")
    print("Updated files should now use external MLflow configuration")

if __name__ == "__main__":
    main()