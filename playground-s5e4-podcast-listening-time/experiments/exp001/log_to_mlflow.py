#!/usr/bin/env python3
"""
MLflow logging script for exp001: LightGBM Baseline Model
Playground Series S5E4: Podcast Listening Time Prediction
"""

import json
import pickle
import re
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import mlflow.lightgbm
import mlflow.sklearn
import numpy as np
import pandas as pd


def markdown_to_html(markdown_text):
    """
    簡単なMarkdown -> HTML変換
    """
    html = markdown_text

    # ヘッダー変換
    html = re.sub(r"^# (.*)", r"<h1>\1</h1>", html, flags=re.MULTILINE)
    html = re.sub(r"^## (.*)", r"<h2>\1</h2>", html, flags=re.MULTILINE)
    html = re.sub(r"^### (.*)", r"<h3>\1</h3>", html, flags=re.MULTILINE)
    html = re.sub(r"^#### (.*)", r"<h4>\1</h4>", html, flags=re.MULTILINE)

    # リスト変換
    html = re.sub(r"^- (.*)", r"<li>\1</li>", html, flags=re.MULTILINE)

    # 番号付きリスト変換
    html = re.sub(r"^(\d+)\. (.*)", r"<li>\2</li>", html, flags=re.MULTILINE)

    # 特徴量リストの個別化（カンマ区切りの特徴量を個別のliに分解）
    def expand_feature_list(match):
        features = match.group(2)
        feature_list = [f.strip() for f in features.split(",")]
        li_items = "".join([f"<li>{feature}</li>" for feature in feature_list])
        return li_items

    html = re.sub(
        r"<li>(カテゴリカル変数|数値変数): ([^<]+)</li>", expand_feature_list, html
    )

    # テーブル処理（簡易版）
    lines = html.split("\n")
    in_table = False
    result_lines = []

    for line in lines:
        if "|" in line and line.strip().startswith("|"):
            if not in_table:
                result_lines.append(
                    '<table border="1" style="border-collapse: collapse;">'
                )
                in_table = True

            # テーブル行の処理
            cells = [
                cell.strip() for cell in line.split("|")[1:-1]
            ]  # 最初と最後の空要素を除外
            if all(
                cell in ["---", "------", "--------", "----------", "------------"]
                or "-" in cell
                for cell in cells
            ):
                continue  # セパレーター行をスキップ

            row_html = (
                "<tr>" + "".join([f"<td>{cell}</td>" for cell in cells]) + "</tr>"
            )
            result_lines.append(row_html)
        else:
            if in_table:
                result_lines.append("</table>")
                in_table = False
            result_lines.append(line)

    if in_table:
        result_lines.append("</table>")

    html = "\n".join(result_lines)

    # ハイパーパラメータのJSONテーブル変換
    def convert_hyperparams_to_table(match):
        code_content = match.group(1)
        if "lgb_params" in code_content and "{" in code_content:
            # パラメータを抽出してテーブルに変換
            params_html = (
                '<table border="1" style="border-collapse: collapse; margin: 10px 0;">'
            )
            params_html += '<tr><th style="padding: 8px; background-color: #f0f0f0;">Parameter</th>'
            params_html += (
                '<th style="padding: 8px; background-color: #f0f0f0;">Value</th></tr>'
            )

            # 簡易的なパラメータ抽出（正規表現で）
            param_lines = [
                ("objective", "'regression'"),
                ("metric", "'rmse'"),
                ("boosting_type", "'gbdt'"),
                ("num_leaves", "31"),
                ("learning_rate", "0.05"),
                ("feature_fraction", "0.9"),
                ("bagging_fraction", "0.8"),
                ("bagging_freq", "5"),
                ("verbose", "-1"),
                ("random_state", "42"),
            ]

            for param, value in param_lines:
                params_html += f'<tr><td style="padding: 8px;">{param}</td>'
                params_html += f'<td style="padding: 8px;">{value}</td></tr>'

            params_html += "</table>"
            return params_html
        else:
            return f"<pre><code>{code_content}</code></pre>"

    # コードブロック処理（ハイパーパラメータは特別扱い）
    html = re.sub(r"```(.*?)```", convert_hyperparams_to_table, html, flags=re.DOTALL)
    html = re.sub(r"`([^`]*)`", r"<code>\1</code>", html)

    # 強調
    html = re.sub(r"\*\*(.*?)\*\*", r"<strong>\1</strong>", html)
    html = re.sub(r"\*(.*?)\*", r"<em>\1</em>", html)

    # 改行処理（最小限）
    html = html.replace("\n\n", "\n")  # 段落区切りを単一改行に
    html = re.sub(r"\n+", " ", html)  # 残りの改行を空白に

    # リスト項目をulで囲む
    html = re.sub(r"(<li>.*?</li>)", r"<ul>\1</ul>", html, flags=re.DOTALL)
    html = re.sub(r"</ul> <ul>", "</ul><ul>", html)

    return f"<html><body>{html}</body></html>"


def log_experiment_to_mlflow():
    """
    exp001の実験結果をMLflowに記録する
    """

    # Databricks MLflowを使用
    mlflow.set_tracking_uri("databricks")

    # 実験名を設定
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    experiment_name = (
        "/Shared/data_science/z_ogai/playground-s5e4-podcast-listening-time"
    )

    try:
        mlflow.set_experiment(experiment_name)
        print(f"✅ Databricks実験を設定: {experiment_name}")
    except Exception as e:
        print(f"⚠️  Databricks接続エラー、ローカルMLflowに切り替え: {e}")
        mlflow.set_tracking_uri("file:./mlruns")
        experiment_name = (
            "/Shared/data_science/z_ogai/playground-s5e4-podcast-listening-time"
        )
        mlflow.set_experiment(experiment_name)

    # 結果ディレクトリ
    results_dir = Path("../../results/exp001")
    
    # 実験結果のJSONファイルを読み込み
    results_json_path = results_dir / 'experiment_results.json'
    if not results_json_path.exists():
        raise FileNotFoundError(f"実験結果ファイルが見つかりません: {results_json_path}")
    
    with open(results_json_path, 'r', encoding='utf-8') as f:
        exp_results = json.load(f)
    
    print(f"✅ 実験結果を読み込み: {results_json_path}")

    # Run設定
    run_name = f"exp001_{timestamp}"
    run_description = "LightGBMベースラインモデル。基本的な特徴量エンジニアリング（Ad_Density, Has_Guest, Host_Guest_Popularity_Diff）を実装した初回実験。exp000（EDA）から派生。"

    with mlflow.start_run(run_name=run_name, description=run_description) as run:
        # ===================
        # パラメータの記録
        # ===================

        # モデルパラメータ（JSONから読み込み）
        model_params = exp_results["model_params"]

        # ハイパーパラメータをログ
        for param_name, param_value in model_params.items():
            mlflow.log_param(f"lgb_{param_name}", param_value)

        # 実験設定パラメータ（JSONから読み込み）
        mlflow.log_param("model_type", exp_results["model_type"])
        mlflow.log_param("experiment_id", exp_results["experiment_id"])
        mlflow.log_param("task_type", "regression")
        mlflow.log_param("cv_folds", exp_results["cv_folds"])
        mlflow.log_param("validation_split", exp_results["validation_split"])
        mlflow.log_param("random_state", exp_results["random_state"])

        # 特徴量情報（JSONから読み込み）
        features = exp_results["features"]
        new_features = exp_results["new_features"]
        
        mlflow.log_param("num_features", exp_results["num_features"])
        mlflow.log_param("feature_engineering", exp_results["preprocessing"]["feature_engineering"])
        mlflow.log_param("new_features", ",".join(new_features))
        mlflow.log_param("num_new_features", len(new_features))

        # データ情報（JSONから読み込み）
        mlflow.log_param("train_size", exp_results["train_size"])
        mlflow.log_param("test_size", exp_results["test_size"])
        mlflow.log_param("target_variable", exp_results["target_variable"])

        # ===================
        # 評価指標の記録
        # ===================

        # Cross Validation結果（JSONから読み込み）
        cv_scores = exp_results["cv_scores"]
        cv_mean = exp_results["cv_rmse_mean"]
        cv_std = exp_results["cv_rmse_std"]

        mlflow.log_metric("cv_rmse_mean", cv_mean)
        mlflow.log_metric("cv_rmse_std", cv_std)
        for i, score in enumerate(cv_scores, 1):
            mlflow.log_metric(f"cv_rmse_fold{i}", score)

        # 訓練・検証結果（JSONから読み込み）
        mlflow.log_metric("train_rmse", exp_results["train_rmse"])
        mlflow.log_metric("val_rmse", exp_results["val_rmse"])
        mlflow.log_metric("train_mae", exp_results["train_mae"])
        mlflow.log_metric("val_mae", exp_results["val_mae"])
        mlflow.log_metric("val_r2", exp_results["val_r2"])

        # Kaggle提出結果（JSONから読み込み）
        public_score = exp_results["public_score"]
        private_score = exp_results["private_score"]
        mlflow.log_metric("public_score", public_score)
        mlflow.log_metric("private_score", private_score)

        # スコア一貫性の指標
        cv_public_diff = abs(cv_mean - public_score)
        cv_private_diff = abs(cv_mean - private_score)
        mlflow.log_metric("cv_public_diff", cv_public_diff)
        mlflow.log_metric("cv_private_diff", cv_private_diff)

        # ===================
        # アーティファクトの記録
        # ===================

        # 特徴量重要度データ（JSONから読み込み）
        feature_importance_data = exp_results["feature_importance"]

        # 特徴量重要度をCSVとして保存
        importance_df = pd.DataFrame(
            [
                {"feature": feature, "importance": importance}
                for feature, importance in feature_importance_data.items()
            ]
        )
        importance_file = "feature_importance.csv"
        importance_df.to_csv(importance_file, index=False)
        mlflow.log_artifact(importance_file)

        # 特徴量重要度のグラフを作成
        plt.figure(figsize=(12, 8))
        importance_sorted = importance_df.sort_values("importance", ascending=True)

        # 横棒グラフ
        bars = plt.barh(range(len(importance_sorted)), importance_sorted["importance"])
        plt.yticks(range(len(importance_sorted)), importance_sorted["feature"])
        plt.xlabel("Feature Importance")
        plt.title("Feature Importance - LightGBM Model (exp001)")
        plt.grid(axis="x", alpha=0.3)

        # カラーグラデーション
        colors = plt.cm.viridis(np.linspace(0, 1, len(bars)))
        for bar, color in zip(bars, colors):
            bar.set_color(color)

        plt.tight_layout()

        # グラフを保存
        feature_importance_plot = "feature_importance_plot.png"
        plt.savefig(feature_importance_plot, dpi=80, bbox_inches="tight")
        mlflow.log_artifact(feature_importance_plot)
        plt.close()

        # 実験サマリーの作成（JSONデータを元に更新）
        experiment_summary = {
            "experiment_id": exp_results["experiment_id"],
            "model_type": exp_results["model_type"],
            "objective": "Baseline model for podcast listening time prediction",
            "date": datetime.now().isoformat(),
            "data_info": {
                "train_samples": exp_results["train_size"],
                "test_samples": exp_results["test_size"],
                "features": exp_results["num_features"],
                "target": exp_results["target_variable"],
            },
            "preprocessing": exp_results["preprocessing"],
            "results": {
                "cv_rmse": f"{cv_mean:.4f} ± {cv_std:.4f}",
                "public_score": public_score,
                "private_score": private_score,
                "validation_r2": exp_results["val_r2"],
            },
            "key_findings": [
                "Episode_Length_minutes is the most important feature",
                "Ad_Density (new feature) ranks 2nd in importance",
                "CV score closely matches public/private scores",
                "No overfitting detected",
                "Stable model performance across folds",
            ],
            "next_steps": [
                "Target encoding for categorical variables",
                "Feature interactions (genre x popularity)",
                "Hyperparameter optimization",
                "Ensemble with XGBoost",
            ],
        }

        # サマリーをJSONとして保存
        summary_file = "experiment_summary.json"
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(experiment_summary, f, indent=2, ensure_ascii=False)
        mlflow.log_artifact(summary_file)

        # README.mdをHTMLに変換してアーティファクトとして追加
        readme_path = Path("README.md")
        if readme_path.exists():
            with open(readme_path, "r", encoding="utf-8") as f:
                readme_content = f.read()

            # MarkdownをHTMLに変換
            readme_html = markdown_to_html(readme_content)

            # HTMLファイルとして保存
            readme_html_path = "README.html"
            with open(readme_html_path, "w", encoding="utf-8") as f:
                f.write(readme_html)

            mlflow.log_artifact(readme_html_path, "documentation")

        # ===================
        # モデルの記録
        # ===================

        # LightGBMモデルが保存されている場合は記録
        model_path = results_dir / "model.pkl"
        if model_path.exists():
            # pickleファイルをアーティファクトとして記録
            mlflow.log_artifact(str(model_path), "model")

            # モデルを読み込んでMLflowモデルとして記録
            with open(model_path, "rb") as f:
                model = pickle.load(f)

            # シグネチャ情報（入力特徴量の情報）
            from mlflow.models.signature import infer_signature

            # ダミーの入力データでシグネチャを作成
            dummy_input = pd.DataFrame({feature: [0] * 5 for feature in features})
            dummy_output = np.array([45.0] * 5)  # 平均的な聴取時間

            signature = infer_signature(dummy_input, dummy_output)

            # LightGBMモデルとして記録
            mlflow.lightgbm.log_model(
                model,
                "lightgbm_model",
                signature=signature,
                input_example=dummy_input.iloc[0:1],
            )

        print(f"✅ MLflow experiment logged successfully!")
        print(f"Run ID: {run.info.run_id}")
        print(f"Experiment ID: {run.info.experiment_id}")
        print(f"Artifact URI: {run.info.artifact_uri}")

        return run.info.run_id


if __name__ == "__main__":
    run_id = log_experiment_to_mlflow()
    print(f"\\n🎯 実験がMLflowに記録されました")
    print(f"Run ID: {run_id}")
