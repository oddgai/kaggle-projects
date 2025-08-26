# exp001 - ベースラインモデル（LightGBM）

## 概要

EDAの結果を基にLightGBMでベースラインモデルを構築。基本的な特徴量エンジニアリングを適用し、生存予測の精度を評価。

## 使用した特徴量

### 基本特徴量
- `Pclass` - チケットクラス（1, 2, 3）
- `Sex` - 性別
- `Age` - 年齢（称号による欠損値補完済み）
- `SibSp` - 同乗した兄弟姉妹・配偶者の数
- `Parch` - 同乗した親・子供の数
- `Fare` - 運賃（中央値で欠損値補完済み）
- `Embarked` - 乗船港（S, C, Q）

### エンジニアリング特徴量
- `Title` - 名前から抽出した称号（Mr, Mrs, Miss, Master, Rare, Other）
- `FamilySize` - 家族サイズ（SibSp + Parch + 1）
- `IsAlone` - 一人旅フラグ（0: 家族連れ, 1: 一人旅）
- `FamilySizeGroup` - 家族サイズのカテゴリ化（Alone, Medium, Large）
- `AgeBin` - 年齢のビニング（Child, Teenager, Adult, Elder）
- `FareBin` - 運賃のビニング（Low, Medium, High, VeryHigh）
- `HasCabin` - 客室情報の有無（0: なし, 1: あり）
- `Sex_Pclass` - 性別とクラスの組み合わせ
- `HasTicketPrefix` - チケット番号にプレフィックスがあるか（0: なし, 1: あり）

**特徴量数:** 16個

## モデル設定

### アルゴリズム
- **LightGBM**（勾配ブースティング）

### ハイパーパラメータ
```python
lgb_params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1,
    'random_state': 42
}
```

### 訓練設定
- `num_boost_round`: 1000（早期停止あり）
- `early_stopping_rounds`: 100
- `validation_split`: 0.2（stratified）

## 検証結果

### 単一検証
- **検証精度**: 0.8212 (82.12%)

### 5-fold クロスバリデーション
- **CV平均スコア**: 0.8496 (84.96%)
- **標準偏差**: ±0.0263

### Kaggle提出結果
- **Public Score**: 0.77272 (77.27%)

### 特徴量重要度（上位5つ）
1. **Sex_Pclass**: 1463 - 性別×クラス組み合わせ（最重要）
2. **Sex**: 878 - 性別
3. **Age**: 709 - 年齢
4. **Fare**: 536 - 運賃
5. **HasCabin**: 162 - 客室情報有無

## 出力ファイル

### 提出ファイル
- `../../results/exp001/submission.csv`
- 形式: PassengerId, Survived

### 予測統計
- **テスト予測生存者数**: 141人
- **テスト予測死亡者数**: 277人
- **テスト予測生存率**: 33.73%

## 実装のポイント

### データ前処理
- 全データを結合して特徴量エンジニアリングを実行
- カテゴリカル変数はLabelEncodingを使用
- 欠損値補完は業務知識を活用（称号による年齢補完など）

### 特徴量エンジニアリング
- EDAで効果的だった特徴量を重点的に作成
- 称号抽出による社会的地位の表現
- 家族構成の多面的な表現（サイズ、一人旅、カテゴリ）
- 数値変数のビニングによるパターン抽出

### モデル選択理由
- LightGBMは高精度で解釈しやすい
- 特徴量重要度で効果的な特徴量を特定可能
- 過学習を防ぐ早期停止機能
- カテゴリカル変数の自然な処理

## 次の改善案

### 特徴量の改善
1. より詳細な称号分析（地域や時代背景考慮）
2. 客室番号からの階層・位置情報抽出
3. チケット番号のパターン分析強化
4. 運賃と他の変数との相互作用特徴量

### モデルの改善
1. ハイパーパラメータチューニング
2. アンサンブル学習（複数モデルの組み合わせ）
3. スタッキング手法の適用
4. 異なるアルゴリズムとの比較（XGBoost, CatBoost等）

### 検証の改善
1. 層化サンプリングの工夫
2. 時系列を考慮した分割手法の検討
3. より詳細な誤分類分析
4. 予測確率の閾値最適化

## 派生元
- exp000（EDA）の知見を活用

## 備考
- OpenMP（libomp）のインストールが必要
- 日本語表示にはjapanize-matplotlibを使用
- 実行環境：Python 3.12 + uv