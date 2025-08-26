# exp006計画: 「Less is More」による精密最適化

## 🎯 基本戦略

exp005の失敗（過学習による汎化性能低下）を受けて、**シンプル化と精密最適化**による堅実な改善を目指す。

### 核心アプローチ: **質的改善 > 量的拡張**

- **特徴量**: 45個 → **15-18個**に削減
- **モデル**: アンサンブルではなく**LightGBM単体**に集中
- **最適化**: **ハイパーパラメータ徹底調整**
- **検証**: **CV vs Kaggle乖離**の最小化重視

## 📊 現状分析と目標設定

### ベースライン
- **exp004**: Kaggle=0.77990, CV=0.8462, 乖離=0.0663 ✅
- **exp005**: Kaggle=0.76315, CV=0.8507, 乖離=0.0876 ❌

### 目標
- **最低**: Kaggle=0.785（exp004から+0.005）
- **理想**: Kaggle=0.795（0.8に接近）
- **乖離**: <0.070（健全な汎化性能）

## 🔍 Phase 1: 戦略的特徴量削減

### 削減戦略

#### 1. 重要度ベース削減
exp004の特徴量重要度から**下位7-8特徴量を除去**
```
保持確定（コア特徴量）:
- Sex_Pclass（最重要交互作用）
- Sex, Age, Fare, Pclass（基本4特徴量）
- Title_Grouped, HasCabin（高重要度カテゴリ）

削減候補（重要度下位）:
- Ticket_Length, Name_Length
- 低重要度統計特徴量
- 効果の薄い交互作用
```

#### 2. カテゴリ粒度調整
高次元カテゴリカル特徴量の**適度な統合**
```
例：Age_Group
7分割 → 4分割（Child, Young, Adult, Senior）

例：Cabin_Deck  
9種類 → 5種類（A-C, D-E, F-G, T, Unknown）
```

#### 3. 冗長特徴量除去
相関>0.8の特徴量ペアから**一方を削除**

### 最終特徴量セット（15-18個予定）
```python
core_features = [
    # 基本特徴量（4個）
    'Sex', 'Pclass', 'Age', 'Fare',
    
    # カテゴリカル（3個）
    'Title_Grouped', 'HasCabin', 'Embarked',
    
    # 家族構成（3個）
    'FamilySize', 'IsAlone', 'Surname_Count',
    
    # 交互作用（2個）
    'Sex_Pclass', 'Age_Fare_Interaction',
    
    # 統計特徴量（3-6個）
    'Age_Rank_SexPclass', 'Fare_Rank_Pclass',
    'Age_Group', 'Fare_Group'  # 条件付き
]
```

## ⚙️ Phase 2: LightGBM精密最適化

### Optunaベース超パラメータ探索

#### 探索空間設計
```python
def objective(trial):
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'verbosity': -1,
        'random_state': 42,
        
        # 探索パラメータ
        'num_leaves': trial.suggest_int('num_leaves', 15, 60),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
        
        # 正則化（重点探索）
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 2.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 2.0),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
        'min_child_weight': trial.suggest_float('min_child_weight', 0.001, 1.0)
    }
    
    # 目標: CV vs Kaggle乖離最小化
    cv_score = cross_validate(params)
    return cv_score  # ただし、過学習監視も並行実施
```

#### 最適化目標
- **主目標**: CV精度最大化
- **制約**: Early Stopping回数監視（50回以下）
- **監視**: CV標準偏差（安定性重視）

### 過学習防止策

#### 1. 厳格なEarly Stopping
```python
callbacks = [
    lgb.early_stopping(stopping_rounds=30),  # 50→30に短縮
    lgb.log_evaluation(100)  # ログ頻度削減
]
```

#### 2. 正則化重点調整
- **reg_alpha**: L1正則化（特徴選択効果）
- **reg_lambda**: L2正則化（過学習抑制）
- **min_child_samples**: リーフの最小サンプル数

#### 3. バリデーション改善
```python
# より厳格な検証
kfold = StratifiedGroupKFold(
    n_splits=7,  # 5→7に増加
    shuffle=True,
    random_state=42
)
```

## 📈 Phase 3: 汎化性能監視システム

### CV vs Kaggle乖離監視
各パラメータ候補に対して：
1. **CV精度**を記録
2. **期待Kaggle精度**を推定（過去実験の経験式）
3. **乖離予測値**を算出
4. 乖離<0.070の候補のみ採用

### 段階的評価プロセス
1. **20特徴量**でベースライン確認
2. **18特徴量**で削減効果検証
3. **15特徴量**で最終調整
4. 各段階でCV vs 期待Kaggle比較

## ⚠️ リスク管理

### 主要リスク
1. **特徴量削減による情報損失**
2. **局所最適解への収束**
3. **改善幅の限界**

### 対策
1. **段階的削減**で情報損失を監視
2. **複数回実行**で最適解の安定性確認
3. **小改善でも価値認識**（+0.003も成功）

### 代替シナリオ
改善が見られない場合：
- **exp007**: Neural Network導入
- **exp008**: 外部データ活用
- **exp009**: 物理的制約モデル

## 🛠 実装スケジュール

### 総予想時間: **4-5時間**

#### Step 1: 準備・分析（30分）
- [ ] exp004重要度分析
- [ ] 削減候補特徴量特定
- [ ] 相関分析による冗長性確認

#### Step 2: 特徴量削減実験（90分）
- [ ] 20特徴量版構築・評価
- [ ] 18特徴量版構築・評価  
- [ ] 15特徴量版構築・評価
- [ ] 最適特徴量セット決定

#### Step 3: ハイパーパラメータ最適化（150分）
- [ ] Optuna環境構築
- [ ] 100回以上の最適化実行
- [ ] 候補パラメータの汎化性能評価
- [ ] 最適解選択

#### Step 4: 最終モデル・評価（30分）
- [ ] 最終パラメータでの訓練
- [ ] テストデータ予測
- [ ] 提出ファイル生成
- [ ] 結果分析・文書化

## 🎯 成功基準

### 技術的成功
- ✅ **Kaggleスコア**: >0.785
- ✅ **CV乖離**: <0.070
- ✅ **特徴量数**: ≤18個
- ✅ **安定性**: CV標準偏差<0.040

### 戦略的成功
- ✅ **過学習制御**の実践的習得
- ✅ **特徴量選択**の系統的手法確立
- ✅ **最適化プロセス**の体系化

### 学習的成功
- ✅ **「Less is More」**原理の実証
- ✅ **汎化性能重視**の意識確立
- ✅ **Titanicドメイン**の理解深化

---

**exp006は「技術的熟練」から「機械学習の本質理解」への重要な転換点となる実験です。**

シンプルで堅実なアプローチにより、持続可能な改善を目指します。