このリポジトリ（またはフォルダ）には、本文中の理解を助ける図と、定理（第3章）の主張をシミュレーションで確認する図を生成する Python スクリプトを配置します。  
添付の出力例は `Figure_1.jpg`（定理検証の2枚組）および `Figure_1_3.jpg`（概念図3枚組）に対応します。[file:50][file:51]

---

## 1. ファイル構成（命名）

### 1.1 概念図（3枚組）を出すスクリプト
- **実行ファイル名**: `plot_concept_figures.py`
- **出力**: `Figure_1_3.jpg`（または `fig_concepts.png` 等、保存名はコード内で統一）
- **内容**（本文の対応セクション）
  - Fig 1: L1正則化の幾何（第2章 2.4節）
  - Fig 2: LASSOパス（第2章 2.6節）
  - Fig 3: 切断正規分布（第4章 4.3節）[file:51]

### 1.2 定理の実験検証（2枚組）を出すスクリプト
- **実行ファイル名**: `exp_lasso_theorems.py`
- **出力**: `Figure_1.jpg`（または `fig_theorem_checks.png` 等、保存名はコード内で統一）
- **内容**（本文の対応セクション）
  - Theorem 3.3.1: 推定誤差（L2）の収束（オラクルレート \(\propto s\log(p)/n\) と比較）
  - Theorem 3.4.1: サポート回復確率（Model Selection Consistency）の上昇 [file:50]

---

## 2. 実行方法

### 2.1 依存ライブラリ
以下が必要です（pip/conda どちらでも可）：

- numpy
- pandas
- matplotlib
- scikit-learn
- scipy（切断正規の図で使用）

例：
```bash
pip install numpy pandas matplotlib scikit-learn scipy
```

### 2.2 実行コマンド

#### 概念図（3枚組）の生成
```bash
python plot_concept_figures.py
```
期待される出力例：`Figure_1_3.jpg` [file:51]

#### 定理検証（2枚組）の生成
```bash
python exp_lasso_theorems.py
```
期待される出力例：`Figure_1.jpg` [file:50]

---

## 3. 出力図の説明（添付との対応）

### 3.1 `Figure_1_3.jpg`（概念図3枚）
- **左：L1正則化の幾何（Section 2.4）**
  - L1制約（ひし形）と損失の等高線の接点が「座標軸上に来やすい」ことを示し、スパース解が生じる直観を可視化。[file:51]
- **中央：LASSOパス（Section 2.6）**
  - \(-\log(\lambda)\) を横軸にとり、\(\lambda\) を弱めると係数が順に活性化していく様子を表示。[file:51]
- **右：切断正規分布（Section 4.3）**
  - 選択イベントにより正規分布が区間で切断される直観を示す（selective inference の入口）。[file:51]

### 3.2 `Figure_1.jpg`（定理検証2枚）
- **左：Theorem 3.3.1（推定誤差の収束）**
  - サンプルサイズ \(n\) を増やすと、推定誤差（実験値）が理論レート \(\propto s\log(p)/n\) と同じ減少傾向になることを確認。[file:50]
- **右：Theorem 3.4.1（モデル選択一貫性）**
  - \(n\) を増やすと、真のサポートを完全回復する確率が上がる様子を確認（ただし有限標本では1に到達しないこともある）。[file:50]

---

## 4. 再現性メモ（推奨設定）
- 乱数シードは固定（例：`np.random.seed(42)`）し、試行回数 `n_trials` を増やすと曲線が滑らかになります。
- `p_dim`, `s_sparsity`, 信号強度（`min_signal` の係数）を変えると、回復確率の立ち上がり位置が変化します（相転移の観察に有用）。

---

## 5. 生成物の保存ルール（推奨）
出力の混乱を避けるため、以下のルールを推奨します。

- !["概念図"]("https://github.com/yugo-nakayama/publish/blob/main/SparseStat/Figure_1_3.png")
- !["定理検証"]("https://github.com/yugo-nakayama/publish/blob/main/SparseStat/Figure_1.png")
