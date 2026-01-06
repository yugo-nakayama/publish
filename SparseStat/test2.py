import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
import matplotlib.pyplot as plt

# 実験設定
np.random.seed(42)
n_trials = 100
n_samples_list = [50, 100, 200, 400, 800]
p_dim = 200       # 高次元設定 (p > n の領域も含む)
s_sparsity = 5    # 真の非ゼロ変数は5個だけ
sigma_noise = 1.0

# 結果格納用
results = []

for n in n_samples_list:
    # 理論的なlambda: C * sigma * sqrt(log(p)/n)
    # 定数Cは実験的に調整（ここではやや大きめに設定してスパース性を確保）
    lambda_theory = sigma_noise * np.sqrt(2 * np.log(p_dim) / n)

    for _ in range(n_trials):
        # 1. データの生成 (y = X*beta + epsilon)
        X = np.random.randn(n, p_dim)
        # 列ごとの正規化 (Lassoの前提)
        X = X / np.linalg.norm(X, axis=0) * np.sqrt(n)
        
        beta_true = np.zeros(p_dim)
        # 信号強度: 最小シグナル条件 (sqrt(log p / n) より十分大きく設定)
        min_signal = 1.5 * np.sqrt(np.log(p_dim) / n)
        true_indices = np.random.choice(p_dim, s_sparsity, replace=False)
        # 信号は {+1, -1} * 強度
        beta_true[true_indices] = np.random.choice([-1, 1], s_sparsity) * (min_signal + 2.0)
        
        epsilon = np.random.randn(n) * sigma_noise
        y = np.dot(X, beta_true) + epsilon
        
        # 2. LASSO推定
        lasso = Lasso(alpha=lambda_theory, fit_intercept=False, max_iter=5000)
        lasso.fit(X, y)
        beta_hat = lasso.coef_
        
        # 3. 評価指標の計算
        
        # (A) L2誤差 ||beta_hat - beta_true||_2^2
        l2_error = np.sum((beta_hat - beta_true)**2)
        
        # (B) サポート回復 (Support Recovery)
        # 閾値処理して非ゼロ判定
        hat_support = set(np.where(np.abs(beta_hat) > 1e-4)[0])
        true_support = set(true_indices)
        
        is_exact_recovery = (hat_support == true_support)
        
        # 理論レート (s * log(p) / n)
        oracle_rate = (s_sparsity * np.log(p_dim)) / n

        results.append({
            'n': n,
            'l2_error': l2_error,
            'oracle_rate': oracle_rate,
            'is_exact_recovery': is_exact_recovery
        })

df = pd.DataFrame(results)
summary = df.groupby('n').mean().reset_index()

# プロット作成（可視化ツールへ渡すデータ整形は省略し、直接描画コードを提示）
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# 左図：L2推定誤差の収束（定理 3.3.1）
# 実験値と理論レートの形状を比較（定数倍でスケール合わせ）
scale = summary['l2_error'].iloc[0] / summary['oracle_rate'].iloc[0]
ax1.plot(summary['n'], summary['l2_error'], 'o-', label='Experimental L2 Error', linewidth=2)
ax1.plot(summary['n'], summary['oracle_rate'] * scale, 'k--', label=r'Theory Rate $\propto \frac{s \log p}{n}$', alpha=0.7)
ax1.set_title('Theorem 3.3.1: Estimation Error Convergence', fontsize=12)
ax1.set_xlabel('Sample Size ($n$)')
ax1.set_ylabel('Squared L2 Error')
ax1.legend()
ax1.grid(True)

# 右図：サポート回復確率（定理 3.4.1）
ax2.plot(summary['n'], summary['is_exact_recovery'], 's-', color='green', linewidth=2, label='Prob. of Exact Recovery')
ax2.axhline(1.0, color='gray', linestyle=':', alpha=0.5)
ax2.set_title('Theorem 3.4.1: Model Selection Consistency', fontsize=12)
ax2.set_xlabel('Sample Size ($n$)')
ax2.set_ylabel('Probability')
ax2.set_ylim(-0.05, 1.05)
ax2.legend(loc='lower right')
ax2.grid(True)

plt.tight_layout()
plt.show()
