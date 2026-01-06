import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import lasso_path
from sklearn.datasets import make_regression
from scipy.stats import norm, truncnorm

# 図のスタイル設定
plt.style.use('seaborn-v0_8-whitegrid')
fig_size = (18, 5)

def plot_l1_geometry(ax):
    """
    図1: L1正則化の幾何学的解釈（スパース性が生まれる理由）
    """
    # グリッド設定
    x = np.linspace(-2, 2, 400)
    y = np.linspace(-2, 2, 400)
    X, Y = np.meshgrid(x, y)
    
    # OLSの損失関数の等高線（中心を(1.2, 1.2)に設定）
    Z = (X - 1.2)**2 + (Y - 0.8)**2
    
    # L1ボール（ひし形）とL2ボール（円）
    L1 = np.abs(X) + np.abs(Y)
    L2 = X**2 + Y**2
    
    # プロット
    levels = [0.1, 0.5, 1.2, 2.2, 3.5]
    ax.contour(X, Y, Z, levels=levels, colors='gray', alpha=0.8, linestyles='dashed')
    ax.plot([1.2], [0.8], 'rx', label='OLS Solution')
    
    # L1制約領域（ひし形）
    ax.contour(X, Y, L1, levels=[1], colors='blue', linewidths=2)
    ax.text(0.1, 0.6, 'L1 Constraint\n(Diamond)', color='blue', fontsize=10, ha='center')
    
    # L2制約領域（円 - 比較用、薄く表示）
    ax.contour(X, Y, L2, levels=[1], colors='green', linewidths=1, alpha=0.5, linestyles='dotted')
    
    # 接点（スパース解）
    ax.plot([1.0], [0.0], 'bo', label='Lasso Solution (Sparse)')
    
    ax.set_title("Fig 1: Geometry of L1 Regularization (Section 2.4)", fontsize=12)
    ax.set_xlabel(r'$\beta_1$')
    ax.set_ylabel(r'$\beta_2$')
    ax.axhline(0, color='black', linewidth=0.5)
    ax.axvline(0, color='black', linewidth=0.5)
    ax.set_aspect('equal')
    ax.legend()

def plot_lasso_path(ax):
    """
    図2: LASSOパス（正則化パラメータによる係数の変化）
    """
    # 合成データの生成
    X, y, w = make_regression(n_samples=100, n_features=10, n_informative=3, 
                              coef=True, random_state=42, noise=1.0)
    
    # LASSOパスの計算
    alphas_lasso, coefs_lasso, _ = lasso_path(X, y, eps=0.001)
    
    # log(lambda) に変換
    log_alphas = -np.log10(alphas_lasso)
    
    # プロット
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    for coef_l, c in zip(coefs_lasso, colors):
        l1 = ax.plot(log_alphas, coef_l, c=c)
        
    ax.set_xlabel(r'$-\log(\lambda)$')
    ax.set_ylabel('Coefficients')
    ax.set_title('Fig 2: LASSO Regularization Path (Section 2.6)', fontsize=12)
    ax.axhline(0, color='black', linestyle='--', linewidth=0.8)
    
    # 説明テキスト
    ax.text(log_alphas[0], 0, 'Large $\lambda$\n(All 0)', ha='left', va='bottom', fontsize=9)
    ax.text(log_alphas[-1], 0, 'Small $\lambda$\n(OLS)', ha='right', va='bottom', fontsize=9)

def plot_truncated_normal(ax):
    """
    図3: 切断正規分布（選択後推論の概念）
    """
    mu, sigma = 0, 1
    a, b = -0.5, 2.0 # 切断区間
    
    x = np.linspace(-4, 4, 1000)
    
    # 元の正規分布
    ax.plot(x, norm.pdf(x, mu, sigma), 'k--', alpha=0.5, label='Original Normal')
    
    # 切断正規分布
    trunc_dist = truncnorm((a - mu) / sigma, (b - mu) / sigma, loc=mu, scale=sigma)
    ax.plot(x, trunc_dist.pdf(x), 'r-', linewidth=2, label='Truncated Normal (Post-Selection)')
    
    # 領域のシェーディング
    x_trunc = np.linspace(a, b, 1000)
    ax.fill_between(x_trunc, trunc_dist.pdf(x_trunc), color='red', alpha=0.1)
    
    # 区間の表示
    ax.axvline(a, color='blue', linestyle=':', linewidth=1.5)
    ax.axvline(b, color='blue', linestyle=':', linewidth=1.5)
    ax.text(a, 0.45, 'Selection Event\nBounds', color='blue', ha='center', fontsize=9)
    
    ax.set_title('Fig 3: Truncated Normal Distribution (Section 4.3)', fontsize=12)
    ax.set_xlabel('Statistic T')
    ax.set_ylabel('Probability Density')
    ax.legend()

# 描画実行
fig, axes = plt.subplots(1, 3, figsize=fig_size)

plot_l1_geometry(axes[0])
plot_lasso_path(axes[1])
plot_truncated_normal(axes[2])

plt.tight_layout()
plt.show()
