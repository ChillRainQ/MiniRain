"""
Chinchilla-style Scaling Law Fitting for MiniMind Pretraining
============================================================

Fits: L(N, D) = E + A/N^α + B/D^β   (N: params, D: tokens, L: loss)

三种方法 (Chinchilla 原文 Approach 1-3):
  Method 1 - 联合参数拟合: 直接用所有数据拟合完整公式
  Method 2 - IsoFLOP 分析: 对每个FLOP预算找最优模型大小
  Method 3 - 分离拟合 + 交叉验证: 先拟合数据项,再拟合参数项

精度提升:
  - 200次随机初始值搜索
  - 500次 Bootstrap 置信区间
  - 5-fold 交叉验证
  - IsoFLOP 三次样条插值
  - RANSAC 鲁棒拟合

Reference: "Training Compute-Optimal Large Language Models"
           Hoffmann et al., 2022
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.optimize import minimize
from scipy.interpolate import CubicSpline
from collections import defaultdict
import re, os, warnings, json

warnings.filterwarnings('ignore')

# ====== 设置 matplotlib 中文字体 ======
import matplotlib.font_manager as fm
zh_fonts = [f.name for f in fm.fontManager.ttflist
            if 'WenQuanYi' in f.name or 'Noto' in f.name or 'CJK' in f.name or 'SimHei' in f.name]
if zh_fonts:
    rcParams['font.family'] = zh_fonts[0]
else:
    rcParams['font.family'] = 'sans-serif'
rcParams['axes.unicode_minus'] = False

# ====================================================================
# 工具函数
# ====================================================================

def huber_loss(delta, r):
    cond = np.abs(r) < delta
    return np.where(cond, 0.5*r**2, delta*(np.abs(r)-0.5*delta))


def chinchilla_law(N, D, E, A, alpha, B, beta):
    """L(N, D) = E + A/N^α + B/D^β"""
    return E + A/(N**alpha) + B/(D**beta)


def huber_log_loss(params, N, D, L):
    """log-space Huber: Σ Huber(δ=10⁻³, log(L_pred) - log(L))"""
    Lp = chinchilla_law(N, D, *params)
    return np.sum(huber_loss(1e-3, np.log(Lp) - np.log(L)))


# ====================================================================
# 1. 数据解析
# ====================================================================

def parse_csvs(data_dir='../train_logs'):
    with open(os.path.join(data_dir, 'loss.csv'), 'r') as f:
        loss_lines = f.readlines()
    with open(os.path.join(data_dir, 'tokens.csv'), 'r') as f:
        tokens_lines = f.readlines()

    loss_header = loss_lines[0].strip().split(',')

    models = []
    for i in range(1, len(loss_header), 2):
        col_name = loss_header[i]
        match = re.search(r'-P([\d.]+p?[\d]*)(M)-', col_name)
        if match:
            param_str = match.group(1).replace('p', '.')
            params = float(param_str)
            models.append({
                'name': col_name,
                'params_m': params,
                'params': params * 1e6,
                'loss_col': i,
            })

    print(f"发现 {len(models)} 个模型:")
    for m in models:
        print(f"  → {m['params_m']:.1f}M 参数")

    loss_data = [l.strip().split(',') for l in loss_lines[1:]]
    tokens_data = [l.strip().split(',') for l in tokens_lines[1:]]

    all_points = []
    for row_idx in range(len(loss_data)):
        lr, tr = loss_data[row_idx], tokens_data[row_idx]
        step = int(lr[0])
        for mi, m in enumerate(models):
            loss = float(lr[m['loss_col']])
            tokens = float(tr[m['loss_col']])
            flops = 6 * m['params'] * tokens
            all_points.append({
                'step': step, 'N': m['params'], 'N_m': m['params_m'],
                'D': tokens, 'loss': loss, 'flops': flops,
                'model_idx': mi,
                'model_name': f"{m['params_m']:.1f}M"
            })

    return models, all_points


# ====================================================================
# Method 1: 联合参数拟合 (Parametric Fit)
# ====================================================================

def method1_parametric_fit(all_points):
    """
    Approach 1 from Chinchilla.
    直接用所有数据拟合 L(N,D) = E + A/N^α + B/D^β
    使用 log-space Huber loss.

    精度提升: 200 次随机初始值搜索 + 多优化器
    """
    N = np.array([p['N'] for p in all_points], dtype=np.float64)
    D = np.array([p['D'] for p in all_points], dtype=np.float64)
    L = np.array([p['loss'] for p in all_points], dtype=np.float64)

    print(f"\n{'='*60}")
    print(f"  Method 1: 联合参数拟合 (Parametric Fit)")
    print(f"  损失: Huber(δ=1e-3, log(L_pred) - log(L_obs))")
    print(f"  数据: {len(all_points)} 点, "
          f"{len(set(p['model_name'] for p in all_points))} 个模型")
    print(f"{'='*60}")

    bounds = [(0.1, 3.0), (0.01, 1e7), (0.01, 1.0),
              (0.01, 1e7), (0.01, 1.0)]

    best_res = None
    best_fun = float('inf')

    # 200 次随机初始值搜索
    np.random.seed(42)
    for _ in range(200):
        p0 = [
            np.random.uniform(0.5, 2.8),
            10**np.random.uniform(0.5, 5.5),
            np.random.uniform(0.05, 0.95),
            10**np.random.uniform(0.5, 5.5),
            np.random.uniform(0.05, 0.95),
        ]
        for method in ['L-BFGS-B', 'TNC']:
            try:
                res = minimize(huber_log_loss, p0, args=(N, D, L),
                               bounds=bounds, method=method,
                               options={'maxiter': 30000, 'ftol': 1e-15})
                if res.success and res.fun < best_fun:
                    best_fun = res.fun
                    best_res = res
            except:
                continue

    if best_res is None:
        print("❌ Method 1 未收敛!")
        return None

    E1, A1, a1, B1, b1 = best_res.x
    L_pred = chinchilla_law(N, D, E1, A1, a1, B1, b1)
    r2 = 1 - np.sum((L - L_pred)**2) / np.sum((L - np.mean(L))**2)
    r2_log = 1 - np.sum((np.log(L_pred) - np.log(L))**2) \
             / np.sum((np.log(L) - np.mean(np.log(L)))**2)

    print(f"\n  最优参数:")
    print(f"    E = {E1:.6f},  A = {A1:.4f},  α = {a1:.6f}")
    print(f"    B = {B1:.4f},  β = {b1:.6f}")
    print(f"  R² (原始) = {r2:.6f},  R² (log) = {r2_log:.6f}")

    # Bootstrap 置信区间 (500 次)
    print(f"\n  Bootstrap 500 次...")
    n_boot = 500
    param_samples = []
    n_pts = len(all_points)

    for b in range(n_boot):
        idx = np.random.choice(n_pts, n_pts, replace=True)
        Nb, Db, Lb = N[idx], D[idx], L[idx]
        try:
            res = minimize(
                lambda p, Nn=Nb, Dd=Db, Ll=Lb: huber_log_loss(p, Nn, Dd, Ll),
                best_res.x, bounds=bounds, method='L-BFGS-B',
                options={'maxiter': 10000, 'ftol': 1e-12})
            if res.success:
                param_samples.append(res.x)
        except:
            continue

    if len(param_samples) > 20:
        ps = np.array(param_samples)
        ci = np.percentile(ps, [5, 50, 95], axis=0)
        print(f"  ({len(ps)} 有效样本)")
        names = ['E', 'A', 'α', 'B', 'β']
        for i, n in enumerate(names):
            print(f"    {n}: median={ci[1,i]:.4f}  "
                  f"[90% CI: {ci[0,i]:.4f}, {ci[2,i]:.4f}]")

    return {
        'E': float(E1), 'A': float(A1), 'alpha': float(a1),
        'B': float(B1), 'beta': float(b1),
        'R2': float(r2), 'R2_log': float(r2_log),
    }


def method1_constrained_fit(all_points):
    """
    约束 α = β 的联合参数拟合。
    此时 D/N 比为常数，更符合 Chinchilla 论文的物理意义。
    """
    N = np.array([p['N'] for p in all_points], dtype=np.float64)
    D = np.array([p['D'] for p in all_points], dtype=np.float64)
    L = np.array([p['loss'] for p in all_points], dtype=np.float64)

    print(f"\n{'='*60}")
    print(f"  Method 1 (约束 α=β): 常数 D/N 比拟合")
    print(f"{'='*60}")

    bounds = [(0.1, 3.0), (0.01, 1e7), (0.01, 1.0), (0.01, 1e7)]

    def loss_eq(params):
        E, A, a, B = params
        Lp = E + A/N**a + B/D**a
        return np.sum(huber_loss(1e-3, np.log(Lp) - np.log(L)))

    best_res = None
    best_fun = float('inf')
    np.random.seed(42)
    for _ in range(200):
        p0 = [np.random.uniform(0.5, 2.8), 10**np.random.uniform(0.5, 5.5),
              np.random.uniform(0.05, 0.95), 10**np.random.uniform(0.5, 5.5)]
        for method in ['L-BFGS-B', 'TNC']:
            try:
                res = minimize(loss_eq, p0, bounds=bounds, method=method,
                               options={'maxiter': 30000, 'ftol': 1e-15})
                if res.success and res.fun < best_fun:
                    best_fun = res.fun
                    best_res = res
            except:
                continue

    if best_res is None:
        print("❌ 约束拟合未收敛!")
        return None

    E1, A1, a1, B1 = best_res.x
    L_pred = E1 + A1/N**a1 + B1/D**a1
    r2 = 1 - np.sum((L - L_pred)**2) / np.sum((L - np.mean(L))**2)
    r2_log = 1 - np.sum((np.log(L_pred)-np.log(L))**2) / np.sum((np.log(L)-np.mean(np.log(L)))**2)
    ratio = (B1/A1)**(1/a1)

    print(f"\n  参数 (α=β):")
    print(f"    E = {E1:.6f},  A = {A1:.4f},  α=β = {a1:.6f},  B = {B1:.4f}")
    print(f"  R² (原始) = {r2:.6f},  R² (log) = {r2_log:.6f}")
    print(f"  D/N = (B/A)^(1/α) = {ratio:.1f} (常数)")
    print(f"  即每个参数约对应 {ratio:.1f} 个 token")

    # Bootstrap
    n_boot = 500
    param_samples = []
    n_pts = len(all_points)
    for b in range(n_boot):
        idx = np.random.choice(n_pts, n_pts, replace=True)
        Nb, Db, Lb = N[idx], D[idx], L[idx]
        try:
            res = minimize(
                lambda p: np.sum(huber_loss(1e-3, np.log(p[0]+p[1]/Nb**p[2]+p[3]/Db**p[2]) - np.log(Lb))),
                best_res.x, bounds=bounds, method='L-BFGS-B',
                options={'maxiter': 10000, 'ftol': 1e-12})
            if res.success:
                param_samples.append(res.x)
        except:
            continue

    if len(param_samples) > 20:
        ps = np.array(param_samples)
        ci = np.percentile(ps, [5, 50, 95], axis=0)
        print(f"  Bootstrap {len(ps)} 次:")
        print(f"    E: median={ci[1,0]:.4f}  [90%: {ci[0,0]:.4f}, {ci[2,0]:.4f}]")
        print(f"    α=β: median={ci[1,2]:.4f}  [90%: {ci[0,2]:.4f}, {ci[2,2]:.4f}]")
        ratios = [(p[3]/p[1])**(1/p[2]) for p in ps]
        r_ci = np.percentile(ratios, [5, 50, 95])
        print(f"    D/N: median={r_ci[1]:.1f}  [90%: {r_ci[0]:.1f}, {r_ci[2]:.1f}]")

    return {
        'E': float(E1), 'A': float(A1), 'alpha': float(a1),
        'B': float(B1), 'beta': float(a1),  # β = α
        'R2': float(r2), 'R2_log': float(r2_log),
        'D_over_N': float(ratio),
    }


# ====================================================================
# Method 2: IsoFLOP 分析
# ====================================================================

def method2_isoflop(all_points):
    """
    Approach 2 from Chinchilla.
    对每个 FLOP 预算 → 找最优 N → 拟合 N_opt ∝ C^a

    精度提升:
    - 三次样条插值平滑 loss-N 曲线
    - 只用后半段训练数据
    - RANSAC 迭代拟合
    """
    print(f"\n{'='*60}")
    print(f"  Method 2: IsoFLOP 分析")
    print(f"  对每个 FLOP 预算找最优参数 → N_opt ∝ C^a")
    print(f"{'='*60}")

    steps = sorted(set(p['step'] for p in all_points))
    steps_use = steps[len(steps)//2:]  # 只用后半段

    opt_points = []
    for st in steps_use:
        pts = [p for p in all_points if p['step'] == st]
        pts.sort(key=lambda x: x['N'])

        Ns = np.array([p['N'] for p in pts])
        Ls = np.array([p['loss'] for p in pts])
        D_step = pts[0]['D']  # 同 step 所有模型 D 相同

        # 三次样条插值找最小值 (log-N 空间)
        try:
            idx_valid = ~(np.isnan(Ls) | np.isinf(Ls))
            cs = CubicSpline(np.log(Ns[idx_valid]), Ls[idx_valid],
                             bc_type='natural')
            N_dense = np.logspace(np.log10(Ns.min()), np.log10(Ns.max()), 500)
            L_dense = cs(np.log(N_dense))
            opt_idx = np.argmin(L_dense)
            N_opt = N_dense[opt_idx]
        except Exception:
            N_opt = Ns[np.argmin(Ls)]

        # 该步骤下最优模型的实际 FLOPs
        C_opt = 6 * N_opt * D_step
        opt_points.append({
            'C': C_opt, 'N_opt': N_opt,
            'D_opt': C_opt / (6 * N_opt),
        })

    C_vals = np.array([p['C'] for p in opt_points])
    N_opts = np.array([p['N_opt'] for p in opt_points])
    D_opts = np.array([p['D_opt'] for p in opt_points])

    # RANSAC 拟合
    best_N = None
    best_r2 = -float('inf')

    for _ in range(500):
        n_sub = max(len(C_vals)//2, 3)
        idx = np.random.choice(len(C_vals), n_sub, replace=False)
        logx, logy_N = np.log(C_vals[idx]), np.log(N_opts[idx])
        logy_D = np.log(D_opts[idx])

        A = np.vstack([np.ones_like(logx), logx]).T
        coeffs_N, *_ = np.linalg.lstsq(A, logy_N, rcond=None)
        coeffs_D, *_ = np.linalg.lstsq(A, logy_D, rcond=None)

        kN, aN = np.exp(coeffs_N[0]), coeffs_N[1]
        kD, aD = np.exp(coeffs_D[0]), coeffs_D[1]

        yN_pred = kN * C_vals**aN
        yD_pred = kD * C_vals**aD
        ss_res_N = np.sum((np.log(N_opts) - np.log(yN_pred))**2)
        ss_tot_N = np.sum((np.log(N_opts) - np.mean(np.log(N_opts)))**2)
        ss_res_D = np.sum((np.log(D_opts) - np.log(yD_pred))**2)
        ss_tot_D = np.sum((np.log(D_opts) - np.mean(np.log(D_opts)))**2)

        r2_N = 1 - ss_res_N / ss_tot_N if ss_tot_N > 0 else -1
        r2_D = 1 - ss_res_D / ss_tot_D if ss_tot_D > 0 else -1

        if r2_N + r2_D > best_r2:
            best_r2 = r2_N + r2_D
            best_N = (kN, aN, r2_N, kD, aD, r2_D)

    if best_N is None:
        # fallback: 用全部数据直接拟合
        logx = np.log(C_vals)
        logy_N = np.log(N_opts)
        logy_D = np.log(D_opts)
        A = np.vstack([np.ones_like(logx), logx]).T
        coeffs_N, *_ = np.linalg.lstsq(A, logy_N, rcond=None)
        coeffs_D, *_ = np.linalg.lstsq(A, logy_D, rcond=None)
        kN, aN = np.exp(coeffs_N[0]), coeffs_N[1]
        kD, aD = np.exp(coeffs_D[0]), coeffs_D[1]
        yN_pred = kN * C_vals**aN
        yD_pred = kD * C_vals**aD
        ss_tot_N = np.sum((np.log(N_opts) - np.mean(np.log(N_opts)))**2)
        ss_tot_D = np.sum((np.log(D_opts) - np.mean(np.log(D_opts)))**2)
        r2_N = 1 - np.sum((np.log(N_opts)-np.log(yN_pred))**2) / ss_tot_N if ss_tot_N > 0 else 0
        r2_D = 1 - np.sum((np.log(D_opts)-np.log(yD_pred))**2) / ss_tot_D if ss_tot_D > 0 else 0
        best_N = (kN, aN, r2_N, kD, aD, r2_D)

    kN, aN, r2_N, kD, aD, r2_D = best_N

    print(f"\n  最优参数:")
    print(f"    N_opt = {kN:.4f} × C^{aN:.6f}   (R²={r2_N:.4f})")
    print(f"    D_opt = {kD:.4f} × C^{aD:.6f}   (R²={r2_D:.4f})")
    print(f"    检验: a_N + a_D = {aN+aD:.4f} (应为 1.0)")

    return {
        'kN': float(kN), 'aN': float(aN), 'r2_N': float(r2_N),
        'kD': float(kD), 'aD': float(aD), 'r2_D': float(r2_D),
        'alpha_over_beta': float(aD / aN) if aN > 0 else 0,
        'opt_points': opt_points,
    }


# ====================================================================
# Method 3: 分离拟合 + k-fold 交叉验证
# ====================================================================

def method3_separable_cv(all_points):
    """
    Approach 3 from Chinchilla.
    ① L(D) = E_N + B/D^β (每个模型独立, 共享 B, β)
    ② E_N = E + A/N^α
    ③ k-fold CV 评估泛化性能
    """
    print(f"\n{'='*60}")
    print(f"  Method 3: 分离拟合 + 交叉验证")
    print(f"  ① L(D) = E_N + B/D^β (每个模型独立)")
    print(f"  ② E_N = E + A/N^α")
    print(f"{'='*60}")

    groups = defaultdict(list)
    for p in all_points:
        groups[p['model_name']].append(p)
    model_names = sorted(groups.keys(), key=lambda x: groups[x][0]['N'])

    # --- Step 1: 拟合共享 B, β ---
    def step1_loss(params):
        B, beta = params
        total = 0
        E_locals = {}
        for mn in model_names:
            pts = groups[mn]
            D = np.array([p['D'] for p in pts])
            L = np.array([p['loss'] for p in pts])
            E_local = np.mean(L - B/(D**beta))
            E_locals[mn] = E_local
            pred = E_local + B/(D**beta)
            total += np.sum(huber_loss(1e-3, np.log(pred) - np.log(L)))
        return total, E_locals

    best_step1 = None
    best_s1_loss = float('inf')

    for B_init in np.logspace(0, 5, 20):
        for beta_init in np.linspace(0.05, 0.95, 10):
            try:
                res = minimize(
                    lambda p: step1_loss(p)[0],
                    [B_init, beta_init],
                    bounds=[(0.01, 1e7), (0.01, 1.0)],
                    method='L-BFGS-B', options={'maxiter': 20000}
                )
                if res.success and res.fun < best_s1_loss:
                    best_s1_loss = res.fun
                    best_step1 = res
            except:
                continue

    B_fit, b_fit = best_step1.x
    _, E_locals = step1_loss([B_fit, b_fit])

    print(f"\n  第一步 (共享数据项):")
    print(f"    B = {B_fit:.4f},  β = {b_fit:.6f}")

    # --- Step 2: 拟合 E_N = E + A/N^α ---
    Ns = np.array([groups[mn][0]['N'] for mn in model_names])
    E_vals = np.array([E_locals[mn] for mn in model_names])

    def step2_loss(params):
        E, A, alpha = params
        pred = E + A/(Ns**alpha)
        return np.sum((E_vals - pred)**2)

    best_step2 = None
    best_s2_loss = float('inf')
    for E_init in np.linspace(0.5, 2.5, 10):
        for A_init in np.logspace(0, 5, 10):
            for a_init in np.linspace(0.05, 0.95, 10):
                try:
                    res = minimize(step2_loss, [E_init, A_init, a_init],
                                   bounds=[(0.1, 3.0), (0.01, 1e7), (0.01, 1.0)],
                                   method='L-BFGS-B')
                    if res.success and res.fun < best_s2_loss:
                        best_s2_loss = res.fun
                        best_step2 = res
                except:
                    continue

    E_fit, A_fit, a_fit = best_step2.x

    print(f"  第二步 (参数项):")
    print(f"    E = {E_fit:.6f},  A = {A_fit:.4f},  α = {a_fit:.6f}")

    # --- 全局评估 ---
    N_arr = np.array([p['N'] for p in all_points])
    D_arr = np.array([p['D'] for p in all_points])
    L_arr = np.array([p['loss'] for p in all_points])
    L_pred = chinchilla_law(N_arr, D_arr, E_fit, A_fit, a_fit, B_fit, b_fit)
    r2 = 1 - np.sum((L_arr - L_pred)**2) / np.sum((L_arr - np.mean(L_arr))**2)
    r2_log = 1 - np.sum((np.log(L_pred)-np.log(L_arr))**2) \
             / np.sum((np.log(L_arr)-np.mean(np.log(L_arr)))**2)

    print(f"\n  全局 R² (原始) = {r2:.6f},  R² (log) = {r2_log:.6f}")

    # --- k-fold 交叉验证 ---
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = []

    for train_idx, val_idx in kf.split(all_points):
        train = [all_points[i] for i in train_idx]
        val = [all_points[i] for i in val_idx]
        N_tr = np.array([p['N'] for p in train])
        D_tr = np.array([p['D'] for p in train])
        L_tr = np.array([p['loss'] for p in train])

        p0_cv = [E_fit, A_fit, a_fit, B_fit, b_fit]
        bounds_cv = [(0.1, 3.0), (0.01, 1e7), (0.01, 1.0),
                     (0.01, 1e7), (0.01, 1.0)]
        res_cv = minimize(
            lambda p: huber_log_loss(p, N_tr, D_tr, L_tr),
            p0_cv, bounds=bounds_cv, method='L-BFGS-B',
            options={'maxiter': 10000}
        )
        if res_cv.success:
            N_va = np.array([p['N'] for p in val])
            D_va = np.array([p['D'] for p in val])
            L_va = np.array([p['loss'] for p in val])
            Lp_va = chinchilla_law(N_va, D_va, *res_cv.x)
            cv_scores.append(
                1 - np.sum((L_va - Lp_va)**2) / np.sum((L_va - np.mean(L_va))**2))

    if cv_scores:
        print(f"\n  5-fold CV R²: μ={np.mean(cv_scores):.4f}, "
              f"σ={np.std(cv_scores):.4f}")

    return {
        'E': float(E_fit), 'A': float(A_fit), 'alpha': float(a_fit),
        'B': float(B_fit), 'beta': float(b_fit),
        'R2': float(r2), 'R2_log': float(r2_log),
        'cv_mean': float(np.mean(cv_scores)) if cv_scores else 0,
        'cv_std': float(np.std(cv_scores)) if cv_scores else 0,
    }


# ====================================================================
# 最优计算分配
# ====================================================================

def compute_optimal_allocation(params, flop_budgets=None):
    E, A, a, B, b = (params['E'], params['A'], params['alpha'],
                     params['B'], params['beta'])

    aN = b / (a + b)
    aD = a / (a + b)

    print(f"\n{'='*60}")
    print(f"  最优计算分配分析")
    print(f"{'='*60}")
    print(f"  N_opt ∝ C^{aN:.4f}")
    print(f"  D_opt ∝ C^{aD:.4f}")
    print(f"  检验: aN + aD = {aN+aD:.4f} (应为 1.0)")
    print(f"  α/β = {a/b:.4f}")

    if flop_budgets:
        print(f"\n  {'C (FLOPs)':>18} {'N_opt':>12} {'D_opt':>14} "
              f"{'D/N':>10}")
        print(f"  {'-'*58}")
        for C in flop_budgets:
            num = a * A
            den = b * B
            exp = 1.0 / (a + b)
            N_opt = (num / den) ** exp * (C**b / (6.0**b)) ** exp
            D_opt = C / (6 * N_opt)
            print(f"  {C:>18.2e} {N_opt:>12.2e} {D_opt:>14.2e} "
                  f"{D_opt/N_opt:>10.1f}")

    return aN, aD


# ====================================================================
# 可视化 — 按 Chinchilla 论文三张标准图
# ====================================================================

def plot_all_results(models, all_points, m1, m2, m3, save_dir='.'):
    """
    三张标准图:
      Fig1 (Method 1): Loss vs FLOPs — 每个模型大小下 loss 随计算量下降
      Fig2 (Method 2): Loss vs Params (IsoFLOP) — 每个FLOP预算下loss随参数量变化
      Fig3 (Method 3): N_opt vs FLOPs — 最优模型大小随FLOP预算的幂律
    """
    colors = plt.cm.viridis(
        np.linspace(0.2, 0.9, len(set(p['model_name'] for p in all_points))))

    groups = defaultdict(list)
    for p in all_points:
        groups[p['model_name']].append(p)
    sorted_models = sorted(groups.items(), key=lambda x: x[1][0]['N'])

    # ================================================================
    # Fig 1: Loss vs FLOPs (每个模型一条曲线) — Method 1
    # ================================================================
    fig1, axes1 = plt.subplots(1, 2, figsize=(16, 6))
    ax1, ax2 = axes1

    for idx, (nm, pts) in enumerate(sorted_models):
        flops = np.array([p['flops'] for p in pts])
        loss = np.array([p['loss'] for p in pts])
        ax1.plot(flops, loss, color=colors[idx], lw=1.5, label=f'N={nm}')
        ax2.loglog(flops, loss, color=colors[idx], lw=1.5, label=f'N={nm}')

    # Method 1 拟合曲线的等N切片
    if m1:
        C_range = np.logspace(17, 20, 200)
        for idx, (nm, pts) in enumerate(sorted_models):
            N_m = pts[0]['N']
            L_fit = chinchilla_law(N_m, C_range/(6*N_m),
                                   m1['E'], m1['A'], m1['alpha'],
                                   m1['B'], m1['beta'])
            ax2.loglog(C_range, L_fit, '--', lw=1,
                       color=colors[idx], alpha=0.5)

    ax1.set_xlabel('FLOPs')
    ax1.set_ylabel('Loss')
    ax1.set_title('线性坐标')
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    ax2.set_xlabel('FLOPs')
    ax2.set_ylabel('Loss')
    ax2.set_title('Log-Log 坐标')
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3, which='both')

    fig1.suptitle('Fig 1 (Method 1): Loss vs FLOPs — 不同模型大小', fontsize=14)
    plt.tight_layout()
    fig1.savefig(os.path.join(save_dir, 'fig1_method1_loss_vs_flops.png'),
                 dpi=150, bbox_inches='tight')
    print(f"  保存: fig1_method1_loss_vs_flops.png")

    # ================================================================
    # Fig 2: Loss vs Params (IsoFLOP轮廓) — Method 2
    # ================================================================
    fig2, axes2 = plt.subplots(1, 2, figsize=(16, 6))
    ax_iso, ax_opt = axes2

    # 2a: IsoFLOP — 对每个FLOP级别, loss vs params
    all_steps_iso = sorted(set(p['step'] for p in all_points))
    step_sample = all_steps_iso[::15]  # 约每15步一条等值线
    if all_steps_iso[-1] not in step_sample:
        step_sample.append(all_steps_iso[-1])

    iso_colors = plt.cm.plasma(np.linspace(0.2, 0.9, len(step_sample)))

    for si, st in enumerate(step_sample):
        pts = [p for p in all_points if p['step'] == st]
        pts.sort(key=lambda x: x['N'])
        Ns = np.array([p['N'] for p in pts])
        Ls = np.array([p['loss'] for p in pts])
        C_avg = np.mean([p['flops'] for p in pts])

        ax_iso.semilogx(Ns, Ls, 'o-', color=iso_colors[si], lw=1.5,
                        label=f'C={C_avg:.1e}', markersize=5)

        # 标注最小值
        min_idx = np.argmin(Ls)
        ax_iso.scatter(Ns[min_idx], Ls[min_idx], color=iso_colors[si],
                       s=60, zorder=5, marker='*', edgecolors='white',
                       linewidths=0.5)

    ax_iso.set_xlabel('Parameters (N)')
    ax_iso.set_ylabel('Loss')
    ax_iso.set_title('IsoFLOP 轮廓 (★ = 该FLOP预算下最优N)')
    ax_iso.legend(fontsize=7, ncol=2)
    ax_iso.grid(alpha=0.3, which='both')

    # 2b: 最优 N vs FLOPs
    # 从每个 step 提取最优 N 和对应 FLOPs
    all_steps = sorted(set(p['step'] for p in all_points))
    C_opts_all = []
    N_opts_all = []
    for st in all_steps:
        pts = [p for p in all_points if p['step'] == st]
        pts.sort(key=lambda x: x['N'])
        Ns = np.array([p['N'] for p in pts])
        Ls = np.array([p['loss'] for p in pts])
        min_idx = np.argmin(Ls)
        # 该 step 下最优模型的 FLOPs
        best_pt = pts[min_idx]
        C_opts_all.append(best_pt['flops'])
        N_opts_all.append(best_pt['N'])

    C_opt_arr = np.array(C_opts_all)
    N_opt_arr = np.array(N_opts_all)

    ax_opt.scatter(C_opt_arr, N_opt_arr, alpha=0.5, s=15, c='steelblue',
                   label='实际最优 N')

    # 幂律拟合: N_opt ∝ C^a
    logC, logN = np.log(C_opt_arr), np.log(N_opt_arr)
    A_mat = np.vstack([np.ones_like(logC), logC]).T
    coeffs, *_ = np.linalg.lstsq(A_mat, logN, rcond=None)
    k_opt, a_opt = np.exp(coeffs[0]), coeffs[1]
    y_pred = k_opt * C_opt_arr**a_opt
    ss_res = np.sum((logN - np.log(y_pred))**2)
    ss_tot = np.sum((logN - np.mean(logN))**2)
    r2_opt = 1 - ss_res/ss_tot

    C_sorted = np.sort(C_opt_arr)
    ax_opt.loglog(C_sorted, k_opt * C_sorted**a_opt, 'r-', lw=2,
                  label=f'N_opt ∝ C^{a_opt:.4f}  (R²={r2_opt:.4f})')

    ax_opt.set_xlabel('C (FLOPs)')
    ax_opt.set_ylabel('N_opt (最优参数量)')
    ax_opt.set_title('最优模型大小 vs FLOP 预算')
    ax_opt.legend(fontsize=9)
    ax_opt.grid(alpha=0.3, which='both')

    plt.tight_layout()
    fig2.savefig(os.path.join(save_dir, 'fig2_method2_isoflop.png'),
                 dpi=150, bbox_inches='tight')
    print(f"  保存: fig2_method2_isoflop.png")

    # ================================================================
    # Fig 3: 拟合效果验证 — Method 3 (分离+CV)
    # ================================================================
    fig3, axes3 = plt.subplots(2, 2, figsize=(14, 12))
    ax_pv, ax_res, ax_comp, ax_surface = axes3[0, 0], axes3[0, 1], \
                                          axes3[1, 0], axes3[1, 1]

    N_all = np.array([p['N'] for p in all_points])
    D_all = np.array([p['D'] for p in all_points])
    L_all = np.array([p['loss'] for p in all_points])

    # 3a: Predicted vs Actual (log-log)
    for method_name, method_result, color, marker in [
        ('M1 (联合)', m1, 'steelblue', 'o'),
        ('M3 (分离+CV)', m3, 'coral', 's'),
    ]:
        if method_result:
            Lp = chinchilla_law(N_all, D_all,
                                method_result['E'], method_result['A'],
                                method_result['alpha'],
                                method_result['B'], method_result['beta'])
            r2 = method_result.get('R2', 0)
            ax_pv.scatter(Lp, L_all, alpha=0.3, s=5, c=color,
                          label=f'{method_name} (R²={r2:.4f})')

    lims = [min(L_all), max(L_all)]
    ax_pv.plot(lims, lims, 'k--', lw=1)
    ax_pv.set_xlabel('Predicted Loss')
    ax_pv.set_ylabel('Actual Loss')
    ax_pv.set_title('预测 vs 实际 Loss (Log-Log)')
    ax_pv.legend(fontsize=9)
    ax_pv.grid(alpha=0.3)
    ax_pv.set_xscale('log')
    ax_pv.set_yscale('log')

    # 3b: 残差分布
    if m1:
        Lp1 = chinchilla_law(N_all, D_all, m1['E'], m1['A'],
                             m1['alpha'], m1['B'], m1['beta'])
        res1 = L_all - Lp1
        ax_res.hist(res1, bins=40, alpha=0.6, color='steelblue',
                    edgecolor='white', label=f'M1 σ={np.std(res1):.4f}')
    if m3:
        Lp3 = chinchilla_law(N_all, D_all, m3['E'], m3['A'],
                             m3['alpha'], m3['B'], m3['beta'])
        res3 = L_all - Lp3
        ax_res.hist(res3, bins=40, alpha=0.5, color='coral',
                    edgecolor='white', label=f'M3 σ={np.std(res3):.4f}')
    ax_res.axvline(0, color='k', ls='--')
    ax_res.set_xlabel('Residual (L_actual - L_pred)')
    ax_res.set_ylabel('Frequency')
    ax_res.set_title('残差分布')
    ax_res.legend(fontsize=9)
    ax_res.grid(alpha=0.3)

    # 3c: 参数对比 (M1 vs M3)
    param_names = ['E', 'log₁₀(A)', 'α', 'log₁₀(B)', 'β']
    x = np.arange(len(param_names))
    width = 0.3

    methods_data = []
    if m1:
        methods_data.append(('M1', [m1['E'], np.log10(m1['A']),
                                    m1['alpha'], np.log10(m1['B']), m1['beta']]))
    if m3:
        methods_data.append(('M3', [m3['E'], np.log10(m3['A']),
                                    m3['alpha'], np.log10(m3['B']), m3['beta']]))

    for i, (nm, vals) in enumerate(methods_data):
        offset = (i - (len(methods_data)-1)/2) * width
        bars = ax_comp.bar(x + offset, vals, width, alpha=0.7, label=nm)
        # 在柱上标数值
        for bar, val in zip(bars, vals):
            ax_comp.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                        f'{val:.3f}', ha='center', va='bottom', fontsize=7)

    ax_comp.set_xticks(x)
    ax_comp.set_xticklabels(param_names)
    ax_comp.set_title('参数对比 (A, B 取 log₁₀)')
    ax_comp.legend(fontsize=9)
    ax_comp.grid(alpha=0.3, axis='y')

    # 3d: 3D surface — loss 随 N, D 变化的等值线
    n_contour = 50
    N_grid = np.logspace(np.log10(4e7), np.log10(6e8), n_contour)
    D_grid = np.logspace(np.log10(3e7), np.log10(3.5e9), n_contour)
    NN, DD = np.meshgrid(N_grid, D_grid)

    if m1:
        LL = chinchilla_law(NN, DD, m1['E'], m1['A'],
                            m1['alpha'], m1['B'], m1['beta'])
        cs = ax_surface.contourf(NN/1e6, DD/1e9, LL, levels=20,
                                  cmap='viridis', alpha=0.7)
        fig3.colorbar(cs, ax=ax_surface, label='Loss')

        # 标注实际数据点
        for nm, pts in sorted_models:
            N_m = pts[0]['N']
            D_m = np.array([p['D'] for p in pts])[::5]  # 每5步取一个点
            ax_surface.scatter([N_m/1e6]*len(D_m), D_m/1e9,
                              c='white', s=3, alpha=0.3)

    ax_surface.set_xlabel('Parameters N (M)')
    ax_surface.set_ylabel('Tokens D (B)')
    ax_surface.set_title('Loss 等值线 (Method 1 拟合)')
    ax_surface.set_xscale('log')

    plt.tight_layout()
    fig3.savefig(os.path.join(save_dir, 'fig3_method3_validation.png'),
                 dpi=150, bbox_inches='tight')
    print(f"  保存: fig3_method3_validation.png")

    plt.close('all')


# ====================================================================
# Main
# ====================================================================

def main():
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'train_logs')
    save_dir = data_dir

    print("=" * 60)
    print("  Chinchilla Scaling Law 拟合")
    print("  3 种方法 + 精度提升")
    print("  MiniMind 预训练数据分析")
    print("=" * 60)

    # 1. 解析
    models, all_points = parse_csvs(data_dir)
    n_models = len(set(p['model_name'] for p in all_points))
    n_steps = len(set(p['step'] for p in all_points))
    print(f"\n数据: {n_models} 个模型 × {n_steps} 步 = {len(all_points)} 点")

    # 2. 三种方法
    m1 = method1_parametric_fit(all_points)
    m2 = method2_isoflop(all_points)
    m3 = method3_separable_cv(all_points)

    # 2b. 约束 α=β 拟合
    m1c = method1_constrained_fit(all_points)

    # 3. 汇总
    m1c_e = f"{m1c['E']:.4f}" if m1c else '—'
    m1c_a = f"{m1c['A']:.2f}" if m1c else '—'
    m1c_al = f"{m1c['alpha']:.6f}" if m1c else '—'
    m1c_r = f"{m1c['R2']:.4f}" if m1c else '—'

    print(f"\n{'='*70}")
    print(f"  四种拟合结果汇总")
    print(f"{'='*70}")
    print(f"{'参数':>8} | {'M1 无约束':>14} {'M1 约束α=β':>14} "
          f"{'M2 IsoFLOP':>14} {'M3 分离+CV':>14}")
    print(f"{'-'*70}")
    print(f"{'E':>8} | {m1['E']:>14.4f} {m1c_e:>14} {'—':>14} {m3['E']:>14.4f}")
    print(f"{'A':>8} | {m1['A']:>14.2f} {m1c_a:>14} {'—':>14} {m3['A']:>14.2f}")
    print(f"{'α':>8} | {m1['alpha']:>14.6f} {m1c_al:>14} {'—':>14} {m3['alpha']:>14.6f}")
    print(f"{'β':>8} | {m1['beta']:>14.6f} {'=α':>14} {'—':>14} {m3['beta']:>14.6f}")
    print(f"{'R²':>8} | {m1['R2']:>14.4f} {m1c_r:>14} {'—':>14} {m3['R2']:>14.4f}")

    if m1c:
        print(f"\n▶ 推荐: 约束 α=β 拟合 — D/N = {m1c['D_over_N']:.1f}:1 (常数)")
        print(f"  (无约束 R²={m1['R2']:.4f} vs 约束 R²={m1c['R2']:.4f}, "
              f"差值仅 {m1['R2']-m1c['R2']:.4f})")

    # 4. 最优分配 (用约束版本)
    flop_budgets = [1e18, 2e18, 5e18, 1e19, 2e19, 5e19, 1e20]
    best_params = m1c if m1c else m1
    ratio = (best_params['B'] / best_params['A']) ** (1 / best_params['alpha'])

    print(f"\n{'='*60}")
    print(f"  最优计算分配 (基于{'约束α=β' if m1c else '无约束'}拟合)")
    print(f"{'='*60}")
    print(f"  每个参数对应约 {ratio:.1f} 个 token (固定比率)" if m1c else "")
    print(f"\n  {'C (FLOPs)':>18} {'N_opt':>12} {'D_opt':>14} {'D/N':>10}")
    print(f"  {'-'*58}")
    for C in flop_budgets:
        if m1c:
            N_opt = np.sqrt(C / (6 * ratio))
        else:
            a, A, b, B = best_params['alpha'], best_params['A'], best_params['beta'], best_params['B']
            exp = 1.0 / (a + b)
            N_opt = ((a * A) / (b * B)) ** exp * (C**b / (6.0**b)) ** exp
        D_opt = C / (6 * N_opt)
        print(f"  {C:>18.2e} {N_opt:>12.2e} {D_opt:>14.2e} {D_opt/N_opt:>10.1f}")

    # 5. 可视化
    print(f"\n{'='*60}")
    print(f"  生成可视化图表...")
    plot_all_results(models, all_points, m1, m2, m3, save_dir)

    # 6. 保存
    output = {
        'method1_parametric': m1,
        'method2_isoflop': {k: v for k, v in m2.items()
                            if k != 'opt_points'} if m2 else None,
        'method3_separable': m3,
        'best_params': m1,
    }
    with open(os.path.join(save_dir, 'scaling_law_params.json'), 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\n结果保存到: scaling_law_params.json")

    # 7. 最终公式
    print(f"\n{'='*60}")
    print(f"  最终 Scaling Law (Method 1)")
    print(f"{'='*60}")
    print(f"  L(N, D) = {m1['E']:.6f} + {m1['A']:.4f}/N^{m1['alpha']:.6f}"
          f" + {m1['B']:.4f}/D^{m1['beta']:.6f}")
    print(f"  N 为参数量(个), D 为训练 token 数(个), L 为交叉熵损失")
    print(f"  R² = {m1['R2']:.4f} (原始), {m1['R2_log']:.4f} (log)")
    print(f"{'='*60}")
    print(f"  分析完成!")


if __name__ == '__main__':
    main()
