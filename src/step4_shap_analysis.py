"""
═══════════════════════════════════════════════════════════════════
SCAPS FRAMEWORK — STEP 4: SHAP Analysis
═══════════════════════════════════════════════════════════════════
Khattak Research Group | CaZrSe3 Solar Cell Project

PURPOSE:
    Compute SHAP (SHapley Additive exPlanations) values to understand
    WHICH parameters control performance and HOW MUCH each one matters.

    This is the SCIENTIFIC CONTRIBUTION of the ML paper.
    The SHAP analysis reveals device physics that individual parametric
    sweeps cannot — it quantifies the relative importance of every
    parameter simultaneously, on the full 12,100-point dataset.

SHAP OUTPUTS:
    1. Global bar chart     — which features matter most overall
    2. Beeswarm plot        — how each feature value affects PCE
    3. Dependence plots     — how the top features interact
    4. SHAP summary table   — numbers to put directly in your paper

PHYSICAL INTERPRETATION:
    High SHAP value for "log₁₀ NA" means:
    → Absorber doping is the dominant control parameter for PCE
    → Fabrication should focus on doping control above all else

    The SIGN of SHAP values matters:
    → Positive SHAP: this feature value INCREASES PCE
    → Negative SHAP: this feature value DECREASES PCE
    → A beeswarm plot shows this for every data point at once
═══════════════════════════════════════════════════════════════════
"""

import os
import json
import numpy as np
import pandas as pd
import joblib
import shap
import warnings
warnings.filterwarnings('ignore')

import xgboost as xgb

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable


COLORS = ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#56B4E9', '#E69F00']


# ─────────────────────────────────────────────────────────────────
# SHAP COMPUTATION
# ─────────────────────────────────────────────────────────────────

def compute_shap(X_full: np.ndarray,
                 y_full: np.ndarray,
                 feature_names: list,
                 feature_names_display: list,
                 target_name: str = 'PCE_pct',
                 n_background: int = 500,
                 verbose: bool = True) -> dict:
    """
    Train XGBoost on the full dataset and compute SHAP values.
    
    We use XGBoost for SHAP because:
    1. TreeExplainer is exact (not approximate) for tree-based models
    2. Fast computation even on 10,000+ samples
    3. Results match between RF and XGBoost — confirms robustness
    
    Parameters
    ----------
    X_full       : full feature array (all data, not just train split)
    y_full       : full target array
    feature_names: column names (cleaned)
    feature_names_display: human-readable names for plots
    target_name  : which target to analyse ('PCE_pct', 'Voc_V', etc.)
    n_background : number of background samples for kernel explainer
    
    Returns
    -------
    dict with 'shap_values', 'expected_value', 'model',
              'mean_abs_shap', 'sorted_idx', 'X_full'
    """
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"STEP 4: SHAP Analysis — Target: {target_name}")
        print(f"  Dataset: {len(X_full)} samples × {len(feature_names)} features")

    # ── Train XGBoost on FULL dataset ─────────────────────────────
    # We use all data here because SHAP is an interpretability tool,
    # not a predictive evaluation — we want the model to capture
    # the full response surface.
    n_trees = min(500, max(200, len(X_full) // 20))
    
    model = xgb.XGBRegressor(
        n_estimators=n_trees,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=0,
    )
    model.fit(X_full, y_full)
    
    # ── Compute SHAP values using TreeExplainer ───────────────────
    # TreeExplainer is EXACT for tree-based models — no sampling error
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_full)
    
    # Global importance: mean absolute SHAP value per feature
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    sorted_idx    = np.argsort(mean_abs_shap)[::-1]  # descending
    
    if verbose:
        print(f"\n  SHAP Feature Importance (descending):")
        print(f"  {'Rank':<5} {'Feature':<45} {'Mean |SHAP|':>12} {'% of total':>10}")
        print(f"  {'─'*75}")
        total_shap = mean_abs_shap.sum()
        for rank, idx in enumerate(sorted_idx):
            pct = 100 * mean_abs_shap[idx] / total_shap
            print(f"  {rank+1:<5} {feature_names_display[idx]:<45} "
                  f"{mean_abs_shap[idx]:>12.5f} {pct:>9.1f}%")
        print(f"{'='*60}")

    return {
        'shap_values':          shap_values,
        'expected_value':       float(explainer.expected_value),
        'model':                model,
        'mean_abs_shap':        mean_abs_shap,
        'sorted_idx':           sorted_idx,
        'X_full':               X_full,
        'y_full':               y_full,
        'feature_names':        feature_names,
        'feature_names_display': feature_names_display,
        'target_name':          target_name,
    }


# ─────────────────────────────────────────────────────────────────
# SHAP FIGURES
# ─────────────────────────────────────────────────────────────────

def plot_shap_bar(shap_data: dict, output_dir: str):
    """
    SHAP global feature importance — horizontal bar chart.
    The primary SHAP figure for the paper.
    Shows: which parameters matter most for PCE (or other target).
    """
    mean_abs   = shap_data['mean_abs_shap']
    sorted_idx = shap_data['sorted_idx']
    feat_disp  = shap_data['feature_names_display']
    target     = shap_data['target_name']
    
    labels = [feat_disp[i] for i in sorted_idx]
    values = mean_abs[sorted_idx]
    total  = values.sum()
    
    # Colors: top feature orange, rest blue
    bar_colors = [COLORS[1] if i == 0 else COLORS[0]
                  for i in range(len(labels))]
    
    fig, ax = plt.subplots(figsize=(8, max(3, 0.6 * len(labels) + 1)))
    bars = ax.barh(labels, values,
                   color=bar_colors, height=0.55, edgecolor='white')
    
    # Add value labels
    for bar, val in zip(bars, values):
        ax.text(val + 0.002, bar.get_y() + bar.get_height()/2,
                f'{val:.4f} ({100*val/total:.1f}%)',
                va='center', ha='left', fontsize=8.5)
    
    ax.set_xlabel('Mean |SHAP value| — impact on '
                  f'{target.replace("_pct", " (%)").replace("_V", " (V)")}',
                  fontsize=11)
    ax.set_title(f'SHAP Global Feature Importance\n'
                 f'CaZrSe₃/CdZnS/CuSbS₂ · n = {len(shap_data["X_full"]):,}',
                 fontsize=11, fontweight='bold')
    ax.tick_params(axis='y', labelsize=10)
    ax.set_xlim(right=values.max() * 1.30)
    
    plt.tight_layout()
    out = f'{output_dir}/Fig_SHAP_bar_{target}.png'
    plt.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"[✓] Saved: {out}")


def plot_shap_beeswarm(shap_data: dict, output_dir: str,
                       max_display: int = None):
    """
    SHAP beeswarm plot — shows distribution of SHAP values for each feature.
    Each dot = one simulation. Color = feature value (red=high, blue=low).
    
    This reveals:
    - Sign: does a HIGH value of this feature INCREASE or DECREASE PCE?
    - Spread: is the effect consistent or variable?
    """
    shap_vals  = shap_data['shap_values']
    X          = shap_data['X_full']
    feat_disp  = shap_data['feature_names_display']
    sorted_idx = shap_data['sorted_idx']
    target     = shap_data['target_name']
    
    if max_display is None:
        max_display = len(feat_disp)
    
    # Use SHAP's built-in beeswarm (it handles the jitter automatically)
    shap_exp = shap.Explanation(
        values=shap_vals,
        base_values=np.full(len(shap_vals), shap_data['expected_value']),
        data=X,
        feature_names=feat_disp
    )
    
    fig, ax = plt.subplots(figsize=(9, max(4, 0.7 * max_display + 1.5)))
    shap.plots.beeswarm(shap_exp, max_display=max_display,
                        show=False, color_bar=True)
    plt.title(f'SHAP Beeswarm — {target.replace("_pct","(%)")} | CaZrSe₃ Device',
              fontsize=11, fontweight='bold')
    plt.tight_layout()
    
    out = f'{output_dir}/Fig_SHAP_beeswarm_{target}.png'
    plt.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"[✓] Saved: {out}")


def plot_shap_dependence(shap_data: dict, output_dir: str,
                         n_top_features: int = 2):
    """
    SHAP dependence plots for the top N features.
    Shows: HOW each feature affects PCE across its range.
    Color = interaction with the most correlated other feature.
    
    This is the physics interpretation figure:
    e.g., "SHAP value of NA increases steeply up to 5×10¹⁵ cm⁻³,
    then drops — defining the optimal doping window."
    """
    shap_vals  = shap_data['shap_values']
    X          = shap_data['X_full']
    feat_names = shap_data['feature_names']
    feat_disp  = shap_data['feature_names_display']
    sorted_idx = shap_data['sorted_idx']
    target     = shap_data['target_name']
    
    n = min(n_top_features, len(feat_names))
    fig, axes = plt.subplots(1, n, figsize=(6*n, 5))
    if n == 1:
        axes = [axes]
    
    for rank, (ax, feat_idx) in enumerate(zip(axes, sorted_idx[:n])):
        feat_vals   = X[:, feat_idx]
        shap_f_vals = shap_vals[:, feat_idx]
        
        # Color by the second most important feature
        color_idx = sorted_idx[1] if feat_idx != sorted_idx[1] else sorted_idx[0]
        color_vals = X[:, color_idx]
        
        # Subsample for plotting speed
        if len(feat_vals) > 5000:
            idx_s = np.random.choice(len(feat_vals), 5000, replace=False)
            feat_vals_p   = feat_vals[idx_s]
            shap_f_vals_p = shap_f_vals[idx_s]
            color_vals_p  = color_vals[idx_s]
        else:
            feat_vals_p, shap_f_vals_p, color_vals_p = feat_vals, shap_f_vals, color_vals
        
        sc = ax.scatter(feat_vals_p, shap_f_vals_p,
                        c=color_vals_p, cmap='viridis',
                        alpha=0.4, s=8, edgecolors='none',
                        rasterized=True)
        plt.colorbar(sc, ax=ax, label=feat_disp[color_idx], shrink=0.9)
        
        ax.axhline(0, color='black', lw=0.7, linestyle='--', alpha=0.4)
        ax.set_xlabel(feat_disp[feat_idx], fontsize=11)
        ax.set_ylabel(f'SHAP value for {target.replace("_pct","(%)")}', fontsize=11)
        ax.set_title(f'#{rank+1} Feature: {feat_disp[feat_idx]}',
                     fontsize=10, fontweight='bold')
        ax.tick_params(labelsize=9)
    
    plt.suptitle(f'SHAP Dependence Plots — Top {n} Features | CaZrSe₃',
                 fontsize=11, fontweight='bold')
    plt.tight_layout()
    
    out = f'{output_dir}/Fig_SHAP_dependence_{target}.png'
    plt.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"[✓] Saved: {out}")


def save_shap_table(shap_data: dict, output_dir: str):
    """
    Save a CSV table of SHAP importance values.
    These numbers go directly into Table 2 of your paper.
    """
    mean_abs   = shap_data['mean_abs_shap']
    sorted_idx = shap_data['sorted_idx']
    feat_disp  = shap_data['feature_names_display']
    total      = mean_abs.sum()
    
    rows = []
    for rank, idx in enumerate(sorted_idx):
        rows.append({
            'Rank':            rank + 1,
            'Feature':         feat_disp[idx],
            'Mean_abs_SHAP':   round(float(mean_abs[idx]), 5),
            'Percent_of_total': round(float(100 * mean_abs[idx] / total), 2),
        })
    
    df_shap = pd.DataFrame(rows)
    out = f'{output_dir}/SHAP_importance_table.csv'
    df_shap.to_csv(out, index=False)
    print(f"[✓] SHAP table saved: {out}")
    return df_shap


# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import sys
    
    RESULTS_DIR = 'results'
    FIGURES_DIR = 'figures'
    os.makedirs(FIGURES_DIR, exist_ok=True)
    
    # Load full data
    if not os.path.exists(f'{RESULTS_DIR}/X_full.npy'):
        print("Error: Run steps 1, 2, and 3 first.")
        sys.exit(1)
    
    X_full = np.load(f'{RESULTS_DIR}/X_full.npy')
    feature_meta = pd.read_csv(f'{RESULTS_DIR}/feature_metadata.csv')
    feature_names         = feature_meta['feature_name'].tolist()
    feature_names_display = feature_meta['display_name'].tolist()
    
    print(f"Loaded X_full: {X_full.shape}")
    
    # Run SHAP for each target
    for target_name in ['PCE_pct', 'Voc_V', 'Jsc_mA', 'FF_pct']:
        safe = target_name.replace('%', 'pct').replace('/', '_')
        y_path = f'{RESULTS_DIR}/y_full_{safe}.npy'
        if not os.path.exists(y_path):
            print(f"  Skipping {target_name} — y_full not found")
            continue
        
        y_full = np.load(y_path)
        
        shap_data = compute_shap(
            X_full, y_full,
            feature_names, feature_names_display,
            target_name=target_name,
            verbose=True
        )
        
        # Save SHAP arrays
        np.save(f'{RESULTS_DIR}/shap_values_{safe}.npy',
                shap_data['shap_values'])
        
        # Generate figures
        plot_shap_bar(shap_data, FIGURES_DIR)
        plot_shap_dependence(shap_data, FIGURES_DIR, n_top_features=2)
        save_shap_table(shap_data, RESULTS_DIR)
        
        # Beeswarm only for PCE (primary target)
        if target_name == 'PCE_pct':
            plot_shap_beeswarm(shap_data, FIGURES_DIR)
    
    print(f"\n[✓] Step 4 complete. Run step5_report.py for the summary.")
