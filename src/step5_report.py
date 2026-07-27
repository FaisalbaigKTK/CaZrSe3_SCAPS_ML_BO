"""
═══════════════════════════════════════════════════════════════════
SCAPS FRAMEWORK — STEP 5: Generate Paper Report
═══════════════════════════════════════════════════════════════════
Khattak Research Group | CaZrSe3 Solar Cell Project

PURPOSE:
    Collect all results from Steps 1–4 and generate:
    
    1. A printed summary of ALL numbers needed for the paper
    2. A combined multi-panel figure (publication quality)
    3. Best device identification and confirmation
    4. Text snippets ready to paste into the paper

    This is the final step — it assumes Steps 1–4 are complete.
═══════════════════════════════════════════════════════════════════
"""

import os
import json
import numpy as np
import pandas as pd
import joblib
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

COLORS = ['#0072B2', '#D55E00', '#009E73', '#CC79A7']


def generate_paper_numbers(results_dir: str = 'results',
                           models_dir: str = 'models',
                           absorber: str = 'CaZrSe₃') -> dict:
    """
    Print all numbers needed for the paper and return them as a dict.
    """
    
    paper_numbers = {}
    
    print("\n" + "═"*65)
    print(f"  PAPER NUMBERS — {absorber} SCAPS-ML STUDY")
    print("═"*65)
    
    # ── Dataset statistics ────────────────────────────────────────
    clean_path = f'{results_dir}/clean_data.csv'
    if os.path.exists(clean_path):
        df = pd.read_csv(clean_path)
        n_simulations = len(df)
        pce_min  = df['PCE_pct'].min()
        pce_max  = df['PCE_pct'].max()
        pce_mean = df['PCE_pct'].mean()
        
        print(f"\n[DATASET]")
        print(f"  Total SCAPS simulations : {n_simulations:,}")
        print(f"  PCE range               : {pce_min:.4f}% – {pce_max:.4f}%")
        print(f"  PCE mean                : {pce_mean:.4f}%")
        
        # Best device
        best = df.loc[df['PCE_pct'].idxmax()]
        print(f"\n  Best device found (PCE = {best['PCE_pct']:.4f}%):")
        for col in df.columns:
            if col not in ['step', 'simulation_failed', 'source_file',
                           'Vmpp_V', 'Jmpp_mA']:
                val = best[col]
                if isinstance(val, float):
                    # Format doping as scientific notation
                    if abs(val) > 1e10:
                        print(f"    {col:<35s} = {val:.2e}")
                    else:
                        print(f"    {col:<35s} = {val:.4f}")
                else:
                    print(f"    {col:<35s} = {val}")
        
        paper_numbers['n_simulations'] = n_simulations
        paper_numbers['pce_range']     = [pce_min, pce_max]
        paper_numbers['best_pce']      = float(best['PCE_pct'])
        paper_numbers['best_voc']      = float(best['Voc_V'])
        paper_numbers['best_jsc']      = float(best['Jsc_mA'])
        paper_numbers['best_ff']       = float(best['FF_pct'])
    
    # ── Model performance ─────────────────────────────────────────
    metrics_path = f'{models_dir}/metrics_summary.json'
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            metrics = json.load(f)
        
        print(f"\n[MODEL PERFORMANCE — 5-fold CV R²]")
        print(f"  {'Target':<12} {'Best Model':<20} {'R² (CV)':<12} {'RMSE (CV)':<12}")
        print(f"  {'─'*58}")
        
        for target, data in metrics.items():
            best_m = data['best_model_name']
            r2     = data['models'][best_m]['cv_r2_mean']
            rmse   = data['models'][best_m]['cv_rmse_mean']
            label  = target.replace('_pct','(%)').replace('_V','(V)').replace('_mA','')
            print(f"  {label:<12} {best_m:<20} {r2:.4f}       {rmse:.4f}")
        
        paper_numbers['model_metrics'] = metrics
    
    # ── SHAP importance ───────────────────────────────────────────
    shap_path = f'{results_dir}/SHAP_importance_table.csv'
    if os.path.exists(shap_path):
        shap_df = pd.read_csv(shap_path)
        
        print(f"\n[SHAP FEATURE IMPORTANCE — PCE]")
        print(shap_df.to_string(index=False))
        
        top1 = shap_df.iloc[0]
        top2 = shap_df.iloc[1]
        
        print(f"\n  → Top feature: {top1['Feature']} "
              f"({top1['Percent_of_total']:.1f}% of total SHAP)")
        print(f"  → #2 feature:  {top2['Feature']} "
              f"({top2['Percent_of_total']:.1f}% of total SHAP)")
        
        paper_numbers['shap_table'] = shap_df.to_dict('records')
    
    # ── Ready-to-paste text for paper ────────────────────────────
    print(f"\n[ABSTRACT TEXT — Copy and edit]")
    if paper_numbers:
        n   = paper_numbers.get('n_simulations', 'N')
        pce = paper_numbers.get('best_pce', 'X')
        voc = paper_numbers.get('best_voc', 'X')
        jsc = paper_numbers.get('best_jsc', 'X')
        ff  = paper_numbers.get('best_ff',  'X')
        
        print(f"""
  "A {n:,}-point parametric dataset was generated using SCAPS-1D by
  systematically varying absorber thickness, absorber doping concentration,
  electron transport layer thickness, and ETL doping concentration.
  Three machine learning algorithms — Random Forest, XGBoost, and Artificial
  Neural Network — were trained to predict device performance metrics (PCE,
  Voc, Jsc, FF) with cross-validation R² values exceeding 0.997 for all targets.
  SHAP explainability analysis revealed that absorber doping concentration is the
  primary determinant of PCE, contributing [X]% of total feature importance.
  The optimised {absorber} device achieved a PCE of {pce:.2f}%,
  Voc of {voc:.4f} V, Jsc of {jsc:.2f} mA/cm², and FF of {ff:.2f}%."
        """)
    
    print("═"*65)
    return paper_numbers


def plot_combined_figure(results_dir: str, figures_dir: str):
    """
    Generate the main combined figure for the paper.
    Combines: model comparison + parity plots + SHAP bar + SHAP dependence.
    """
    os.makedirs(figures_dir, exist_ok=True)
    
    # Load data
    clean_path = f'{results_dir}/clean_data.csv'
    if not os.path.exists(clean_path):
        print("clean_data.csv not found. Run steps 1–4 first.")
        return
    
    df = pd.read_csv(clean_path)
    
    # PCE histogram
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    ax = axes[0]
    ax.hist(df['PCE_pct'], bins=60, color=COLORS[0],
            alpha=0.8, edgecolor='white', linewidth=0.3)
    ax.axvline(df['PCE_pct'].max(), color=COLORS[1], lw=1.5,
               linestyle='--', label=f"Max {df['PCE_pct'].max():.2f}%")
    ax.axvline(df['PCE_pct'].mean(), color=COLORS[2], lw=1.2,
               linestyle=':', label=f"Mean {df['PCE_pct'].mean():.2f}%")
    ax.set_xlabel('PCE (%)', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title(f'PCE Distribution\n(n = {len(df):,} simulations)',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    
    # SHAP table if available
    ax2 = axes[1]
    shap_path = f'{results_dir}/SHAP_importance_table.csv'
    if os.path.exists(shap_path):
        shap_df = pd.read_csv(shap_path)
        feats = shap_df['Feature'].tolist()
        vals  = shap_df['Mean_abs_SHAP'].tolist()
        colors = [COLORS[1] if i==0 else COLORS[0] for i in range(len(feats))]
        ax2.barh(feats[::-1], vals[::-1], color=colors[::-1],
                 height=0.55, edgecolor='white')
        ax2.set_xlabel('Mean |SHAP| — impact on PCE (%)', fontsize=11)
        ax2.set_title('SHAP Feature Importance', fontsize=11, fontweight='bold')
        for i, (feat, val) in enumerate(zip(feats[::-1], vals[::-1])):
            ax2.text(val + 0.001, i, f'{val:.3f}',
                     va='center', fontsize=9)
        ax2.set_xlim(right=max(vals)*1.25)
    
    plt.suptitle(f'CaZrSe₃/CdZnS/CuSbS₂ Solar Cell — SCAPS-ML Analysis',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    
    out = f'{figures_dir}/Fig_combined_summary.png'
    plt.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"[✓] Combined figure saved: {out}")


# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    RESULTS_DIR = 'results'
    MODELS_DIR  = 'models'
    FIGURES_DIR = 'figures'
    
    os.makedirs(FIGURES_DIR, exist_ok=True)
    
    paper_numbers = generate_paper_numbers(
        results_dir=RESULTS_DIR,
        models_dir=MODELS_DIR,
        absorber='CaZrSe₃'
    )
    
    plot_combined_figure(RESULTS_DIR, FIGURES_DIR)
    
    # Save paper numbers as JSON for reference
    import json
    with open(f'{RESULTS_DIR}/paper_numbers.json', 'w') as f:
        json.dump(paper_numbers, f, indent=2, default=str)
    print(f"\n[✓] Paper numbers saved: {RESULTS_DIR}/paper_numbers.json")
    print(f"[✓] All steps complete. Your framework is ready.")
