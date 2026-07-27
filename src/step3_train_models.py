"""
═══════════════════════════════════════════════════════════════════
SCAPS FRAMEWORK — STEP 3: Train ML Models
═══════════════════════════════════════════════════════════════════
Khattak Research Group | CaZrSe3 Solar Cell Project

PURPOSE:
    Train three ML models (Random Forest, XGBoost, ANN) on the
    prepared SCAPS dataset and evaluate their performance using
    5-fold cross-validation.

    For each target (PCE, Voc, Jsc, FF):
    - Train all three models
    - Evaluate with R², RMSE, MAE
    - Generate parity plots (predicted vs actual)
    - Select the best model

    Results are saved for use in Step 4 (SHAP analysis).

WHY THREE MODELS?
    - Random Forest: robust, handles non-linearity, good baseline
    - XGBoost: often best accuracy, fast on tabular data
    - ANN: catches complex interactions, needs scaled features
    Comparing all three is required for a publishable ML paper.

WHY 5-FOLD CV?
    With large datasets (n > 1000), 5-fold CV gives reliable
    performance estimates. It trains on 80% and validates on 20%,
    repeated 5 times — every sample is used for validation once.
═══════════════════════════════════════════════════════════════════
"""

import os
import json
import numpy as np
import pandas as pd
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import xgboost as xgb

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# ─────────────────────────────────────────────────────────────────
# MODEL DEFINITIONS
# Hyperparameters chosen for solar cell SCAPS datasets
# ─────────────────────────────────────────────────────────────────

def get_models(n_samples: int) -> dict:
    """
    Return model definitions appropriate for the dataset size.
    Automatically scales complexity with sample count.
    """
    # Scale estimators with dataset size
    n_trees = min(500, max(100, n_samples // 20))
    
    models = {
        'Random Forest': RandomForestRegressor(
            n_estimators=n_trees,
            max_depth=None,          # grow full trees
            min_samples_leaf=2,      # prevent overfitting on small leaves
            max_features='sqrt',     # standard RF heuristic
            random_state=42,
            n_jobs=-1,               # use all CPU cores
        ),
        'XGBoost': xgb.XGBRegressor(
            n_estimators=n_trees,
            max_depth=6,             # good default for tabular data
            learning_rate=0.05,      # slow learning = better generalisation
            subsample=0.8,           # row sampling per tree
            colsample_bytree=0.8,    # column sampling per tree
            min_child_weight=3,
            random_state=42,
            verbosity=0,
            n_jobs=-1,
        ),
        'ANN': MLPRegressor(
            hidden_layer_sizes=(128, 64, 32),  # 3-layer network
            activation='relu',
            solver='adam',
            learning_rate_init=0.001,
            max_iter=500,
            early_stopping=True,     # stop when validation loss plateaus
            validation_fraction=0.1,
            n_iter_no_change=20,
            random_state=42,
        ),
    }
    return models


# ─────────────────────────────────────────────────────────────────
# TRAINING AND EVALUATION
# ─────────────────────────────────────────────────────────────────

def train_and_evaluate(X_train, X_train_sc, y_train_dict,
                       X_test, X_test_sc, y_test_dict,
                       feature_names, verbose=True) -> dict:
    """
    Train all models on all targets. Return full results.
    
    Parameters
    ----------
    X_train, X_test         : raw features (for RF, XGBoost)
    X_train_sc, X_test_sc   : scaled features (for ANN)
    y_train_dict, y_test_dict : {target_name: array}
    feature_names           : list of feature column names
    
    Returns
    -------
    dict with model objects, metrics, predictions, best model per target
    """
    results = {}
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    n_samples = len(X_train)
    models_def = get_models(n_samples)
    
    for target_name, y_train in y_train_dict.items():
        y_test = y_test_dict[target_name]
        
        if verbose:
            print(f"\n{'─'*55}")
            print(f"  Target: {target_name}")
            print(f"  Range: [{y_train.min():.4f}, {y_train.max():.4f}]")
            print(f"{'─'*55}")

        target_results = {}
        
        for model_name, model in models_def.items():
            # ANN uses scaled features, others use raw
            use_scaled = (model_name == 'ANN')
            X_tr = X_train_sc if use_scaled else X_train
            X_te = X_test_sc  if use_scaled else X_test
            
            # ── Cross-validation ──────────────────────────────────
            cv_r2   = cross_val_score(model, X_tr, y_train, cv=kf,
                                       scoring='r2', n_jobs=-1)
            cv_rmse = np.sqrt(-cross_val_score(
                model, X_tr, y_train, cv=kf,
                scoring='neg_mean_squared_error', n_jobs=-1))
            cv_mae  = -cross_val_score(model, X_tr, y_train, cv=kf,
                                        scoring='neg_mean_absolute_error',
                                        n_jobs=-1)

            # ── Train on full training set, predict test set ──────
            model.fit(X_tr, y_train)
            y_pred_test = model.predict(X_te)
            
            test_r2   = r2_score(y_test, y_pred_test)
            test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
            test_mae  = mean_absolute_error(y_test, y_pred_test)

            # ── Out-of-fold predictions for parity plots ──────────
            # This gives an honest prediction for EVERY training sample
            y_oof = np.zeros_like(y_train)
            for train_idx, val_idx in kf.split(X_tr):
                m = get_models(n_samples)[model_name]
                m.fit(X_tr[train_idx], y_train[train_idx])
                y_oof[val_idx] = m.predict(X_tr[val_idx])
            
            if verbose:
                print(f"\n  {model_name}:")
                print(f"    CV  R²:   {cv_r2.mean():.4f} ± {cv_r2.std():.4f}")
                print(f"    CV  RMSE: {cv_rmse.mean():.4f} ± {cv_rmse.std():.4f}")
                print(f"    Test R²:  {test_r2:.4f}")
                print(f"    Test RMSE:{test_rmse:.4f}")

            target_results[model_name] = {
                'model':          model,
                'cv_r2_mean':     float(cv_r2.mean()),
                'cv_r2_std':      float(cv_r2.std()),
                'cv_rmse_mean':   float(cv_rmse.mean()),
                'cv_rmse_std':    float(cv_rmse.std()),
                'cv_mae_mean':    float(cv_mae.mean()),
                'test_r2':        float(test_r2),
                'test_rmse':      float(test_rmse),
                'test_mae':       float(test_mae),
                'y_pred_test':    y_pred_test,
                'y_oof':          y_oof,
                'use_scaled':     use_scaled,
            }

        # Determine best model (by CV R²)
        best_model_name = max(
            target_results,
            key=lambda m: target_results[m]['cv_r2_mean']
        )
        if verbose:
            print(f"\n  ★ Best: {best_model_name} "
                  f"(CV R² = {target_results[best_model_name]['cv_r2_mean']:.4f})")
        
        results[target_name] = {
            'models':          target_results,
            'best_model_name': best_model_name,
            'y_train':         y_train,
            'y_test':          y_test,
        }

    return results


# ─────────────────────────────────────────────────────────────────
# FIGURE GENERATION
# ─────────────────────────────────────────────────────────────────

def plot_parity_plots(results: dict, output_dir: str,
                      feature_names_display: list = None):
    """
    Generate parity plots (predicted vs actual) for all targets.
    One panel per target, showing best model's out-of-fold predictions.
    """
    targets = list(results.keys())
    n_targets = len(targets)
    
    COLORS = ['#0072B2', '#D55E00', '#009E73', '#CC79A7']
    
    fig, axes = plt.subplots(1, n_targets, figsize=(4.5 * n_targets, 4.5))
    if n_targets == 1:
        axes = [axes]
    
    for i, (target, ax) in enumerate(zip(targets, axes)):
        best_name = results[target]['best_model_name']
        best      = results[target]['models'][best_name]
        y_true    = results[target]['y_train']
        y_pred    = best['y_oof']
        
        # Subsample if very large for faster plotting
        if len(y_true) > 5000:
            idx = np.random.choice(len(y_true), 5000, replace=False)
            y_true_plot, y_pred_plot = y_true[idx], y_pred[idx]
        else:
            y_true_plot, y_pred_plot = y_true, y_pred
        
        ax.scatter(y_true_plot, y_pred_plot,
                   alpha=0.3, s=6, color=COLORS[i % len(COLORS)],
                   edgecolors='none', rasterized=True)
        
        lims = [min(y_true.min(), y_pred.min()) * 0.97,
                max(y_true.max(), y_pred.max()) * 1.03]
        ax.plot(lims, lims, 'k--', lw=0.9, alpha=0.6)
        
        r2   = best['test_r2']
        rmse = best['test_rmse']
        ax.text(0.05, 0.93,
                f'R² = {r2:.4f}\nRMSE = {rmse:.4f}',
                transform=ax.transAxes, fontsize=9,
                bbox=dict(boxstyle='round,pad=0.3',
                          facecolor='white', alpha=0.9))
        
        unit_map = {'PCE_pct': '%', 'Voc_V': 'V',
                    'Jsc_mA': 'mA/cm²', 'FF_pct': '%'}
        unit = unit_map.get(target, '')
        label = target.replace('_pct', ' (%)').replace('_V', ' (V)').replace('_mA', ' (mA/cm²)')
        
        ax.set_xlabel(f'SCAPS-1D {label}', fontsize=10)
        ax.set_ylabel(f'Predicted {label}', fontsize=10)
        ax.set_title(f'{label}\n({best_name})', fontsize=10, fontweight='bold')
        ax.set_xlim(lims); ax.set_ylim(lims)
        ax.tick_params(labelsize=9)

    plt.suptitle('Parity Plots: ML Predicted vs SCAPS-1D Simulated Values',
                 fontsize=11, fontweight='bold')
    plt.tight_layout()
    
    out_path = f'{output_dir}/Fig_parity_plots.png'
    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"[✓] Saved: {out_path}")


def plot_model_comparison(results: dict, output_dir: str):
    """
    Heatmap comparing all models across all targets.
    Shows CV R² for every combination.
    """
    targets     = list(results.keys())
    model_names = list(list(results.values())[0]['models'].keys())
    
    r2_matrix = np.array([
        [results[t]['models'][m]['cv_r2_mean'] for m in model_names]
        for t in targets
    ])

    fig, ax = plt.subplots(figsize=(6, 3.5))
    im = ax.imshow(r2_matrix, cmap='YlGnBu', vmin=0.95, vmax=1.0, aspect='auto')
    
    ax.set_xticks(range(len(model_names)))
    ax.set_xticklabels(model_names, fontsize=9)
    ax.set_yticks(range(len(targets)))
    labels = [t.replace('_pct', ' (%)').replace('_V', ' (V)').replace('_mA', '') for t in targets]
    ax.set_yticklabels(labels, fontsize=9)
    
    for i in range(len(targets)):
        for j in range(len(model_names)):
            v = r2_matrix[i, j]
            ax.text(j, i, f'{v:.4f}',
                    ha='center', va='center', fontsize=8.5,
                    color='white' if v > 0.998 else 'black',
                    fontweight='bold')
    
    plt.colorbar(im, ax=ax, shrink=0.9, label='R² (5-fold CV)')
    ax.set_title('Model Performance Comparison (R², 5-fold CV)', fontweight='bold', fontsize=10)
    plt.tight_layout()
    
    out_path = f'{output_dir}/Fig_model_comparison.png'
    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"[✓] Saved: {out_path}")


def plot_learning_curve(X_train, y_train, model_name='XGBoost',
                        output_dir='figures'):
    """
    Plot learning curve — shows whether more data would help.
    Important for justifying the dataset size in the paper.
    """
    from sklearn.model_selection import learning_curve
    
    n_samples = len(X_train)
    model = get_models(n_samples)[model_name]
    
    train_sizes = np.linspace(0.1, 1.0, 10)
    train_sz, train_scores, val_scores = learning_curve(
        model, X_train, y_train,
        train_sizes=train_sizes,
        cv=5, scoring='r2',
        n_jobs=-1, random_state=42
    )
    
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.fill_between(train_sz,
                    train_scores.mean(1) - train_scores.std(1),
                    train_scores.mean(1) + train_scores.std(1),
                    alpha=0.2, color='#0072B2')
    ax.fill_between(train_sz,
                    val_scores.mean(1) - val_scores.std(1),
                    val_scores.mean(1) + val_scores.std(1),
                    alpha=0.2, color='#D55E00')
    ax.plot(train_sz, train_scores.mean(1), 'o-', color='#0072B2',
            lw=1.8, ms=5, label='Training R²')
    ax.plot(train_sz, val_scores.mean(1), 's-', color='#D55E00',
            lw=1.8, ms=5, label='Validation R²')
    
    ax.set_xlabel('Training set size', fontsize=11)
    ax.set_ylabel('R² Score', fontsize=11)
    ax.set_title(f'Learning Curve — {model_name}\n(PCE prediction)',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.set_ylim([0.8, 1.01])
    ax.grid(True, alpha=0.3, lw=0.5)
    plt.tight_layout()
    
    out_path = f'{output_dir}/Fig_learning_curve.png'
    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"[✓] Saved: {out_path}")


# ─────────────────────────────────────────────────────────────────
# SAVE / LOAD
# ─────────────────────────────────────────────────────────────────

def save_models(results: dict, output_dir: str = 'models'):
    """Save all trained model objects and metrics."""
    os.makedirs(output_dir, exist_ok=True)
    
    metrics_summary = {}
    
    for target, target_data in results.items():
        safe_target = target.replace('%', 'pct').replace('/', '_')
        target_metrics = {}
        
        for model_name, model_data in target_data['models'].items():
            safe_model = model_name.replace(' ', '_')
            
            # Save model object
            joblib.dump(model_data['model'],
                        f'{output_dir}/{safe_target}_{safe_model}.pkl')
            
            # Collect metrics (no model objects — not JSON serializable)
            target_metrics[model_name] = {
                'cv_r2_mean':   model_data['cv_r2_mean'],
                'cv_r2_std':    model_data['cv_r2_std'],
                'cv_rmse_mean': model_data['cv_rmse_mean'],
                'cv_rmse_std':  model_data['cv_rmse_std'],
                'cv_mae_mean':  model_data['cv_mae_mean'],
                'test_r2':      model_data['test_r2'],
                'test_rmse':    model_data['test_rmse'],
                'test_mae':     model_data['test_mae'],
            }
        
        metrics_summary[target] = {
            'models':          target_metrics,
            'best_model_name': target_data['best_model_name'],
        }
    
    with open(f'{output_dir}/metrics_summary.json', 'w') as f:
        json.dump(metrics_summary, f, indent=2)
    
    print(f"[✓] Models saved to: {output_dir}/")
    print(f"[✓] Metrics saved:   {output_dir}/metrics_summary.json")


# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import sys
    
    RESULTS_DIR = 'results'
    FIGURES_DIR = 'figures'
    MODELS_DIR  = 'models'
    
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR,  exist_ok=True)
    
    # ── Load prepared data from Step 2 ───────────────────────────
    if not os.path.exists(f'{RESULTS_DIR}/X_train.npy'):
        print("Error: Step 2 output not found. Run step2_prepare_features.py first.")
        sys.exit(1)
    
    X_train    = np.load(f'{RESULTS_DIR}/X_train.npy')
    X_test     = np.load(f'{RESULTS_DIR}/X_test.npy')
    X_train_sc = np.load(f'{RESULTS_DIR}/X_train_sc.npy')
    X_test_sc  = np.load(f'{RESULTS_DIR}/X_test_sc.npy')
    
    feature_meta = pd.read_csv(f'{RESULTS_DIR}/feature_metadata.csv')
    feature_names = feature_meta['feature_name'].tolist()
    feature_names_display = feature_meta['display_name'].tolist()
    
    target_names = ['PCE_pct', 'Voc_V', 'Jsc_mA', 'FF_pct']
    y_train_dict = {}
    y_test_dict  = {}
    
    for t in target_names:
        safe = t.replace('%', 'pct').replace('/', '_')
        train_path = f'{RESULTS_DIR}/y_train_{safe}.npy'
        test_path  = f'{RESULTS_DIR}/y_test_{safe}.npy'
        if os.path.exists(train_path):
            y_train_dict[t] = np.load(train_path)
            y_test_dict[t]  = np.load(test_path)
    
    if not y_train_dict:
        print("Error: No target data found in results/. Run step2 first.")
        sys.exit(1)
    
    print(f"Loaded: X_train {X_train.shape}, targets: {list(y_train_dict.keys())}")
    
    # ── Train all models ──────────────────────────────────────────
    print(f"\n{'='*60}")
    print("STEP 3: Training ML Models")
    print(f"{'='*60}")
    
    results = train_and_evaluate(
        X_train, X_train_sc, y_train_dict,
        X_test, X_test_sc, y_test_dict,
        feature_names, verbose=True
    )
    
    # ── Generate figures ──────────────────────────────────────────
    print(f"\nGenerating figures...")
    plot_model_comparison(results, FIGURES_DIR)
    plot_parity_plots(results, FIGURES_DIR, feature_names_display)
    
    # Learning curve for PCE (the primary target)
    if 'PCE_pct' in y_train_dict:
        plot_learning_curve(X_train, y_train_dict['PCE_pct'],
                            model_name='XGBoost', output_dir=FIGURES_DIR)
    
    # ── Save models and metrics ───────────────────────────────────
    save_models(results, MODELS_DIR)
    
    print(f"\n[✓] Step 3 complete. Run step4_shap_analysis.py next.")
