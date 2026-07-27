"""
═══════════════════════════════════════════════════════════════════
SCAPS FRAMEWORK — STEP 2: Prepare Features for ML
═══════════════════════════════════════════════════════════════════
Khattak Research Group | CaZrSe3 Solar Cell Project

PURPOSE:
    Take the raw parsed DataFrame from Step 1 and prepare it
    for machine learning:
    
    1. Auto-detect which columns are features vs targets
    2. Apply log-transform to doping/density columns
       (they span many orders of magnitude — 10¹³ to 10¹⁸)
    3. Remove failed simulations and outliers
    4. Split into train/test sets
    5. Scale features for ANN training
    6. Save everything needed for Step 3 (ML training)

WHY LOG-TRANSFORM DOPING?
    Doping spans 5+ orders of magnitude (10¹³ to 10¹⁸).
    Without log-transform, a linear model treats the difference
    between 10¹³ and 10¹⁴ the same as between 10¹⁷ and 10¹⁸.
    After log: log(10¹³)=13, log(10¹⁸)=18 — equal spacing,
    which is physically correct for carrier recombination physics.

FUTURE-PROOF:
    When you add HTL parameters or band offset, just re-run this
    script. It auto-detects new feature columns.
═══════════════════════════════════════════════════════════════════
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib  # for saving scalers


# ─────────────────────────────────────────────────────────────────
# CONFIGURATION — edit these to match your device
# ─────────────────────────────────────────────────────────────────

# Target outputs — what we want to predict
TARGET_COLUMNS = ['PCE_pct', 'Voc_V', 'Jsc_mA', 'FF_pct']

# Shockley-Queisser limits per absorber (for sanity checking)
SQ_LIMITS = {
    'CaZrSe3': 30.4,   # Eg ~1.35 eV
    'BaZrSe3': 30.0,
    'default': 33.0,
}

# Columns that should be log-transformed (auto-detected if None)
# Set to None to auto-detect based on column names
LOG_TRANSFORM_KEYWORDS = [
    'doping', 'donor', 'acceptor', 'density',
    'NA', 'ND', 'Nt', 'concentration', 'defect'
]

# Columns to always exclude (metadata, flags)
EXCLUDE_COLUMNS = ['step', 'simulation_failed', 'source_file']


# ─────────────────────────────────────────────────────────────────
# MAIN PREPARATION FUNCTION
# ─────────────────────────────────────────────────────────────────

def prepare_features(df: pd.DataFrame,
                     absorber: str = 'CaZrSe3',
                     test_size: float = 0.2,
                     random_state: int = 42,
                     verbose: bool = True) -> dict:
    """
    Prepare a parsed SCAPS DataFrame for ML training.
    
    Parameters
    ----------
    df : pd.DataFrame
        Output from step1_parse_iv.parse_iv_file()
    absorber : str
        Absorber material name (for S-Q limit check)
    test_size : float
        Fraction of data for test set (default 0.2 = 20%)
    random_state : int
        Random seed for reproducibility
    verbose : bool
        Print detailed information
    
    Returns
    -------
    dict with keys:
        'X_train', 'X_test'        : raw feature arrays
        'X_train_sc', 'X_test_sc'  : scaled feature arrays (for ANN)
        'y_train', 'y_test'        : target dicts {target_name: array}
        'feature_names'            : list of feature column names
        'feature_names_display'    : human-readable names
        'target_names'             : list of target column names
        'scaler'                   : fitted StandardScaler
        'df_clean'                 : cleaned DataFrame
        'log_transformed_cols'     : which columns were log-transformed
        'transform_map'            : original col → transformed col name
    """

    if verbose:
        print(f"\n{'='*60}")
        print(f"STEP 2: Feature Preparation")
        print(f"  Input: {len(df)} rows × {len(df.columns)} columns")

    # ── 1. Remove failed simulations ─────────────────────────────
    n_before = len(df)
    df_clean = df[~df.get('simulation_failed', pd.Series(False, index=df.index))].copy()
    df_clean = df_clean[df_clean['PCE_pct'] >= 0].copy()
    
    if verbose:
        n_removed = n_before - len(df_clean)
        if n_removed > 0:
            print(f"\n  Removed {n_removed} failed simulations")

    # ── 2. Apply S-Q limit filter ─────────────────────────────────
    sq_limit = SQ_LIMITS.get(absorber, SQ_LIMITS['default'])
    sq_violations = df_clean[df_clean['PCE_pct'] > sq_limit]
    if len(sq_violations) > 0:
        print(f"\n  ⚠ {len(sq_violations)} rows exceed S-Q limit "
              f"({sq_limit}% for {absorber}). Removing.")
        df_clean = df_clean[df_clean['PCE_pct'] <= sq_limit].copy()

    # ── 3. Auto-detect feature columns ───────────────────────────
    # Features = all numeric columns that are not targets and not excluded
    all_cols = df_clean.columns.tolist()
    feature_cols = []
    for col in all_cols:
        if col in TARGET_COLUMNS:
            continue
        if col in EXCLUDE_COLUMNS:
            continue
        if col.startswith('Vmpp') or col.startswith('Jmpp'):
            continue  # derived output, not an input feature
        if df_clean[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
            feature_cols.append(col)

    if verbose:
        print(f"\n  Feature columns detected ({len(feature_cols)}):")
        for c in feature_cols:
            print(f"    {c}")

    # ── 4. Log-transform doping/density columns ───────────────────
    log_transformed_cols = []
    transform_map = {}  # original_col → transformed_col

    df_features = df_clean[feature_cols].copy()

    for col in feature_cols:
        # Check if this column should be log-transformed
        should_log = any(kw.lower() in col.lower()
                         for kw in LOG_TRANSFORM_KEYWORDS)
        
        if should_log:
            # Verify all values are positive (required for log)
            if (df_features[col] > 0).all():
                new_col = f'log10_{col}'
                df_features[new_col] = np.log10(df_features[col])
                df_features.drop(columns=[col], inplace=True)
                log_transformed_cols.append(col)
                transform_map[col] = new_col
                
                if verbose:
                    min_v = df_clean[col].min()
                    max_v = df_clean[col].max()
                    print(f"\n  Log-transformed: {col}")
                    print(f"    Original range: {min_v:.2e} – {max_v:.2e}")
                    print(f"    Log range:      {np.log10(min_v):.1f} – {np.log10(max_v):.1f}")

    # Final feature column names after transformation
    final_feature_cols = df_features.columns.tolist()

    if verbose:
        print(f"\n  Final feature set ({len(final_feature_cols)} features):")
        for col in final_feature_cols:
            vals = df_features[col]
            print(f"    {col:<45s} [{vals.min():.3f} – {vals.max():.3f}]")

    # ── 5. Create human-readable display names ────────────────────
    def make_display_name(col: str) -> str:
        """Convert cleaned column name to readable label."""
        col = col.replace('log10_', 'log₁₀ ')
        col = col.replace('_', ' ')
        col = col.replace('um', '(µm)')
        col = col.replace('1cm3', '(cm⁻³)')
        col = col.replace('cm3', '(cm⁻³)')
        col = col.replace('eV', '(eV)')
        col = col.replace('pct', '(%)')
        col = col.replace('V ', 'V ')
        return col.strip()

    display_names = {col: make_display_name(col) for col in final_feature_cols}

    # ── 6. Build X (features) and y (targets) arrays ─────────────
    X = df_features.values.astype(np.float64)
    
    # Check for NaN
    nan_count = np.isnan(X).sum()
    if nan_count > 0:
        print(f"\n  ⚠ {nan_count} NaN values in features. "
              f"Filling with column median.")
        for j in range(X.shape[1]):
            col_mask = np.isnan(X[:, j])
            if col_mask.any():
                X[col_mask, j] = np.nanmedian(X[:, j])

    y = {}
    for target in TARGET_COLUMNS:
        if target in df_clean.columns:
            y[target] = df_clean[target].values.astype(np.float64)

    # ── 7. Train/test split ───────────────────────────────────────
    indices = np.arange(len(X))
    idx_train, idx_test = train_test_split(
        indices, test_size=test_size,
        random_state=random_state, shuffle=True
    )

    X_train = X[idx_train]
    X_test  = X[idx_test]
    y_train = {t: arr[idx_train] for t, arr in y.items()}
    y_test  = {t: arr[idx_test]  for t, arr in y.items()}

    # ── 8. Scale features (needed for ANN, optional for RF/XGB) ──
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    if verbose:
        print(f"\n  Train set: {len(X_train)} samples ({100*(1-test_size):.0f}%)")
        print(f"  Test set:  {len(X_test)} samples ({100*test_size:.0f}%)")
        print(f"\n  Target statistics (train set):")
        for t, arr in y_train.items():
            print(f"    {t:<12s}  min={arr.min():.4f}  "
                  f"max={arr.max():.4f}  mean={arr.mean():.4f}")
        print(f"{'='*60}\n")

    return {
        'X_train':             X_train,
        'X_test':              X_test,
        'X_train_sc':          X_train_sc,
        'X_test_sc':           X_test_sc,
        'y_train':             y_train,
        'y_test':              y_test,
        'feature_names':       final_feature_cols,
        'feature_names_display': [display_names[c] for c in final_feature_cols],
        'target_names':        list(y.keys()),
        'scaler':              scaler,
        'df_clean':            df_clean,
        'df_features':         df_features,
        'log_transformed_cols': log_transformed_cols,
        'transform_map':       transform_map,
        'idx_train':           idx_train,
        'idx_test':            idx_test,
        'X_full':              X,
        'y_full':              y,
    }


def save_prepared_data(prepared: dict, output_dir: str = 'results'):
    """
    Save all prepared data to disk so you can reload without re-running.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save arrays
    np.save(f'{output_dir}/X_train.npy',    prepared['X_train'])
    np.save(f'{output_dir}/X_test.npy',     prepared['X_test'])
    np.save(f'{output_dir}/X_train_sc.npy', prepared['X_train_sc'])
    np.save(f'{output_dir}/X_test_sc.npy',  prepared['X_test_sc'])
    np.save(f'{output_dir}/X_full.npy',     prepared['X_full'])
    
    for target in prepared['target_names']:
        safe = target.replace('%', 'pct').replace('/', '_')
        np.save(f'{output_dir}/y_train_{safe}.npy', prepared['y_train'][target])
        np.save(f'{output_dir}/y_test_{safe}.npy',  prepared['y_test'][target])
        np.save(f'{output_dir}/y_full_{safe}.npy',  prepared['y_full'][target])
    
    # Save scaler
    joblib.dump(prepared['scaler'], f'{output_dir}/scaler.pkl')
    
    # Save metadata as CSV for inspection
    meta = pd.DataFrame({
        'feature_name':  prepared['feature_names'],
        'display_name':  prepared['feature_names_display'],
    })
    meta.to_csv(f'{output_dir}/feature_metadata.csv', index=False)
    
    # Save clean DataFrame
    prepared['df_clean'].to_csv(f'{output_dir}/clean_data.csv', index=False)
    
    print(f"[✓] Prepared data saved to: {output_dir}/")
    print(f"    Files: X_train/test.npy, y_*.npy, scaler.pkl, "
          f"feature_metadata.csv, clean_data.csv")


# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from step1_parse_iv import load_dataset
    
    # Load data saved from Step 1
    DATA_PATH = 'results/parsed_data.csv'
    
    if not os.path.exists(DATA_PATH):
        print(f"Error: {DATA_PATH} not found.")
        print("Run step1_parse_iv.py first.")
        sys.exit(1)
    
    df = load_dataset(DATA_PATH)
    
    # Prepare features
    prepared = prepare_features(
        df,
        absorber='CaZrSe3',
        test_size=0.2,
        random_state=42,
        verbose=True
    )
    
    # Save for Step 3
    save_prepared_data(prepared, output_dir='results')
    
    print("[✓] Step 2 complete. Run step3_train_models.py next.")
