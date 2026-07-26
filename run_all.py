"""
═══════════════════════════════════════════════════════════════════
SCAPS FRAMEWORK — run_all.py
MASTER PIPELINE: Runs all 5 steps in sequence
═══════════════════════════════════════════════════════════════════
Khattak Research Group | CaZrSe3 Solar Cell Project

USAGE:
    python run_all.py                           # uses default settings
    python run_all.py path/to/your/file.iv     # single file
    python run_all.py file1.iv file2.iv ...    # multiple files combined

WHAT IT DOES:
    Step 1: Parse .iv file(s) → DataFrame
    Step 2: Engineer features → train/test split
    Step 3: Train RF, XGBoost, ANN → evaluate with 5-fold CV
    Step 4: SHAP analysis → feature importance
    Step 5: Generate paper report and figures

OUTPUT:
    results/  → all data files (.csv, .npy)
    models/   → trained model files (.pkl)
    figures/  → all publication figures (.png)

ADDING NEW .iv FILES LATER:
    Just run: python run_all.py new_file1.iv new_file2.iv
    The pipeline will merge them with your existing data automatically
    if you set MERGE_WITH_EXISTING = True below.
═══════════════════════════════════════════════════════════════════
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

# ── Configuration ─────────────────────────────────────────────────
RESULTS_DIR  = 'results'
MODELS_DIR   = 'models'
FIGURES_DIR  = 'figures'

# Set to True to merge new .iv files with previously parsed data
MERGE_WITH_EXISTING = False

# Absorber name for S-Q limit check and report labels
ABSORBER_NAME = 'CaZrSe3'

# Train/test split ratio
TEST_SIZE = 0.2


def main():
    # ── Parse command line arguments ──────────────────────────────
    parser = argparse.ArgumentParser(
        description='SCAPS-ML Framework: parse .iv files and run ML pipeline'
    )
    parser.add_argument(
        'iv_files', nargs='*',
        help='Path(s) to SCAPS .iv file(s). Multiple files are combined.',
        default=[]
    )
    parser.add_argument('--absorber', default=ABSORBER_NAME,
                        help='Absorber material name (for labels)')
    parser.add_argument('--test-size', type=float, default=TEST_SIZE,
                        help='Test set fraction (default 0.2)')
    parser.add_argument('--skip-training', action='store_true',
                        help='Skip Steps 3–4 (just parse and prepare)')
    parser.add_argument('--shap-only', action='store_true',
                        help='Only run SHAP analysis (Steps 1–3 must be done)')
    args = parser.parse_args()
    
    # ── Create output directories ──────────────────────────────────
    for d in [RESULTS_DIR, MODELS_DIR, FIGURES_DIR]:
        os.makedirs(d, exist_ok=True)
    
    # ── Locate .iv files ───────────────────────────────────────────
    iv_files = args.iv_files
    if not iv_files:
        # Auto-detect in data/ folder
        data_dir = Path('data')
        if data_dir.exists():
            iv_files = list(data_dir.glob('*.iv'))
            if iv_files:
                print(f"Auto-detected {len(iv_files)} .iv file(s) in data/:")
                for f in iv_files:
                    print(f"  {f}")
            else:
                print("No .iv files found in data/ folder.")
                print("Usage: python run_all.py path/to/file.iv")
                sys.exit(1)
        else:
            print("No .iv files specified and no data/ folder found.")
            print("Usage: python run_all.py path/to/file.iv")
            sys.exit(1)
    
    iv_files = [str(f) for f in iv_files]
    
    # ─────────────────────────────────────────────────────────────
    # STEP 1: Parse .iv files
    # ─────────────────────────────────────────────────────────────
    print("\n" + "█"*60)
    print("  STEP 1: Parse SCAPS .iv Files")
    print("█"*60)
    
    sys.path.insert(0, os.path.dirname(__file__) or '.')
    from step1_parse_iv import parse_iv_file, load_multiple_iv_files, save_dataset, load_dataset
    
    if len(iv_files) == 1:
        df_new = parse_iv_file(iv_files[0], verbose=True)
    else:
        df_new = load_multiple_iv_files(iv_files, verbose=True)
    
    # Optionally merge with previously parsed data
    existing_csv = f'{RESULTS_DIR}/parsed_data.csv'
    if MERGE_WITH_EXISTING and os.path.exists(existing_csv):
        print(f"\nMerging with existing data: {existing_csv}")
        df_existing = load_dataset(existing_csv)
        df = pd.concat([df_existing, df_new], ignore_index=True, sort=False)
        df = df.drop_duplicates().reset_index(drop=True)
        print(f"Merged dataset: {len(df)} total rows")
    else:
        df = df_new
    
    save_dataset(df, f'{RESULTS_DIR}/parsed_data.csv')
    
    if args.skip_training:
        print("\n[--skip-training] Stopping after Step 1.")
        return
    
    # ─────────────────────────────────────────────────────────────
    # STEP 2: Prepare Features
    # ─────────────────────────────────────────────────────────────
    print("\n" + "█"*60)
    print("  STEP 2: Feature Engineering")
    print("█"*60)
    
    from step2_prepare_features import prepare_features, save_prepared_data
    
    prepared = prepare_features(
        df,
        absorber=args.absorber,
        test_size=args.test_size,
        random_state=42,
        verbose=True
    )
    save_prepared_data(prepared, output_dir=RESULTS_DIR)
    
    # ─────────────────────────────────────────────────────────────
    # STEP 3: Train Models
    # ─────────────────────────────────────────────────────────────
    if not args.shap_only:
        print("\n" + "█"*60)
        print("  STEP 3: Train ML Models")
        print("█"*60)
        
        from step3_train_models import (train_and_evaluate, save_models,
                                         plot_parity_plots, plot_model_comparison,
                                         plot_learning_curve)
        
        results = train_and_evaluate(
            prepared['X_train'],    prepared['X_train_sc'],
            prepared['y_train'],
            prepared['X_test'],     prepared['X_test_sc'],
            prepared['y_test'],
            prepared['feature_names'],
            verbose=True
        )
        
        print(f"\nGenerating training figures...")
        plot_model_comparison(results, FIGURES_DIR)
        plot_parity_plots(results, FIGURES_DIR,
                          prepared['feature_names_display'])
        
        if 'PCE_pct' in prepared['y_train']:
            plot_learning_curve(
                prepared['X_train'],
                prepared['y_train']['PCE_pct'],
                output_dir=FIGURES_DIR
            )
        
        save_models(results, MODELS_DIR)
    
    # ─────────────────────────────────────────────────────────────
    # STEP 4: SHAP Analysis
    # ─────────────────────────────────────────────────────────────
    print("\n" + "█"*60)
    print("  STEP 4: SHAP Analysis")
    print("█"*60)
    
    from step4_shap_analysis import (compute_shap, plot_shap_bar,
                                      plot_shap_beeswarm, plot_shap_dependence,
                                      save_shap_table)
    
    for target_name in prepared['y_full'].keys():
        shap_data = compute_shap(
            prepared['X_full'],
            prepared['y_full'][target_name],
            prepared['feature_names'],
            prepared['feature_names_display'],
            target_name=target_name,
            verbose=True
        )
        
        safe = target_name.replace('%', 'pct').replace('/', '_')
        np.save(f'{RESULTS_DIR}/shap_values_{safe}.npy',
                shap_data['shap_values'])
        
        plot_shap_bar(shap_data, FIGURES_DIR)
        plot_shap_dependence(shap_data, FIGURES_DIR, n_top_features=2)
        save_shap_table(shap_data, RESULTS_DIR)
        
        if target_name == 'PCE_pct':
            plot_shap_beeswarm(shap_data, FIGURES_DIR)
    
    # ─────────────────────────────────────────────────────────────
    # STEP 5: Report
    # ─────────────────────────────────────────────────────────────
    print("\n" + "█"*60)
    print("  STEP 5: Generate Paper Report")
    print("█"*60)
    
    from step5_report import generate_paper_numbers, plot_combined_figure
    
    paper_numbers = generate_paper_numbers(
        results_dir=RESULTS_DIR,
        models_dir=MODELS_DIR,
        absorber=args.absorber.replace('3', '₃').replace('e', 'e')
    )
    
    plot_combined_figure(RESULTS_DIR, FIGURES_DIR)
    
    import json
    with open(f'{RESULTS_DIR}/paper_numbers.json', 'w') as f:
        json.dump(paper_numbers, f, indent=2, default=str)
    
    # ─────────────────────────────────────────────────────────────
    # FINAL SUMMARY
    # ─────────────────────────────────────────────────────────────
    print("\n" + "█"*60)
    print("  COMPLETE — All Steps Finished")
    print("█"*60)
    print(f"\n  Output locations:")
    print(f"    Parsed data   → {RESULTS_DIR}/parsed_data.csv")
    print(f"    Clean data    → {RESULTS_DIR}/clean_data.csv")
    print(f"    SHAP table    → {RESULTS_DIR}/SHAP_importance_table.csv")
    print(f"    Paper numbers → {RESULTS_DIR}/paper_numbers.json")
    print(f"    Trained models → {MODELS_DIR}/*.pkl")
    print(f"    Figures       → {FIGURES_DIR}/*.png")
    print(f"\n  To add more .iv files later:")
    print(f"    python run_all.py new_sweep.iv")
    print(f"\n  Figures ready for paper:")
    for png in sorted(Path(FIGURES_DIR).glob('*.png')):
        print(f"    {png.name}")


if __name__ == '__main__':
    main()
