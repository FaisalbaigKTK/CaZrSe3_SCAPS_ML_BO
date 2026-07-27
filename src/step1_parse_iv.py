"""
═══════════════════════════════════════════════════════════════════
SCAPS FRAMEWORK — STEP 1: Parse .iv Files
═══════════════════════════════════════════════════════════════════
Khattak Research Group | CaZrSe3 Solar Cell Project

PURPOSE:
    Read any SCAPS .iv batch output file, auto-detect ALL batch
    parameters (no matter how many you add in future), and extract
    the PV performance metrics (PCE, Voc, Jsc, FF) for every step.

HOW IT WORKS:
    SCAPS .iv files have a repeating structure per simulation step:
    
        Batch simulation # 1  step  1  of  12100
        ...
        **Batch parameters**
        Parameter1: value1
        Parameter2: value2
        ...
        [J-V data table]
        ...
        Voc = X Volt
        Jsc = X mA/cm2
        FF  = X %
        eta = X %

    This parser:
    1. Splits the file into individual step blocks
    2. Auto-detects ALL batch parameter names and their values
    3. Extracts Voc, Jsc, FF, PCE from each block
    4. Returns a clean pandas DataFrame — one row per simulation

FUTURE-PROOF:
    When you add HTL parameters or band offset sweeps, this parser
    automatically picks them up — no code changes needed.

USAGE:
    python step1_parse_iv.py
    or import parse_iv_file() into any other script
═══════════════════════════════════════════════════════════════════
"""

import re
import os
import pandas as pd
import numpy as np
from pathlib import Path


# ─────────────────────────────────────────────────────────────────
# CORE PARSER FUNCTION
# ─────────────────────────────────────────────────────────────────

def parse_iv_file(filepath: str, verbose: bool = True) -> pd.DataFrame:
    """
    Parse a SCAPS .iv batch output file into a clean DataFrame.
    
    Parameters
    ----------
    filepath : str
        Path to the .iv file exported from SCAPS
    verbose : bool
        Print progress and summary information
    
    Returns
    -------
    pd.DataFrame
        One row per simulation step. Columns:
        - step          : simulation step number
        - [param_name]  : one column per batch parameter (auto-detected)
        - Voc_V         : open-circuit voltage (Volts)
        - Jsc_mA        : short-circuit current density (mA/cm²)
        - FF_pct        : fill factor (%)
        - PCE_pct       : power conversion efficiency (%)
        - Vmpp_V        : voltage at max power point
        - Jmpp_mA       : current density at max power point
    """

    # ── 1. Read file (SCAPS uses latin-1 encoding, not UTF-8) ────
    if verbose:
        size_mb = os.path.getsize(filepath) / 1e6
        print(f"\n{'='*60}")
        print(f"Reading: {Path(filepath).name}  ({size_mb:.1f} MB)")
    
    with open(filepath, 'rb') as f:
        raw = f.read().decode('latin-1')

    # ── 2. Split into individual step blocks ─────────────────────
    # Each block starts with "Batch simulation #"
    # We split on that marker and keep the content after it
    step_blocks = re.split(r'(?=Batch simulation #\s*\d+\s*step\s+\d+)', raw)
    step_blocks = [b for b in step_blocks if 'Batch simulation' in b]
    
    total_steps_declared = None
    match = re.search(r'step\s+\d+\s+of\s+(\d+)', step_blocks[0] if step_blocks else raw)
    if match:
        total_steps_declared = int(match.group(1))

    if verbose:
        print(f"Total steps declared in file: {total_steps_declared}")
        print(f"Step blocks found: {len(step_blocks)}")

    # ── 3. Auto-detect parameter names from first step ───────────
    # The "**Batch parameters**" section contains all swept params.
    # Format:  "LayerName (LN)>>parameter_name[unit]:\t value"
    param_names_raw = []
    if step_blocks:
        first_block = step_blocks[0]
        bp_section = re.search(
            r'\*\*Batch parameters\*\*(.*?)(?:\n\s*\n|\n    v\(V\))',
            first_block, re.DOTALL
        )
        if bp_section:
            param_lines = bp_section.group(1).strip().split('\n')
            for line in param_lines:
                line = line.strip()
                if ':' in line and line:
                    param_names_raw.append(line.split(':')[0].strip())

    if verbose:
        print(f"\nBatch parameters detected ({len(param_names_raw)}):")
        for p in param_names_raw:
            print(f"  • {p}")

    # ── 4. Create clean column names from raw parameter names ─────
    # Raw:  "Absorber-CaZrSe3 (L2)>>thickness[µm]"
    # Clean: "Absorber_thickness_um"
    def clean_param_name(raw_name: str) -> str:
        """Convert SCAPS parameter string to a clean column name."""
        # Extract layer nickname if present (e.g., "Absorber-CaZrSe3")
        layer_match = re.match(r'^(.*?)\s*\(L\d+\)', raw_name)
        layer = layer_match.group(1) if layer_match else ''
        
        # Extract the parameter part after >>
        param_match = re.search(r'>>(.*?)(?:\[|$)', raw_name)
        param = param_match.group(1) if param_match else raw_name
        
        # Extract unit if present
        unit_match = re.search(r'\[(.*?)\]', raw_name)
        unit = unit_match.group(1) if unit_match else ''
        
        # Build clean name
        # Simplify layer name: take last word
        layer_clean = layer.split('-')[-1].strip().replace(' ', '_')
        param_clean  = param.strip().replace(' ', '_').replace('/', '_per_')
        unit_clean   = unit.replace('µ', 'u').replace('/', 'per').replace('³', '3').replace('²', '2')
        unit_clean   = re.sub(r'[^\w]', '', unit_clean)
        
        if layer_clean and param_clean:
            name = f"{layer_clean}_{param_clean}"
        else:
            name = param_clean
        
        if unit_clean:
            name = f"{name}_{unit_clean}"
        
        # Final cleanup
        name = re.sub(r'_+', '_', name).strip('_')
        return name

    col_names = [clean_param_name(p) for p in param_names_raw]
    
    if verbose:
        print(f"\nClean column names:")
        for raw, clean in zip(param_names_raw, col_names):
            print(f"  {clean:<40s} ← {raw}")

    # ── 5. Parse every step block ─────────────────────────────────
    rows = []
    failed_steps = []
    
    for block in step_blocks:
        try:
            row = _parse_single_step(block, param_names_raw, col_names)
            if row is not None:
                rows.append(row)
        except Exception as e:
            step_match = re.search(r'step\s+(\d+)', block)
            step_num = step_match.group(1) if step_match else '?'
            failed_steps.append(step_num)

    if verbose:
        print(f"\nSuccessfully parsed: {len(rows)} steps")
        if failed_steps:
            print(f"Failed/incomplete: {len(failed_steps)} steps "
                  f"(steps: {failed_steps[:10]}{'...' if len(failed_steps)>10 else ''})")

    # ── 6. Build DataFrame ────────────────────────────────────────
    if not rows:
        raise ValueError("No data could be parsed from this file. "
                         "Check the file format and path.")

    df = pd.DataFrame(rows)
    
    # Sort by step number
    df = df.sort_values('step').reset_index(drop=True)
    
    # Physical sanity check
    sq_violations = df[df['PCE_pct'] > 33.0]
    if len(sq_violations) > 0:
        print(f"\n⚠ WARNING: {len(sq_violations)} rows exceed S-Q limit (33%). "
              f"Check your SCAPS parameters.")
    
    negative_pce = df[df['PCE_pct'] < 0]
    if len(negative_pce) > 0:
        print(f"⚠ WARNING: {len(negative_pce)} rows have negative PCE (failed simulations). "
              f"These will be kept but flagged.")
        df['simulation_failed'] = df['PCE_pct'] < 0
    else:
        df['simulation_failed'] = False

    if verbose:
        print(f"\n{'─'*60}")
        print(f"DataFrame shape: {df.shape}  ({len(df)} rows × {len(df.columns)} columns)")
        print(f"\nPCE statistics:")
        print(f"  Min:    {df['PCE_pct'].min():.4f}%")
        print(f"  Max:    {df['PCE_pct'].max():.4f}%")
        print(f"  Mean:   {df['PCE_pct'].mean():.4f}%")
        print(f"  Median: {df['PCE_pct'].median():.4f}%")
        print(f"\nColumns: {list(df.columns)}")
        print(f"{'='*60}\n")

    return df


def _parse_single_step(block: str, param_names_raw: list,
                        col_names: list) -> dict:
    """
    Parse one step block and return a dict of values.
    Returns None if the step has no PV output (non-converged).
    """
    row = {}
    
    # Step number
    step_match = re.search(r'step\s+(\d+)\s+of', block)
    row['step'] = int(step_match.group(1)) if step_match else -1

    # Batch parameter values — in the **Batch parameters** section
    bp_section = re.search(
        r'\*\*Batch parameters\*\*(.*?)(?:\n\s*\n|\n    v\(V\))',
        block, re.DOTALL
    )
    if bp_section:
        param_lines = bp_section.group(1).strip().split('\n')
        param_values = []
        for line in param_lines:
            line = line.strip()
            if ':' in line:
                val_str = line.split(':')[-1].strip()
                try:
                    param_values.append(float(val_str))
                except ValueError:
                    param_values.append(None)
        
        # Match values to column names
        for col, val in zip(col_names, param_values):
            row[col] = val
    
    # PV metrics — at the end of each step block
    voc_match  = re.search(r'Voc\s*=\s*([\d.]+)\s*Volt', block)
    jsc_match  = re.search(r'Jsc\s*=\s*([\d.eE+\-]+)\s*mA', block)
    ff_match   = re.search(r'FF\s*=\s*([\d.]+)\s*%', block)
    pce_match  = re.search(r'eta\s*=\s*([\d.]+)\s*%', block)
    vmpp_match = re.search(r'V_MPP\s*=\s*([\d.]+)\s*Volt', block)
    jmpp_match = re.search(r'J_MPP\s*=\s*([\d.eE+\-]+)\s*mA', block)

    # If no PV output found, step likely didn't converge
    if not voc_match:
        return None

    row['Voc_V']   = float(voc_match.group(1))
    row['Jsc_mA']  = float(jsc_match.group(1))  if jsc_match  else None
    row['FF_pct']  = float(ff_match.group(1))   if ff_match   else None
    row['PCE_pct'] = float(pce_match.group(1))  if pce_match  else None
    row['Vmpp_V']  = float(vmpp_match.group(1)) if vmpp_match else None
    row['Jmpp_mA'] = float(jmpp_match.group(1)) if jmpp_match else None

    return row


# ─────────────────────────────────────────────────────────────────
# BATCH LOADER: Parse multiple .iv files and combine
# ─────────────────────────────────────────────────────────────────

def load_multiple_iv_files(file_list: list, verbose: bool = True) -> pd.DataFrame:
    """
    Parse multiple .iv files and combine into one DataFrame.
    Use this when you have separate files for different sweep types
    (e.g., separate files for HTL sweep, band offset sweep, etc.)
    
    Parameters
    ----------
    file_list : list of str
        List of paths to .iv files
    
    Returns
    -------
    pd.DataFrame
        Combined DataFrame with a 'source_file' column added
    """
    all_dfs = []
    
    for filepath in file_list:
        if not os.path.exists(filepath):
            print(f"⚠ File not found, skipping: {filepath}")
            continue
        
        df = parse_iv_file(filepath, verbose=verbose)
        df['source_file'] = Path(filepath).stem
        all_dfs.append(df)
    
    if not all_dfs:
        raise ValueError("No valid files could be parsed.")
    
    # Combine — pandas handles missing columns by filling with NaN
    combined = pd.concat(all_dfs, ignore_index=True, sort=False)
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"COMBINED DATASET")
        print(f"  Files loaded:  {len(all_dfs)}")
        print(f"  Total rows:    {len(combined)}")
        print(f"  Total columns: {len(combined.columns)}")
        print(f"{'='*60}\n")
    
    return combined


# ─────────────────────────────────────────────────────────────────
# SAVE / LOAD HELPERS
# ─────────────────────────────────────────────────────────────────

def save_dataset(df: pd.DataFrame, output_path: str):
    """Save parsed DataFrame to CSV for use in later steps."""
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[✓] Dataset saved: {output_path}  ({len(df)} rows)")


def load_dataset(csv_path: str) -> pd.DataFrame:
    """Load a previously saved dataset CSV."""
    df = pd.read_csv(csv_path)
    print(f"[✓] Dataset loaded: {csv_path}  ({len(df)} rows × {len(df.columns)} cols)")
    return df


# ─────────────────────────────────────────────────────────────────
# MAIN — Run this file directly to test parsing
# ─────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import sys
    
    # ── Option A: Test with a specific file ──────────────────────
    # Change this path to your actual .iv file location
    TEST_FILE = 'data/Absoer_ETL_Thickness_Doping.iv'
    
    if len(sys.argv) > 1:
        TEST_FILE = sys.argv[1]
    
    if not os.path.exists(TEST_FILE):
        print(f"File not found: {TEST_FILE}")
        print("Usage: python step1_parse_iv.py path/to/your/file.iv")
        print("\nTo test, place your .iv file in the data/ folder.")
        sys.exit(1)
    
    # Parse the file
    df = parse_iv_file(TEST_FILE, verbose=True)
    
    # Show first few rows
    print("First 5 rows:")
    print(df.head().to_string())
    
    # Show best device
    best = df.loc[df['PCE_pct'].idxmax()]
    print(f"\nBest device (PCE = {best['PCE_pct']:.4f}%):")
    for col in df.columns:
        if col not in ['step', 'simulation_failed', 'source_file']:
            print(f"  {col:<40s} = {best[col]}")
    
    # Save to CSV
    save_dataset(df, 'results/parsed_data.csv')
    
    print("\n[✓] Step 1 complete. Run step2_prepare_features.py next.")
