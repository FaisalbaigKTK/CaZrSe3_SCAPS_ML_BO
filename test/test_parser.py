"""
Unit tests for the processed dataset and data loading utilities.
Run: pytest tests/test_parser.py -v

Note: parse_iv_file requires authentic SCAPS-1D batch output
("Batch simulation # N  step M" format). These tests validate the
processed clean_data.csv which is the output used by all downstream steps.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
import pandas as pd
import numpy as np

CLEAN_CSV = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'clean_data.csv')


@pytest.fixture
def df():
    if not os.path.exists(CLEAN_CSV):
        pytest.skip("clean_data.csv not found — run generate_dataset.py first")
    return pd.read_csv(CLEAN_CSV)


class TestDatasetIntegrity:

    def test_row_count(self, df):
        assert len(df) == 12100, f"Expected 12100 rows, got {len(df)}"

    def test_required_columns_present(self, df):
        required = ['CaZrSe3_thickness_um',
                    'CaZrSe3_shallow_acceptor_density_1percm3',
                    'CdZnS_thickness_um',
                    'CdZnS_shallow_donor_density_1percm3',
                    'PCE_pct', 'Voc_V', 'Jsc_mA', 'FF_pct']
        for col in required:
            assert col in df.columns, f"Missing column: {col}"

    def test_pce_global_maximum(self, df):
        assert abs(df['PCE_pct'].max() - 10.4927) < 0.001

    def test_voc_at_optimum(self, df):
        best = df.loc[df['PCE_pct'].idxmax()]
        assert abs(best['Voc_V'] - 0.5388) < 0.002

    def test_jsc_at_optimum(self, df):
        best = df.loc[df['PCE_pct'].idxmax()]
        assert abs(best['Jsc_mA'] - 27.4327) < 0.05

    def test_ff_at_optimum(self, df):
        best = df.loc[df['PCE_pct'].idxmax()]
        assert abs(best['FF_pct'] - 70.9909) < 0.05

    def test_no_nan_values(self, df):
        assert not df[['PCE_pct','Voc_V','Jsc_mA','FF_pct']].isnull().any().any()

    def test_no_negative_pce(self, df):
        assert (df['PCE_pct'] >= 0).all()

    def test_pce_below_sq_limit(self, df):
        SQ_LIMIT = 30.4  # CaZrSe3, Eg=1.35 eV
        assert (df['PCE_pct'] <= SQ_LIMIT).all()

    def test_absorber_thickness_range(self, df):
        assert df['CaZrSe3_thickness_um'].min() >= 0.09
        assert df['CaZrSe3_thickness_um'].max() <= 1.51

    def test_etl_thickness_range(self, df):
        assert df['CdZnS_thickness_um'].min() >= 0.01
        assert df['CdZnS_thickness_um'].max() <= 0.21

    def test_acceptor_doping_range(self, df):
        log_NA = np.log10(df['CaZrSe3_shallow_acceptor_density_1percm3'])
        assert log_NA.min() >= 12.9
        assert log_NA.max() <= 18.1

    def test_donor_doping_range(self, df):
        log_ND = np.log10(df['CdZnS_shallow_donor_density_1percm3'])
        assert log_ND.min() >= 14.9
        assert log_ND.max() <= 18.1

    def test_ff_physical_range(self, df):
        assert (df['FF_pct'] >= 0).all()
        assert (df['FF_pct'] <= 100).all()

    def test_optimal_params_match_paper(self, df):
        """Verify the optimal device parameters match Table 2 of the paper."""
        best = df.loc[df['PCE_pct'].idxmax()]
        # ETL should be at upper doping bound
        assert np.log10(best['CdZnS_shallow_donor_density_1percm3']) >= 17.5
        # ETL thickness near 0.10 µm
        assert abs(best['CdZnS_thickness_um'] - 0.10) < 0.05
