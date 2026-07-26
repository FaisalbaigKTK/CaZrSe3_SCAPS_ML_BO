"""
Unit tests for step2_prepare_features.py and step3_train_models.py
Run: pytest tests/test_models.py -v
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
import numpy as np
import pandas as pd

from step2_prepare_features import prepare_features


@pytest.fixture
def sample_df():
    """200-row toy dataset with correct column names."""
    np.random.seed(42)
    n = 200
    log_NA = np.random.uniform(13, 18, n)
    log_ND = np.random.uniform(15, 18, n)
    t_abs  = np.random.uniform(0.1, 1.5, n)
    return pd.DataFrame({
        'CaZrSe3_thickness_um': t_abs,
        'CaZrSe3_shallow_acceptor_density_1percm3': 10**log_NA,
        'CdZnS_thickness_um': np.random.uniform(0.02, 0.2, n),
        'CdZnS_shallow_donor_density_1percm3': 10**log_ND,
        'PCE_pct': np.random.uniform(1, 10.49, n),
        'Voc_V':   np.random.uniform(0.3, 0.54, n),
        'Jsc_mA':  np.random.uniform(15, 27, n),
        'FF_pct':  np.random.uniform(45, 71, n),
    })


class TestPrepareFeatures:

    def test_returns_dict_with_required_keys(self, sample_df):
        out = prepare_features(sample_df, absorber='CaZrSe3',
                               test_size=0.2, verbose=False)
        for k in ['X_train', 'X_test', 'X_train_sc', 'X_test_sc',
                  'y_train', 'y_test', 'feature_names', 'scaler']:
            assert k in out

    def test_train_test_split_sizes(self, sample_df):
        out = prepare_features(sample_df, test_size=0.2, verbose=False)
        n = len(sample_df)
        assert len(out['X_train']) == pytest.approx(n * 0.8, abs=2)
        assert len(out['X_test'])  == pytest.approx(n * 0.2, abs=2)

    def test_log_transform_applied(self, sample_df):
        out = prepare_features(sample_df, verbose=False)
        # Doping columns should be log-transformed (values in 13–18 range)
        log_cols = [c for c in out['feature_names'] if 'log10' in c]
        assert len(log_cols) >= 1, "Expected at least one log10-transformed column"
        for col_idx, col in enumerate(out['feature_names']):
            if 'log10' in col:
                vals = out['X_full'][:, col_idx]
                assert vals.min() >= 12.0
                assert vals.max() <= 19.0

    def test_sq_limit_filter(self, sample_df):
        # Inject a row with PCE above SQ limit
        bad = sample_df.copy()
        bad.loc[0, 'PCE_pct'] = 99.0
        out = prepare_features(bad, absorber='CaZrSe3', verbose=False)
        # That row should be removed
        assert out['y_full']['PCE_pct'].max() <= 30.4

    def test_scaler_fitted(self, sample_df):
        out = prepare_features(sample_df, verbose=False)
        # Scaled train set should have near-zero mean
        assert abs(out['X_train_sc'].mean()) < 0.5

    def test_no_nan_in_features(self, sample_df):
        out = prepare_features(sample_df, verbose=False)
        assert not np.isnan(out['X_train']).any()
        assert not np.isnan(out['X_test']).any()

    def test_targets_present(self, sample_df):
        out = prepare_features(sample_df, verbose=False)
        for t in ['PCE_pct', 'Voc_V', 'Jsc_mA', 'FF_pct']:
            assert t in out['y_train']
            assert t in out['y_test']
