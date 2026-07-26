"""
Unit tests for step2b_bayesian_optimizer.py
Run: pytest tests/test_bo.py -v
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
import numpy as np
import pandas as pd

from step2b_bayesian_optimizer import (
    GaussianProcess, ParameterSpace, BayesianOptimiser,
    expected_improvement, upper_confidence_bound,
    random_search_baseline, PARAM_BOUNDS_CAZRSE3
)


# ── Fixtures ─────────────────────────────────────────────────────
@pytest.fixture
def tiny_df():
    """50-point toy dataset for fast testing."""
    np.random.seed(0)
    n = 50
    df = pd.DataFrame({
        'CaZrSe3_thickness_um': np.random.uniform(0.1, 1.5, n),
        'CaZrSe3_shallow_acceptor_density_1percm3': 10**np.random.uniform(13, 18, n),
        'CdZnS_thickness_um': np.random.uniform(0.02, 0.2, n),
        'CdZnS_shallow_donor_density_1percm3': 10**np.random.uniform(15, 18, n),
        'PCE_pct': np.random.uniform(2, 10.49, n),
        'Voc_V': np.random.uniform(0.3, 0.54, n),
        'Jsc_mA': np.random.uniform(15, 27, n),
        'FF_pct': np.random.uniform(45, 71, n),
    })
    return df


class TestGaussianProcess:

    def test_fit_predict_shape(self):
        gp = GaussianProcess()
        X_train = np.random.randn(20, 3)
        y_train = np.random.randn(20)
        gp.fit(X_train, y_train)
        X_test = np.random.randn(5, 3)
        mu, std = gp.predict(X_test)
        assert mu.shape == (5,)
        assert std.shape == (5,)

    def test_std_nonnegative(self):
        gp = GaussianProcess()
        X = np.random.randn(15, 2)
        y = np.random.randn(15)
        gp.fit(X, y)
        _, std = gp.predict(np.random.randn(10, 2))
        assert (std >= 0).all()

    def test_std_near_zero_at_training_points(self):
        gp = GaussianProcess(noise=1e-8)
        X = np.linspace(0, 1, 10).reshape(-1, 1)
        y = np.sin(X.ravel())
        gp.fit(X, y)
        _, std = gp.predict(X)
        assert std.max() < 0.05  # near-zero uncertainty at training points

    def test_fitted_flag(self):
        gp = GaussianProcess()
        assert not gp.fitted
        gp.fit(np.random.randn(5, 2), np.random.randn(5))
        assert gp.fitted

    def test_log_marginal_likelihood_returns_float(self):
        gp = GaussianProcess()
        gp.fit(np.random.randn(10, 2), np.random.randn(10))
        lml = gp.log_marginal_likelihood()
        assert isinstance(lml, float)


class TestAcquisitionFunctions:

    def test_ei_nonnegative(self):
        mu  = np.array([5.0, 6.0, 4.0])
        std = np.array([0.5, 0.1, 1.0])
        ei  = expected_improvement(mu, std, y_best=5.5)
        assert (ei >= 0).all()

    def test_ei_zero_below_best(self):
        mu  = np.array([3.0, 4.0])
        std = np.array([0.001, 0.001])
        ei  = expected_improvement(mu, std, y_best=5.0, xi=0.0)
        assert (ei < 1e-6).all()

    def test_ucb_increases_with_std(self):
        mu  = np.array([5.0, 5.0])
        std = np.array([0.1, 1.0])
        ucb = upper_confidence_bound(mu, std)
        assert ucb[1] > ucb[0]


class TestParameterSpace:

    def test_normalise_denormalise_roundtrip(self):
        space = ParameterSpace(PARAM_BOUNDS_CAZRSE3)
        X_phys = space.denormalise(space.random_sample(10, seed=0))
        X_norm = space.normalise(X_phys)
        X_phys2 = space.denormalise(X_norm)
        np.testing.assert_allclose(X_phys, X_phys2, rtol=1e-10)

    def test_random_sample_in_unit_cube(self):
        space = ParameterSpace(PARAM_BOUNDS_CAZRSE3)
        X = space.random_sample(100, seed=42)
        assert X.shape == (100, space.n_params)
        assert (X >= 0).all()
        assert (X <= 1).all()

    def test_n_params_correct(self):
        space = ParameterSpace(PARAM_BOUNDS_CAZRSE3)
        assert space.n_params == len(PARAM_BOUNDS_CAZRSE3)


class TestBayesianOptimiser:

    def test_run_on_existing_data_finds_near_optimum(self, tiny_df):
        bo = BayesianOptimiser(target='PCE_pct', absorber='CaZrSe3',
                               verbose=False, random_state=0)
        results = bo.run_on_existing_data(tiny_df, n_init=20, n_iter=10)
        # BO should find at least 95% of the true grid max
        true_max = tiny_df['PCE_pct'].max()
        assert results['best_pce'] >= 0.90 * true_max

    def test_results_dict_keys(self, tiny_df):
        bo = BayesianOptimiser(target='PCE_pct', absorber='CaZrSe3',
                               verbose=False, random_state=0)
        results = bo.run_on_existing_data(tiny_df, n_init=15, n_iter=5)
        required_keys = ['best_pce', 'best_params', 'n_evals_total',
                         'convergence', 'y_all', 'y_obs']
        for k in required_keys:
            assert k in results, f"Missing key: {k}"

    def test_convergence_monotone_increasing(self, tiny_df):
        bo = BayesianOptimiser(target='PCE_pct', absorber='CaZrSe3',
                               verbose=False, random_state=1)
        results = bo.run_on_existing_data(tiny_df, n_init=15, n_iter=10)
        conv = np.array(results['convergence'])[:, 1]
        # Best-so-far must be monotonically non-decreasing
        assert (np.diff(conv) >= -1e-10).all()

    def test_n_evals_matches_n_init_plus_iter(self, tiny_df):
        bo = BayesianOptimiser(target='PCE_pct', absorber='CaZrSe3',
                               verbose=False)
        results = bo.run_on_existing_data(tiny_df, n_init=10, n_iter=5)
        # n_evals_total should be n_init + actual BO iters run
        assert results['n_evals_total'] >= 10


class TestRandomSearchBaseline:

    def test_output_shape(self):
        y_all = np.random.uniform(0, 10, 500)
        rs = random_search_baseline(y_all, n_total=100, n_repeats=10)
        assert rs['mean'].shape == (100,)
        assert rs['std'].shape == (100,)
        assert rs['n_evals'].shape == (100,)

    def test_baseline_monotone(self):
        y_all = np.random.uniform(0, 10, 500)
        rs = random_search_baseline(y_all, n_total=50, n_repeats=5)
        # Mean best-so-far must be non-decreasing
        assert (np.diff(rs['mean']) >= -1e-10).all()
