import numpy as np
import pandas as pd

from empirical_copula.significance import _bootstrap_independently, significance_from_bootstrap


def test__bootstrap_independently():
    samples = pd.DataFrame(
        data=[['A', 'A', 'B', 'B', 'B'],
              [100, 200, 100, 300, 500]],
        index=['a', 'b']
    ).T

    random_state = np.random.RandomState(98)
    n_bootstraps = 3
    bootstrap_counts = _bootstrap_independently(samples, n_bootstraps, random_state=random_state)

    # There should be no missing values, those are counts of 0
    assert (~bootstrap_counts.isnull()).all().all()
    assert bootstrap_counts.shape[1] == n_bootstraps
    # only <= because we cannot guarantee that all values are bootstrapped
    # 2: A, B;  4: 100, 200, 300, 500
    assert bootstrap_counts.shape[0] <= 2 * 4


def test_significance_from_bootstrap():
    # Marginal distributions are uniform
    # 'A' and 100 are very dependent (they always appear together)
    # 'A', 'B' are independent of 200, 300 (they appear equally often together)
    samples = pd.DataFrame(
        data=[['A', 'A', 'A', 'A', 'B', 'B', 'B', 'B', 'C', 'C', 'C', 'C'],
              [100, 100, 100, 100, 200, 300, 200, 300, 200, 300, 200, 300]],
        index=['c', 'd']
    ).T

    random_state = np.random.RandomState(98)
    quantile_levels, quantile_levels_labels, significance = significance_from_bootstrap(
        samples, n_bootstraps=100, p_levels_low=[0.01, 0.1], random_state=random_state)

    assert quantile_levels == [0.01, 0.1, 0, 0.9, 0.99]
    assert quantile_levels_labels == [-2, -1, 0, 1, 2]
    assert significance.loc['A', 100] == 2
    assert (significance.loc['B':'C', 200:300] == 0).all().all()
