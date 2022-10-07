import pandas as pd
from pandas.testing import assert_frame_equal, assert_series_equal

from empirical_copula import (
    empirical_joint_pmf_details,
    empirical_marginal_pmf,
    independent_pmf,
    joint_counts,
)


def test_empirical_marginals_categorical():
    samples = pd.Series(['A', 'B', 'A', 'F', 'B', 'B'])
    expected = pd.Series(data=[3.0/6.0, 2.0/6.0, 1.0/6.0], index=['B', 'A', 'F'])
    pmf = empirical_marginal_pmf(samples, is_ordinal=False)
    assert_series_equal(expected, pmf)


def test_empirical_marginals_ordinal():
    samples = pd.Series(['A', 'B', 'A', 'F', 'B', 'B'])
    expected = pd.Series(data=[2.0/6.0, 3.0/6.0, 1.0/6.0], index=['A', 'B', 'F'])
    pmf = empirical_marginal_pmf(samples, is_ordinal=True)
    assert_series_equal(expected, pmf)


def test_joint_counts():
    samples = pd.DataFrame(
        data=[['A', 'A', 'A', 'B', 'B', 'F'],
              [100, 200, 100, 100, 300, 300]],
        index=['v1', 'v2']
    ).T
    order1 = ['B', 'A', 'F']
    order2 = [100, 200, 300]
    expected = pd.DataFrame(
        index=['B', 'A', 'F'],
        columns=[100, 200, 300],
        data=[
            [1, 0, 1],
            [2, 1, 0],
            [0, 0, 1],
        ]
    )
    counts = joint_counts(samples, order1=order1, order2=order2)
    assert_frame_equal(expected, counts, check_names=False)


def test_independent_pmf():
    pmf1 = pd.Series(data=[0.4, 0.3, 0.1], index=['A', 'B', 'C'])
    pmf2 = pd.Series(data=[0.2, 0.3, 0.5], index=[1, 2, 3])
    expected = pd.DataFrame(
        index=pmf1.index,
        columns=pmf2.index,
        data=[
            [0.4 * 0.2, 0.4 * 0.3, 0.4 * 0.5],
            [0.3 * 0.2, 0.3 * 0.3, 0.3 * 0.5],
            [0.1 * 0.2, 0.1 * 0.3, 0.1 * 0.5],
        ]
    )
    pmf = independent_pmf(pmf1, pmf2)
    assert_frame_equal(expected, pmf)


def test_empirical_pmf():
    samples = pd.DataFrame(
        data=[['A', 'A', 'A', 'B', 'B', 'F', 'B', 'B'],
              [100, 200, 100, 100, 200, 200, 200, 200]],
        index=['v1', 'v2']
    ).T

    pmf1, pmf2, empirical_pmf, others = empirical_joint_pmf_details(
        samples, is_col1_ordinal=True, is_col2_ordinal=True)

    n = float(samples.shape[0])
    expected_joint_freq = pd.DataFrame(
        index=['A', 'B', 'F'],
        columns=[100, 200],
        data=[
            [2.0/n, 1.0/n],
            [1.0/n, 3.0/n],
            [0.0/n, 1.0/n],
        ]
    )
    assert_frame_equal(expected_joint_freq, others['joint_freq'])

    expected_empirical_pmf = pd.DataFrame(
        index=['A', 'B', 'F'],
        columns=[100, 200],
        data=[
            [2.0 * n / (3 * 3), 1.0 * n / (3 * 5)],
            [1.0 * n / (4 * 3), 3.0 * n / (4 * 5)],
            [0.0, 1.0 * n  / (1 * 5)],
        ]
    )
    assert_frame_equal(expected_empirical_pmf, empirical_pmf)
