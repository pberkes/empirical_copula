import numpy as np
import pandas as pd

from empirical_copula import joint_counts


def _bootstrap_independently(samples, n_bootstraps, random_state=np.random):
    """ Create datasets with empirical distribution as samples but all dependencies removed.

    The two columns of `samples` are sampled with repetition, independently and with number of
    samples as in the original dataset. The goal is to create data where each column
    has the same empirical distribution as the original columns, but the dependencies are
    gone.

    Parameters
    ----------
    samples : DataFrame
        Pandas DataFrame with two columns, each row representing a sample from two
        discrete random variables. The dtype of the columns does not matter, but it is expected
        a number of unique values per column much smaller than the number of samples.
    n_bootstraps : int
        Number of bootstrapped datasets.
    random_state : numpy.RandomState
        Random number generator. Default is `numpy.random`.

    Returns
    -------
    bootstrap_counts : DataFrame
        A table containing the counts of the joint distribution of the two variables. Each row
        is a combination of the values of the first and second variable, and each column is a
        bootstrap resampling.

    """
    col1, col2 = samples.columns
    bootstrap_counts = []
    for _ in range(n_bootstraps):
        bootstrap_samples = pd.DataFrame(
            data={
                col1: random_state.choice(samples[col1], size=samples.shape[0], replace=True),
                col2: random_state.choice(samples[col2], size=samples.shape[0], replace=True),
            }
        )
        counts = joint_counts(bootstrap_samples)
        bootstrap_counts.append(counts.unstack())

    bootstrap_counts = pd.concat(bootstrap_counts, axis=1).fillna(0)
    return bootstrap_counts


def significance_from_bootstrap(samples, n_bootstraps, p_levels_low, random_state=np.random):
    """ Compute thresholds for significance under the 0-hypothesis of independence.

    Parameters
    ----------
    samples : DataFrame
        Pandas DataFrame with two columns, each row representing a sample from two
        discrete random variables. The dtype of the columns does not matter, but it is expected
        a number of unique values per column much smaller than the number of samples.
    n_bootstraps : int
        Number of bootstrapped datasets.
    p_levels_low: list of floats
        List of significance levels for the low tail, between 0 and 0.5 .
        The function adds significance levels for the high tail.
    random_state : numpy.RandomState
        Random number generator. Default is `numpy.random`.

    Returns
    -------
    quantile_levels : list of floats
        List of significance levels for the low and high tail.
    quantile_levels_labels : list of int
        List of labels for the significance levels.
    significance : DataFrame
        A table containing the significance label for all combinations of values for the two
        variables.
    """

    n_levels = len(p_levels_low)
    p_levels_high = [1.0 - l for l in reversed(p_levels_low)]
    quantile_levels = p_levels_low + [0] + p_levels_high
    quantile_levels_labels = (
            [-(n_levels - i) for i in range(n_levels)]
            + [0]
            + [i+1 for i in range(n_levels)]
    )

    bootstrap_counts = _bootstrap_independently(samples, n_bootstraps, random_state)
    bootstrap_quantiles = bootstrap_counts.quantile(quantile_levels, axis=1).stack()

    samples_counts = joint_counts(samples)
    significance = np.zeros_like(samples_counts)
    # More frequent than uniform
    quantile_levels_high = p_levels_high
    quantile_levels_labels_high = [i+1 for i in range(n_levels)]
    for v, q in zip(quantile_levels_labels_high, quantile_levels_high):
        significance[samples_counts >= bootstrap_quantiles.loc[q]] = v

    # Less frequent than uniform
    quantile_levels_low = p_levels_low[::-1]
    quantile_levels_labels_low = [-(i+1) for i in range(n_levels)]
    for v, q in zip(quantile_levels_labels_low, quantile_levels_low):
        significance[samples_counts <= bootstrap_quantiles.loc[q]] = v
    significance = pd.DataFrame(
        data=significance,
        index=samples_counts.index,
        columns=samples_counts.columns
    )

    return quantile_levels, quantile_levels_labels, significance
