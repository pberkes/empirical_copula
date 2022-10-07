__version__ = 0.1

import pandas as pd


def empirical_marginal_pmf(samples, is_ordinal=False):
    """ Compute the empirical marginals of a series of discrete variables.
    """
    pmf = samples.value_counts(normalize=True)

    if is_ordinal:
        pmf.sort_index(ascending=True, inplace=True)
    else:
        pmf.sort_values(ascending=False, inplace=True)

    return pmf


def joint_counts(samples, order1, order2):
    index_label = samples.columns[0]
    columns_label = samples.columns[1]
    counts = (
        samples
        .pivot_table(index=index_label, columns=columns_label, aggfunc='size', fill_value=0)
        .loc[order1, order2]
    )
    return counts


def independent_pmf(pmf1, pmf2):
    independent_pmf = pd.DataFrame(
        data=pmf1.values[:, None] * pmf2.values[None, :],
        index=pmf1.index,
        columns=pmf2.index,
    )
    return independent_pmf


def empirical_joint_pmf_details(samples, is_col1_ordinal=False, is_col2_ordinal=False):
    pmf1 = empirical_marginal_pmf(samples.iloc[:, 0], is_ordinal=is_col1_ordinal)
    pmf2 = empirical_marginal_pmf(samples.iloc[:, 1], is_ordinal=is_col2_ordinal)
    counts = joint_counts(samples, pmf1.index, pmf2.index)
    joint_freq = counts / counts.sum().sum()
    ind_pmf = independent_pmf(pmf1, pmf2)
    empirical_pmf = joint_freq / ind_pmf
    return pmf1, pmf2, empirical_pmf, {'counts': counts, 'joint_freq': joint_freq, 'ind_pmf': ind_pmf}


def empirical_joint_pmf(samples, is_col1_ordinal=False, is_col2_ordinal=False):
    pmf1, pmf2, empirical_pmf, _ = empirical_joint_pmf_details(
        samples, is_col1_ordinal=is_col1_ordinal, is_col2_ordinal=is_col2_ordinal)
    return pmf1, pmf2, empirical_pmf
