__version__ = 0.1

import pandas as pd


def empirical_marginal_pmf(samples):
    """
    Compute the empirical marginals of a series of discrete variables.

    Keyword arguments:
    samples -- a pandas Series of discrete variables
    is_ordinal -- if True, the samples are assumed to be ordered

    Returns:
    independent_pmf -- a pandas Series of the empirical marginal probabilities
    """
    pmf = samples.value_counts(normalize=True)
    return pmf


def joint_counts(samples):
    """
    DataFrame of counts of the joint distribution of two variables.

    Keyword arguments:
    samples -- a pandas DataFrame with two columns

    Returns:
    counts -- a DataFrame of counts of the joint distribution of the two variables
    """
    index_label = samples.columns[0]
    columns_label = samples.columns[1]
    counts = (
        samples
        .pivot_table(index=index_label, columns=columns_label, aggfunc='size', fill_value=0)
    )
    return counts


def independent_pmf(pmf1, pmf2):
    """
    Independent dataframe

    Keyword arguments:
    pmf1 -- a pandas Series of the empirical marginal probabilities of the first variable
    pmf2 -- a pandas Series of the empirical marginal probabilities of the second variable

    Returns:
    independent_pmf -- a pandas DataFrame of the independent probability mass function
    """
    independent_pmf = pd.DataFrame(
        data=pmf1.values[:, None] * pmf2.values[None, :],
        index=pmf1.index,
        columns=pmf2.index,
    )
    return independent_pmf


def empirical_joint_pmf_details(samples):
    """ Compute the empirical joint pmf and return all the details about it.

    Keyword arguments:
    samples -- a pandas DataFrame with two columns

    Returns:
    pmf1 -- a pandas Series of the empirical marginal probabilities of the first column
    pmf2 -- a pandas Series of the empirical marginal probabilities of the second column
    empirical_pmf -- a pandas DataFrame of the empirical joint probability mass function
    dict -- counts, joint_freqs, ind_pmf
    """
    pmf1 = empirical_marginal_pmf(samples.iloc[:, 0])
    pmf2 = empirical_marginal_pmf(samples.iloc[:, 1])
    counts = joint_counts(samples)
    joint_freq = counts / counts.sum().sum()
    ind_pmf = independent_pmf(pmf1, pmf2)
    empirical_pmf = joint_freq / ind_pmf
    return pmf1, pmf2, empirical_pmf, {'counts': counts, 'joint_freq': joint_freq, 'ind_pmf': ind_pmf}


def empirical_joint_pmf(samples):
    """
    empirical joint pmf function

    Keyword arguments:
    samples -- a pandas DataFrame with two columns

    Returns:
    pmf1 -- a pandas Series of the empirical marginal probabilities of the first variable
    pmf2 -- a pandas Series of the empirical marginal probabilities of the second variable
    empirical_pmf -- a pandas DataFrame of the empirical joint probability mass function
    """
    pmf1, pmf2, empirical_pmf, _ = empirical_joint_pmf_details(samples)
    return pmf1, pmf2, empirical_pmf


def order_pmf(pmf, is_ordinal=False):
    """ Order the pmf depending on the kind of discrete variable it contains.

    An ordinal variable is ordered in ascending order of the values of the variable, a non-ordinal
    variable is ordered by descending pmf values.

    Parameters
    ----------
    pmf : Series
         The pmf values for each value of a discrete variable.
    is_ordinal : bool
         True if the variable is ordinal.

    Returns
    -------
    ordered_pmf : Series
         The pmf values for each value of a discrete variable, ordered according to its ordinality.
    """
    if is_ordinal:
        ordered_pmf = pmf.sort_index(ascending=True)
    else:
        ordered_pmf = pmf.sort_values(ascending=False)
    return ordered_pmf
