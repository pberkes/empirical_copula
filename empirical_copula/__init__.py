import pandas as pd


def empirical_marginal_pmf(samples):
    """ Compute the empirical marginal probability of a discrete variable.

    Parameters
    ----------
    samples : Series
        The observed values of a discrete variable.

    Returns
    -------
    pmf : Series
        Empirical marginal probability of each value of the discrete variable.
    """
    pmf = samples.value_counts(normalize=True)
    return pmf


def joint_counts(samples):
    """ Compute the joint counts of two discrete variables.

    Parameters
    ----------
    samples : DataFrame of size (n, 2)
        The observed values of two discrete variables (two columns).

    Returns
    -------
    counts : DataFrame
        Counts for each combination of the values of the two variables.
    """
    index_label = samples.columns[0]
    columns_label = samples.columns[1]
    counts = (
        samples
        .pivot_table(index=index_label, columns=columns_label, aggfunc='size', fill_value=0)
    )
    return counts


def independent_pmf(pmf1, pmf2):
    """ Joint probabilities of two variables, assuming they are independent.

    Parameters
    ----------
    pmf1 : Series
        The pmf values for each value of the first discrete variable.
    pmf2 : Series
        The pmf values for each value of the second discrete variable.

    Returns
    -------
    independent_pmf : DataFrame
        Joint probability of each combination of values of the two variables, assuming they are
        independent.
    """
    independent_pmf = pd.DataFrame(
        data=pmf1.values[:, None] * pmf2.values[None, :],
        index=pmf1.index,
        columns=pmf2.index,
    )
    return independent_pmf


def empirical_joint_pmf_details(samples):
    """ Compute the empirical joint probability of two variable.

    This version of `empirical_joint_pmf` also returns a dictionary with intermediate results.

    Parameters
    ----------
    samples : DataFrame of size (n, 2)
        The observed values of two discrete variables (two columns).

    Returns
    -------
    pmf1 : Series
        The pmf values for each value of the first discrete variable.
    pmf2 : Series
        The pmf values for each value of the second discrete variable.
    empirical_pmf : DataFrame
        Joint probability of each combination of values of the two variables.
    others : dict
        Dictionary containing intermediate results.
        `'counts'`: Joint counts of each combination
        `'joint_freq'`: Joint frequency of each combination
        `'ind_pmf'`: Joint probability of each combination of values of the two variables,
                     assuming they are independent.
    """
    pmf1 = empirical_marginal_pmf(samples.iloc[:, 0])
    pmf2 = empirical_marginal_pmf(samples.iloc[:, 1])
    counts = joint_counts(samples)
    joint_freq = counts / counts.sum().sum()
    ind_pmf = independent_pmf(pmf1, pmf2)
    empirical_pmf = joint_freq / ind_pmf

    others = {'counts': counts, 'joint_freq': joint_freq, 'ind_pmf': ind_pmf}
    return pmf1, pmf2, empirical_pmf, others


def empirical_joint_pmf(samples):
    """ Compute the empirical joint probability of two variable.

    Parameters
    ----------
    samples : DataFrame of size (n, 2)
        The observed values of two discrete variables (two columns).

    Returns
    -------
    pmf1 : Series
        The pmf values for each value of the first discrete variable.
    pmf2 : Series
        The pmf values for each value of the second discrete variable.
    empirical_pmf : DataFrame
        Joint probability of each combination of values of the two variables.
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
