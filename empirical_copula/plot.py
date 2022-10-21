import numpy as np


def _create_copula_axes(fig, pmf1, pmf2, grid_lw):
    """
    Function to create the axes for the copula plot

    Keyword arguments:
    fig -- a matplotlib figure
    pmf1 -- a pandas Series of the empirical marginal probabilities of the first variable
    pmf2 -- a pandas Series of the empirical marginal probabilities of the second variable
    grid_lw -- line width of the grid lines

    Returns:
    ax -- a matplotlib Axes object
    x_mesh -- a numpy array of the x coordinates of the grid
    y_mesh -- a numpy array of the y coordinates of the grid
    """
    cdf1 = pmf1.cumsum()
    labels1 = cdf1.index.tolist()
    values1 = [0.0] + cdf1.values.tolist()

    cdf2 = pmf2.cumsum()
    labels2 = cdf2.index.tolist()
    values2 = [0.0] + cdf2.values.tolist()

    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_aspect(1)

    xticks = values1
    xticks_labels = [f'{t:.02f}' for t in xticks]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticks_labels, rotation=90)
    for x in xticks:
        ax.axvline(x, color='k', lw=grid_lw)
    trans = ax.get_xaxis_transform()  # x in data units, y in axes fraction
    for x0, x1, label in zip(xticks[:-1], xticks[1:], labels1):
        ax.annotate(label, xy=((x0 + x1) / 2, -0.15), xycoords=trans, fontsize=15,
                    horizontalalignment='center')

    yticks = values2
    yticks_labels = [f'{t:.02f}' for t in yticks]
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticks_labels)
    for y in yticks:
        ax.axhline(y, color='k', lw=grid_lw)
    trans = ax.get_yaxis_transform()  # y in data units, x in axes fraction
    for y0, y1, label in zip(yticks[:-1], yticks[1:], labels2):
        ax.annotate(label, xy=(-0.15, (y0 + y1) / 2), xycoords=trans, fontsize=15,
                    verticalalignment='center')

    y_mesh, x_mesh = np.meshgrid(values2, values1)
    return ax, x_mesh, y_mesh


def copula_pcolormesh(fig, pmf1, pmf2, data, grid_lw=2, **pcolormesh_kwargs):
    """ Create a copula plot.

    The values in `data` are re-ordered according to the ordering of the indices of `pmf1` and
    `pmf2`.

    Parameters
    ----------
    fig : Figure
        Matplotlib Figure object to plot on.
    pmf1 : Series
        The pmf values for each value of the first discrete variable.
    pmf2 : Series
        The pmf values for each value of the second discrete variable.
    data : DataFrame
        A value to plot for each combination of the values of the x-axis variable (index) and of
        the y-axis variable (columns)
    grid_lw : int
        Line width of the grid lines. Default is 2.

    **pcolormesh_kwargs : dict
        Additional keyword arguments are passed on to `pcolormesh`.

    Returns:
    ax : Axes
       Matplotlib Axes object
    pcm : QuadMesh
       Matplotlib object returned by `pcolormesh`.
    """

    # Re-order data to match pmf1, pmf2
    data = data.loc[pmf1.index, pmf2.index]
    # Create axes for the copula plot
    ax, x_mesh, y_mesh = _create_copula_axes(fig, pmf1, pmf2, grid_lw=grid_lw)
    # Plot the copula data on a pcolormesh
    pcm = ax.pcolormesh(x_mesh, y_mesh, data, **pcolormesh_kwargs)
    return ax, pcm
