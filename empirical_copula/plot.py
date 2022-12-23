from matplotlib.colors import ListedColormap
import numpy as np


def _create_copula_axes(fig, pmf1, pmf2, grid_lw, annotation_fontsize=15):
    """ Function to create the axes for the copula plot.

    Parameters
    ----------
    fig : Figure
        Matplotlib Figure object to plot on.
    pmf1 : Series
        The pmf values for each value of the first discrete variable.
    pmf2 : Series
        The pmf values for each value of the second discrete variable.
    grid_lw : int
        Line width of the grid lines. Default is 2.
    annotation_fontsize : int
        Font size of the category annotation next to the axis.

    Returns
    -------
    ax : Axes
        Matplotlib Axes object
    x_mesh : array
        The x coordinates of the grid
    x_mesh : array
        The y coordinates of the grid
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
        ax.annotate(label, xy=((x0 + x1) / 2, -0.15), xycoords=trans, fontsize=annotation_fontsize,
                    verticalalignment='top', horizontalalignment='center')

    yticks = values2
    yticks_labels = [f'{t:.02f}' for t in yticks]
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticks_labels)
    for y in yticks:
        ax.axhline(y, color='k', lw=grid_lw)
    trans = ax.get_yaxis_transform()  # y in data units, x in axes fraction
    for y0, y1, label in zip(yticks[:-1], yticks[1:], labels2):
        ax.annotate(label, xy=(-0.15, (y0 + y1) / 2), xycoords=trans, fontsize=annotation_fontsize,
                    verticalalignment='center', horizontalalignment='right')

    y_mesh, x_mesh = np.meshgrid(values2, values1)
    return ax, x_mesh, y_mesh


def copula_pcolormesh(fig, pmf1, pmf2, data, grid_lw=2,
                      annotation_fontsize=15, **pcolormesh_kwargs):
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
    annotation_fontsize : int
        Font size of the category annotation next to the axis.

    **pcolormesh_kwargs : dict
        Additional keyword arguments are passed on to `pcolormesh`.

    Returns
    -------
    ax : Axes
       Matplotlib Axes object
    pcm : QuadMesh
       Matplotlib object returned by `pcolormesh`.
    """

    # Re-order data to match pmf1, pmf2
    data = data.loc[pmf1.index, pmf2.index]
    # Create axes for the copula plot
    ax, x_mesh, y_mesh = _create_copula_axes(fig, pmf1, pmf2, grid_lw=grid_lw,
                                             annotation_fontsize=annotation_fontsize)
    # Plot the copula data on a pcolormesh
    pcm = ax.pcolormesh(x_mesh, y_mesh, data, **pcolormesh_kwargs)
    return ax, pcm


def significance_copula_pcolormesh(fig, pmf1, pmf2, significance, quantile_levels, grid_lw=2,
                                   annotation_fontsize=15, **pcolormesh_kwargs):
    """ Create a copula plot from significance data.

    The significance copula plot has a colormap specialized to display low- and high-tail
    significance. It is displayed with a colorbar.

    Parameters
    ----------
    fig : Figure
        Matplotlib Figure object to plot on.
    pmf1 : Series
        The pmf values for each value of the first discrete variable.
    pmf2 : Series
        The pmf values for each value of the second discrete variable.
    significance : DataFrame
        A value to plot for each combination of the values of the x-axis variable (index) and of
        the y-axis variable (columns). These values will usually be generated from the function
        `significance_from_bootstrap`.
    quantile_levels : list of floats
        List of significance levels for the low and high tail. These values will usually be
        generated from the function `significance_from_bootstrap`.
    grid_lw : int
        Line width of the grid lines. Default is 2.
    annotation_fontsize : int
        Font size of the category annotation next to the axis.
    **pcolormesh_kwargs : dict
        Additional keyword arguments are passed on to `pcolormesh`.

    Returns
    -------
    ax : Axes
       Matplotlib Axes object
    pcm : QuadMesh
       Matplotlib object returned by `pcolormesh`.
    """

    n_levels = len(quantile_levels) // 2

    # Create specialized discrete colormap
    colors_neg = [[0.0, g, 1.0] for g in np.linspace(0, 1, n_levels)]
    colors_pos = [[1.0, g, 0.0] for g in reversed(np.linspace(0, 1, n_levels))]
    colors_levels = colors_neg + [[0.5,0.5,0.5]] + colors_pos
    significance_cmap = ListedColormap(colors_levels)

    ax, pcm = copula_pcolormesh(
        fig, pmf1, pmf2, significance,
        vmin=-n_levels-0.5, vmax=n_levels + 0.5, cmap=significance_cmap, grid_lw=grid_lw,
        annotation_fontsize=annotation_fontsize, **pcolormesh_kwargs,
    )
    cbar = fig.colorbar(pcm, ax=ax)
    cbar.ax.set_yticks(np.arange(-n_levels, n_levels+1))
    cbar.ax.set_yticklabels(quantile_levels)
    cbar.set_label('significance level', fontsize=15, rotation=270)

    return significance_cmap
