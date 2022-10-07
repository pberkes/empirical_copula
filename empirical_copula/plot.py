import numpy as np


def create_copula_axes(fig, pmf1, pmf2, grid_lw=1):
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
