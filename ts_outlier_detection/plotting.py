import numpy as np

from copy import deepcopy
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Default formatting
DEF_PLOT_STYLE = {'color': 'c', 'linewidth': 1}
DEF_SCAT_STYLE = {'s': 2, 'facecolors': 'k', 'marker': '.'}
DEF_POINT_STYLE = {'alpha': 0.3, 's': 5, 'facecolors': 'k', 'edgecolors': 'none'}
DEF_VLINE_STYLE = {'colors': 'r', 'linewidth': 1}


def _check_time_data_agreement(data, times):
    if data.shape[0] != times.shape[0]:
        raise ValueError(
            'data and time labels must be the same shape in first dimension')


def reduce_dimensions(data, dims=2, perplexity=20):
    '''
    Reduce data dimensionality via PCA and t-SNE

    :param numpy.ndarray data: Input data of shape (n_samples, n_dimensions)
    :param int dims: (Optional) Desired number of output dimensions (default 2)
    :param float perplexity: (Optional) Perplexity to use for t-SNE (default 20)
    '''
    output = data
    if data.shape[1] > dims:
        if data.shape[1] > 50 and dims < 50:
            pca = PCA(50)
            pca.fit(data.transpose())
            output = pca.components_.transpose()
        output = TSNE(dims, perplexity=perplexity).fit_transform(output)
    return output


def plot_2d_phase_space(
    data, ax, outlier_ids=None,
    plot_style=DEF_PLOT_STYLE,
    scatter_style=DEF_SCAT_STYLE
):
    '''
    Draw a 2D phase-space, reducing dimensions if necessary

    :param numpy.ndarray data: Input data of shape (n_samples, n_dimensions)
    :param matplotlib.axes.Axes ax: Axes object to plot onto
    :param numpy.ndarray outlier_ids:   (Optional) Data indices corresponding to outliers
    :param dict plot_style:             (Optional) Custom keyword arguments to pass to plt.plot
    :param dict scatter_style:          (Optional) Custom keyword arguments to pass to plt.scatter

    :return: list of PathCollections of plotted scatter plots
    :rtype: [matplotlib.collections.PathCollection]
    '''
    scatter_data = reduce_dimensions(data)
    plots = [ax.plot(scatter_data[:,0], scatter_data[:,1], **plot_style, zorder=1)[0]]
    if outlier_ids is not None:
        plots.append(
            ax.scatter(
                scatter_data[outlier_ids][:,0],
                scatter_data[outlier_ids][:,1],
                **scatter_style, zorder=2
            )
        )
    return plots


def plot_2d_scatter(
    data, ax, outlier_ids=None,
    scatter_style=DEF_POINT_STYLE,
    outlier_style=DEF_SCAT_STYLE
):
    '''
    Draw a 2D scatter plot, reducing dimensions if necessary

    :param numpy.ndarray data: Input data of shape (n_samples, n_dimensions)
    :param matplotlib.axes.Axes ax: Axes object to plot onto
    :param numpy.ndarray outlier_ids:   (Optional) Data indices corresponding to outliers
    :param dict scatter_style:          (Optional) Custom keyword arguments to pass to plt.scatter
    :param dict outlier_style:          (Optional) Custom keyword arguments to pass to plt.scatter at outlying points

    :return: list of PathCollections of plotted scatter plots
    :rtype: [matplotlib.collections.PathCollection]
    '''
    scatter_data = reduce_dimensions(data)
    plots = [ax.scatter(scatter_data[:,0], scatter_data[:,1], **scatter_style)]
    if outlier_ids is not None:
        plots.append(
            ax.scatter(
                scatter_data[outlier_ids][:,0],
                scatter_data[outlier_ids][:,1],
                **outlier_style
            )
        )
    return plots


def plot_ts_outliers(
    ts_outlier, axs, detections=True,
    plot_style=DEF_PLOT_STYLE,
    scatter_style=DEF_SCAT_STYLE
):
    '''
    Plot time series and corresponding outlier scores

    :param TimeSeriesOutlier ts_outlier: Model to plot
    :param (matplotlib.axes.Axes, matplotlib.axes.Axes) axs: Array of two axes to plot onto
    :param bool detections:     (Optional) Whether or not to scatter plot detected outliers (default True)
    :param dict plot_style:     (Optional) Custom keyword arguments to pass to plt.plot
    :param dict scatter_style:  (Optional) Custom keyword arguments to pass to plt.scatter

    :return: list of Line2D and PathCollections of plotted line and scatter plots
    :rtype: [matplotlib.collections.PathCollection]
    '''
    data, times = ts_outlier.get_truncated_data()
    scores = ts_outlier.get_outlier_factors()

    plots = [
        axs[0].plot(times, data, **plot_style, zorder=1)[0],
        axs[1].plot(times, scores, **plot_style, zorder=1)[0]
    ]
    
    if detections:
        outliers = ts_outlier.get_outlier_indices()
        scatters = [
            axs[0].scatter(times[outliers], data[outliers], **scatter_style, zorder=2),
            axs[1].scatter(times[outliers], scores[outliers], **scatter_style, zorder=2)
        ]
        plots.extend(scatters)
    
    return plots

def plot_nearest_neighbors(
    ts_outlier, axs,
    plot_style=DEF_PLOT_STYLE,
    scatter_style=DEF_SCAT_STYLE,
    vline_style=DEF_VLINE_STYLE
):
    '''
    Plot nearest neighbors of each point in time series

    :param TimeSeriesOutlier ts_outlier: Model to plot
    :param (matplotlib.axes.Axes, matplotlib.axes.Axes) axs: Array of three axes to plot onto
    :param dict plot_style:     (Optional) Custom keyword arguments to pass to plt.plot
    :param dict scatter_style:  (Optional) Custom keyword arguments to pass to plt.scatter
    :param dict vline_style:    (Optional) Custom keyword arguments to pass to plt.vlines

    :return: list of Line2D and PathCollections of plotted line and scatter plots
    :rtype: [matplotlib.collections.PathCollection]
    '''
    data, times = ts_outlier.get_truncated_data()
    matrix_y = ts_outlier.neighbor_indices_.flatten()
    matrix_x = np.tile(np.arange(data.size).reshape(-1, 1), (1, ts_outlier.n_neighbors)).flatten()

    vline_style = deepcopy(vline_style)
    vline_style['zorder'] = 1
    scatter_style = deepcopy(scatter_style)
    scatter_style['zorder'] = 2
    plot_style = deepcopy(plot_style)
    plot_style['zorder'] = 4
    plot_style['alpha'] = 0.5
    plots = [
        axs[0].vlines(times[ts_outlier.get_outlier_indices()], 0, times[-1], **vline_style),
        axs[0].scatter(matrix_x, matrix_y, **scatter_style),
        axs[0].plot([0, times[-1]], [0, times[-1]], 'b-', zorder=3)[0],
        axs[1].plot(times, data, **plot_style)[0],
        axs[2].plot(data, times, **plot_style)[0],
    ]
    return plots
