from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Default formatting
DEF_PLOT_STYLE = {'color': 'c', 'linewidth': 1}
DEF_SCAT_STYLE = {'facecolors': 'k', 'marker': '+'}
DEF_POINT_STYLE = {'alpha': 0.3, 's': 5, 'facecolors': 'k', 'edgecolors': 'none'}


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
    '''
    scatter_data = reduce_dimensions(data)
    ax.plot(scatter_data[:,0], scatter_data[:,1], **plot_style, zorder=1)
    if outlier_ids is not None:
        ax.scatter(
            scatter_data[outlier_ids][:,0],
            scatter_data[outlier_ids][:,1],
            **scatter_style, zorder=2
        )


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
    :param dict scatter_style:             (Optional) Custom keyword arguments to pass to plt.plot
    :param dict outlier_style:          (Optional) Custom keyword arguments to pass to plt.scatter
    '''
    scatter_data = reduce_dimensions(data)
    ax.scatter(scatter_data[:,0], scatter_data[:,1], **scatter_style)
    if outlier_ids is not None:
        ax.scatter(
            scatter_data[outlier_ids][:,0],
            scatter_data[outlier_ids][:,1],
            **outlier_style
        )


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
    '''
    data, times = ts_outlier.get_truncated_data()
    scores = ts_outlier.get_outlier_factors()

    axs[0].plot(times, data, **plot_style, zorder=1)
    axs[1].plot(times, scores, **plot_style, zorder=1)
    
    if detections:
        outliers = ts_outlier.get_outlier_indices()
        axs[0].scatter(times[outliers], data[outliers], **scatter_style, zorder=2)
        axs[1].scatter(times[outliers], scores[outliers], **scatter_style, zorder=2)
