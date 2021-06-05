import numpy as np
from sklearn.neighbors import LocalOutlierFactor

from ts_outlier_detection.window_lof import WindowedLocalOutlierFactor

def make_polar_scatter(data, times):
    domain = times[-1] - times[0]
    angles = 2 * np.pi * (times - times[0]) / domain
    xdata = data * np.cos(angles)
    ydata = data * np.sin(angles)
    return np.stack((xdata, ydata), axis=-1)


def make_polar_mean_scatter(data, times):
    return make_polar_scatter(data - np.mean(data), times)


def make_cartesian_scatter(data, times):
    return np.stack((times, data), axis=-1)


GENERATE_SCATTER = {
    'polar': make_polar_scatter,
    'polar_mean': make_polar_mean_scatter,
    'cartesian': make_cartesian_scatter
}


class SpatialLocalOutlierFactor(WindowedLocalOutlierFactor):
    def __init__(self, style='polar', **kwargs):
        self._scatter = GENERATE_SCATTER[style]
        super().__init__(**kwargs)


    def fit(self, data, times=None):
        if times is None:
            times = np.arange(data.size)
        if times.size != data.size:
            raise ValueError(
                'data and times arrays must have the same number of elements')
        self._set_embedded_data(self._scatter(data, times))
        self.clf.fit(self.get_embedded_data())
        self.lofs_ = -self.clf.negative_outlier_factor_
        self._set_truncated_data(data, times, self.lofs_.size)
