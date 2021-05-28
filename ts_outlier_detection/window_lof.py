import numpy as np
from sklearn.neighbors import LocalOutlierFactor

from ts_outlier_detection.time_series_outlier import TimeSeriesOutlier

ARG_DEFAULTS = {
    'window': 20,
    'offset': 0.5,
    'crit_lof': 1.0,
    'wrap': True
}

class WindowedLocalOutlierFactor(LocalOutlierFactor, TimeSeriesOutlier):
    def __init__(self, **kwargs):
        '''
        Detects temporal outliers in one-dimensional time series data
        using a sliding window spatial embedding scheme and Local Outlier Factor

        :param int window:      (Optional) Length of sliding window (default 20)
        :param float offset:    (Optional) Relative position within window to map sample to embedded coordinate (default 0.5)
        :param float crit_lof:  (Optional) Value of LOF above which a point is considered outlying (default 1.0)
        :param bool wrap:       (Optional) Whether or not to wrap data to preserve number of samples (default: true)
        :kwargs:                (Optional) Parameters to pass to sklearn.neighbors.LocalOutlierFactor constructor
        '''
        for key, val in ARG_DEFAULTS.items():
            setattr(self, key, kwargs.get(key, val))
            try:
                del kwargs[key]
            except KeyError:
                pass
        super().__init__(**kwargs)

    def _get_rolling_windows(self, data):
        '''
        Generate an array representing the n-dimensional embedded space for the windowed time series

        :param numpy.ndarray data: 1D time series to be embedded
        :return: array of resulting phase space with shape (n_samples, window)
                (n_samples will be truncated by (window-1) if wrap=False)
        :rtype: numpy.ndarray
        '''
        offset = self._offset
        window = self._window
        offset_idx = int(offset*window)
        end_padding = (window - offset_idx) - 1
        # Padding by wrapping
        if self._wrap:
            data_length = data.size
            if offset_idx > 0:
                data = np.insert(data, 0, data[-offset_idx:])
            if end_padding > 0:
                data = np.append(data, data[0:end_padding])
        else:
            data_length = data.size - (window-1)
        # Stack offset data to make higher-dimensional embedding
        frames = [data[i:data_length+i] for i in range(window)]
        return np.stack(frames, axis=-1)

    def fit(self, data, times=None):
        '''
        Populate internal parameters based on input time series

        :param numpy.ndarray data: 1D time series to be processed
                                   (will be reshaped to (-1,) if multi-dimensional)
        :param numpy.ndarray times: (Optional) Corresponding time labels (must have same first dimension as data)
        :return: time series and corresponding time labels (if provided); truncated if wrap=False
        :rtype: numpy.ndarray
        '''
        if times is not None and data.shape[0] != times.shape[0]:
            raise ValueError(
                f'Expected times {times.shape} to have the same number of entries as data {data.shape}')
        data = data.reshape(-1)
        windowed_data = self._get_rolling_windows(data)
        super().fit(windowed_data)
        super()._set_embedded_data(windowed_data)
        super()._set_truncated_data(data, times, self.negative_outlier_factor_.size)


    def get_outlier_indices(self):
        return np.where(-self.negative_outlier_factor_ > self.crit_lof)[0]

    
    def get_outlier_factors(self):
        return -self.negative_outlier_factor_
