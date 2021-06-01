import numpy as np
from sklearn.neighbors import LocalOutlierFactor

from ts_outlier_detection.time_series_outlier import set_arg_defaults, TimeSeriesOutlier

ARG_DEFAULTS = {
    'crit_lof': (1.0, False),
    'crit_sigma': (None, False),
}

class WindowedLocalOutlierFactor(TimeSeriesOutlier):
    def __init__(self, **kwargs):
        '''
        Detects temporal outliers in one-dimensional time series data
        using a sliding window spatial embedding scheme and Local Outlier Factor

        :param float crit_sigma:    (Optional) Alternative to specifying crit_lof; number of sigmas from mean to consider a point outlying (overrides crit_lof)
        :param bool wrap:           (Optional) Whether or not to wrap data to preserve number of samples (default: true)

        Remaining parameters are passed to sklearn.neighbors.LocalOutlierFactor
        '''
        kwargs = set_arg_defaults(self, ARG_DEFAULTS, kwargs)
        super().__init__(**kwargs)
        self.clf = LocalOutlierFactor(**self.unused_kwargs)


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
        self._time_delay_embed(data)
        self.clf.fit(self.get_embedded_data())
        self.lofs_ = -self.clf.negative_outlier_factor_
        super()._set_truncated_data(data, times, self.lofs_.size)


    def get_outlier_indices(self):
        if self.crit_sigma is not None:
            mean_lof = np.mean(self.lofs_)
            lof_std = np.std(self.lofs_)
            self.crit_lof  = self.crit_sigma*lof_std + mean_lof
        return np.where(self.lofs_ > self.crit_lof)[0]

    
    def get_outlier_factors(self):
        return self.lofs_
