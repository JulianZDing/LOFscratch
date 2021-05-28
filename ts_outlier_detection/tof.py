import numpy as np
from sklearn.neighbors import NearestNeighbors

from ts_outlier_detection.time_series_outlier import TimeSeriesOutlier

class TemporalOutlierFactor(TimeSeriesOutlier):
    def __init__(
        self, dims=3, delay=1, q=2,
        n_neighbors=None, dist_metric='minkowski', p=2, metric_params=None,
        event_length=80, wrap=True
    ):
        '''
        Detects unique events ("unicorns") in one-dimensional time series data

        :param int dims:            (Optional) Embedding dimension (default 3)
        :param int delay:           (Optional) Number of indices in the data to offset for time delay (default 1)
        :param int q:               (Optional) Exponent degree to use in TOF calculation (default 2)
        :param int n_neighbors:     (Optional) Number of nearest neighbors to consider (default dims+1)
        :param str dist_metric:     (Optional) Distance metric to pass to sklearn.neighbors.NearestNeighbors (default minkowski)
        :param int p:               (Optional) Minkowski degree to pass to sklearn.neighbors.NearestNeighbors (default 2)
        :param dict metric_params:  (Optional) Additional args to pass to sklearn.neighbors.NearestNeighbors
        :param int event_length:    (Optional) Maximum detectable event length (samples); sets TOF detection threshold (default 80)
        :param bool wrap:           (Optional) Whether or not to wrap data to preserve number of samples (default true)
        '''
        self.dims = dims
        self.delay = delay
        self.q = q
        self.n_neighbors = (dims+1) if n_neighbors is None else n_neighbors
        self.kNN = NearestNeighbors(
            n_neighbors=self.n_neighbors, metric=dist_metric, p=p, metric_params=metric_params)
        self._set_threshold(event_length)
        self.wrap = wrap
    

    def _set_threshold(self, event_length):
        '''
        Set TOF threshold based on desired detectable event length

        :param int event_length: Maximum detectable event length
        '''
        indices = np.arange(0, self.dims)
        M = np.full(indices.shape, event_length)
        self.threshold = np.sqrt(np.sum(np.power(M-indices, 2)) / self.dims)


    def _set_tof(self, neighbor_indices):
        '''
        Helper function to set TOF parameters

        :param numpy.ndarray neighbor_indices: (n_samples, n_neighbors) array of n_neighbors nearest neighbors
                                               to each point in time-embedded phase space
        '''
        indices = np.stack([np.arange(0, neighbor_indices.shape[0]) for _ in range(self.n_neighbors)], axis=-1)
        self.tofs_ = np.power(
            np.sum(
                np.power(
                    np.absolute(
                        np.add(indices, -neighbor_indices)),
                    self.q
                ),
                axis=-1
            ) / self.dims,
            1 / self.q
        )
        self.tof_detections_ = np.where(self.tofs_ < self.threshold)[0]


    def _time_delay_embed(self, data):
        '''
        Generate an array representing the phase space of the time-embedded time series

        :param numpy.ndarray data: 1D time series to be embedded
        :return: array of resulting phase space with shape (n_samples, dims)
                (n_samples will be truncated by delay*(dims-1) if wrap=False)
        :rtype: numpy.ndarray
        '''
        overflow = self.delay*(self.dims-1)
        if self.wrap:
            data_length = data.size
            extension = data[0:overflow]
            data = np.append(data, extension)
        else:
            data_length = data.size - overflow
        embedded_data = [data[(i*self.delay):(i*self.delay + data_length)] for i in range(self.dims)]
        return np.stack(embedded_data, axis=1)


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
        embedded_data = self._time_delay_embed(data)
        self.kNN.fit(embedded_data)
        self._set_tof(self.kNN.kneighbors(return_distance=False))
        super()._set_embedded_data(embedded_data)
        super()._set_truncated_data(data, times, self.tofs_.size)


    def get_outlier_indices(self):
        return self.tof_detections_

    
    def get_outlier_factors(self):
        return self.tofs_
