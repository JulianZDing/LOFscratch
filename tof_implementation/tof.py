import numpy as np
from sklearn.neighbors import NearestNeighbors

class TemporalOutlierFactor:
    def __init__(
        self, dims=3, delay=1, q=2,
        n_neighbors=None, dist_metric='minkowski', p=2, metric_params=None,
        event_length=80
    ):
        '''
        Detects unique events ("unicorns") in one-dimensional time series data

        :param int dims: Embedding dimension
        :param int delay: Number of indices in the data to offset for time delay
        :param int q: Exponent degree to use in TOF calculation
        :param int n_neighbors: Number of nearest neighbors to consider (default: dims+1)
        :param str dist_metric: Distance metric to pass to sklearn.neighbors.NearestNeighbors
        :param int p: Minkowski degree to pass to sklearn.neighbors.NearestNeighbors
        :param dict metric_params: Additional args to pass to sklearn.neighbors.NearestNeighbors
        :param int event_length: Maximum detectable event length (samples); sets TOF detection threshold
        '''
        self.dims = dims
        self.delay = delay
        self.q = q
        self.n_neighbors = (dims+1) if n_neighbors is None else n_neighbors
        self.kNN = NearestNeighbors(
            n_neighbors=self.n_neighbors, metric=dist_metric, p=p, metric_params=metric_params)
        self._set_threshold(event_length)
    

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
        :return: (n_samples, dims) array of resulting phase space
        :rtype: numpy.ndarray
        '''
        embedded_data = []
        for i in range(self.dims):
            if i+1 == self.dims:
                offset_data = data[i:]
            else:
                offset_data = data[i:(i+1-self.dims)]
            embedded_data.append(offset_data)
        return np.stack(embedded_data, axis=1)


    def fit(self, data):
        '''
        Populate internal parameters based on input time series

        :param numpy.ndarray data: 1D time series to be processed
                                   (will be reshaped to (-1,) if multi-dimensional)
        '''
        data = data.reshape(-1)
        self.kNN.fit(self._time_delay_embed(data))
        self._set_tof(self.kNN.kneighbors(return_distance=False))
