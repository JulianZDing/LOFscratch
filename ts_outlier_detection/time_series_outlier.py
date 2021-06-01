import numpy as np

ARG_DEFAULTS = {
    'dims': (3, False),
    'delay': (1, False),
    'offset': (0.0, False),
    'wrap': (True, False),
    'n_neighbors': (20, True)
}


def set_arg_defaults(obj, defaults, kwargs={}, delete=True):
    '''
    Function for setting an object's attributes using keyword arguments

    :param object obj: Target object
    :param dict defaults: Dictionary {key: (default value, share?)}
                          If share is False, key will be deleted from kwargs if present
    :param dict kwargs: (Optional) Dictionary containing override values
    '''
    for key, (val, share) in defaults.items():
        setattr(obj, key, kwargs.get(key, val))
        if not share:
            try:
                del kwargs[key]
            except KeyError:
                pass
    return kwargs


class TimeSeriesOutlier:
    def __init__(self, **kwargs):
        '''
        Superclass providing standard access methods for LOF and TOF

        :param int dims:            (Optional) Embedding dimension (default 3)
        :param int delay:           (Optional) Number of indices in the data to offset for time delay (default 1)
        :param float offset:        (Optional) Relative position within window to map sample to embedded coordinate (default 0)
        :param bool wrap:           (Optional) Whether or not to wrap data to preserve number of samples (default true)
        
        (Additional parameters will be saved in self.kwargs)
        '''
        self.unused_kwargs = set_arg_defaults(self, ARG_DEFAULTS, kwargs)

    
    def _time_delay_embed(self, data):
        '''
        Generate and save an array representing the time-embedded phase space of an input time series

        :param numpy.ndarray data: 1D time series to be embedded
        '''
        width = int(self.dims*self.delay)
        data_length = data.size
        if self.wrap:
            offset_idx = int(self.offset*width)
            end_padding = width - offset_idx - 1
            if offset_idx > 0:
                data = np.insert(data, 0, data[-offset_idx:])
            if end_padding > 0:
                data = np.append(data, data[0:end_padding])
        else:
            data_length -= (width-1)
        frames = [data[(i*self.delay):(i*self.delay + data_length):self.delay] for i in range(self.dims)]
        self.embedded_data = np.stack(frames, axis=-1)


    def _set_truncated_data(self, data, times, length):
        self.trunc_data = data[:length]
        if times is not None:
            self.trunc_times = times[:length]
        else:
            self.trunc_times = np.arange(0, length)
    

    def get_embedded_data(self):
        '''
        Get array of embedded data

        :return: data array of shape (n_samples, n_dimensions)
        :rtype: numpy.ndarray
        '''
        return self.embedded_data


    def get_truncated_data(self):
        '''
        Get data and time labels corresponding to all processed samples

        :return: data array, time label array
        :rtype: (numpy.ndarray, numpy.ndarray)
        '''
        return self.trunc_data, self.trunc_times

    def get_outlier_indices(self):
        '''
        Get a list of data indices corresponding to detected outliers
        :return: array of indices
        :rtype: numpy.ndarray
        '''
        pass

    def get_outlier_factors(self):
        '''
        Get a list of all the labelled outlier factors for the fitted time series

        :return: array of floats
        :rtype: numpy.ndarray
        '''
        pass
