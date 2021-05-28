import numpy as np

class TimeSeriesOutlier:
    '''Abstract class providing standard access methods for LOF and TOF'''

    def _set_embedded_data(self, data):
        self.embedded_data = data


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
