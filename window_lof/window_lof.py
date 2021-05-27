import numpy as np
from sklearn.neighbors import LocalOutlierFactor

class WindowedLocalOutlierFactor(LocalOutlierFactor):
    def __init__(self, **kwargs):
        self._window = kwargs.get('window', 20)
        self._offset = kwargs.get('offset', 0.5)
        self._wrap = kwargs.get('wrap', True)
        try:
            del kwargs['window']
            del kwargs['offset']
            del kwargs['wrap']
        except KeyError:
            pass
        super().__init__(**kwargs)

    def _get_rolling_windows(self, data):
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
        windows = np.stack(frames, axis=-1)
        return windows

    def fit(self, data, times=None):
        if times is not None and data.shape[0] != times.shape[0]:
            raise ValueError(
                f'Expected times {times.shape} to have the same number of entries as data {data.shape}')
        data = data.reshape(-1)
        windowed_data = self._get_rolling_windows(data)
        super().fit(windowed_data)
