import numpy as np

from inspect import signature
from matplotlib import pyplot as plt
from gwpy.timeseries import TimeSeries

## Data fetching and pre-processing functions

BANDPASS = (20, 500)


def get_processed_event(
    detector, gps_time,
    length=10, offset=0, edges=None, downsample=1, bp=[BANDPASS], return_raw=False
):
    edges = length/5 if edges is None else edges
    ts = get_raw_event(detector, gps_time, length, offset, edges)
    border = int(edges/ts.dt.value)
    processed_ts = [s[border:-border:downsample] for s in preprocess_timeseries(ts, bp=bp)]
    if return_raw:
        return processed_ts, ts
    return processed_ts


def get_raw_event(detector, gps_time, length, offset, edges):
    bracket = edges + length/2
    t0 = gps_time - bracket + offset
    t1 = gps_time + bracket + offset
    return TimeSeries.fetch_open_data(detector, t0, t1, cache=False)
    

def preprocess_timeseries(ts, bp=[BANDPASS]):
    # Whiten
    ts = ts.whiten()
    # Bandpass
    output = []
    for band in bp:
        output.append(ts.bandpass(*band))
    return output


## Validation functions

def diff_loss(true_time, detected_times):
    if detected_times.size < 2:
        return 1000
    return np.sum(np.square(detected_times-true_time))


def inverse_diff_loss(true_time, detected_times):
    diffs = detected_times - true_time + 1e-7
    if diffs.size > 0:
        return np.mean(-1 / np.abs(diffs))
    return 0


def stdev_loss(true_time, detected_times):
    if detected_times.size < 2:
        return 0
    return np.mean(np.abs(detected_times - true_time)) - detected_times.size / np.std(detected_times)


def roc_loss(true_time, detected_times, range=1, offset=0):
    start = true_time + offset*range
    end = true_time + (offset+1)*range
    true_positive_idx = (detected_times >= start) & (detected_times <= end)
    true_positives = detected_times[true_positive_idx]
    false_positives = detected_times[np.logical_not(true_positive_idx)]
    roc = true_positives.size / false_positives.size
    return -roc


def first_order_derivative(f, x, h):
    return (f(x+h) - f(x)) / h


def estimate_gradient(f, x, eps=1e-4):
    dims = len(signature(f).parameters)
    if len(x) != dims:
        raise ValueError(
            f'Function {f.__name__} requires inputs of length {dims}; given {len(x)}')
    return np.array([
        first_order_derivative(lambda a: f(*x[:i], a, *x[i+1:]), j, eps) for i, j in enumerate(x)
    ])
