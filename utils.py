import numpy as np

from inspect import signature
from matplotlib import pyplot as plt
from gwpy.signal import filter_design
from gwpy.timeseries import TimeSeries

from ts_outlier_detection import *
from ts_outlier_detection.plotting import *

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


## Plotting functions

def plot_omegascan_detections(ax, ts, detections, ypos=BANDPASS[0]-10, **kwargs):
    image = ax.pcolormesh(ts.q_transform(**kwargs), vmin=0, vmax=15, zorder=1)
    markers, = ax.plot(detections, np.full(detections.shape, ypos), 'k.', zorder=2)
    ax.set_ylim(0, 300)
    cbar = ax.colorbar()
    cbar.set_label('Normalized energy')
    return [image, markers]

def plot_outlier_comparison(hdata, rdata, title=None, fontsize=None, tso=TemporalOutlierFactor, **kwargs):
    fig = plt.figure(figsize=(18, 6), constrained_layout=True)
    gs = fig.add_gridspec(2, 2)
    bax = fig.add_subplot(gs[1,0])
    bax.set_ylabel(tso.__name__, fontsize=fontsize)
    bax.set_xlabel('GPS Time (s)', fontsize=fontsize)
    tax = fig.add_subplot(gs[0,0], sharex=bax)
    tax.set_ylabel('Strain', fontsize=fontsize)
    if title is not None:
        tax.set_title(title, fontsize=fontsize)
    oax = fig.add_subplot(gs[:,1])
    oax.set_ylabel('Frequency (Hz)', fontsize=fontsize)
    oax.set_xlabel('GPS Time (s)', fontsize=fontsize)

    ctof = tso(**kwargs)
    data = hdata.value
    times = hdata.times.value
    ctof.fit(data, times)
    plot_ts_outliers(ctof, (tax, bax))
    plot_omegascan_detections(
        oax, rdata, hdata.times.value[ctof.get_outlier_indices()],
        outseg=(hdata.times.value[0], hdata.times.value[-1])
    )
    
def plot_outlier_comparison_vertical(hdata, rdata, title=None, fontsize=None, tso=TemporalOutlierFactor, **kwargs):
    fig = plt.figure(figsize=(9, 9), constrained_layout=True)
    gs = fig.add_gridspec(3, 1)
    tax = fig.add_subplot(gs[2])
    tax.set_ylabel(tso.__name__, fontsize=fontsize)
    tax.set_xlabel('GPS Time (s)', fontsize=fontsize)
    oax = fig.add_subplot(gs[0:2], sharex=tax)
    oax.set_ylabel('Frequency (Hz)', fontsize=fontsize)
    if title is not None:
        oax.set_title(title, fontsize=fontsize)
    
    dummyfig, dummyax = plt.subplots(1,1)
    ctof = tso(**kwargs)
    data = hdata.value
    times = hdata.times.value
    ctof.fit(data, times)
    plot_ts_outliers(ctof, (dummyax, tax))
    plt.close(dummyfig)
    plot_omegascan_detections(
        oax, rdata, hdata.times.value[ctof.get_outlier_indices()],
        outseg=(hdata.times.value[0], hdata.times.value[-1])
    )

def plot_omega_phase(hdata, rdata, title=None, fontsize=None, tso=TemporalOutlierFactor, **kwargs):
    fig, axs = plt.subplots(1, 2, figsize=(18, 7))
    oax, pax = axs
    oax.set_ylabel('Frequency (Hz)', fontsize=fontsize)
    oax.set_xlabel('GPS Time (s)', fontsize=fontsize)
    if title is not None:
        oax.set_title(title, fontsize=fontsize)
    pax.set_xlabel('$x(t)$', fontsize=fontsize)
    pax.set_ylabel('$x(t+𝜏)$', fontsize=fontsize)
    pax.set_title('Time-embedded phase space', fontsize=fontsize)

    ctof = tso(**kwargs)
    data = hdata.value
    times = hdata.times.value
    ctof.fit(data, times)
    
    plot_omegascan_detections(
        oax, rdata, hdata.times.value[ctof.get_outlier_indices()],
        outseg=(hdata.times.value[0], hdata.times.value[-1])
    )
    plot_2d_phase_space(
        ctof.get_embedded_data(), pax, outlier_ids=ctof.get_outlier_indices())
    fig.subplots_adjust(hspace=0.5)

def plot_outlier_phase(hdata, rdata, title=None, fontsize=None, tso=TemporalOutlierFactor, **kwargs):
    fig = plt.figure(figsize=(18, 8), constrained_layout=True)
    gs = fig.add_gridspec(2, 2)
    bax = fig.add_subplot(gs[1,0])
    bax.set_ylabel(tso.__name__, fontsize=fontsize)
    bax.set_xlabel('GPS Time (s)', fontsize=fontsize)
    tax = fig.add_subplot(gs[0,0], sharex=bax)
    tax.set_ylabel('Strain', fontsize=fontsize)
    if title is not None:
        tax.set_title(title, fontsize=fontsize)
    pax = fig.add_subplot(gs[:,1])
    pax.set_xlabel('$x(t)$', fontsize=fontsize)
    pax.set_ylabel('$x(t+𝜏)$', fontsize=fontsize)
    pax.set_title('Time-embedded phase space', fontsize=fontsize)

    scatter_style={'s': 10, 'facecolors': 'k'}
    ctof = tso(**kwargs)
    data = hdata.value
    times = hdata.times.value
    ctof.fit(data, times)
    plot_ts_outliers(ctof, (tax, bax), scatter_style=scatter_style)
    plot_2d_phase_space(
        ctof.get_embedded_data(), pax, outlier_ids=ctof.get_outlier_indices(),
        scatter_style=scatter_style
    )


## Validation functions

def diff_loss(true_time, detected_times, scale=np.square):
    return np.sum(scale(detected_times-true_time))


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


def estimate_gradient(f, x, eps=1e-4, approximator=first_order_derivative):
    dims = len(signature(f).parameters)
    if len(x) != dims:
        raise ValueError(
            f'Function {f.__name__} requires inputs of length {dims}; given {len(x)}')
    return np.array([
        first_order_derivative(lambda a: f(*x[:i], a, *x[i+1:]), j, eps) for i, j in enumerate(x)
    ])
