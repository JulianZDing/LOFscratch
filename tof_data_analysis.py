import os
import numpy as np

from collections import defaultdict
from matplotlib import pyplot as plt
from gwpy.timeseries import TimeSeries
from gwpy.time import tconvert
from tqdm.auto import tqdm

from ts_outlier_detection import *
from ts_outlier_detection.plotting import *
from utils import *


def _scale_width_asymptotic(base, width, maximum):
    return maximum * (1 - np.exp(-base * (width + 1) / maximum))


def _generate_windows(start, stop, level, offset):
    step = offset * level
    left = np.arange(start, stop, step)
    right = left + level
    right[right > stop] = stop
    return np.stack((left, right), axis=-1)


def tof_detections_in_open_data(
    det, start, stop,
    padding=2, levels=[4, 30, 120], base_level=None, offset=1,
    plot=True, plot_windows=False, save_dir=None, colors=['red', 'blue', 'black'],
    **kwargs
):
    '''
    Plot and return detections made by TOF within a time frame in open data from a GW detector

    :param str det: String naming target detector (ex. H1, L1, V1)
    :param float start: GPS time for start of time window
    :param float stop: GPS time for end of time window
    
    :param float padding: (Optional) 
    :param list levels: (Optional) 
    :param float base_level: (Optional) 
    :param float offset: (Optional) 
    :param list colors: (Optional) 

    Remaining parameters are passed to TemporalOutlierFactor constructor
    '''
    rdata = TimeSeries.fetch_open_data(det, start-padding, stop+padding, cache=False)

    if base_level is None:
        base_level = levels[0]

    if plot:
        width = round((stop - start) / base_level)
        scaled_width = _scale_width_asymptotic(4, width, 100)
        scaled_base_level = base_level * width / scaled_width
        fig, ax = plt.subplots(1, 1, figsize=(scaled_width, 6))
        ax.set_title(f'TOF detections on {det} between {tconvert(start)} and {tconvert(stop)}')
        base_windows = _generate_windows(start, stop, scaled_base_level, 1)
        plot_high_resolution_qscan(ax, rdata, tqdm(base_windows, desc='Main plot'), vmin=0, vmax=15)
        ax.set_xlim(start, stop)
        ax.set_xlabel('GPS Time (s)')
        ax.set_ylabel('Frequency (Hz)')
        ax.set_ylim(-5*len(levels), 300)
    
    level_windows = {level: _generate_windows(start, stop, level, offset) for level in levels}
    
    outlier_times = defaultdict(lambda: np.array([]))
    ctof = TemporalOutlierFactor(**kwargs)
    hdata = preprocess_timeseries(rdata)[0][(rdata.times.value >= start) & (rdata.times.value < stop)]
    samples = base_level * rdata.sample_rate
    
    for level, windows in level_windows.items():
        resampled_data = hdata.resample(samples / level)
        for left, right in tqdm(windows, desc=f'Windows of length {level}'):
            wdata = resampled_data[(resampled_data.times.value >= left) & (resampled_data.times.value < right)]
            data = wdata.value
            times = wdata.times.value
            ctof.fit(data, times)
            outliers = times[ctof.get_outlier_indices()]
            outlier_times[level] = np.append(outlier_times[level], outliers)

            if plot_windows:
                fig1 = plt.figure(figsize=(9, 9), constrained_layout=True)
                gs = fig1.add_gridspec(4, 1)
                bax = fig1.add_subplot(gs[3])
                bax.set_ylabel('TOF')
                bax.set_xlabel('GPS Time (s)')
                tax = fig1.add_subplot(gs[2], sharex=bax)
                tax.set_ylabel('Strain')
                plot_ts_outliers(ctof, (tax, bax))
                oax = fig1.add_subplot(gs[0:2], sharex=bax)
                oax.set_ylabel('Frequency (Hz)')
                oax.set_title(f'TOF detections on {det} between {tconvert(left)} and {tconvert(right)}')
                plot_qscan(oax, rdata, q_kwargs={'outseg': (left, right)}, vmin=0, vmax=15)
                oax.set_ylim(-5, 300)
                if save_dir is not None:
                    fig1.savefig(os.path.join(save_dir, f'{det}_{left}_{right}.png'), bbox_inches='tight')
                plt.close(fig1)
    
    if plot:    
        for i, (level, outliers) in enumerate(outlier_times.items()):
            c = colors[i % len(colors)]
            plot_detections(ax, outliers, ypos=-5*i, label=f'{level}s window size', color=c)

        ax.set_xticks(np.arange(start, stop, scaled_base_level))
        ax.legend()
        ax.grid(False)
        plt.tight_layout()
        if save_dir:
            fig.savefig(os.path.join(save_dir, f'multi_window_{det}_{left}_{right}.jpg'), bbox_inches='tight')

    return np.unique(np.concatenate(list(outlier_times.values()), axis=None))
