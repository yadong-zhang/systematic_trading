import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm, trange

# from https://jfin-swufe.springeropen.com/articles/10.1186/s40854-020-00187-0#Sec7
# code from https://github.com/pecu/Series2GAF/blob/master/series2gaf.py
def GenerateGAF(all_ts, window_size, rolling_length, fname, normalize_window_scaling=1.0, method='summation',
                scale='[0,1]'):

    n = len(all_ts)

    moving_window_size = int(window_size * normalize_window_scaling)

    # np.floor rounds the decimals downwards
    # finds the number of windows we can make with our rollign window size, which dictates the step szie before calculating another GAN.
    n_rolling_data = int(np.floor((n - moving_window_size) / rolling_length))


    gramian_field = []


    # Prices = []

    # trange seems to just do the same as xrange but with more comentary and progress updates
    for i_rolling_data in trange(n_rolling_data, desc="Generating...", ascii=True):


        start_flag = i_rolling_data * rolling_length


        full_window_data = list(all_ts[start_flag: start_flag + moving_window_size])


        # Prices.append(full_window_data[-int(window_size*(normalize_window_scaling-1)):])


        rescaled_ts = np.zeros((moving_window_size, moving_window_size), float)
        min_ts, max_ts = np.min(full_window_data), np.max(full_window_data)
        if scale == '[0,1]':
            diff = max_ts - min_ts
            if diff != 0:
                rescaled_ts = (full_window_data - min_ts) / diff
        if scale == '[-1,1]':
            diff = max_ts - min_ts
            if diff != 0:
                rescaled_ts = (2 * full_window_data - diff) / diff


        rescaled_ts = rescaled_ts[-int(window_size * (normalize_window_scaling - 1)):]

        #for each window size we create a gam field?
        this_gam = np.zeros((window_size, window_size), float)
        #Given an interval, values outside the interval are clipped to the interval edges.
        # For example, if an interval of [0, 1] is specified, values smaller than 0 become 0, and values larger than 1 become 1.
        sin_ts = np.sqrt(np.clip(1 - rescaled_ts ** 2, 0, 1))
        if method == 'summation':
            #Given two vectors, a = [a0, a1, ..., aM] and b = [b0, b1, ..., bN], the outer product [1] is:
            # [[a0*b0  a0*b1 ... a0*bN ]
            #  [a1*b0    .
            #  [ ...          .
            #  [aM*b0            aM*bN ]]
            this_gam = np.outer(rescaled_ts, rescaled_ts) - np.outer(sin_ts, sin_ts)
        if method == 'difference':

            this_gam = np.outer(sin_ts, rescaled_ts) - np.outer(rescaled_ts, sin_ts)
        # for each of those window sizes, we then append each of those to the number of times we can cycle through the
        gramian_field.append(this_gam)


        del this_gam

    # 輸出 Gramian Angular Field
    np.array(gramian_field).dump('%s_gaf.pkl' % fname)


    return gramian_field


def PlotHeatmap(all_img, save_dir='output_img'):

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)


    total_length = all_img.shape[0]


    fname_zero_padding_size = int(np.ceil(np.log10(total_length)))


    for img_no in trange(total_length, desc="Output Heatmaps...", ascii=True):
        this_fname = str(img_no).zfill(fname_zero_padding_size)
        plt.imshow(all_img[img_no], cmap='hot', interpolation='nearest')
        plt.axis('off')
        plt.savefig("%s/%s.png" % (save_dir, this_fname), bbox_inches='tight', pad_inches=0, transparent=True)
        plt.clf()


#
#
# DEMO
#
#
if __name__ == '__main__':
    random_series = np.random.uniform(low=50.0, high=150.0, size=(200,))
    from series2gaf import *
    import pandas as pd
    from matplotlib import pyplot as plt

    EURUSD = pd.DataFrame(pd.read_csv(r"C:\Users\edgil\Documents\SysTrade\candlesticks\EURUSDOHLC.csv"))
    EURUSD['Datetime'] = pd.to_datetime(EURUSD['Datetime'], format="%d/%m/%Y %H:%M")

    testdata = EURUSD[:200]
    testy = testdata[["Open", "High", "Low", "Close"]].values
    timeSeries = list(random_series)
    windowSize = 50
    rollingLength = 10
    fileName = 'demo_%02d_%02d' % (windowSize, rollingLength)
    GenerateGAF(all_ts=testy,
                window_size=windowSize,
                rolling_length=rollingLength,
                fname=fileName,
                normalize_window_scaling=1.0)

    ts_img = np.load('%s_gaf.pkl' % fileName)
    PlotHeatmap(ts_img)