import tushare as ts
import pandas as pd
import time
import os.path
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import DB
import os
import datetime
import imageio
import glob
from multiprocessing import Process
pd.options.mode.chained_assignment = None  # default='warn'

def create_gif(ts_code="000002.SZ"):
    images = []
    for jpgfile in glob.iglob(os.path.join("Media/Plot/stock/" + ts_code, "*.jpg")):
        images.append(imageio.imread(jpgfile))
        print(f"{ts_code} load image", jpgfile)
    output_file = f"Media/Plot/stock/{ts_code}_{datetime.datetime.now().strftime('%Y_%M_%d_%H_%M_%S')}.gif"
    print("Plotting...please wait...")
    imageio.mimsave(output_file, images, duration=0.005)

def support_resistance_once_plot(window=1000, rolling_freq=20, ts_code="000002.SZ", step=5):
    def support_resistance_acc(df, freq, max_rs, s_minmax, adj_start_date, end_date, df_asset):
        s_occurence_bins = s_minmax.value_counts(bins=100)
        for counter, (index, value) in enumerate(s_occurence_bins.iteritems()):
            df.loc[adj_start_date:end_date, f"rs{freq}_{counter}"] = index.left
            df[f"rs{freq}_{counter}"].replace(0, np.nan, inplace=True)

    # 1 to 6 means 5 resistance freq with each 2 pieces
    dict_rs = {int(round(window / (2 ** i))): 4 for i in range(0, 6)}

    # calculate all min max for acceleration used for later simulation
    df_asset = DB.get_asset(ts_code=ts_code)
    s_minall = df_asset["close"].rolling(rolling_freq).min()
    s_maxall = df_asset["close"].rolling(rolling_freq).max()

    # only consider close and add rsi for plotting reason
    df_asset = df_asset[["close"]]

    # iterate over past data as window
    for row in range(4000, len(df_asset), step):
        start_date = df_asset.index[row]
        try:
            end_date = df_asset.index[row + window]
        except:  # hits the end
            break
        df_partcial = df_asset.loc[start_date: end_date]
        print(f"resistance {ts_code} {start_date} to {end_date}")

        for freq, max_rs in dict_rs.items():
            adj_start = df_asset.index[row + window - freq]
            s_minmax = (s_minall.loc[adj_start:end_date]).append(s_maxall.loc[adj_start:end_date])
            support_resistance_acc(df=df_partcial, freq=freq, max_rs=max_rs, s_minmax=s_minmax, adj_start_date=adj_start, end_date=end_date, df_asset=df_asset)

        # plot graph and save it
        df_partcial.reset_index(inplace=True, drop=True)
        df_partcial.plot(legend=False)

        newpath = f"Media/Plot/stock/{ts_code}/"
        if not os.path.exists(newpath):
            os.makedirs(newpath)
        plt.savefig(newpath + f"{start_date}_{end_date}.jpg")
        # plt.show()
        # df_partcial.to_csv(f"resistance{row}.csv", index=False)
        plt.close()

def support_resistance_plot_multiple(step=1):
    ts_code = "000002.SZ"
    support_resistance_once_plot(ts_code=ts_code, step=1)
    create_gif(ts_code=ts_code)

    # for step in [10, 20]:  # performance : how many days should I refresh the future rs line. Ultimate step should be 1
    #     for start_window in [240]:  # how long is the starting window
    #         for rolling_freq in [1]:  # how many past days should I use to calculate
    #             for thresh in [[4, 0.2]]:  # how far is the spread from current price to the line. bigger spread better
    #                 for rs_count in [8, 4]:  # how many lines for abv and und current price. more is better
    #                     for bins in [8]:  # performance:  how big is the distance between the lines themselves. smaller bin better
    #                         for delay in [1, 3, 5, 10, 20]:

if __name__ == '__main__':
    # create_gif(ts_code="000001.SZ")
    # support_resistance_multiple()
    pass
