import tushare as ts
import pandas as pd
import time
import os.path
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import Util
import DB
import os
import datetime
import imageio
import glob
import shutil

pd.options.mode.chained_assignment = None  # default='warn'


def create_gif(folder_path="000002.SZ"):
    images = []
    for jpgfile in glob.iglob(os.path.join("Media/Plot/stock/" + folder_path, "*.jpg")):
        images.append(imageio.imread(jpgfile))
        print("load image", jpgfile)
    output_file = 'Gif-%s.gif' % datetime.datetime.now().strftime('%Y-%M-%d-%H-%M-%S')
    print("Plotting...please wait...")
    imageio.mimsave(output_file, images, duration=0.005)


def support_resistance_once(df, freq, max_rs, s_minmax, adj_start_date, end_date):
    # with bins, we can find clustered resistance

    s_occurence_bins = s_minmax.value_counts(bins=2000)

    counter = 1
    for index, value in s_occurence_bins.iteritems():
        if counter <= max_rs:
            df.loc[adj_start_date:end_date, f"rs{freq}_{counter}"] = index.left
            df[f"rs{freq}_{counter}"].replace(0, np.nan, inplace=True)
            counter = counter + 1
        else:
            break


def support_resistance_multiple(window=1000, rolling_freq=240, step=1):
    dict_rs = {int(round(window / 5) * i): 2 for i in range(1, 6)}  # 1 to 6 means 5 resistance freq with each 2 pieces

    df_ts_code = DB.get_ts_code()
    df_ts_code = df_ts_code[df_ts_code["ts_code"] == "000002.SZ"]

    for ts_code in df_ts_code["ts_code"]:
        df_asset = DB.get_asset(ts_code=ts_code)[::step]  # only see every 5th day

        # calculate all min max for acceleration used for later simulation
        s_minall = df_asset["close"].rolling(240).min()
        s_maxall = df_asset["close"].rolling(240).max()

        # only consider close and add rsi for plotting reason
        df_asset = df_asset[["close"]]
        for freq, counter in dict_rs.items():
            for i in range(1, counter):
                df_asset[f"rs{freq}_{i}"] = 0

        # iterate over past data as window
        for row in range(1000, len(df_asset)):
            start_date = df_asset.index[row]
            end_date = df_asset.index[row + window]
            df_partcial = df_asset.loc[start_date: end_date]
            print(f"resistance {ts_code} {start_date} to {end_date}")

            for freq, max_rs in dict_rs.items():
                adj_start = df_asset.index[row + window - freq]
                s_minmax = s_minall.loc[adj_start:end_date].append(s_maxall.loc[adj_start:end_date])
                support_resistance_once(df=df_partcial, freq=freq, max_rs=max_rs, s_minmax=s_minmax, adj_start_date=adj_start, end_date=end_date)

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

if __name__ == '__main__':
    support_resistance_multiple()
    #create_gif()
