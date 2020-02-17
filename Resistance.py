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
    output_file = 'Gif-%s.gif' % datetime.datetime.now().strftime('%Y-%M-%d-%H-%M-%S')
    print("Plotting...please wait...")
    imageio.mimsave(output_file, images, duration=0.005)


def support_resistance_once(df, rs_needed, s_minmax):
    # with bins, we can find clustered resistance
    series_occurence_bins = s_minmax.value_counts(bins=2000)

    counter = 0
    for index, value in series_occurence_bins.iteritems():
        if counter < rs_needed:
            df[f"rs{counter}"] = index.left
            counter = counter + 1
        else:
            break


def support_resistance_multiple(rs=6, window=500, rolling_freq=240, step=1):
    df_ts_code = DB.get_ts_code()
    df_ts_code = df_ts_code[df_ts_code["ts_code"] == "000002.SZ"]

    for ts_code in df_ts_code["ts_code"]:
        df_asset = DB.get_asset(ts_code=ts_code)[::step]  # only see every 5th day
        s_min_all = df_asset["close"].rolling(rolling_freq).min()
        s_max_all = df_asset["close"].rolling(rolling_freq).max()

        df_asset = df_asset[["close"]]

        for i in range(0, rs):
            df_asset[f"rs{i}"] = 0

        for row in range(0, len(df_asset)):
            start_date = df_asset.index[row]
            end_date = df_asset.index[row + window]
            df_partcial = df_asset.loc[start_date: end_date]
            print(f"resistance {ts_code} {start_date} to {end_date}")

            s_minmax = s_min_all.loc[start_date:end_date].append(s_max_all.loc[start_date:end_date])
            support_resistance_once(df=df_partcial, rs_needed=rs, s_minmax=s_minmax)

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
    # create_gif()
