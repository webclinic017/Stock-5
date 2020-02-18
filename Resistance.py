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
        for (index, value), counter in zip(s_occurence_bins.iteritems(), range(0, max_rs)):
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


def support_resistance_once_calc(start_window=1000, rolling_freq=5, ts_code="000002.SZ", step=1, thresh=[2, 0.5], bins=100, dict_rs={"abv": 2, "und": 2}):
    def support_resistance_acc(abv_und, max_rs, s_minmax, adj_start_date, end_date, f_end_date, df_asset):
        # 1. step calculate all relevant resistance = relevant earlier price close to current price
        current_price = df_asset.at[end_date, "close"]

        if abv_und == "abv":
            s_minmax = s_minmax[(s_minmax / current_price < thresh[0]) & ((s_minmax / current_price > 1))]
        elif abv_und == "und":
            s_minmax = s_minmax[(s_minmax / current_price < 1) & ((s_minmax / current_price > thresh[1]))]

        # find the max occurence of n values
        try:
            s_occurence_bins = s_minmax.value_counts(bins=bins)
            a_rs = []
            for (index, value), counter in zip(s_occurence_bins.iteritems(), range(0, max_rs)):
                a_rs.append(index.left)

            # sort the max occurence values and assign them as rs
            a_rs.sort()  # small value first
            for item, i in zip(a_rs, range(0, len(a_rs))):
                df_asset.loc[end_date:f_end_date, f"rs{abv_und}{i}"] = item
        except:
            pass

    # calculate all min max for acceleration used for later simulation
    df_asset = DB.get_asset(ts_code=ts_code)
    s_minall = df_asset["close"].rolling(rolling_freq).min()
    s_maxall = df_asset["close"].rolling(rolling_freq).max()

    # only consider close and add rsi for plotting reason
    a_pgain = []
    a_fgain = []
    for i in Util.c_rolling_freqs():
        a_fgain.append(f"fgain{i}")
        a_pgain.append(f"pgain{i}")
    df_asset = df_asset[["close", "pct_chg"] + a_pgain + a_fgain]

    # iterate over past data as window
    for row in range(0, len(df_asset), step):
        if row + start_window > len(df_asset) - 1:
            break

        start_date = df_asset.index[0]
        end_date = df_asset.index[row + start_window]
        print(f"resistance {ts_code} {start_date} to {end_date}")
        for abv_und, max_rs in dict_rs.items():
            if row + start_window + step > len(df_asset) - 1:
                break

            f_end_date = df_asset.index[row + start_window + step]
            s_minmax = (s_minall.loc[start_date:end_date]).append(s_maxall.loc[start_date:end_date])
            support_resistance_acc(abv_und=abv_und, max_rs=max_rs, s_minmax=s_minmax, adj_start_date=start_date, end_date=end_date, f_end_date=f_end_date, df_asset=df_asset)

    for key, count in dict_rs.items():
        for i in range(0, count):
            try:
                df_asset[f"rs{key}{i}_abv"] = (df_asset["close"] < df_asset[f"rs{key}{i}"]).astype(int)
                df_asset[f"rs{key}{i}_cross"] = df_asset[f"rs{key}{i}_abv"].diff().replace(0, np.nan)
            except:
                pass

    return df_asset


def rs_evaluator(ts_code, df_evaluate, df_result, dict_rs):
    # Normalize:  devide the pct_chg gain by the stocks mean gain to calculate the relative performance against mean
    for fgain_freq in Util.c_rolling_freqs():
        freq_mean = df_evaluate[f"fgain{fgain_freq}"].mean()
        for key, count in dict_rs.items():
            for i in range(0, count):
                for cross in [1, -1]:
                    try:
                        df_result.at[ts_code, f"rs{key}{i}_cross{cross}_fgain{fgain_freq}"] = df_evaluate.loc[df_evaluate[f"rs{key}{i}_cross"] == cross, f"fgain{fgain_freq}"].mean() / freq_mean
                    except:
                        pass


if __name__ == '__main__':
    # create_gif(ts_code="000001.SZ")
    # support_resistance_multiple()

    df_result = pd.DataFrame()
    df_result_summary = pd.DataFrame()
    dict_asset = DB.preload(load="asset", step=37)

    for step in [20, 120]:  # performance : how many days should I refresh the future rs line
        for start_window in [1000]:  # how long is the starting window
            for rolling_freq in [1, 240]:  # how many past days should I use to calculate
                for bins in [20, 200]:  # performance:  how big is the distance between the lines themselves
                    for thresh in [[3, 0.33], [1.5, 0.66]]:  # how far is the spread from current price to the line
                        for rs_count in [2, 4]:  # how many lines for abv and und current price

                            dict_rs = {"abv": rs_count, "und": rs_count}
                            for ts_code, df_asset in dict_asset.items():
                                # ultimate RS search
                                df_evaluate = support_resistance_once_calc(start_window=start_window, rolling_freq=rolling_freq, ts_code=ts_code, step=step, thresh=thresh, bins=bins, dict_rs=dict_rs)
                                rs_evaluator(ts_code, df_evaluate, df_result, dict_rs)

                                # df_evaluate.to_csv(f"{ts_code}.csv")
                                df_result.to_csv(f"Market/CN/RS/summary_step{step}_window{start_window}_rolling_freq{rolling_freq}_bins{bins}_rs_count{rs_count}_thresh{thresh[0]}_{thresh[1]}.csv")

                                # evaluator summarizer
                                df_result_summary = df_result_summary.append(df_result.mean(), ignore_index=True, sort=False)
                                df_result_summary.to_csv(f"Market/CN/RS/summary.csv")
