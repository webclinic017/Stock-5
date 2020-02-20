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





def resistance_bruteforce():
    df_result_summary = pd.DataFrame()
    dict_asset = DB.preload(load="asset", step=10)

    for step in [10, 20]:  # performance : how many days should I refresh the future rs line. Ultimate step should be 1
        for start_window in [240]:  # how long is the starting window
            for rolling_freq in [1]:  # how many past days should I use to calculate
                for thresh in [[4, 0.2]]:  # how far is the spread from current price to the line. bigger spread better
                    for rs_count in [8, 4]:  # how many lines for abv and und current price. more is better
                        for bins in [8]:  # performance:  how big is the distance between the lines themselves. smaller bin better
                            for delay in [1, 3, 5, 10, 20]:
                                df_result = pd.DataFrame()
                                dict_rs = {"abv": rs_count, "und": rs_count}

                                for ts_code, df_asset in dict_asset.items():
                                    # ultimate RS search
                                    df_evaluate = DB.support_resistance_horizontal(start_window=start_window, rolling_freq=rolling_freq, step=step, thresh=thresh, bins=bins, dict_rs=dict_rs, df_asset=df_asset)
                                    rs_evaluator(ts_code, df_evaluate, df_result, dict_rs, df_asset)

                                    # df_evaluate.to_csv(f"{ts_code}.csv")
                                    df_result.to_csv(f"Market/CN/RS/DETAIL_step{step}_window{start_window}_rolling_freq{rolling_freq}_bins{bins}_rs_count{rs_count}_thresh{thresh[0]}_{thresh[1]}.csv")

                                # evaluator summarizer
                                df_result_summary[f"step{step}_window{start_window}_rolling_freq{rolling_freq}_bins{bins}_thresh{thresh[0], thresh[1]}_rscount{rs_count}"] = df_result.mean()
                                df_result_summary.to_csv(f"Market/CN/RS/SUM_step{step}_window{start_window}_rolling_freq{rolling_freq}_bins{bins}_rs_count{rs_count}_thresh{thresh[0]}_{thresh[1]}.csv")


if __name__ == '__main__':
    # create_gif(ts_code="000001.SZ")
    # support_resistance_multiple()
    df_result = pd.DataFrame()
    dict_asset = DB.preload(load="asset", step=1)

    for ts_code, df_asset in dict_asset.items():
        for fgain_freq in Util.c_rolling_freq():
            for abv_und in ["und", "abv"]:
                for i in [0, 1, 2, 3, 4, 5, 6, 7]:
                    for cross in [1, -1]:
                        try:
                            occurence = len(df_asset[(df_asset[f"rs{abv_und}{i}_cross"] == cross)]) / len(df_asset)
                            mean = df_asset.loc[df_asset[f"rs{abv_und}{i}_cross"] == cross, f"fgain{fgain_freq}"].mean()
                            df_result.at[ts_code, f"rs{abv_und}{i}_cross{cross}_occ"] = occurence
                            df_result.at[ts_code, f"rs{abv_und}{i}_cross{cross}_mean"] = mean
                        except:
                            pass

                    df_result.to_csv("occurence.csv")
