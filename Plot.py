import tushare as ts
import pandas as pd
import time
import os.path
import numpy as np
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import DB
import os
import datetime
import imageio
import glob
import scipy.fftpack
import LB
from scipy.signal import find_peaks
from pandas.plotting import autocorrelation_plot
import Alpha
from multiprocessing import Process
pd.options.mode.chained_assignment = None  # default='warn'


def create_gif(ts_code="000002.SZ"):
    images = []
    for jpgfile in glob.iglob(os.path.join(f"Media/Plot/stock/{ts_code}", "*.jpg")):
        images.append(imageio.imread(jpgfile))
        print(f"{ts_code} load image", jpgfile)
    output_file = f"Media/Plot/stock/{ts_code}_{datetime.datetime.now().strftime('%Y_%M_%d_%H_%M_%S')}.gif"
    print("Plotting...please wait...")
    imageio.mimsave(output_file, images, duration=0.005)

def plot_support_resistance(window=1000, rolling_freq=20, ts_code="000002.SZ", step=5):
    def support_resistance_acc(df, freq, max_rs, s_minmax, adj_start_date, end_date, df_asset):
        s_occurence_bins = s_minmax.value_counts(bins=100)
        for counter, (index, value) in enumerate(s_occurence_bins.iteritems()):
            df.loc[adj_start_date:end_date, f"rs{freq}_{counter}"] = index.left
            df[f"rs{freq}_{counter}"].replace(0, np.nan, inplace=True)

    # 1 to 6 means 5 resistance freq with each 2 pieces
    d_rs = {int(round(window / (2 ** i))): 4 for i in range(0, 6)}

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

        for freq, max_rs in d_rs.items():
            adj_start = df_asset.index[row + window - freq]
            s_minmax = (s_minall.loc[adj_start:end_date]).append(s_maxall.loc[adj_start:end_date])
            support_resistance_acc(df=df_partcial, freq=freq, max_rs=max_rs, s_minmax=s_minmax, adj_start_date=adj_start, end_date=end_date, df_asset=df_asset)

        # plot graph and save it
        df_partcial.reset_index(inplace=True, drop=True)
        df_partcial.plot(legend=False)

        newpath = f"Media/Plot/stock/{ts_code}/"
        if not os.path.exists(newpath):
            os.makedirs(newpath)
        plt.savefig(f"{newpath}{start_date}_{end_date}.jpg")
        # plt.show()
        # df_partcial.to_csv(f"resistance{row}.csv", index=False)
        plt.close()


def plot_fft(ts_code="000002.SZ"):
    df = DB.get_asset(ts_code=ts_code)

    # Number of samplepoints
    N = 2000
    # sample spacing
    T = 1.0 / 800.0
    x = np.linspace(0.0, N * T, N)
    y = np.sin(50.0 * 2.0 * np.pi * x) + 0.5 * np.sin(80.0 * 2.0 * np.pi * x)
    y = df["close"].to_numpy()
    yf = scipy.fftpack.fft(y)
    xf = np.linspace(0.0, 1.0 / (2.0 * T), int(N / 2))

    fig, ax = plt.subplots()
    ax.plot(xf, 2.0 / N * np.abs(yf[:N // 2]))
    plt.show()

def plot_polynomials(df):
    df = df.copy().reset_index()
    df["poly1"] = Alpha.poly_fit(df=df, abase="close",degree=1,inplace=False)
    df["poly2"] = Alpha.poly_fit(df=df, abase="close",degree=2,inplace=False)
    df["poly3"] = Alpha.poly_fit(df=df, abase="close",degree=3,inplace=False)
    df["poly4"] = Alpha.poly_fit(df=df, abase="close",degree=4,inplace=False)
    df["poly5"] = Alpha.poly_fit(df=df, abase="close",degree=5,inplace=False)
    df = df[["close", "poly1", "poly2", "poly3", "poly4", "poly5"]]
    df.plot(legend=True)
    plt.show()
    plt.close()


def plot_histo(series):
    plt.hist(series)  # use this to draw histogram of your data
    plt.show()

def plot_distribution(df, abase="close",rfreq=10):
    df["norm"] = df[abase].rolling(rfreq).apply(Alpha.normalize_apply, raw=False)
    plot_histo(df["norm"])

def plot_autocorrelation(series):
    autocorrelation_plot(series.dropna())
    plt.show()

def plot_chart(df, columns):
    df_copy = df[columns].copy().reset_index(drop=True)
    df_copy.plot(legend=True)
    plt.show()

def plot_peaks(df, abase, distance=120, height=""):
    y=df[abase]
    peaks,_=find_peaks(df[abase],distance=distance)
    plt.plot(df[abase].index,df[abase])
    plt.plot(peaks,y[peaks],"x")
    plt.show()

if __name__ == '__main__':
    # support_resistance_multiple()
    pass
    matplotlib.use("TkAgg")

else:
    pass
    matplotlib.use("TkAgg")


