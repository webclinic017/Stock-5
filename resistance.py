import tushare as ts
import pandas as pd
import time
import math
import timeit
from datetime import timedelta
import os.path
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import Util
import DB

from collections import Counter
import glob
import os
import sys
from collections import Counter, defaultdict
from itertools import groupby
from operator import itemgetter
from timeit import timeit

import sys
from itertools import groupby
from operator import itemgetter
from timeit import timeit

pro = ts.pro_api('c473f86ae2f5703f58eecf9864fa9ec91d67edbc01e3294f6a4f9c32')
ts.set_token("c473f86ae2f5703f58eecf9864fa9ec91d67edbc01e3294f6a4f9c32")


def most_common_2(lst, firstn):
    if not lst:
        return float("NaN")
    result = Counter(lst).most_common(firstn)
    return result


def support_resistance(with_time_rs, df):
    # NEED to be reversed and reindexed before continue
    rolling_period = 240
    first_n_resistance = 6
    spread_threashold_1 = 0.8
    spread_threashold_2 = 1.2

    list_min = (np.array(df.close.rolling(rolling_period).min()))
    list_max = (np.array(df.close.rolling(rolling_period).max()))
    list_min_max = np.concatenate([list_min, list_max])

    min_max_u, min_max_count = np.unique(list_min_max, return_index=False, return_inverse=False, return_counts=True)
    min_max_sort = np.argsort(-min_max_count)
    sorted_min_max_u = min_max_u[min_max_sort]
    # sorted_count=count[count_sort_ind]

    list_of_rs = []
    for i in range(0, sorted_min_max_u.size):
        possible_rs = sorted_min_max_u[i]
        if (possible_rs == float("nan") or np.isnan(possible_rs)):
            continue
        too_close = False
        for existing_rs in list_of_rs:
            if (spread_threashold_1 <= (existing_rs / possible_rs) <= spread_threashold_2):
                too_close = True
                break

        if (not too_close):
            list_of_rs.append(possible_rs)
            if (with_time_rs):
                first_date = np.where(list_min == possible_rs)
                if (first_date[0].size == 0):
                    first_date = np.where(list_max == possible_rs)
                df.loc[first_date[0][0]:len(df.index), "rs" + str(len(list_of_rs) - 1)] = possible_rs
            else:
                df["sr" + str(len(list_of_rs) - 1)] = possible_rs

        if (len(list_of_rs) > first_n_resistance):
            break

    # df["min"]=df.close.rolling(rolling_period).min()
    # df["max"]=df.close.rolling(rolling_period).max()
    return df


def comp_support_resistance(with_time_rs, df, list_min_max):
    # NEED to be reversed and reindexed before continue
    rolling_period = 240
    first_n_resistance = 6
    spread_threashold_1 = 0.8
    spread_threashold_2 = 1.2

    list_min = (np.array(df.close[len(df.index)].rolling(rolling_period).min()))
    list_max = (np.array(df.close[len(df.index)].rolling(rolling_period).max()))
    list_min_max = list_min_max.append(list_min)
    list_min_max = list_min_max.append(list_max)

    min_max_u, min_max_count = np.unique(list_min_max, return_index=False, return_inverse=False, return_counts=True)
    min_max_sort = np.argsort(-min_max_count)
    sorted_min_max_u = min_max_u[min_max_sort]
    # sorted_count=count[count_sort_ind]

    list_of_rs = []
    for i in range(0, sorted_min_max_u.size):
        possible_rs = sorted_min_max_u[i]
        if (possible_rs == float("nan") or np.isnan(possible_rs)):
            continue
        too_close = False
        for existing_rs in list_of_rs:
            if (spread_threashold_1 <= (existing_rs / possible_rs) <= spread_threashold_2):
                too_close = True
                break

        if (not too_close):
            list_of_rs.append(possible_rs)
            if (with_time_rs):
                first_date = np.where(list_min == possible_rs)
                if (first_date[0].size == 0):
                    first_date = np.where(list_max == possible_rs)
                df.loc[first_date[0][0]:len(df.index), "rs" + str(len(list_of_rs) - 1)] = possible_rs
            else:
                df["sr" + str(len(list_of_rs) - 1)] = possible_rs

        if (len(list_of_rs) > first_n_resistance):
            break

    # df["min"]=df.close.rolling(rolling_period).min()
    # df["max"]=df.close.rolling(rolling_period).max()
    return [df, list_min_max]


# df = pro.daily(ts_code='000002.SZ', start_date='20100701', end_date='20180718')
# df = pd.read_csv("data.csv")
# df=Library_Main.df_reverse_reindex(df)
# Library_Main.columns_remove(df,["open","high","low","pre_close","pct_chg","change","amount","vol"])

all_ts_code = DB.get_ts_code()

for ts_code in all_ts_code.ts_code:
    df = DB.get_asset(ts_code=ts_code)
    Util.columns_remove(df, ["open", "high", "low", "pre_close", "pct_chg", "change", "amount", "vol"])
    for i in range(260, 262):

        print(ts_code, "day", i)
        egal = df.loc[0:i]
        date = egal.trade_date.at[i]
        ts_code_id = egal.ts_code.at[0]
        egal[ts_code_id] = egal["trade_date"]
        egal.index = egal[ts_code_id].astype("str")
        Util.columns_remove(egal, ["trade_date", "ts_code", ts_code_id])
        df_csv = support_resistance(False, egal)

        df_csv.plot.line(legend=False)
        plt.setp(plt.gca().get_xticklabels(), visible=False)

        newpath = "plot/stock/" + str(ts_code) + "/"
        if not os.path.exists(newpath):
            os.makedirs(newpath)
        plt.savefig(newpath + str(date) + ".jpg")
        plt.clf()
        # plt.show()

        print()

    df_csv.to_csv("resistance.csv", index=False)

# TODO use rolling.groupby("trade_date") to iterate over a long df and test if it is faster than loc
if __name__ == '__main__':
    print("lol")
    pass
