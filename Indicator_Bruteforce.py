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
import copy
import imageio
import glob
import itertools
from multiprocessing import Process

pd.options.mode.chained_assignment = None  # default='warn'


# NOTE DIFFERENCE BETWEEN BACKTEST AND EVAL: backtest can choose on date basis, while eval only chooses on asset basis
# backtest: top 10 of today. Eval: if condition happens, then fgain


def to_mystring(my_dict):
    string_result = ""
    for key, value in my_dict.items():
        string_result = string_result + f"{key}{value}_"
    return string_result


def all_indicator_variables(indicator="rs"):
    dict_all_indicators_variables = {
        "rs": {
            "column": ["rs_und", "rs_abv"],
            "step": [10, 20],
            "start_window": [240],
            "rolling_freq": [1],
            "thresh": [[4, 0.2]],
            "rs_count": [8, 4],
            "bins": [8],
            "delay": [1, 3, 5, 10, 20],
        },

        "trend2": {
            "column": ["trend2"],
        },

        "trend": {
            "column": ["trend"],
        },

    }
    return dict_all_indicators_variables[indicator]


# inout a dict with all variables. Output a list of all possible combinations
def all_indicator_settings(dict_one_indicator_variables):
    # 1. only get values form above dict
    # 2. create cartesian product of the list
    # 3. create dict out of list
    a_product_result = []
    for one_combination in itertools.product(*dict_one_indicator_variables.values()):
        dict_result = dict(zip(dict_one_indicator_variables.keys(), one_combination))
        a_product_result.append(dict_result)

    return a_product_result


def bruteforce_summary():
    pass


def eval_fgain_mean(df, ts_code, column, dict_fgain_mean_detail):
    dict_ts_code_mean = {}
    for fgain, df_fgain_mean in dict_fgain_mean_detail.items():
        # general ts_code pgain
        df_fgain_mean.at[ts_code, fgain] = dict_ts_code_mean[fgain] = df[fgain].mean()
        # general ts_code pearson with fgain
        df_fgain_mean.at[ts_code, f"{fgain}_pearson"] = df[column].corr(df[fgain], method="pearson")

    # evaluate after percentile
    p_setting = [(0, 0.18), (0.18, 0.5), (0.5, 0.82), (0.82, 1)]
    for low_quant, high_quant in p_setting:
        low_val, high_val = list(df[column].quantile([low_quant, high_quant]))
        df_percentile = df[df[column].between(low_val, high_val)]
        for fgain, df_fgain_mean in dict_fgain_mean_detail.items():
            df_fgain_mean.at[ts_code, f"{fgain}_p{low_quant, high_quant}"] = df_percentile[fgain].mean() / dict_ts_code_mean[fgain]

    # evaluate after occurence bins
    try:
        o_setting = 4
        s_occurence = df[column].value_counts(bins=o_setting)
        for (index, value), counter in zip(s_occurence.iteritems(), range(0, o_setting)):
            df_occ = df[df[column].between(index.left, index.right)]
            for fgain, df_fgain_mean in dict_fgain_mean_detail.items():
                df_fgain_mean.at[ts_code, f"{fgain}_o{counter}"] = df_occ[fgain].mean() / dict_ts_code_mean[fgain]
    except Exception as e:
        print("Occurence did not work")

    # evaluate after probability/ occurence TODO

    # evaluate after seasonality
    for trend_freq in Util.c_rolling_freqs():
        for trend in [1, 0]:
            for fgain, df_fgain_mean in dict_fgain_mean_detail.items():
                df_fgain_mean.at[ts_code, f"{fgain}_trend{trend_freq}{trend}"] = df.loc[df[f"trend{trend_freq}"] == trend, fgain].mean() / dict_ts_code_mean[fgain]


# 1. iterate through all variable
# 2. find fgain mean
def bruteforce():
    setting = {
        "target": "asset",  # date
        "step": 37,  # 1 or any other integer
        "onthefly": False,  # if the indicator needs to be created during eval or is already there
        "indicator": "trend",
        "group_result": False,
        "path_general": "Market/CN/Bruteforce/result/",
        "path_result": "Market/CN/Bruteforce/",
        "big_update": False
    }

    dict_df_asset = DB.preload(load=setting["target"], step=setting["step"])
    dict_df_summary = {f"fgain{freq}": pd.DataFrame() for freq in Util.c_rolling_freqs()}
    numeric_col = Util.get_numeric_df_columns(DB.get_asset())

    for column, counter in zip(numeric_col, range(len(numeric_col))):
        # for one_indicator_setting in all_indicator_settings(all_indicator_variables(indicator=setting["indicator"])):

        # if small update, then continue if file exists
        if not setting["big_update"]:
            for key in Util.c_rolling_freqs():
                path = setting["path_general"] + f"{column}_fgain{key}_mean.xlsx"
                if not os.path.exists(path):
                    print("=" * 30)
                    print("SMALL UPDATE: File NOT EXIST. DO. -> ", f"({counter}/{len(numeric_col)}) {column}")
                    print("=" * 30)
                    break  # go into the column
            else:
                print("=" * 30)
                print("SMALL UPDATE: File Exists. Skip. -> ", f"({counter}/{len(numeric_col)}) {column}")
                print("=" * 30)
                continue  # go to next column
        else:
            print("=" * 30)
            print(f"BIG UPDATE: {column} ({counter}/{len(numeric_col)})")
            print("=" * 30)

        # ts_code and fgain
        dict_fgain_mean_detail = {f"fgain{freq}": pd.DataFrame() for freq in Util.c_rolling_freqs()}
        for ts_code, df_asset in dict_df_asset.items():
            print("ts_code", ts_code, column)
            try:
                df_asset = df_asset[df_asset.index >= 20050101 & (df_asset["period"] > 240)]
            except Exception as e:
                print(f"{ts_code} too young")
                continue

            # if the indicator needs to be created on the fly
            if setting["onthefly"]:
                pass

            # evaluate function
            eval_fgain_mean(df=df_asset, ts_code=ts_code, column=column, dict_fgain_mean_detail=dict_fgain_mean_detail)

        # save
        for key, df in dict_fgain_mean_detail.items():
            # save column result
            df.index.name = "ts_code"
            # path = setting["path_general"] + setting["indicator"] + f"/{to_mystring(one_indicator_setting)}_{key}_mean.xlsx"
            path = setting["path_general"] + f"{column}_{key}_mean.xlsx"
            DB.ts_code_series_to_excel(df_ts_code=df, path=path, sort=[key, False], asset="E", group_result=setting["group_result"])

            # save Summary
            row_summary = df.mean()
            row_summary["bruteforce"] = f"{column}_{key}_mean"
            dict_df_summary[key] = dict_df_summary[key].append(row_summary, sort=False, ignore_index=True)
            Util.to_csv_feather(df=dict_df_summary[key], skip_feather=True, a_path=Util.a_path(setting["path_result"] + f"{key}_summary"))


if __name__ == '__main__':
    bruteforce()
