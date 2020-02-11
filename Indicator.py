import tushare as ts
import pandas as pd
import numpy as np
import time as mytime
import math
import talib
import time
import DB
import os
from itertools import combinations
from itertools import permutations
import Util
from time import sleep
from progress.bar import PixelBar
from datetime import datetime
import traceback
import Backtest_Util
import copy
import cProfile
from tqdm import tqdm
import operator


def c_all_indicators():
    return ["period", "open", "high", "low", "close", "pct_chg", "past_gain", "pjump_up", "pjump_down", "ivola", "vol", "turnover_rate", "pb", "ps_ttm", "dv_ttm", "total_share", "total_mv", "n_cashflow_act", "n_cashflow_inv_act", "n_cash_flows_fnc_act", "profit_dedt", "netprofit_yoy", "or_yoy",
            "grossprofit_margin", "netprofit_margin", "debt_to_assets", "pledge_ratio", "candle_net_pos", "trend"]


#
def indicator_once(df, a_indicators=[]):
    pass


def indicator_multiple():
    dict_indicators = {
        "pct_chg": {
            "ts_code_a": pd.DataFrame(),
            "ts_code_20": pd.DataFrame(),
            "ts_code_50": pd.DataFrame(),
            "ts_code_80": pd.DataFrame(),
            "ts_code_100": pd.DataFrame(),
            "date_a": pd.DataFrame(),
            "date_20": pd.DataFrame(),
            "date_50": pd.DataFrame(),
            "date_80": pd.DataFrame(),
            "date_100": pd.DataFrame(),
        }
    }

    # check if saved tab exists

    rolling_freqs_big = [2, 5, 10, 20, 60, 240]
    rolling_freqs_small = [2, 5, 20]

    dict_year = {}
    for start_year, counter in zip(range(1999, 2021), range(100)):
        end_year = (start_year + 1) * 10000
        start_year = start_year * 10000
        dict_year[counter] = (start_year, end_year)

    df_ts_codes = DB.get_ts_code(asset="E")
    df_trade_dates = DB.get_trade_date(start_date="20000000", end_date="20200101", freq="D")

    dict_df_ts_code = DB.preload("asset")
    dict_df_date = DB.preload("trade_date")

    df_stock_market = DB.get_stock_market_all()
    df_stock_market = df_stock_market[["trade_date", "trend2", "trend5", "trend10", "trend20", "trend60", "trend240"]]

    # ts_code
    for indicator_label, dict_dfs in dict_indicators.items():

        if os.path.isfile(f"Market/CN/Indicator/result/{indicator_label}.xlsx"):
            print(f"{indicator_label} already exist")
            continue

        # asset view
        for ts_code in df_ts_codes["ts_code"]:
            print("ts_code,", ts_code)
            df_saved_asset = dict_df_ts_code[ts_code]

            if df_saved_asset.empty:
                continue

            period = len(df_saved_asset)
            if period <= 240:
                continue

            df_saved_asset = df_saved_asset[df_saved_asset["period"] > 240]

            for part in ["a", "20", "50", "80", "100"]:
                if part == "a":  # todo do quantile instead of only two ends
                    a_perc = np.nanpercentile(a=df_saved_asset[indicator_label], q=[0, 100])
                elif part == "20":
                    a_perc = np.nanpercentile(a=df_saved_asset[indicator_label], q=[0, 20])
                elif part == "50":
                    a_perc = np.nanpercentile(a=df_saved_asset[indicator_label], q=[20, 50])
                elif part == "80":
                    a_perc = np.nanpercentile(a=df_saved_asset[indicator_label], q=[50, 80])
                elif part == "100":
                    a_perc = np.nanpercentile(a=df_saved_asset[indicator_label], q=[8, 100])
                else:
                    a_perc = [0, 100]
                df_filtered = df_saved_asset[df_saved_asset[indicator_label].between(a_perc[0], a_perc[1])]

                df_result = dict_dfs[f"ts_code_{part}"]
                df_result.at[ts_code, f"period"] = period

                for trend in rolling_freqs_big:
                    for trend_op_label, trend_op in Util.c_ops().items():
                        df_filtered_trend = df_filtered[trend_op(df_filtered[f"trend{trend}"], 0.5)]

                        for rolling_freq in rolling_freqs_small:
                            # mean
                            mean = df_filtered_trend[f"future_gain{rolling_freq}"].mean()
                            df_result.at[ts_code, f"mean_future_gain{rolling_freq}_trend{trend}{trend_op_label}"] = mean

                        for rolling_freq in rolling_freqs_small:
                            # std
                            std = df_filtered_trend[f"future_gain{rolling_freq}"].std()
                            df_result.at[ts_code, f"std_future_gain{rolling_freq}_trend{trend}{trend_op_label}"] = std

                        for rolling_freq in rolling_freqs_small:
                            # pearson
                            correlation = df_filtered_trend[indicator_label].corr(other=df_filtered_trend[f"future_gain{rolling_freq}"], method="pearson")
                            df_result.at[ts_code, f"pearson_future_gain{rolling_freq}_trend{trend}{trend_op_label}"] = correlation

        # date view
        for trade_date in df_trade_dates["trade_date"]:
            df_saved_date = dict_df_date[trade_date]
            df_saved_date = df_saved_date[df_saved_date["period"] > 240]

            df_result = dict_dfs[f"ts_code_a"]

            for trend in rolling_freqs_big:
                for trend_op_label, trend_op in Util.c_ops().items():
                    df_filtered_trend = df_saved_date[trend_op(df_saved_date[f"trend{trend}"], 0.5)]

                    for rolling_freq in rolling_freqs_small:
                        # mean
                        mean = df_filtered_trend[f"future_gain{rolling_freq}"].mean()
                        df_result.at[trade_date, f"mean_future_gain{rolling_freq}_trend{trend}{trend_op_label}"] = mean

                    for rolling_freq in rolling_freqs_small:
                        # std
                        std = df_filtered_trend[f"future_gain{rolling_freq}"].std()
                        df_result.at[trade_date, f"std_future_gain{rolling_freq}_trend{trend}{trend_op_label}"] = std

                    for rolling_freq in rolling_freqs_small:
                        # pearson
                        correlation = df_filtered_trend[indicator_label].corr(other=df_filtered_trend[f"future_gain{rolling_freq}"], method="pearson")
                        df_result.at[trade_date, f"pearson_future_gain{rolling_freq}_trend{trend}{trend_op_label}"] = correlation

            dict_dfs[f"ts_code_l"] = df_result.nlargest(n=20, columns=indicator_label)
            dict_dfs[f"ts_code_s"] = df_result.nsmallest(n=20, columns=indicator_label)

        for indicator_label, dict_indicators in dict_indicators.items():
            df_source = dict_indicators["date"]
            df_source["trade_date"] = df_source.index
            print("df_source", df_source)
            df_result = dict_indicators["date_return"]
            for year_counter, a_date in dict_year.items():
                for column in [x for x in df_source.columns if x not in ["trade_date"]]:
                    df_result.at[a_date[1], column] = df_source.loc[df_source["trade_date"].between(int(a_date[0]), int(a_date[1])), column].mean()

            Util.columns_remove(df_source, ["trade_date"])

        for indicator_label, dict_dfs in dict_indicators.items():
            path = f"Market/CN/Indicator/result/{indicator_label}.xlsx"
            writer = pd.ExcelWriter(path, engine='xlsxwriter')
            for sheet_label, df in dict_dfs.items():
                print(f"saving{indicator_label, sheet_label}...")
                df.to_excel(writer, sheet_name=sheet_label, index=True)
            Util.pd_writer_save(pd_writer=writer, path=path)


if __name__ == '__main__':
    try:
        indicator_multiple()
        pass
    except Exception as e:
        traceback.print_exc()
        Util.sound("error.mp3")
