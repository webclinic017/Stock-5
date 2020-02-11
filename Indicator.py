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
            "a0_10": pd.DataFrame(),
            "a0_2": pd.DataFrame(),
            "a2_5": pd.DataFrame(),
            "a5_8": pd.DataFrame(),
            "a8_10": pd.DataFrame(),
            "d0_10": pd.DataFrame(),
            "d0_2": pd.DataFrame(),
            "d2_5": pd.DataFrame(),
            "d5_8": pd.DataFrame(),
            "d8_10": pd.DataFrame(),
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

    step = 1
    df_ts_codes = DB.get_ts_code(asset="E")[::step]
    df_trade_dates = DB.get_trade_date(start_date="20000000", end_date="20200101", freq="D")[::step]

    dict_df_ts_code = DB.preload("asset", step=step)
    dict_df_date = DB.preload("trade_date", step=step)

    df_stock_market = DB.get_stock_market_all()
    df_stock_market = df_stock_market[["trend2", "trend5", "trend10", "trend20", "trend60", "trend240"]]

    # ts_code
    for indicator_label, dict_dfs in dict_indicators.items():

        # check if file exist. Each file is atomic. Can only exist or not exist. But not updated
        if os.path.isfile(f"Market/CN/Indicator/result/{indicator_label}.xlsx"):
            print(f"{indicator_label} already exist")
            continue

        # asset view
        for ts_code in df_ts_codes["ts_code"]:
            print(f"ts_code {ts_code}")
            df_saved_asset = dict_df_ts_code[ts_code]

            if df_saved_asset.empty:
                continue

            period = len(df_saved_asset)
            if period <= 240:
                continue

            df_saved_asset = df_saved_asset[df_saved_asset["period"] > 240]
            indicator_asset(indicator_label, ts_code, df_saved_asset, period, dict_dfs, rolling_freqs_big, rolling_freqs_small, "a")

        # date view
        for trade_date in df_trade_dates["trade_date"]:
            print(f"trade_date {trade_date}")
            df_saved_date = dict_df_date[trade_date]
            df_saved_date = df_saved_date[df_saved_date["period"] > 240]
            indicator_date(indicator_label, trade_date, df_saved_date, period, dict_dfs, rolling_freqs_big, rolling_freqs_small, "d")

        # for indicator_label, dict_indicators in dict_indicators.items():
        #     df_source = dict_indicators["date"]
        #     df_source["trade_date"] = df_source.index
        #     print("df_source", df_source)
        #     df_result = dict_indicators["date_return"]
        #     for year_counter, a_date in dict_year.items():
        #         for column in [x for x in df_source.columns if x not in ["trade_date"]]:
        #             df_result.at[a_date[1], column] = df_source.loc[df_source["trade_date"].between(int(a_date[0]), int(a_date[1])), column].mean()
        #
        #     Util.columns_remove(df_source, ["trade_date"])

        for indicator_label, dict_dfs in dict_indicators.items():
            path = f"Market/CN/Indicator/result/{indicator_label}.xlsx"
            writer = pd.ExcelWriter(path, engine='xlsxwriter')
            for sheet_label, df in dict_dfs.items():
                print(f"saving{indicator_label, sheet_label}...")
                df.to_excel(writer, sheet_name=sheet_label, index=True)
            Util.pd_writer_save(pd_writer=writer, path=path)


def indicator_asset(indicator_label, ts_code, df_saved_asset, period, dict_dfs, rolling_freqs_big, rolling_freqs_small, keyword):
    indicator_common(indicator_label=indicator_label, index=ts_code, df_saved=df_saved_asset, period=period, dict_result_dfs=dict_dfs, rolling_freqs_big=rolling_freqs_big, rolling_freqs_small=rolling_freqs_small, keyword=keyword)


def indicator_date(indicator_label, trade_date, df_saved_date, period, dict_dfs, rolling_freqs_big, rolling_freqs_small, keyword):
    indicator_common(indicator_label=indicator_label, index=trade_date, df_saved=df_saved_date, period=period, dict_result_dfs=dict_dfs, rolling_freqs_big=rolling_freqs_big, rolling_freqs_small=rolling_freqs_small, keyword=keyword)


def indicator_common(indicator_label, index, df_saved, period, dict_result_dfs, rolling_freqs_big, rolling_freqs_small, keyword):
    for perc1, perc2 in [(0, 100), (0, 20), (20, 50), (50, 80), (80, 100)]:
        a_perc = np.nanpercentile(a=df_saved[indicator_label], q=[perc1, perc2])
        df_filtered = df_saved[df_saved[indicator_label].between(a_perc[0], a_perc[1])]

        df_result = dict_result_dfs[f"{keyword}{int(perc1 / 10)}_{int(perc2 / 10)}"]
        if keyword == "a":
            df_result.at[index, f"period"] = period

        for trend in rolling_freqs_big:
            for trend_op_label, trend_op in Util.c_ops().items():
                df_filtered_trend = df_filtered[trend_op(df_filtered[f"trend{trend}"], 0.5)]

                for rolling_freq in rolling_freqs_small:
                    # mean
                    mean = df_filtered_trend[f"fgain{rolling_freq}"].mean()
                    df_result.at[index, f"mean_fgain{rolling_freq}_trend{trend}{trend_op_label}"] = mean

                for rolling_freq in rolling_freqs_small:
                    # std
                    std = df_filtered_trend[f"fgain{rolling_freq}"].std()
                    df_result.at[index, f"std_fgain{rolling_freq}_trend{trend}{trend_op_label}"] = std

                for rolling_freq in rolling_freqs_small:
                    # pearson
                    correlation = df_filtered_trend[indicator_label].corr(other=df_filtered_trend[f"fgain{rolling_freq}"], method="pearson")
                    df_result.at[index, f"pearson_fgain{rolling_freq}_trend{trend}{trend_op_label}"] = correlation



if __name__ == '__main__':
    try:
        indicator_multiple()
        pass
    except Exception as e:
        traceback.print_exc()
        Util.sound("error.mp3")
