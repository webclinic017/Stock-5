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

import Indicator_Create
import LB
import operator
from time import sleep
from progress.bar import PixelBar
from datetime import datetime
import traceback
import Backtest_LB
import copy
import cProfile
from tqdm import tqdm
import operator


def increaselimit(increase=1):
    df_result_asset = pd.DataFrame()
    df_summary_asset = pd.DataFrame()
    df_ts_code = DB.get_ts_code()
    for ts_code in df_ts_code["ts_code"]:
        print("ts_code", ts_code)
        df = DB.get_asset(ts_code)
        try:
            df = df[(df["period"] > 240) & (df.index > int(19990101))]
            df.reset_index(inplace=True, drop=False)
        except:
            continue

        for days in [1, 2, 3, 5, 10, -1, -2, -3, -5, -10]:
            # find 涨停day and then pick the previous one day
            try:
                if increase == 1:
                    df_helper = df[df["pct_chg"] > 9]
                else:
                    df_helper = df[df["pct_chg"] < -9]
                df_gainlimit = df.loc[df_helper.index + days]
                df_gainlimit.dropna(inplace=True)

                for gain in ["pgain", "fgain"]:
                    for freq in LB.c_bfreq():
                        df_result_asset.at[ts_code, f"days{days}_{gain}{freq}"] = df_gainlimit[f"{gain}{freq}"].mean()
            except:
                for gain in ["pgain", "fgain"]:
                    for freq in LB.c_bfreq():
                        df_result_asset.at[ts_code, f"days{days}_{gain}{freq}"] = np.nan

    for gain in ["pgain", "fgain"]:
        for days in [1, 2, 3, 5, 10, -1, -2, -3, -5, -10]:
            for freq in LB.c_bfreq():
                df_summary_asset.at[f"days{days}", f"{gain}{freq}"] = df_result_asset[f"days{days}_{gain}{freq}"].mean()

    path = "Market/CN/Indicator/increaselimit.xlsx" if increase == 1 else "Market/CN/Indicator/decreaselimit.xlsx"
    LB.to_excel(path_excel=path, dict_df={"a": df_result_asset, "a_sum": df_summary_asset})


def overma():
    # asset
    df_result_asset = pd.DataFrame()
    df_summary_asset = pd.DataFrame()
    df_ts_code = DB.get_ts_code()
    for ts_code in df_ts_code["ts_code"]:
        print("ts_code", ts_code)
        df = DB.get_asset(ts_code)
        try:
            df = df[(df["period"] > 240) & (df.index > int(19990101))]
        except:
            continue

        for rolling_freq in LB.c_bfreq():
            Indicator_Create.mean(df=df, rolling_freq=rolling_freq, add_from="close")

        for lower in LB.c_bfreq():
            for upper in LB.c_bfreq():
                mean = df.loc[df[f"close{upper}"] > df[f"close{lower}"], "fgain2"].mean()
                df_result_asset.at[ts_code, f"lower{lower}_upper{upper}"] = mean

    # asset summary
    for lower in LB.c_bfreq():
        for upper in LB.c_bfreq():
            df_summary_asset.at[f"close{lower}", f"close{upper}_over"] = df_result_asset[f"lower{lower}_upper{upper}"].mean()

    # date summary
    df_result_date = pd.DataFrame()
    df_stock_market_all = DB.get_stock_market_all()

    for rolling_freq in LB.c_bfreq():
        Indicator_Create.mean(df=df_stock_market_all, rolling_freq=rolling_freq, add_from="close")

    for lower in LB.c_bfreq():
        for upper in LB.c_bfreq():
            mean = df_stock_market_all.loc[df_stock_market_all[f"close{upper}"] > df_stock_market_all[f"close{lower}"], "fgain2"].mean()
            df_result_date.at[f"close{lower}", f"closer{upper}_over"] = mean

    path = "Market/CN/Indicator/overma.xlsx"
    LB.to_excel(path_excel=path, dict_df={"a": df_result_asset, "a_sum": df_summary_asset, "d": df_stock_market_all, "d_sum": df_result_date})


def crossma():
    # asset
    df_result_asset = pd.DataFrame()
    df_summary_asset = pd.DataFrame()
    df_ts_code = DB.get_ts_code()[::1]
    for ts_code in df_ts_code["ts_code"]:
        print("ts_code", ts_code)
        df = DB.get_asset(ts_code)
        try:
            df = df[(df["period"] > 240) & (df.index > int(19990101))]
        except:
            continue

        # add ma
        for rolling_freq in LB.c_bfreq():
            Indicator_Create.mean(df=df, rolling_freq=rolling_freq, add_from="close")

        # add flag above ma and find cross over point
        for lower in LB.c_bfreq():
            for upper in LB.c_bfreq():
                if lower != upper:
                    df[f"upper{upper}_abv_lower{lower}"] = ((df[f"close{upper}"] > df[f"close{lower}"]).astype(int))
                    df[f"upper{upper}_cross_lower{lower}"] = (df[f"upper{upper}_abv_lower{lower}"].diff()).replace(0, np.nan)

        # calculate future pgain based on crossover
        for lower in LB.c_bfreq():
            for upper in LB.c_bfreq():
                if lower != upper:
                    for cross in [1, -1]:
                        mean = df.loc[df[f"upper{upper}_cross_lower{lower}"] == cross, "fgain2"].mean()
                        df_result_asset.at[ts_code, f"upper{upper}_cross{cross}_lower{lower}"] = mean

    # asset summary
    for cross in [1, -1]:
        for lower in LB.c_bfreq():
            for upper in LB.c_bfreq():
                try:
                    df_summary_asset.at[f"close{lower}", f"close{upper}_cross{cross}"] = df_result_asset[f"upper{upper}_cross{cross}_lower{lower}"].mean()
                except:
                    df_summary_asset.at[f"close{lower}", f"close{upper}_cross{cross}"] = np.nan

    # date
    df_result_date = pd.DataFrame()
    df_stock_market_all = DB.get_stock_market_all()
    for rolling_freq in LB.c_bfreq():
        Indicator_Create.mean(df=df_stock_market_all, rolling_freq=rolling_freq, add_from="close")

    # add flag above ma and find cross over point
    for lower in LB.c_bfreq():
        for upper in LB.c_bfreq():
            if lower != upper:
                df_stock_market_all[f"upper{upper}_abv_lower{lower}"] = ((df_stock_market_all[f"close{upper}"] > df_stock_market_all[f"close{lower}"]).astype(int))
                df_stock_market_all[f"upper{upper}_cross_lower{lower}"] = (df_stock_market_all[f"upper{upper}_abv_lower{lower}"].diff()).replace(0, np.nan)

    # date summary
    for cross in [1, -1]:
        for lower in LB.c_bfreq():
            for upper in LB.c_bfreq():
                try:
                    df_result_date.at[f"close{lower}", f"close{upper}_cross{cross}"] = df_stock_market_all.loc[df_stock_market_all[f"upper{upper}_cross_lower{lower}"] == cross, "fgain2"].mean()
                except:
                    df_result_date.at[f"close{lower}", f"close{upper}_cross{cross}"] = np.nan

    # date summary
    path = "Market/CN/Indicator/crossma.xlsx"
    LB.to_excel(path_excel=path, dict_df={"a": df_result_asset, "a_sum": df_summary_asset, "d": df_stock_market_all, "d_sum": df_result_date})



# auto correlation = relation between past price and future price
def auto_corr_multiple():
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

    # dict_df_ts_code = DB.preload("asset", step=step)
    # dict_df_date = DB.preload("trade_date", step=step)

    df_stock_market = DB.get_stock_market_all()

    df_pearson = pd.DataFrame()

    dict_df = {}
    for pct_chg in ["pgain2", "pgain5", "pgain10", "pgain20", "pgain60", "pgain240"]:
        dict_df[pct_chg] = pd.DataFrame()
    df_result_date = pd.DataFrame()

    a_lower = [x for x in [-1, -0.7, -0.5, -0.2, -0.1, -0.05, 0, 0.05, 0.1, 0.2, 0.5, 0.7]]
    a_upper = [x for x in [-0.7, -0.5, -0.2, -0.1, -0.05, 0, 0.05, 0.1, 0.2, 0.5, 0.7, 1, ]]
    # asset view
    for ts_code in df_ts_codes["ts_code"]:
        # prep
        # print(f"ts_code {ts_code}")
        # df_saved_asset = DB.get_asset(ts_code)
        # if df_saved_asset.empty:
        #     continue
        # period = len(df_saved_asset)
        # if period <= 240:
        #     continue
        # df_saved_asset = df_saved_asset[df_saved_asset["period"] > 240]

        # pearson
        for past_freq in [2, 5, 10, 20, 60, 240]:
            for future_freq in [2, 5, 10, 20, 60, 240]:
                try:
                    # pearson = df_saved_asset[f"fgain{future_freq}"].corr(df_saved_asset[f"pgain{past_freq}"])
                    # print(ts_code, "pearson ", past_freq, future_freq, pearson)
                    # df_pearson.at[ts_code, f"pgain{past_freq}_fgain{future_freq}"] = pearson
                    pass
                except:
                    pass

        # Distribution
        # for pgain in ["pgain2","pgain5","pgain10","pgain20","pgain60","pgain240"]:
        #     for pgain_lower, pgain_upper in zip(a_lower,a_upper):
        #         df_filtere = df_saved_asset[df_saved_asset[pgain].between(pgain_lower, pgain_upper)]
        #         total_counts = len(df_filtere)
        #         if total_counts==0:
        #             continue
        #         dict_df[pgain].at[ts_code, f"total_counts{pgain_lower, pgain_upper}"] = total_counts
        #         for fgain_lower, fgain_upper in zip(a_lower,a_upper):
        #             condition_count=len(df_filtere.loc[df_filtere["fgain2"].between(fgain_lower,fgain_upper)])
        #             dict_df[pgain].at[ts_code,f"{pgain}_{pgain_lower,pgain_upper}_fgain2_{fgain_lower,fgain_upper}"]=condition_count/total_counts

    for ma in LB.c_bfreq():
        Indicator_Create.mean(df_stock_market, rolling_freq=ma, add_from="pct_chg", complete_new_update=True)

    df_mean = pd.DataFrame()

    for pct_chg in ["pct_chg2"]:  # "pct_chg5", "pct_chg10", "pct_chg20", "pct_chg60", "pct_chg240"
        for pgain_lower, pgain_upper in zip(a_lower, a_upper):

            df_filtere = df_stock_market[df_stock_market[pct_chg].between(pgain_lower, pgain_upper)]

            total_counts = len(df_filtere)
            print(f"{pct_chg} filtere stock market between{pgain_lower, pgain_upper, total_counts}")
            if total_counts == 0:
                continue

            Expected_return = 0
            for fgain_lower, fgain_upper in zip(a_lower, a_upper):
                condition_count = len(df_filtere.loc[df_filtere["fgain2"].between(fgain_lower, fgain_upper)])
                prob = condition_count / total_counts
                df_result_date.at[pct_chg, f"{pgain_lower, pgain_upper}_fgain2_{fgain_lower, fgain_upper}"] = prob

                mean_field = (fgain_lower + fgain_upper) / 2
                Expected_return = Expected_return + mean_field * prob

            df_mean.at[f"{pct_chg}_{pgain_lower, pgain_upper}", "expected mean for fgain2"] = Expected_return

    # calculate mean Expected return

    path = "Market/CN/Indicator/auto_regression_date.xlsx"
    portfolio_writer = pd.ExcelWriter(path, engine='xlsxwriter')
    for key, df in dict_df.items():
        pass
        # df.to_excel(portfolio_writer, sheet_name=f"{key}_fgain2", index=True, encoding='utf-8_sig')
    df_result_date.to_excel(portfolio_writer, sheet_name="probability_dis_date", index=True, encoding='utf-8_sig')
    df_mean.to_excel(portfolio_writer, sheet_name="probability_dis_date_mean", index=True, encoding='utf-8_sig')

    for i in range(0, 10):
        try:
            portfolio_writer.save()
            return
        except Exception as e:
            LB.close_file(path)
            LB.sound("close_excel.mp3")
            print(e)
            time.sleep(10)


def cross_corr_multiple():
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
            writer.save()


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
            for trend_op_label, trend_op in LB.c_op().items():
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
        # cross_corr_multiple()
        # overma()
        # crossma()
        increaselimit(-1)
        # auto_corr_multiple()
        pass
    except Exception as e:
        traceback.print_exc()
        LB.sound("error.mp3")
