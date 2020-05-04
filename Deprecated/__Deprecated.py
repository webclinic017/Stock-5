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

import Alpha
import LB
import operator
from time import sleep
from progress.bar import PixelBar
from datetime import datetime
import traceback
import copy
import cProfile
from tqdm import tqdm
import operator
import matplotlib as plt

"""This .py contains deprecated or wrong or old functions that are no longer in use"""

def increaselimit(increase=1):
    """test 涨停，跌停"""
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
    LB.to_excel(path=path, d_df={"a": df_result_asset, "a_sum": df_summary_asset})




def auto_corr_multiple():
    """ auto correlation = relation between past price and future price
    TODO this should be replaced by a better quantile bin version
    """
    # check if saved tab exists

    rolling_freqs_big = [2, 5, 10, 20, 60, 240]
    rolling_freqs_small = [2, 5, 20]

    dict_year = {}
    for start_year, counter in zip(range(1999, 2021), range(100)):
        end_year = (start_year + 1) * 10000
        start_year = start_year * 10000
        dict_year[counter] = (start_year, end_year)

    step = 1
    df_ts_codes = DB.get_ts_code(a_asset=["E"])[::step]
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
        Alpha.mean(df_stock_market, rolling_freq=ma, add_from="pct_chg", complete_new_update=True)

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
    df_ts_codes = DB.get_ts_code(a_asset=["E"])[::step]
    df_trade_dates = DB.get_trade_date(start_date="20000000", end_date="20200101", freq="D")[::step]

    dict_df_ts_code = DB.preload("E", step=step)
    dict_df_date = DB.preload("E",on_asset=False, step=step)

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



def rsi_sim_no_bins_multiple():
    """
    proven to be useless: because past data can not 1:1 predict future data.
    the bins and no bins variation all conclude a inverse relationship. Maybe this is correct regarding law of big data"""

    def sim_no_bins_once(df_result, ts_code):
        # create target freq and target period
        for target in all_target:
            df_result[f"tomorrow{target}"] = df_result["open"].shift(-target) / df_result["open"].shift(-1)

        # pre calculate all rsi
        for column in all_column:
            for freq in all_freq:
                df_result[f"{column}.rsi{freq}"] = talib.RSI(df_result[column], timeperiod=freq)

        # go through each day and measure similarity
        for trade_date in df_result.index:
            if trade_date < start_date:
                continue

            # df_yesterday = df_result.loc[(df_result.index < trade_date)]
            trade_date_index_loc_minus_past = df_result.index.get_loc(trade_date) - 280
            date_lookback = df_result.index[trade_date_index_loc_minus_past]
            df_past_ref = df_result.loc[(df_result.index < date_lookback)]
            s_final_sim = pd.Series(data=1, index=df_past_ref.index)  # an array of days with the highest or lowest simlarity to today

            print(f"{ts_code} {trade_date}", f"reference until {date_lookback}")

            # check for each rsi column combination their absolute derivation
            for freq in all_freq:
                for column in all_column:
                    freq_today_value = df_result.at[trade_date, f"{column}.rsi{freq}"]

                    # IN yesterdays df you can see how similar one column.freq is to today
                    column_freq_sim = ((freq_today_value - df_past_ref[f"{column}.rsi{freq}"]).abs()) / 100
                    column_freq_sim = 1 + column_freq_sim
                    column_freq_sim = column_freq_sim.fillna(2)  # 1 being lowest, best. 2 or any other number higher 1 being highest, worst

                    column_freq_sim_weight = all_weight[f"{column}.{freq}"]
                    weighted_column_freq_sim = column_freq_sim ** column_freq_sim_weight
                    s_final_sim = s_final_sim.multiply(weighted_column_freq_sim, fill_value=1)

            # calculate a final similarity score (make the final score cross columns)
            # remove the last 240 items to prevent the algo know whats going on future
            nsmallest = s_final_sim.nsmallest(past_samples)
            # print(f"on trade_date {trade_date} the most similar days are: {list(nsmallest.index)}")
            df_similar = df_result.loc[nsmallest.index]
            df_result.at[trade_date, "similar"] = str(list(nsmallest.index))
            for target in all_target:
                df_result.at[trade_date, f"final.rsi.tomorrow{target}.score"] = df_similar[f"tomorrow{target}"].mean()

        LB.to_csv_feather(df_result, LB.a_path(f"sim_no_bins/result/similar_{ts_code}.{str(all_column)}"))
        return df_result

    # setting
    all_weight = {
        "close.2": 0.5,
        "close.5": 0.7,
        "close.10": 0.9,
        "close.20": 1,
        "close.40": 1.2,
        "close.60": 1.3,
        "close.120": 1.4,
        "close.240": 1.6,
        # "ivola.2": 0.0,
        # "ivola.5": 0.0,
        # "ivola.10": 0.3,
        # "ivola.20": 0.3,
        # "ivola.40": 0.3,
        # "ivola.60": 0.3,
        # "ivola.120": 0.3,
        # "ivola.240": 0.3,
    }

    all_column = ["close"]
    all_target = [2, 5, 10, 20, 40, 60, 120, 240]  # -60, -1 means future 60 days return to future 1 day, means in
    all_freq = [10, 20, 120, 240]  # 780, 20, 520， 2, 5,
    past_samples = 2
    start_date = 20050101

    df_summary = pd.DataFrame()
    df_ts_code = DB.get_ts_code()
    df_ts_code = df_ts_code[df_ts_code.index == "000001.SZ"]

    for ts_code in df_ts_code.index[::100]:
        df_result = DB.get_asset(ts_code=ts_code)
        df_result = df_result[df_result["period"] > 240]

        try:
            df_result = sim_no_bins_once(df_result, ts_code)
        except:
            continue

        for target in all_target:
            try:
                df_summary.at[ts_code, f"tomorrow{target}_pearson"] = df_result[f"tomorrow{target}"].corr(df_result[f"final.rsi.tomorrow{target}.score"])
            except:
                pass

    DB.to_excel_with_static_data(df_ts_code=df_summary, path=f"sim_no_bins/summary.{str(all_column)}.xlsx", sort=[], a_assets=["E"], group_result=True)


def rsi_sim_bins():
    """
    proven to be useless: because past data can not 1:1 predict future data.
    the bins and no bins variation all conclude a inverse relationship. Maybe this is correct regarding law of big data"""
    df_ts_code = DB.get_ts_code()
    df_result_summary = pd.DataFrame()

    for ts_code in df_ts_code.index[::100]:

        df_result = DB.get_asset(ts_code)
        df_result = df_result[df_result["period"] > 240]
        df_result = df_result[df_result.index > 20000101]

        # setting
        all_column = ["close"]
        all_target = [2, 5, 10, 20, 40, 60, 120, 240]  # -60, -1 means future 60 days return to future 1 day, means in
        all_q = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
        all_freq = [2, 5, 10, 20, 120, 260, 520]  # 780
        all_freq_comb = [(2, 120), (2, 260), (2, 520), (5, 120), (5, 260), (5, 520), (10, 120), (10, 260), (10, 520), (20, 120), (20, 260), (20, 520)]  # (2,780),(5,780),(10,780),(20,780),(120,260),(120,520),(120,780),(260,520),(260,780),(520,780)

        # create target freq and target period
        for target in all_target:
            df_result[f"tomorrow{target}"] = df_result["open"].shift(-target) / df_result["open"].shift(-1)

        # pre calculate all rsi
        for column in all_column:
            for freq in all_freq:
                # plain pct_chg pearson 0.01, plain tor pearson, 0.07, rsi dv_ttm is crap,ivola very good
                df_result[f"{column}.rsi{freq}"] = talib.RSI(df_result[column], timeperiod=freq)

        # for each day, check in which bin that rsi is
        for trade_date in df_result.index:
            df_today = df_result.loc[(df_result.index < trade_date)]
            if len(df_today) < 620:
                continue
            last_Day = df_today.index[-600]
            df_today = df_result.loc[(df_result.index < last_Day)]
            print(f"{ts_code} {trade_date}. last day is {last_Day}", )

            d_accel = {}
            # create quantile
            for column in all_column:
                for freq in all_freq:
                    # divide all past values until today by bins/gategories/quantile using quantile:  45<50<55
                    a_q_result = list(df_today[f"{column}.rsi{freq}"].quantile(all_q))
                    for counter, item in enumerate(a_q_result):
                        df_result.at[trade_date, f"{column}.freq{freq}_q{counter}"] = item
                        d_accel[f"{column}.freq{freq}_q{counter}"] = item

            # for each day check what category todays rsi belong
            for column in all_column:
                for freq in all_freq:
                    for counter in range(0, len(all_q) - 1):
                        under_limit = d_accel[f"{column}.freq{freq}_q{counter}"]
                        above_limit = d_accel[f"{column}.freq{freq}_q{counter + 1}"]
                        today_rsi_value = df_result.at[trade_date, f"{column}.rsi{freq}"]
                        if under_limit <= today_rsi_value <= above_limit:
                            df_result.at[trade_date, f"{column}.rsi{freq}_bin"] = int(counter)
                            break

            # calculate simulated value

            for column in all_column:
                d_column_target_results = {key: [] for key in all_target}
                for small, big in all_freq_comb:
                    try:
                        small_bin = df_result.at[trade_date, f"{column}.rsi{small}_bin"]
                        big_bin = df_result.at[trade_date, f"{column}.rsi{big}_bin"]
                        df_filtered = df_today[(df_today[f"{column}.rsi{big}_bin"] == big_bin) & (df_today[f"{column}.rsi{small}_bin"] == small_bin)]
                        for target in d_column_target_results.keys():
                            d_column_target_results[target] = df_filtered[f"tomorrow{target}"].mean()
                    except Exception as e:
                        print("error", e)

                for target, a_estimates in d_column_target_results.items():
                    df_result.at[trade_date, f"{column}.score{target}"] = pd.Series(data=a_estimates).mean()

        # create sume of all results
        for freq in all_freq:
            try:
                df_result[f"sum.score{freq}"] = bi.sum([df_result[f"{x}.score{freq}"] for x in all_column])
            except:
                pass

        # initialize setting result
        LB.to_csv_feather(df_result, LB.a_path(f"trade/result/trading_result_{ts_code}.{str(all_column)}"))

        for target in all_target:
            try:
                pearson = df_result[f"{column}.score{target}"].corr(df_result[f"tomorrow{target}"])
                df_result_summary.at[ts_code, f"pearson_{target}"] = pearson
            except:
                pass

    LB.to_csv_feather(df_result_summary, LB.a_path(f"trade/summary.{str(all_column)}"))


def price_statistic_train(a_freq=[1, 2, 5, 10, 20, 60, 120, 240, 500, 750], past=10, q_step=5, df=DB.get_stock_market_all()):
    """use quantile to count insted of fixed price gaps
    This method does not work because past can not predict future due to unstable period. This means that at the same stage of two past days with the exact same situation. One can go up and one can go down due to unstable period of the next happening thing
    """
    df_result = pd.DataFrame()
    # for future in a_freq:
    #     df[f"tomorrow{future}"] = df["close"].shift(-future) / df["close"]
    #     df[f"past{future}"] = df["close"] / df["close"].shift(future)

    for key, df_filtered in  LB.custom_quantile(df=df, column=f"past{past}", p_setting=[x/100 for x in range(0, 101, q_step)]).items():
        df_result.at[key, "count"] = len(df_filtered)
        df_result.at[key, "q1"] ,df_result.at[key, "q2"] ,df_result.at[key, "q1_val"] ,df_result.at[key, "q2_val"]= [float(x) for x in key.split(",")]
        for future in a_freq:
            # df_result.at[f"{from_price,to_price}", f"tomorrow{future}_mean"] = (df_filtered[f"tomorrow{future}"].mean())
            # df_result.at[f"{from_price,to_price}", f"tomorrow{future}_std"] = (df_filtered[f"tomorrow{future}"].std())
            df_result.at[key, f"tomorrow{future}gmean"] = gmean(df_filtered[f"tomorrow{future}"].dropna())

        # a_path=LB.a_path(f"Market/CN/Atest/seasonal/all_date_price_statistic_past_{past}")
        # LB.to_csv_feather(df_result,a_path,skip_feather=True)
    return df_result

"""does not work here. Past can not predict future here"""
def price_statistic_predict(a_all_freq=[1, 2, 5, 10, 20, 60, 120, 240, 500, 750]):
    """performs a strategy based on past experience of price structure
    This method does not work because past can not predict future due to unstable period. This means that at the same stage of two past days with the exact same situation. One can go up and one can go down due to unstable period of the next happening thing

    Test 1: single past
    1. train expanding window data set on past gain future gain
    2. use the past trained data on to predict future gain
    """
    a_all_freq = [750]
    a_past_freq=a_all_freq
    a_future_freq=[750]

    df=DB.get_stock_market_all()

    for freq in a_all_freq:
        df[f"tomorrow{freq}"] = df["close"].shift(-freq) / df["close"]
        df[f"past{freq}"] = df["close"] / df["close"].shift(freq)
    df_result = df.copy()

    #simulate past by expanding
    for trade_date,df_past in LB.custom_expand(df=df, min_freq=1000).items():

        #1. cut df_past AGAIN: instead of expanding until today, we expand until couple days before that. So that latest value does not disturb calculation
        df_past=df_past.iloc[0:len(df_past)-500]

        #get result of past quantile and their predicted future gain
        for past_freq in a_all_freq:

            #1. train past values and create matrix
            df_pred_matrix=price_statistic_train(a_freq=a_all_freq,past=past_freq, q_step=10,df=df_past)

            for future_freq in a_future_freq:

                # predict what happens in the future using past trained value
                todays_value = float(df.at[trade_date, f"past{past_freq}"])
                try:
                    #todays value has been happened in the past
                    predicted_value=df_pred_matrix.loc[ (df_pred_matrix["q1_val"]<=todays_value) & (todays_value<=df_pred_matrix["q2_val"]), f"tomorrow{future_freq}gmean"].values[0]
                except :
                    #todays value is extrem value, either maxima or minima.
                    if todays_value > 1:#maxima
                        predicted_value=df_pred_matrix.tail(1)[f"tomorrow{future_freq}gmean"].values[0]
                    else: #minima
                        predicted_value=df_pred_matrix.head(1)[f"tomorrow{future_freq}gmean"].values[0]
                print(f"{trade_date} past{past_freq} predicted future{future_freq} =", predicted_value)
                df_result.at[trade_date, f"past{past_freq}_pred_future{future_freq}"] = predicted_value

    #combine the score using mean
    for future_freq in a_future_freq:
        #combined score
        df_result[f"pred_future{future_freq}"]=sum([df_result[f"past{past_freq}_pred_future{future_freq}"] for past_freq in a_past_freq]) / len(a_past_freq)

        #combined score bin
        df_result[f"pred_future{future_freq}_bin"] =pd.qcut(df_result[f"pred_future{future_freq}"], q=10, labels=False)

    df_result.to_csv("past_test.csv")
    df_pred_matrix.to_csv(("last_pred_matrix.csv"))




def kalman_filter():
    """
        From stack overflow, might be just wrong. Kalman filter is generally very noisy. Not smooth at all.
        Reasons why kalman filter is not the best for measure stock price:
        1. The assmumes the true value exist, Whereas stock price true value might always have a lot of noise, so no true value
        2. The kalman filter itself deviates a lot with a lot of error


        Parameters:
        x: initial state 4-tuple of location and velocity: (x0, x1, x0_dot, x1_dot)
        P: initial uncertainty convariance matrix
        measurement: observed position
        R: measurement noise
        motion: external motion added to state vector x
        Q: motion noise (same shape as P)
        """

    def kalman(x, P, measurement, R, motion, Q, F, H):
        '''


        Parameters:
        x: initial state
        P: initial uncertainty convariance matrix
        measurement: observed position (same shape as H*x)
        R: measurement noise (same shape as H)
        motion: external motion added to state vector x
        Q: motion noise (same shape as P)
        F: next state function: x_prime = F*x
        H: measurement function: position = H*x

        Return: the updated and predicted new values for (x, P)

        See also http://en.wikipedia.org/wiki/Kalman_filter

        This version of kalman can be applied to many different situations by
        appropriately defining F and H
        '''
        # UPDATE x, P based on measurement m
        # distance between measured and current position-belief
        y = np.matrix(measurement).T - H * x
        S = H * P * H.T + R  # residual convariance
        K = P * H.T * S.I  # Kalman gain
        x = x + K * y
        I = np.matrix(np.eye(F.shape[0]))  # identity matrix
        P = (I - K * H) * P

        # PREDICT x, P based on motion
        x = F * x + motion
        P = F * P * F.T + Q
        return x, P

    def kalman_xy(x, P, measurement, R, motion=np.matrix('0. 0. 0. 0.').T, Q=np.matrix(np.eye(4))):
        """
        Parameters:
        x: initial state 4-tuple of location and velocity: (x0, x1, x0_dot, x1_dot)
        P: initial uncertainty convariance matrix
        measurement: observed position
        R: measurement noise
        motion: external motion added to state vector x
        Q: motion noise (same shape as P)
        """
        return kalman(x, P, measurement, R, motion, Q, F=np.matrix('''  1. 0. 1. 0.;
                                                                          0. 1. 0. 1.;
                                                                          0. 0. 1. 0.;
                                                                          0. 0. 0. 1.
                                                                          '''),
                      H=np.matrix('''
                                                                          1. 0. 0. 0.;
                                                                          0. 1. 0. 0.'''))

    x = np.matrix('0. 0. 0. 0.').T
    P = np.matrix(np.eye(4)) * 1000  # initial uncertainty

    df = DB.get_asset()

    N = len(df["close"])
    observed_x = range(0, N)
    observed_y = df["close"]
    plt.plot(observed_x, observed_y)
    result = []
    R = 800 ** 2  # the bigger the noise, the more the lag . Otherwise too close to actual price, no need to filter
    for meas in zip(observed_x, observed_y):
        x, P = kalman_xy(x, P, meas, R)
        result.append((x[:2]).tolist())
    kalman_x, kalman_y = zip(*result)
    plt.plot(kalman_x, kalman_y)
    plt.show()



"""incompleted"""
def swissarmy(type, s_high, s_low, n):
    """http://www.mesasoftware.com/papers/SwissArmyKnifeIndicator.pdf
        Various indicators together in one place
    """
    delta = 0.1
    N = 0
    s_price = (s_high + s_low) / 2

    a_result = []
    if type == "EMA":
        for i in range(0, len(s_high)):
            if i < n:
                result = s_price.iloc[i]
                alpha = (math.cos(np.radians(360) / n) + math.sin((np.radians(360) / n)) - 1) / math.cos(np.radians(360) / n)
                b0 = alpha
                a1 = 1 - alpha



def my_monte_carlo(s_close,n,m):
    """
    s_close=close series
    n= n days forcast into the future
    m= amount of simulations
    Basically useless simulation!
    It is just a random std
    """
    from scipy.stats import norm
    log_returns = np.log(1 + s_close.pct_change())
    u=log_returns.mean()
    var=log_returns.var()
    drift=u-(0.5*var)
    stdev=log_returns.std()

    daily_returns=np.exp(drift + stdev*norm.ppf(np.random.rand(n,m)))

    #takes last day as starting point
    s0=s_close.iat[-1]
    price_list=np.zeros_like(daily_returns)
    price_list[0]=s0

    #apply monte carlo
    for t in range(1, n):
        price_list[t]=price_list[t-1]*daily_returns[t]


    plt.plot(price_list)
    plt.show()

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
