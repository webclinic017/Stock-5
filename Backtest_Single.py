import os.path
import numpy as np
import matplotlib.pyplot as plt
import os
from collections import Counter, defaultdict
import sys
import pandas as pd
import numpy as np
import time
import math
import talib
import Util
import DB


# WHO WAS GOOD DURING THAT TIME PERIOD
# ASSET INFORMATION
# measures the fundamentals aspect
def asset_fundamental(start_date, end_date, freq, assets=["E"]):
    asset = assets[0]
    ts_codes = DB.get_ts_code(asset)
    a_result_mean = []
    a_result_std = []

    ts_codes = ts_codes[::-1]
    small = ts_codes[(ts_codes["exchange"] == "创业板") | (ts_codes["exchange"] == "中小板")]
    big = ts_codes[(ts_codes["exchange"] == "主板")]

    print("small size", len(small))
    print("big size", len(big))

    ts_codes = ts_codes

    for ts_code in ts_codes.ts_code:
        print("start appending to asset_fundamental", ts_code)

        # get asset
        df_asset = DB.get_asset(ts_code, asset, freq)
        if df_asset.empty:
            continue
        df_asset = df_asset[df_asset["trade_date"].between(int(start_date), int(end_date))]

        # get all label
        fun_balancesheet_label_list = ["pe_ttm", "ps_ttm", "pb", "total_mv", "profit_dedt", "total_cur_assets", "total_nca", "total_assets", "total_cur_liab", "total_ncl", "total_liab"]
        fun_cashflow_label_list = ["n_cashflow_act", "n_cashflow_inv_act", "n_cash_flows_fnc_act"]
        fun_indicator_label_list = ["netprofit_yoy", "or_yoy", "grossprofit_margin", "netprofit_margin", "debt_to_assets", "turn_days"]
        fun_pledge_label_list = ["pledge_ratio"]
        fun_label_list = fun_balancesheet_label_list + fun_cashflow_label_list + fun_indicator_label_list + fun_pledge_label_list
        df_asset = df_asset[["ts_code", "period"] + fun_label_list]

        # calc reduced result
        ts_code = df_asset.at[0, "ts_code"]
        period = df_asset.at[len(df_asset) - 1, "period"]

        # calc result
        fun_result_mean_list = [df_asset[label].mean() for label in fun_label_list]
        fun_result_std_list = [df_asset[label].std() for label in fun_label_list]

        a_result_mean.append(list([asset, ts_code, period] + fun_result_mean_list))
        a_result_std.append(list([asset, ts_code, period] + fun_result_std_list))

    # create tab Asset View
    df_result_mean = pd.DataFrame(a_result_mean, columns=["asset"] + list(df_asset.columns))
    df_result_std = pd.DataFrame(a_result_std, columns=["asset"] + list(df_asset.columns))

    # create std rank
    # THE LESS STD THE BETTER
    df_result_mean["std_growth_rank"] = df_result_std["netprofit_yoy"] + df_result_std["or_yoy"]
    df_result_mean["std_margin_rank"] = df_result_std["grossprofit_margin"] + df_result_std["netprofit_margin"]
    df_result_mean["std_cashflow_op_rank"] = df_result_std["n_cashflow_act"]
    df_result_mean["std_cashflow_inv_rank"] = df_result_std["n_cashflow_inv_act"]
    df_result_mean["std_cur_asset_rank"] = df_result_std["total_cur_assets"]
    df_result_mean["std_cur_liab_rank"] = df_result_std["total_cur_liab"]
    df_result_mean["std_plus_rank"] = df_result_mean["std_growth_rank"] + df_result_mean["std_margin_rank"] + df_result_mean["std_cashflow_op_rank"] * 2 + df_result_mean["std_cashflow_inv_rank"] + df_result_mean["std_cur_asset_rank"] * 3

    df_result_mean["std_growth_rank"] = df_result_mean["std_growth_rank"].rank(ascending=False)
    df_result_mean["std_margin_rank"] = df_result_mean["std_margin_rank"].rank(ascending=False)
    df_result_mean["std_cashflow_op_rank"] = df_result_mean["std_cashflow_op_rank"].rank(ascending=False)
    df_result_mean["std_cashflow_inv_rank"] = df_result_mean["std_cashflow_inv_rank"].rank(ascending=False)
    df_result_mean["std_cur_asset_rank"] = df_result_mean["std_cur_asset_rank"].rank(ascending=False)
    df_result_mean["std_cur_liab_rank"] = df_result_mean["std_cur_liab_rank"].rank(ascending=False)
    df_result_mean["std_plus_rank"] = df_result_mean["std_plus_rank"].rank(ascending=False)

    # create mean rank

    # 7  asset rank
    # SMALLER BETTER, rank LOWER BETTER
    # the bigger the company, the harder to get good asset ratio
    df_result_mean["asset_score"] = (df_result_mean["debt_to_assets"] + df_result_mean["pledge_ratio"] * 3) * np.sqrt(df_result_mean["total_mv"])
    df_result_mean["asset_rank"] = df_result_mean["asset_score"].rank(ascending=True)

    # 0 mv score
    # Higher BETTER, the bigger the company the better return
    # implies that value stock are better than value stock
    df_result_mean["mv_score"] = df_result_mean["total_mv"]
    df_result_mean["mv_rank"] = df_result_mean["mv_score"].rank(ascending=False)

    # 6 cashflow rank
    # SMALLER BETTER, rank LOWER BETTER
    # cashflow the closer to profit the better
    df_result_mean["cashflow_o_rank"] = 1 - abs(df_result_mean["n_cashflow_act"] / df_result_mean["profit_dedt"])
    df_result_mean["cashflow_o_rank"] = df_result_mean["cashflow_o_rank"].rank(ascending=True)

    # higher the better
    df_result_mean["cashflow_netsum_rank"] = (df_result_mean["n_cashflow_act"] + df_result_mean["n_cashflow_inv_act"] + df_result_mean["n_cash_flows_fnc_act"]) / df_result_mean["total_mv"]
    df_result_mean["cashflow_netsum_rank"] = df_result_mean["cashflow_netsum_rank"].rank(ascending=False)

    df_result_mean["non_current_asset_ratio"] = df_result_mean["total_nca"] / df_result_mean["total_assets"]
    df_result_mean["non_current_liability_ratio"] = df_result_mean["total_ncl"] / df_result_mean["total_liab"]
    df_result_mean["current_liability_to_mv"] = df_result_mean["total_cur_assets"] / df_result_mean["total_mv"]

    # 8 other rank
    # SMALLER BETTER, rank LOWER BETTER
    df_result_mean["other_rank"] = df_result_mean["turn_days"]
    df_result_mean["other_rank"] = df_result_mean["other_rank"].rank(ascending=True)

    # 1 margin score
    # HIGHER BETTER, rank LOWER BETTER
    # the bigger and longer a company, the harder to get high margin
    df_result_mean["margin_score"] = (df_result_mean["grossprofit_margin"] * 0.5 + df_result_mean["netprofit_margin"] * 0.5) * (np.sqrt(df_result_mean["total_mv"])) * (df_result_mean["period"])
    df_result_mean["margin_rank"] = df_result_mean["margin_score"].rank(ascending=False)

    # 2 growth rank
    # the longer a firm exists, the bigger a company, the harder to keep growth rate
    # the higher the margin, the higher the growthrate, the faster it grow
    # HIGHER BETTER, rank LOWER BETTER
    df_result_mean["average_growth"] = df_result_mean["netprofit_yoy"] * 0.2 + df_result_mean["or_yoy"] * 0.8
    df_result_mean["period_growth_score"] = ((df_result_mean["average_growth"]) * (df_result_mean["margin_score"]))
    df_result_mean["period_growth_rank"] = df_result_mean["period_growth_score"].rank(ascending=False)

    # the bigger the better
    df_result_mean["test_score"] = df_result_mean["average_growth"] * (df_result_mean["grossprofit_margin"] * 0.5 + df_result_mean["netprofit_margin"] * 0.5) * np.sqrt(np.sqrt(np.sqrt(df_result_mean["total_mv"]))) * np.sqrt(df_result_mean["period"]) * (100 - df_result_mean["pledge_ratio"])
    df_result_mean["test_rank"] = df_result_mean["test_score"].rank(ascending=False)

    # 3 PEG
    # SMALLER BETTER, rank LOWER BETTER
    df_result_mean["peg_rank"] = df_result_mean["pe_ttm"] / df_result_mean["average_growth"]
    df_result_mean["peg_rank"] = df_result_mean["peg_rank"].rank(ascending=True)

    # 4 PSG
    # SMALLER BETTER, rank LOWER BETTER
    df_result_mean["psg_rank"] = df_result_mean["ps_ttm"] / df_result_mean["average_growth"]
    df_result_mean["psg_rank"] = df_result_mean["psg_rank"].rank(ascending=True)

    # 5 PBG
    # SMALLER BETTER, rank LOWER BETTER
    df_result_mean["pbg_rank"] = df_result_mean["pb"] / df_result_mean["average_growth"]
    df_result_mean["pbg_rank"] = df_result_mean["pbg_rank"].rank(ascending=True)

    # final rank
    df_result_mean["final_fundamental_rank"] = df_result_mean["margin_rank"] * 0.40 + \
                                               df_result_mean["period_growth_rank"] * 0.2 + \
                                               df_result_mean["peg_rank"] * 0.0 + \
                                               df_result_mean["psg_rank"] * 0.0 + \
                                               df_result_mean["pbg_rank"] * 0.0 + \
                                               df_result_mean["cashflow_o_rank"] * 0.0 + \
                                               df_result_mean["cashflow_netsum_rank"] * 0.1 + \
                                               df_result_mean["asset_rank"] * 0.05 + \
                                               df_result_mean["other_rank"] * 0.05 + \
                                               df_result_mean["std_plus_rank"] * 0.2
    df_result_mean["final_fundamental_rank"] = df_result_mean["final_fundamental_rank"].rank(ascending=True)

    # add static data and sort by final rank
    df_result_mean = DB.add_static_data(df_result_mean, assets=assets)
    df_result_mean = DB.add_asset_final_analysis_rank(df_result_mean, assets, freq, "bullishness")
    df_result_mean = DB.add_asset_final_analysis_rank(df_result_mean, assets, freq, "volatility")
    df_result_mean.sort_values(by=["final_fundamental_rank"], ascending=True, inplace=True)

    path = "Market/" + "CN" + "/Backtest_Single/" + "fundamental" + "/" + ''.join(assets) + "_" + freq + "_" + start_date + "_" + end_date + ".xlsx"
    ts_code_series_to_excel(df_result_mean, path=path, sort=["final_fundamental_rank", True], asset=assets)


# measures the volatility aspect
def asset_volatility(start_date, end_date, assets, freq):
    a_result = []
    for asset in assets:
        ts_codes = DB.get_ts_code(asset)
        for ts_code in ts_codes.ts_code:
            print("start appending to asset_volatility", ts_code)

            # get asset
            df_asset = DB.get_asset(ts_code, asset, freq)
            if df_asset.empty:
                continue
            df_asset = df_asset[df_asset["trade_date"].between(int(start_date), int(end_date))]

            # get all label
            close_std_label_list = [s for s in df_asset.columns if "close_std" in s]
            ivola_std_label_list = [s for s in df_asset.columns if "ivola_std" in s]
            turnover_rate_std_label_list = [s for s in df_asset.columns if "turnover_rate_std" in s]
            beta_list = [s for s in df_asset.columns if "beta" in s]  # TODO add beta for E,I

            std_label_list = close_std_label_list + ivola_std_label_list + turnover_rate_std_label_list + beta_list
            df_asset = df_asset[["ts_code", "period"] + std_label_list]

            # calc reduced result
            ts_code = df_asset.at[0, "ts_code"]
            period = df_asset.at[len(df_asset) - 1, "period"]

            # calc result
            std_result_list = [df_asset[label].mean() for label in std_label_list]

            df_asset_reduced = [asset, ts_code, period] + std_result_list
            a_result.append(list(df_asset_reduced))

    # create tab Asset View
    df_result = pd.DataFrame(a_result, columns=["asset"] + list(df_asset.columns))

    # ranking
    # price: the higher the volatility between close prices each D the better
    # interday: the higher interday volatility the better
    # volume: the lower tor the better
    # beta: the lower the beta the better

    # calculate score
    df_result["close_score"] = sum([df_result[label] for label in close_std_label_list]) / len(close_std_label_list)
    df_result["ivola_score"] = sum([df_result[label] for label in ivola_std_label_list]) / len(ivola_std_label_list)
    if (asset == "E"):
        df_result["turnover_rate_score"] = sum([df_result[label] for label in turnover_rate_std_label_list]) / len(turnover_rate_std_label_list)
    # df_result["beta_score"]=sum([df_result[label] for label in beta_list])

    # rank them
    df_result["close_rank"] = df_result["close_score"].rank(ascending=False)
    df_result["ivola_rank"] = df_result["ivola_score"].rank(ascending=False)
    if (asset == "E"):  # TODO add turnover_rate for I ,FD
        df_result["turnover_rate_rank"] = df_result["turnover_rate_score"].rank(ascending=True)
    # df_result["beta_rank"] = df_result["beta_score"].rank(ascending=True)

    # final rank
    if (asset == "E"):
        df_result["final_volatility_rank"] = df_result["close_rank"] + df_result["ivola_rank"] + df_result["turnover_rate_rank"]
    else:
        df_result["final_volatility_rank"] = df_result["close_rank"] + df_result["ivola_rank"]
    df_result["final_volatility_rank"] = df_result["final_volatility_rank"].rank(ascending=True)

    # add static data and sort by final rank
    df_result = DB.add_static_data(df_result, assets)
    df_result = DB.add_asset_final_analysis_rank(df_result, assets, freq, "bullishness")
    df_result.sort_values(by=["final_volatility_rank"], ascending=True, inplace=True)

    path = "Market/" + "CN" + "/Backtest_Single/" + "volatility" + "/" + ''.join(assets) + "_" + freq + "_" + start_date + "_" + end_date + ".xlsx"
    ts_code_series_to_excel(df_result, path=path, sort=["final_volatility_rank", True], asset=assets)


# measures the result of a stock
# works for multiple assets, like assets=["E","FD","I"]
def asset_bullishness(start_date, end_date, assets, freq):
    a_result = []
    for asset in assets:
        ts_codes = DB.get_ts_code(asset)

        for ts_code in ts_codes.ts_code:
            print("start appending to asset_bullishness", ts_code, asset, freq)

            # get asset
            df_asset = DB.get_asset(ts_code, asset, freq)
            if df_asset.empty:
                continue

            df_asset = df_asset[df_asset["trade_date"].between(int(start_date), int(end_date))]

            if df_asset.empty:
                continue
            df_asset = Util.df_reindex(df_asset)

            # comp_gain
            asset_first_period_open = df_asset.open.loc[df_asset.open.first_valid_index()]
            asset_last_period_close = df_asset.at[len(df_asset) - 1, "close"]
            asset_period_comp_gain = round(asset_last_period_close / asset_first_period_open, 2)

            # get all label
            close_abv_m_label_list = [s for s in df_asset.columns if "close_abv_m" in s]
            df_asset = df_asset[["ts_code", "period", "total_mv", "beta240sh", "pct_chg"] + close_abv_m_label_list]

            # calc reduced result
            period = df_asset.at[len(df_asset) - 1, "period"]
            beta240sh = df_asset["beta240sh"].mean()
            total_mv = df_asset["total_mv"].mean()
            pct_chg = df_asset["pct_chg"].mean()

            close_abv_m_result_list = [df_asset[label].mean() for label in close_abv_m_label_list]
            df_asset_reduced = [ts_code, period, total_mv, beta240sh, pct_chg] + close_abv_m_result_list + [asset_period_comp_gain]
            a_result.append(list(df_asset_reduced))

    # create tab Asset View
    # df_result=pd.DataFrame(a_result,columns=["asset"]+list(df_asset.columns)+["comp_gain"])
    df_result = pd.DataFrame(a_result, columns=["ts_code", "period", "total_mv", "beta240sh", "pct_chg", "close_abv_m5", "close_abv_m20", "close_abv_m60", "close_abv_m240"] + ["comp_gain"])

    # final score is based on three scores:
    # 1. normal_score: the longer it can hold above ma the better: per period
    # score the HIGHER the better,rank the LOWER the better
    # normalized from 0 to 1, with 0.5 being median
    df_result["normal_score"] = sum([df_result[label] for label in close_abv_m_label_list]) / len(close_abv_m_label_list)
    df_result["normal_rank"] = df_result["normal_score"].rank(ascending=False)

    # 2. sustained_score: the longer a company can perform good, the better it is:
    # score the HIGHER the better,rank the LOWER the better
    # normalized from -0.5*sqrt(period) to 0.5*sqrt(period)
    df_result["sustained_score"] = (df_result["normal_score"] - 0.5) * np.sqrt(df_result['period'])
    df_result["sustained_rank"] = df_result["sustained_score"].rank(ascending=False)

    # 3. efficiency_score: The quicker a company can gain return, the better: per period
    # score the HIGHER the better,rank the LOWER the better
    # normalized from
    df_result["efficiency_score"] = (df_result["comp_gain"] - 1) / df_result["period"]
    df_result["efficiency_rank"] = df_result["efficiency_score"].rank(ascending=False)

    # combine into final score
    # score the HIGHER the better,rank the LOWER the better
    df_result["final_bullishness_rank"] = df_result["normal_rank"] * 0.55 + df_result["sustained_rank"] * 0.15 + df_result["efficiency_rank"] * 0.30
    df_result["final_bullishness_rank"] = df_result["final_bullishness_rank"].rank(ascending=True)  # not nessesary but makes the ranks all integer, looks better

    # df_result["calculated_beta"]=
    # add beta for reading

    print("df_result,len", len(df_result))
    # add static data and sort by final rank, only if there is one asset

    df_result = DB.add_static_data(df_result, assets)
    df_result.sort_values(by=["final_bullishness_rank"], ascending=True, inplace=True)

    path = "Market/" + "CN" + "/Backtest_Single/" + "bullishness" + "/" + ''.join(assets) + "_" + freq + "_" + start_date + "_" + end_date + ".xlsx"
    ts_code_series_to_excel(df_result, path=path, sort=["final_bullishness_rank", True], asset=assets)


# input df = df_date with abov_ma indicator for a period e.g. 20000101 to 20191111
# SIMILAR TO FUNCTION BACKTEST_MULTILPLE.Setup_Stock_Market_abv_ma()
# BUT only for individual ts_code instead for get_all_stock_market
def asset_trend_once(ts_code, start_time, end_time):
    df = DB.get_asset(ts_code=ts_code, asset="E", freq="D")

    try:  # put all errors in one try, if any error happens return nothing default row
        df = df[df["trade_date"].between(int(start_time), int(end_time))]
        if df.empty:
            return [ts_code] + [float("nan") for _ in range(0, 16)]
    except:
        return [ts_code] + [float("nan") for _ in range(0, 16)]

    df["rsi5"] = talib.RSI(df["close"], timeperiod=5) / 100
    df["rsi20"] = talib.RSI(df["close"], timeperiod=20) / 100
    df["rsi60"] = talib.RSI(df["close"], timeperiod=60) / 100
    df["rsi240"] = talib.RSI(df["close"], timeperiod=240) / 100

    # first abv_ma
    # second using RSI

    mean_even = (df["rsi5"].mean() + df["rsi20"].mean() + df["rsi60"].mean() + df["rsi240"].mean()) / 4
    mean_short = df["rsi5"].mean() * 0.428 + df["rsi20"].mean() * 0.285 + df["rsi60"].mean() * 0.142 + df["rsi240"].mean() * 0.142
    mean_long = df["rsi5"].mean() * 0.142 + df["rsi20"].mean() * 0.142 + df["rsi60"].mean() * 0.285 + df["rsi240"].mean() * 0.428

    # IMPORTANT NOTE!
    # mean rsi_60 seems to be the BEST frequency to setup as the one and only MEAN
    # in a test with 5,20,60,240, rsi60 has the highest mean_pct_chg rate for all stocks in average
    max240 = df["rsi60"].mean() + 0
    min240 = df["rsi60"].mean() - 0
    thresh60 = 0.0

    # 1 means uptrend
    # 0 means downtrend
    df["phase240"] = df["rsi240"].apply(lambda x: 1 if x > max240 + thresh60 * 0 else 0 if x < min240 - thresh60 * 0 else float("nan"))
    df["phase60"] = df["rsi60"].apply(lambda x: 1 if x > max240 + thresh60 * 0 else 0 if x < min240 - thresh60 * 0 else float("nan"))
    df["phase20"] = df["rsi20"].apply(lambda x: 1 if x > max240 + thresh60 * 0 else 0 if x < min240 - thresh60 * 0 else float("nan"))
    df["phase5"] = df["rsi5"].apply(lambda x: 1 if x > max240 + thresh60 * 0 else 0 if x < min240 - thresh60 * 0 else float("nan"))

    ##1 for uptrend
    ##0 for downtrend
    # 物极必反效应
    # mean reverse on the long run.
    # Crazy time is peak 60 on peak 240
    # Bad time is low 60 on low 240
    # connect peak and low, and we have a trend

    helper_low = ["5", "20", "60"]
    helper_high = ["20", "60", "240"]

    for rolling_freq_low, rolling_freq_high in zip(helper_low, helper_high):
        trend_name = "trend" + rolling_freq_high
        df[trend_name] = float("nan")
        df.loc[(df["phase" + rolling_freq_high] == 1) & (df["phase" + rolling_freq_low] == 1), trend_name] = 1
        df.loc[(df["phase" + rolling_freq_high] == 0) & (df["phase" + rolling_freq_low] == 0), trend_name] = 0

        # fill na based on the trigger points
        df[trend_name].fillna(method='bfill', inplace=True)

        last_trade = df.loc[df.last_valid_index(), trend_name]
        if last_trade == 1:
            df[trend_name].fillna(value=0, inplace=True)
        else:
            df[trend_name].fillna(value=1, inplace=True)

    df["trendtest"] = df["trend240"] * 0.5 + df["trend60"] * 0.25 + df["trend20"] * 0.25

    df["trend240_pct_chg"] = df.loc[(df["trend240"] == 1), "pct_chg"]
    df["trend60_pct_chg"] = df.loc[(df["trend60"] == 1), "pct_chg"]
    df["trend20_pct_chg"] = df.loc[(df["trend20"] == 1), "pct_chg"]
    df["trendtest_pct_chg"] = df.loc[(df["trendtest"] == 1), "pct_chg"]

    df["trend240_pct_chg"] = df["trend240_pct_chg"].fillna(value=0, inplace=False)
    df["trend60_pct_chg"] = df["trend60_pct_chg"].fillna(value=0, inplace=False)
    df["trend20_pct_chg"] = df["trend20_pct_chg"].fillna(value=0, inplace=False)

    df["trend240_comp_chg"] = Util.column_add_comp_chg(df["trend240_pct_chg"])
    df["trend60_comp_chg"] = Util.column_add_comp_chg(df["trend60_pct_chg"])
    df["trend20_comp_chg"] = Util.column_add_comp_chg(df["trend20_pct_chg"])

    # print out trend

    print(ts_code + " mean is", mean_even)
    print()
    # rsi_mean, trend20len, trend20_pct_chg_mean, trend20_pct_chg_std, trend20_comp_chg
    period = len(df)
    total_mv = df["total_mv"].mean()
    a_result = [ts_code, period, total_mv, mean_even, mean_short, mean_long]
    for freq in ["20", "60", "240"]:
        period = len(df)
        trend_n_len = (df["trend" + freq].sum()) / period
        _pct_chg_mean = df["trend" + freq + "_pct_chg"].mean()
        _pct_chg_std = df["trend" + freq + "_pct_chg"].std()
        _comp_chg = df["trend" + freq + "_comp_chg"].tail(1).values[0]

        print(ts_code + " trend" + freq + " len is", trend_n_len)
        print(ts_code + " trend" + freq + "_pct_chg_mean is", _pct_chg_mean)
        print(ts_code + " trend" + freq + "_pct_chg_std is", _pct_chg_std)
        print(ts_code + " trend" + freq + "_comp_chg is", _comp_chg)
        print()
        a_result.append(trend_n_len)
        a_result.append(_pct_chg_mean)
        a_result.append(_pct_chg_std)
        a_result.append(_comp_chg)

    # path = "test.csv"
    # Library_Main.close_file(path)
    # df.to_csv(path, index=False)
    # Library_Main.open_file(path)

    return a_result


# IMPORTANT function to calculate trend for an asset.
# This is probably the most important base for all strategy
def asset_trend_multiple(asset="E", start_time="20050101", end_time="20191111"):
    # meta
    path = "Market/CN/Backtest_Single/rsi/rsi_summary.xlsx"
    df_ts_code = DB.get_ts_code(asset)
    a_a_result = []
    df_result_column = []

    # create column helper
    for x in ["20", "60", "240"]:
        for y in ["len", "pct_chg_mean", "pct_chg_std", "comp_chg"]:
            df_result_column.append("trend" + x + "_" + y)

    # append result to array
    for ts_code in df_ts_code.ts_code:
        a_a_result.append(asset_trend_once(ts_code, start_time, end_time))

    # create df and to excel
    df_result = pd.DataFrame(a_a_result, columns=["ts_code", "period", "total_mv", "rsi_mean_even", "rsi_mean_short", "rsi_mean_long"] + df_result_column)
    ts_code_series_to_excel(df_ts_code_series=df_result, path=path, sort=["rsi_mean_even", False], asset=["E"])


# mainly used for save ts_code series into excel with groups
def ts_code_series_to_excel(df_ts_code_series, path, sort=["column_name", True], asset=["E"]):
    pdwriter = pd.ExcelWriter(path, engine='xlsxwriter')

    # add static data
    df_ts_code_series = DB.add_static_data(df_ts_code_series, assets=asset)
    df_ts_code_series.to_excel(pdwriter, sheet_name="Overview", index=False, encoding='utf-8_sig')

    # tab group
    for group_column, group_instancce in Util.c_groups_dict(asset).items():
        df_groupbyhelper = df_ts_code_series.groupby(group_column)
        try:
            df_group = df_groupbyhelper.mean()
            df_group["count"] = df_groupbyhelper.size()
            if sort:
                df_group.sort_values(by=sort[0], ascending=sort[1], inplace=True)
            df_group.to_excel(pdwriter, sheet_name=group_column, index=True, encoding='utf-8_sig')
        except:
            pass

    # save
    Util.close_file(path)
    pdwriter.save()


def excel_to_summary(folder_path, sort_setting=["", False]):
    dict_tab = {x: pd.DataFrame() for x in ["asset", "industry1", "industry2", "industry3", "area", "exchange", "is_hs"]}
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if "summary" in file:  # do not catch the summary file itself which is in the same folder
                continue
            file_path = root + file
            print("summarizing file", file_path)
            xls = pd.ExcelFile(file_path)

            for key, df in dict_tab.items():
                df_helper = pd.read_excel(xls, sheet_name=key)
                df_helper["filepath"] = file
                dict_tab[key] = df.append(df_helper, sort=False, ignore_index=True)

    summary_path = folder_path + "/excel_summary.xlsx"
    Util.close_file(summary_path)
    pdwriter = pd.ExcelWriter(summary_path, engine='xlsxwriter')
    for key, df in dict_tab.items():
        df.sort_values(by=sort_setting[0], ascending=sort_setting[1], inplace=True)
        df.to_excel(pdwriter, sheet_name=key, index=False, encoding='utf-8_sig')
    pdwriter.save()


def asset_technical_analysis_MA(ts_code, df, rolling_freq, future_gain_freqs, ma_version="SMA", ma_dict={}):
    ma_dict_local = {"SMA": ["close_m" + rolling_freq, talib.SMA, "My MA created with pandas "],
                     "DEMA": ["DEMA" + rolling_freq, talib.DEMA, "Double Exponential Moving Average"],
                     "EMA": ["EMA" + rolling_freq, talib.EMA, "Exponential Moving Average"],
                     "KAMA": ["KAMA" + rolling_freq, talib.KAMA, "Kaufman Adaptive Moving Average"],
                     "MAMA": ["MAMA" + rolling_freq, talib.MAMA, "MESA Adaptive Moving Average"],
                     "MAVP": ["MAVP" + rolling_freq, talib.MAVP, "Moving average with variable period"],
                     "T3": ["T3" + rolling_freq, talib.T3, "Triple Exponential Moving Average (T3)"],
                     "TEMA": ["TEMA" + rolling_freq, talib.TEMA, "Triple Exponential Moving Average"],
                     "TRIMA": ["TRIMA" + rolling_freq, talib.TRIMA, "Triangular Moving Average"],
                     "WMA": ["WMA" + rolling_freq, talib.WMA, "Weighted Moving Average"],
                     "MIDPOINT ": ["WMA" + rolling_freq, talib.MIDPOINT, "Weighted Moving Average"],
                     "RSI ": ["RSI" + rolling_freq, talib.RSI, "Weighted Moving Average"],
                     "MIDPRICE ": ["WMA" + rolling_freq, talib.MIDPRICE, "Weighted Moving Average"]}

    # create non-default MA during runtime with ta-lib
    if ma_version != "SMA":  # if non default MA
        print("subname", ma_version)
        func = ma_dict_local[ma_version][1]
        print("what")
        for key, value in ma_dict.items():
            print("start")
            df[ma_version + "_" + key] = func(df["close"].values, timeperiod=int(key))
            print("end")
    else:  # Default MA
        ma_version = "close_m"

    # loop over all ma to handle condition
    special_requirement = True
    strategy_name = ma_version
    for key, value in ma_dict.items():
        if value:
            special_requirement = special_requirement & (df["close"] > df[ma_version + key])
            strategy_name = strategy_name + "abv" + key + "_"
        else:
            special_requirement = special_requirement & (df["close"] < df[ma_version + key])
            strategy_name = strategy_name + "und" + key + "_"

    df["pattern"] = special_requirement

    df_grouped = df.groupby(by="pattern").agg(["mean", "count", "std"])

    occurence = df_grouped["pct_chg"]["count"][True] / len(df)
    triggerday_pct_chg_mean = df_grouped["pct_chg"]["mean"][True]
    triggerday_pct_chg_std = df_grouped["pct_chg"]["std"][True]
    a_result = [strategy_name, ts_code, occurence, triggerday_pct_chg_mean, triggerday_pct_chg_std]

    for future_gain_freq in future_gain_freqs:
        gain_mean = df_grouped["future_gain" + future_gain_freq]["mean"][True]
        gain_std = df_grouped["future_gain" + future_gain_freq]["std"][True]
        a_result.append(gain_mean)
        a_result.append(gain_std)

    for key, value in ma_dict.items():
        a_result.append(value)

    return a_result


def asset_technical_analysis_once(rolling_freq, name, sub_name="", start_date="20050101", end_date="20191111", ma_dict={}):
    path_helper = ""
    for key, value in ma_dict.items():
        if value:
            path_helper = path_helper + "abv" + str(key) + "_"
        else:
            path_helper = path_helper + "und" + str(key) + "_"
    path = "Market/CN/Backtest_Single/technical/" + name + "_abv_" + sub_name + "_" + path_helper + ".xlsx"  # not csv because we write everything to excel

    df_ts_code = DB.get_ts_code("E")[::5]
    future_gain_freqs = [str(x) for x in [2, 5, 10, 20, 60]]
    future_gain_freqs = [str(x) for x in [2, 5, 10]]
    data = []

    for key, value in ma_dict.items():
        print("ma_dict", key, value)

    df_ts_code = df_ts_code[df_ts_code["exchange"] == "中小板"]
    print("there are that many stocks in df_ts_code", len(df_ts_code))
    for ts_code, counter in zip(df_ts_code.ts_code, range(0, len(df_ts_code))):

        df = DB.get_asset(ts_code=ts_code, asset="E", freq="D")
        if len(df) > 300:
            print(counter, ts_code, "start technical Analysis", name, sub_name, rolling_freq, len(df))
        else:
            print(counter, ts_code, "Continue because stock is new", name, sub_name, rolling_freq, len(df))
            continue

        try:
            df = df[df["trade_date"].between(int(start_date), int(end_date))]
            # from here on it should not throw any error
            a_result = asset_technical_analysis_MA(ts_code, df, rolling_freq, future_gain_freqs, ma_version=sub_name, ma_dict=ma_dict)
            data.append(a_result)
        except Exception as e:
            print(ts_code, counter, "is occured error, continue")
            print(e)

    columns_label = ["strategy_name", "ts_code", "occurence", "triggerday_pct_chg_mean", "triggerday_pct_chg_std"]
    for gain_freq in future_gain_freqs:
        columns_label.append("future_gain" + gain_freq + "_mean")
        columns_label.append("future_gain" + gain_freq + "_std")

    for key, value in ma_dict.items():
        columns_label.append("abv_ma" + key)

    df_result = pd.DataFrame(data=data, columns=columns_label)
    ts_code_series_to_excel(df_ts_code_series=df_result, path=path, sort=[], asset=["E"])
    return df_result


def asset_technical_analysis_multiple():
    path = "Market/CN/Backtest_Single/technical/overview.csv"
    names = ["MA"]
    sub_names = ["SMA", "EMA", "DEMA", "KAMA", "TEMA", "WMA", "TRIMA", "MIDPOINT", "MIDPRICE"]
    sub_names = ["RSI"]

    for technical_name in names:
        for sub_names in sub_names:
            for rolling_freq in [str(x) for x in [5]][::1]:
                for abv_ma5 in [True, False]:
                    for abv_ma10 in [True, False]:
                        for abv_ma20 in [True, False]:
                            for abv_ma60 in [True, False]:
                                for abv_ma240 in [True, False]:
                                    ma_dict = {"5": abv_ma5, "10": abv_ma10, "20": abv_ma20, "60": abv_ma60, "240": abv_ma240}
                                    df = asset_technical_analysis_once(rolling_freq, technical_name, sub_names, start_date="20050101", end_date=DB.get_last_trade_date(), ma_dict=ma_dict)
                                    strategy_name = df.at[0, "strategy_name"]
                                    df = df.mean()
                                    df["strategy_name"] = strategy_name

                                    try:
                                        df_saved = pd.read_csv(path)
                                    except:
                                        df_saved = pd.DataFrame()
                                    df_saved = df_saved.append(df, ignore_index=True, sort=False)

                                    Util.close_file(path)
                                    df_saved.to_csv(path, index=False)
                                    time.sleep(7)


# checks for each asset what the next freq mean gain is based on certain condition
# e.g. if today 涨停，what is tomorrow mean gain
# e.g. if today 涨停，what is chance of tomorrow 涨停
def asset_price_analysis_once(start_date, end_date, freq="D", from_price=0.0, to_price=1.0, subfolder_name="", query=""):
    df_ts_code = DB.get_ts_code("E")

    condition_name = "price" + "_" + str(from_price) + "_" + str(to_price)
    path = "Market/CN/Backtest_Single/price/" + subfolder_name + condition_name + ".xlsx"

    a_result = []
    for ts_code in df_ts_code.ts_code:
        print("ts_code", ts_code)
        df_ts_code = DB.get_asset(ts_code, asset="E", freq=freq)

        try:
            df_ts_code = df_ts_code.tail(len(df_ts_code) - 100)  # remove stocks that are just ipod. The ipo stock can disturb the data
            if df_ts_code.empty:
                continue
        except Exception as e:
            print(e)

        df_filtered = df_ts_code.query(query)
        df_filtered = df_filtered[["future_gain2", "future_gain5", "future_gain10", "future_gain20"]]
        # df_filtered=df_ts_code.loc[(df_ts_code["pct_chg"].between(from_price,to_price)),["future_gain2","future_gain5","future_gain10","future_gain20"]]

        df_filtered["future_gain2_std"] = df_filtered["future_gain2"].std()
        df_filtered["future_gain5_std"] = df_filtered["future_gain5"].std()
        df_filtered["future_gain10_std"] = df_filtered["future_gain10"].std()
        df_filtered["future_gain20_std"] = df_filtered["future_gain20"].std()
        df_filtered["occurence"] = len(df_filtered) / len(df_ts_code)
        df_filtered = df_filtered.mean()
        df_filtered["ts_code"] = ts_code

        a_result.append(df_filtered)

    df_result = pd.DataFrame(a_result, columns=["ts_code", "future_gain2", "future_gain5", "future_gain10", "future_gain20", "future_gain2_std", "future_gain5_std", "future_gain10_std", "future_gain20_std", "occurence"])
    print(df_result)
    ts_code_series_to_excel(df_result, path, sort=["future_gain2", False], asset=["E"])


# loop over assets
def asset_price_analysis_multiple(freq_day=2):
    if freq_day == 1:  # looking at past day pct_chg
        # freq_day 1
        from_range = [-11, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        to_range = [-9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11]
    else:  # looking at past_gain2,5,10,20, etc
        # freq_day 2,5,10,20,60,240
        from_range = [x for x in [-10, -8, -6, -4, -2, 0, 2, 4, 6, 8]]
        to_range = [x for x in [-8, -6, -4, -2, 0, 2, 4, 6, 8, 10]]

    for from_price, to_price in zip(from_range, to_range):
        if freq_day != 1:
            print()
            from_price = (1 + from_price * 0.01) ** freq_day
            to_price = (1 + to_price * 0.01) ** freq_day
            query = '(past_gain{2} >= {0}) & (past_gain{2} <= {1})'.format(from_price, to_price, str(freq_day))
        else:
            query = '(pct_chg >= {0}) & (pct_chg <= {1})'.format(from_price, to_price)
        print("query is", query)
        asset_price_analysis_once("00000000", Util.today(), freq="D", from_price=from_price, to_price=to_price, query=query, subfolder_name="past_gain{}/".format(str(freq_day)))

    excel_to_summary(folder_path="D:/GoogleDrive/私人/私人 Stock 2.0/Market/CN/Backtest_Single/price/past_gain{}/".format(str(freq_day)), sort_setting=["future_gain2", False])


# measures which day of week/month performs generally better
# need only to calculate one time since it is the same for ALL periods
def date_seasonal_stats(start_date, end_date):
    path = "Market/CN/Backtest_Single/seasonal/all_date_seasonal.xlsx"
    pdwriter = pd.ExcelWriter(path, engine='xlsxwriter')

    print(start_date, end_date, "start date_period_statistc " + "...")
    path_all_dates = "Market/CN/Backtest_Multiple/Setup/Stock_Market/all_stock_market.csv"
    df_all_dates = pd.read_csv(path_all_dates)
    df_all_dates = df_all_dates[(df_all_dates["trade_date"].between(int(start_date), int(end_date)))]

    # get all different groups
    a_groups = [[Util.get_trade_date_datetime_dayofweek, "dayofweek"],
                [Util.get_trade_date_datetime_d, "dayofmonth"],
                [Util.get_trade_date_datetime_weekofyear, "weekofyear"],
                [Util.get_trade_date_datetime_dayofyear, "dayofyear"],
                [Util.get_trade_date_datetime_m, "monthofyear"]]

    # transform all trade_date into different format
    for group in a_groups:
        df_all_dates[group[1]] = df_all_dates["trade_date"].apply(lambda x: group[0](x))

    # OPTIONAL filter all ipo stocks and pct_chg greater than 10
    df_all_dates = df_all_dates[(df_all_dates["pct_chg"] < 11) & (df_all_dates["pct_chg"] > -11)]
    df_all_dates = df_all_dates[["pct_chg"] + [item[1] for item in a_groups]]

    # create and sort single groups
    for group in a_groups:
        print("group", group[1])
        df_group = df_all_dates[[group[1], "pct_chg"]].groupby(group[1]).mean()
        df_group.sort_values(by=[group[1]], ascending=True, inplace=True)
        group.append(df_group)
        df_group.to_excel(pdwriter, sheet_name=group[1], index=True, encoding='utf-8_sig')

    # create and sort multiple groups
    df_group = df_all_dates.groupby(["dayofweek", "dayofmonth"]).mean()
    df_group = df_group["pct_chg"]
    df_group.to_excel(pdwriter, sheet_name="dayofweekandmonth", index=True, encoding='utf-8_sig')
    pdwriter.save()


def trend_advanced_technique():  # TODO test it with shanghai index
    df = DB.get_stock_market_all()
    array = [1, 0]
    df["trend_all"] = df["trend2"] * 0.7 + df["trend5"] * 0.10 + df["trend20"] * 0.10 + df["trend60"] * 0.05 + df["trend240"] * 0.05

    a_tresh = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    df_result = pd.DataFrame(index=a_tresh, columns=["occ", "future_gain2", "future_gain5", "future_gain20"])
    for tresh in a_tresh:
        df_copy = df[df["trend_all"] >= tresh]
        df_result.at[tresh, "occ"] = len(df_copy)
        for freq in ["2", "5", "20"]:
            df_result.at[tresh, "future_gain" + freq] = df_copy["future_gain" + freq].mean()

    # df_result = pd.DataFrame(index=[range(0, 100)], columns=["ma2", "ma5", "ma20", "ma60", "ma240", "occ", "future_gain2", "future_gain5", "future_gain20"])

    # for ma2 in array:
    #     for ma5 in array:
    #         for ma20 in array:
    #             for ma60 in array:
    #                 for ma240 in array:
    #                     c_ma2 = df["trend2"] == ma2
    #                     c_ma5 = df["trend5"] == ma5
    #                     c_ma20 = df["trend20"] == ma20
    #                     c_ma60 = df["trend60"] == ma60
    #                     c_ma240 = df["trend240"] == ma240
    #
    #                     df_copy = df[c_ma2 & c_ma5 & c_ma20 & c_ma60 & c_ma240]
    #
    #                     print("ma: ", ma2, ma5, ma20, ma60, ma240)
    #                     print("occ", len(df_copy) / len(df))
    #                     print("future_gain2", df_copy["future_gain2"].mean())
    #                     print("future_gain5", df_copy["future_gain5"].mean())
    #                     print("future_gain20", df_copy["future_gain20"].mean())
    #
    #                     occ = len(df_copy) / len(df)
    #                     future_gain2 = (df_copy["future_gain2"].mean())
    #                     future_gain5 = (df_copy["future_gain5"].mean())
    #                     future_gain20 = (df_copy["future_gain20"].mean())
    #
    #                     s_result = pd.Series(data=[ma2, ma5, ma20, ma60, ma240, occ, future_gain2, future_gain5, future_gain20], index=["ma2", "ma5", "ma20", "ma60", "ma240", "occ", "future_gain2", "future_gain5", "future_gain20"])
    #                     df_result = df_result.append(s_result, sort=False, ignore_index=True)
    #                     print()

    df_result.to_csv("df_date3.csv")


def asset_candlestick_analysis_once(ts_code, pattern, func):
    df_asset = DB.get_asset(ts_code)
    rolling_freqs = [2, 5, 10, 20, 60, 240]
    # labels
    candle_1 = ["future_gain" + str(i) + "_1" for i in rolling_freqs] + ["future_gain" + str(i) + "_std_1" for i in rolling_freqs]
    candle_0 = ["future_gain" + str(i) + "_0" for i in rolling_freqs] + ["future_gain" + str(i) + "_std_0" for i in rolling_freqs]

    try:
        df_asset = df_asset[df_asset["period"] > 240]
        df_asset[pattern] = func(open=df_asset["open"], high=df_asset["high"], low=df_asset["low"], close=df_asset["close"])
    except:
        s_interim = pd.Series(index=["candle", "ts_code", "occurence_1", "occurence_0"] + candle_1 + candle_0)
        s_interim["ts_code"] = ts_code
        s_interim["candle"] = pattern
        return s_interim

    occurence_1 = len(df_asset[df_asset[pattern] == 100]) / len(df_asset)
    occurence_0 = len(df_asset[df_asset[pattern] == -100]) / len(df_asset)

    a_future_gain_1_mean = []
    a_future_gain_1_std = []
    a_future_gain_0_mean = []
    a_future_gain_0_std = []

    for freq in rolling_freqs:
        a_future_gain_1_mean.append(df_asset.loc[df_asset[pattern] == 100, "future_gain" + str(freq)].mean())
        a_future_gain_1_std.append(df_asset.loc[df_asset[pattern] == 100, "future_gain" + str(freq)].std())
        a_future_gain_0_mean.append(df_asset.loc[df_asset[pattern] == -100, "future_gain" + str(freq)].mean())
        a_future_gain_0_std.append(df_asset.loc[df_asset[pattern] == -100, "future_gain" + str(freq)].std())

    data = [pattern, ts_code, occurence_1, occurence_0] + a_future_gain_1_mean + a_future_gain_1_std + a_future_gain_0_mean + a_future_gain_0_std
    s_result = pd.Series(data=data, index=["candle", "ts_code", "occurence_1", "occurence_0"] + candle_1 + candle_0)
    return s_result


def asset_candlestick_analysis_multiple():
    dict_pattern = Util.c_candle()
    df_all_ts_code = DB.get_ts_code(asset="E")

    for key, array in dict_pattern.items():
        function = array[0]
        a_result = []
        for ts_code in df_all_ts_code.ts_code:
            print("start candlestick with", key, ts_code)
            a_result.append(asset_candlestick_analysis_once(ts_code=ts_code, pattern=key, func=function))

        df_result = pd.DataFrame(data=a_result)
        path = "Market/CN/Backtest_Single/candlestick/" + key + ".csv"
        df_result.to_csv(path, index=False)
        print("SAVED candlestick", key, ts_code)

    a_all_results = []
    for key, array in dict_pattern.items():
        path = "Market/CN/Backtest_Single/candlestick/" + key + ".csv"
        df_pattern = pd.read_csv(path)
        df_pattern = df_pattern.mean()
        df_pattern["candle"] = key
        a_all_results.append(df_pattern)
    df_all_result = pd.DataFrame(data=a_all_results)
    path = "Market/CN/Backtest_Single/candlestick/summary.csv"
    df_all_result.to_csv(path, index=True)


def asset_resistance_once(asset="E"):
    df_asset = DB.get_asset(ts_code="000002.SZ")

    i = 240

    df_select = df_asset.loc[i:i + 240, ["ts_code", "close"]]
    Util.df_reindex(df_select)
    print(df_select)
    df_select["regression"] = Util.get_linear_regression_s(df_select.index, df_select["close"])
    df_select.to_csv("regression.csv", index=False)


def most_common_2(lst, firstn):
    if not lst:
        return float("NaN")
    result = Counter(lst).most_common(firstn)
    return result


def support_resistance(with_time_rs, df):
    # NEED to be reversed and reindexed before continue
    rolling_period = 20
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


def asset_support_resistance_multiple():
    all_ts_code = DB.get_ts_code()
    all_ts_code = all_ts_code[all_ts_code["ts_code"] == "000002.SZ"]
    for ts_code in all_ts_code.ts_code:
        df = DB.get_asset(ts_code=ts_code)
        df = df[["ts_code", "trade_date", "open", "high", "low", "close"]]

        i = 240
        df_select = df.loc[i:i + 240]

        df_csv = support_resistance(False, df_select)

        df_csv.plot.line(legend=False)
        # plt.setp(plt.gca().get_xticklabels(), visible=False)
        plt.setp(plt.gca().get_xticklabels()[::1], visible=True)

        newpath = "plot/stock/" + str(ts_code) + "/"
        if not os.path.exists(newpath):
            os.makedirs(newpath)
        plt.savefig(newpath + str(df_select.at[i, "trade_date"]) + ".jpg")
        plt.clf()
        plt.show()
        # df_csv.to_csv("temp/day_"+str(i)+".csv",index=False)
        print()


if __name__ == '__main__':
    trend_advanced_technique()

    # asset_support_resistance_multiple()
    # asset_resistance_once()
    # asset_candlestick_analysis_multiple()
    # technical_analysis_multiple()
    # date_trend_multiple()
    # trend_for_asset_multiple("E","20050101","20191111")
    # asset_bullishness("20050101", "20191111", assets=["E"],freq= "D")
    # asset_volatility("00000000", MyLibrary.today(), assets=["E","I","FD"],freq= "D")
    # asset_fundamental("00000000", MyLibrary.today(), assets=["E"],freq= "D")
