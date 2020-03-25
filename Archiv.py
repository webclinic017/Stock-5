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
import LB
import DB
from scipy.stats.mstats import gmean
import Sandbox


# measures which day of week/month performs generally better
def date_seasonal_stats(group_instance="asset_E"):
    path = f"Market/CN/Backtest_Single/seasonal/all_date_seasonal_{group_instance}.xlsx"
    pdwriter = pd.ExcelWriter(path, engine='xlsxwriter')

    # perform seasonal stats for all stock market or for some groups only
    df_group = DB.get_stock_market_all().reset_index() if group_instance=="" else  DB.get_group_instance(group_instance=group_instance).reset_index()

    # get all different groups
    a_groups = [[LB.get_trade_date_datetime_dayofweek, "dayofweek"],
                [LB.get_trade_date_datetime_d, "dayofmonth"],
                [LB.get_trade_date_datetime_weekofyear, "weekofyear"],
                [LB.get_trade_date_datetime_dayofyear, "dayofyear"],
                [LB.get_trade_date_datetime_m, "monthofyear"]]

    # transform all trade_date into different date format
    for group in a_groups:
        df_group[group[1]] = df_group["trade_date"].apply(lambda x: group[0](x))


    # OPTIONAL filter all ipo stocks and pct_chg greater than 10
    df_group = df_group[(df_group["pct_chg"] < 11) & (df_group["pct_chg"] > -11)]

    # create and sort single groups
    for group in a_groups:
        df_result = df_group[[group[1], "pct_chg"]].groupby(group[1]).agg(["mean", "std", my_gmean, my_mean, my_std, my_mean_std_diff])
        df_result.to_excel(pdwriter, sheet_name=group[1], index=True, encoding='utf-8_sig')

    pdwriter.save()


def price_statistic_train(a_freq=[1, 2, 5, 10, 20, 60, 120, 240, 500, 750], past=10, q_step=5, df=DB.get_stock_market_all()):
    """use quantile to count insted of fixed price gaps"""

    df_result = pd.DataFrame()
    # for future in a_freq:
    #     df[f"tomorrow{future}"] = df["close"].shift(-future) / df["close"]
    #     df[f"past{future}"] = df["close"] / df["close"].shift(future)

    for key, df_filtered in  LB.custom_quantile(df=df, column=f"past{past}", p_setting=range(0, 101, q_step)).items():
        df_result.at[key, "count"] = len(df_filtered)
        df_result.at[key, "q1"] ,df_result.at[key, "q2"] ,df_result.at[key, "q1_val"] ,df_result.at[key, "q2_val"]= [float(x) for x in key.split(",")]
        for future in a_freq:
            # df_result.at[f"{from_price,to_price}", f"tomorrow{future}_mean"] = (df_filtered[f"tomorrow{future}"].mean())
            # df_result.at[f"{from_price,to_price}", f"tomorrow{future}_std"] = (df_filtered[f"tomorrow{future}"].std())
            df_result.at[key, f"tomorrow{future}gmean"] = gmean(df_filtered[f"tomorrow{future}"].dropna())

        # a_path=LB.a_path(f"Market/CN/Backtest_Single/seasonal/all_date_price_statistic_past_{past}")
        # LB.to_csv_feather(df_result,a_path,skip_feather=True)
    return df_result

"""does not work here. Past can not predict future here"""
def price_statistic_predict(a_all_freq=[1, 2, 5, 10, 20, 60, 120, 240, 500, 750]):
    """performs a strategy based on past experience of price structure

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
            df_predict_matrix=price_statistic_train(a_freq=a_all_freq,past=past_freq, q_step=10,df=df_past)

            for future_freq in a_future_freq:

                # predict what happens in the future using past trained value
                todays_value = float(df.at[trade_date, f"past{past_freq}"])
                try:
                    #todays value has been happened in the past
                    predicted_value=df_predict_matrix.loc[ (df_predict_matrix["q1_val"]<=todays_value) & (todays_value<=df_predict_matrix["q2_val"]), f"tomorrow{future_freq}gmean"].values[0]
                except :
                    #todays value is extrem value, either maxima or minima.
                    if todays_value > 1:#maxima
                        predicted_value=df_predict_matrix.tail(1)[f"tomorrow{future_freq}gmean"].values[0]
                    else: #minima
                        predicted_value=df_predict_matrix.head(1)[f"tomorrow{future_freq}gmean"].values[0]
                print(f"{trade_date} past{past_freq} predicted future{future_freq} =", predicted_value)
                df_result.at[trade_date, f"past{past_freq}_predict_future{future_freq}"] = predicted_value

    #combine the score using mean
    for future_freq in a_future_freq:
        #combined score
        df_result[f"predict_future{future_freq}"]=sum([df_result[f"past{past_freq}_predict_future{future_freq}"] for past_freq in a_past_freq]) / len(a_past_freq)

        #combined score bin
        df_result[f"predict_future{future_freq}_bin"] =pd.qcut(df_result[f"predict_future{future_freq}"], q=10, labels=False)

    df_result.to_csv("past_test.csv")
    df_predict_matrix.to_csv(("last_predict_matrix.csv"))





# WHO WAS GOOD DURING THAT TIME PERIOD
# ASSET INFORMATION
# measures the fundamentals aspect
"""does not work in general. past can not predict future here"""
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
    DB.ts_code_series_to_excel(df_result_mean, path=path, sort=["final_fundamental_rank", True], asset=assets)


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
            beta_list = [s for s in df_asset.columns if "beta" in s]

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
    DB.ts_code_series_to_excel(df_result, path=path, sort=["final_volatility_rank", True], asset=assets)


# measures the overall bullishness of an asset using GEOMEAN. replaces bullishness
def asset_bullishness():
    from scipy.stats import gmean
    df_ts_code_E = DB.get_ts_code(asset="E")[::1]
    df_ts_code_I = DB.get_ts_code(asset="I")[::1]
    df_ts_code=df_ts_code_E.append(df_ts_code_I)
    df_result = pd.DataFrame()
    a_freqs = [10, 20, 60, 120, 240]

    df_sh_index = DB.get_asset(ts_code="000001.SH", asset="I")
    df_sh_index["sh_close"] = df_sh_index["close"]
    df_cy_index = DB.get_asset(ts_code="399006.SZ", asset="I")
    df_cy_index["cy_close"] = df_cy_index["close"]
    for ts_code, asset in zip(df_ts_code.index, df_ts_code["asset"]):
        print("ts_code", ts_code)
        df_asset = DB.get_asset(ts_code=ts_code, asset=asset)

        df_result.at[ts_code, "period"] = len(df_asset)
        try:
            df_asset = df_asset[(df_asset["period"] > 240)]
        except:
            continue

        if len(df_asset) > 100:
            # assed gained from lifetime. bigger better
            df_result.at[ts_code, "comp_gain"] = df_asset["close"].iat[len(df_asset) - 1] / df_asset["close"].iat[0]

            # period. the longer the better
            df_result.at[ts_code, "period"] = len(df_asset)

            # Geomean.
            helper = 1 + (df_asset["pct_chg"] / 100)
            df_result.at[ts_code, "geomean"] = gmean(helper)

            # times above ma, bigger better
            df_asset["abv_ma"] = 0
            for freq in [240]:
                df_asset[f"highpass{freq}"] = Sandbox.highpass(df_asset["close"], freq)
                # df_asset[f"ma{freq}"] = df_asset["close"] - df_asset[f"highpass{freq}"]
                df_asset[f"ma{freq}"] = df_asset["close"].rolling(freq).mean()
                df_asset[f"abv_ma{freq}"] = (df_asset["close"] > df_asset[f"ma{freq}"]).astype(int)
                df_asset["abv_ma"] = df_asset["abv_ma"] + df_asset[f"abv_ma{freq}"]
            df_result.at[ts_code, "abv_ma"] = df_asset["abv_ma"].mean()

            # trend swap. how long a trend average lasts
            for freq in [240]:
                df_result.at[ts_code, f"abv_ma_days{freq}"] = LB.trend_swap(df_asset, f"abv_ma{freq}", 1)

            # volatility of the high pass, the smaller the better
            highpass_mean = 0
            for freq in [240]:
                highpass_mean = highpass_mean + df_asset[f"highpass{freq}"].mean()
            df_result.at[ts_code, "highpass_mean"] = highpass_mean

            # volatility pct_ chg, less than better
            df_result.at[ts_code, "rapid_down"] = len(df_asset[df_asset["pct_chg"] <= (-5)]) / len(df_asset)

            # beta, lower the better
            df_result.at[ts_code, "beta_sh"] = LB.calculate_beta(df_asset["close"], df_sh_index["sh_close"])
            df_result.at[ts_code, "beta_cy"] = LB.calculate_beta(df_asset["close"], df_cy_index["cy_close"])
            df_result.at[ts_code, "beta"] = abs(df_result.at[ts_code, "beta_sh"]) * abs(df_result.at[ts_code, "beta_cy"])

            # is_max. How long the current price is around the all time high. higher better
            df_asset["expanding_max"] = df_asset["close"].expanding(240).max()
            df_result.at[ts_code, "is_max"] = len(df_asset[(df_asset["close"] / df_asset["expanding_max"]).between(0.9, 1.1)]) / len(df_asset)

    df_result["final_rank"] = df_result["geomean"].rank(ascending=False) * 0.70 \
                              + df_result["is_max"].rank(ascending=False) * 0.10 \
                              + df_result["beta"].rank(ascending=True) * 0.05 \
                              + df_result["period"].rank(ascending=False) * 0.04 \
                              + df_result["abv_ma_days240"].rank(ascending=False) * 0.05 \
                              + df_result["highpass_mean"].rank(ascending=False) * 0.02 \
                              + df_result["abv_ma"].rank(ascending=False) * 0.02 \
                              + df_result["rapid_down"].rank(ascending=True) * 0.02

    DB.ts_code_series_to_excel(df_ts_code=df_result, sort=["final_rank", True], path="Market/CN/Backtest_single/bullishness.xlsx", group_result=True)


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
    dict_pattern = LB.c_candle()
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






if __name__ == '__main__':
    # for freq in [1,2,5,10,20,60,120,240,500,750]:
    #     price_statistic_train(past=freq, q_step=10)
    # df_asset=DB.get_asset()
    # df_asset["pcut"]=pd.qcut(df_asset["pct_chg"],10,labels=False)
    # df_asset[["pct_chg","pcut"]].to_csv("qcut.csv")
    asset_bullishness()
    # date_seasonal_stats(group_instance="")
    # date_seasonal_stats(group_instance="Exchange_主板")
    # date_seasonal_stats(group_instance="Exchange_中小板")
    # date_seasonal_stats(group_instance="Exchange_创业板")
    #price_statistic_predict()