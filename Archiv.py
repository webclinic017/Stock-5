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
    a_groups = [[LB.get_trade_date_datetime_dayofweek, "dayofweek"],
                [LB.get_trade_date_datetime_d, "dayofmonth"],
                [LB.get_trade_date_datetime_weekofyear, "weekofyear"],
                [LB.get_trade_date_datetime_dayofyear, "dayofyear"],
                [LB.get_trade_date_datetime_m, "monthofyear"]]

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
    DB.ts_code_series_to_excel(df_result, path=path, sort=["final_volatility_rank", True], asset=assets)


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
            df_asset = df_asset[df_asset["trade_date"].between(int(start_date), int(end_date))]

            if df_asset.empty:
                continue
            df_asset = LB.df_reindex(df_asset)

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
    DB.ts_code_series_to_excel(df_result, path=path, sort=["final_bullishness_rank", True], asset=assets)



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
    pass
