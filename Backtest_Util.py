import tushare as ts
import time as mytime
import copy
import pandas as pd
import numpy as np
import time
import DB
import Util
from datetime import datetime
from numba import njit
from numba import jit
from pathlib import Path

pro = ts.pro_api('c473f86ae2f5703f58eecf9864fa9ec91d67edbc01e3294f6a4f9c32')
ts.set_token("c473f86ae2f5703f58eecf9864fa9ec91d67edbc01e3294f6a4f9c32")


def report_portfolio(setting_original, dict_trade_h, df_stock_market_all, backtest_start_time, setting_count):
    current_trend = Util.c_rolling_freq()
    beta_against = "_000001.SH"

    a_time = []
    now = mytime.time()

    # deep copy setting
    setting = copy.deepcopy(setting_original)
    p_compare = setting["p_compare"]
    s_weight1 = setting["s_weight1"]
    s_weight2 = setting["s_weight2"]

    # convertes dict in dict to string
    for key, value in setting.items():
        if type(value) == dict:
            setting[key] = Util.groups_dict_to_string_iterable(setting[key])
        elif type(value) in [list, np.ndarray]:
            if (key == "p_compare"):
                setting["p_compare"] = ', '.join([x[1] for x in setting["p_compare"]])
            else:
                setting[key] = ', '.join(value)

    # create trade_h and port_h from array
    # everything under this line has integer as index
    # everything above this line has ts_code and trade_date as index
    trade_h_helper = []
    for day, info in dict_trade_h.items():
        for trade_type, a_trade_context in info.items():
            for single_trade in a_trade_context:
                trade_h_helper.append({"trade_date": day, "trade_type": trade_type, **single_trade})
    df_trade_h = pd.DataFrame(data=trade_h_helper, columns=trade_h_helper[-1].keys())


    # use trade date on all stock market to see which day was not traded
    df_merge_helper = df_stock_market_all.reset_index(inplace=False, drop=False)
    df_merge_helper = df_merge_helper.loc[df_merge_helper["trade_date"].between(int(setting["start_date"]), int(setting["end_date"])), "trade_date"]
    next_trade_date = DB.get_next_trade_date(freq="D")
    df_merge_helper = df_merge_helper.append(pd.Series(data=int(next_trade_date), index=[0]), ignore_index=False)  # before merge_ add future trade date as proxy
    df_merge_helper = df_merge_helper.rename("trade_date")

    # merge
    df_trade_h = pd.merge(df_merge_helper, df_trade_h, how='left', on=["trade_date"], suffixes=["", ""], sort=False)

    if np.nan in df_trade_h["ts_code"] or float("nan") in df_trade_h:
        print("file has nan ts_codes")
        raise ValueError

    df_port_c = df_trade_h.groupby("trade_date").agg("mean")
    df_port_c["port_pearson"] = df_trade_h.groupby("trade_date").apply(lambda x: x["rank_final"].corr(x["pct_chg"]))
    df_port_c["port_size"] = df_trade_h[df_trade_h["trade_type"].isin(["hold", "buy"])].groupby("trade_date").size()
    df_port_c["port_cash"] = df_trade_h.groupby("trade_date").apply(lambda x: x.at[x.last_valid_index(), "port_cash"])
    df_port_c["buy"] = df_trade_h[df_trade_h["trade_type"].isin(["buy"])].groupby("trade_date").size()
    df_port_c["hold"] = df_trade_h[df_trade_h["trade_type"].isin(["hold"])].groupby("trade_date").size()
    df_port_c["sell"] = df_trade_h[df_trade_h["trade_type"].isin(["sell"])].groupby("trade_date").size()
    df_port_c["port_close"] = df_trade_h[df_trade_h["trade_type"].isin(["hold", "buy"])].groupby("trade_date").sum()["value_close"]
    df_port_c["all_close"] = df_port_c["port_close"] + df_port_c["port_cash"]
    df_port_c["all_pct_chg"] = df_port_c["all_close"].pct_change().fillna(0) + 1
    df_port_c["all_comp_chg"] = df_port_c["all_pct_chg"].cumprod()


    # add competitor
    for competitor in p_compare:
        df_port_c = DB.add_asset_comparison(df=df_port_c, freq=setting["freq"], asset=competitor[0], ts_code=competitor[1], a_compare_label=["pct_chg"])
        df_port_c["comp_chg_" + competitor[1]] = Util.column_add_comp_chg(df_port_c["pct_chg_" + competitor[1]])

    # df_port_c add trend2,10,20,60,240
    a_current_trend_label = []
    for i in current_trend:  # do not add trend1 since it does not exist
        a_current_trend_label.append(f"market_trend{i}")
    df_port_c = pd.merge(left=df_stock_market_all.loc[int(setting["start_date"]):int(setting["end_date"]), a_current_trend_label], right=df_port_c, on="trade_date", how="left", sort=False)

    # tab_overview
    df_port_overview = pd.DataFrame(float("nan"), index=range(len(p_compare) + 1), columns=[]).astype(object)
    end_time_date = datetime.now()
    day = end_time_date.strftime('%Y/%m/%d')
    time = end_time_date.strftime('%H:%M:%S')
    duration = (end_time_date - backtest_start_time).seconds
    duration, Duration_rest = divmod(duration, 60)
    df_port_overview["SDate"] = day
    df_port_overview["STime"] = time
    df_port_overview["SDuration"] = str(duration) + ":" + str(Duration_rest)
    df_port_overview["strategy"] = setting["id"] = (str(setting_original["id"]) + "__" + str(setting_count))
    df_port_overview["start_date"] = setting["start_date"]
    df_port_overview["end_date"] = setting["end_date"]

    # portfolio strategy specific overview
    df_port_c["tomorrow_pct_chg"] = df_port_c["all_pct_chg"].shift(-1)  # add a future pct_chg 1 for easier target
    period = len(df_trade_h.groupby("trade_date"))
    df_port_overview.at[0, "period"] = period
    df_port_overview.at[0, "pct_days_involved"] = 1 - (len(df_port_c[df_port_c["port_size"] == 0]) / len(df_port_c))

    df_port_overview.at[0, "sell_mean_T+"] = df_trade_h.loc[(df_trade_h["trade_type"] == "sell"), "T+"].mean()
    df_port_overview.at[0, "asset_winrate"] = len(df_trade_h.loc[(df_trade_h["trade_type"] == "sell") & (df_trade_h["comp_chg"] > 1)]) / len(df_trade_h.loc[(df_trade_h["trade_type"] == "sell")])
    df_port_overview.at[0, "all_winrate"] = len(df_port_c.loc[df_port_c["all_pct_chg"] >= 1]) / len(df_port_c)

    df_port_overview.at[0, "asset_pct_chg"] = df_trade_h.loc[(df_trade_h["trade_type"] == "sell"), "comp_chg"].mean()
    df_port_overview.at[0, "all_pct_chg"] = df_port_c["all_pct_chg"].mean()

    df_port_overview.at[0, "asset_pct_chg_std"] = df_trade_h.loc[(df_trade_h["trade_type"] == "sell"), "comp_chg"].std()
    df_port_overview.at[0, "all_pct_chg_std"] = df_port_c["all_pct_chg"].std()

    try:
        df_port_overview.at[0, "all_comp_chg"] = df_port_c.at[df_port_c["all_comp_chg"].last_valid_index(), "all_comp_chg"]
    except:
        df_port_overview.at[0, "all_comp_chg"] = float("nan")

    df_port_overview.at[0, "port_beta"] = Util.calculate_beta(df_port_c["all_pct_chg"], df_port_c["pct_chg" + beta_against])
    df_port_overview.at[0, "buy_imp"] = len(df_trade_h.loc[(df_trade_h["buy_imp"] == 1) & (df_trade_h["trade_type"] == "buy")]) / len(df_trade_h.loc[(df_trade_h["trade_type"] == "buy")])
    df_port_overview.at[0, "port_pearson"] = df_port_c["port_pearson"].mean()

    for trade_type in ["buy", "sell", "hold"]:
        try:
            df_port_overview.at[0, f"{trade_type}_count"] = df_trade_h["trade_type"].value_counts()[trade_type]
        except:
            df_port_overview.at[0, f"{trade_type}_count"] = float("nan")

    for lower_year, upper_year in [(20000101, 20050101), (20050101, 20100101), (20100101, 20150101), (20150101, 20200101)]:
        df_port_overview.at[0, f"all_pct_chg{upper_year}"] = df_port_c.loc[(df_port_c.index > lower_year) & (df_port_c.index < upper_year), "all_pct_chg"].mean()


    # overview win rate and pct_chg mean
    condition_trade = df_port_c["tomorrow_pct_chg"].notna()
    condition_win = df_port_c["tomorrow_pct_chg"] >= 0
    for one_zero in [1, 0]:
        for rolling_freq in current_trend:
            try:  # calculate the pct_chg of all stocks 1 day after the trend shows buy signal
                df_port_overview.at[0, "market_trend" + str(rolling_freq) + "_" + str(one_zero) + "_pct_chg_mean"] = df_port_c.loc[df_port_c["market_trend" + str(rolling_freq)] == one_zero, "tomorrow_pct_chg"].mean()
                # condition_one_zero=df_port_c["market_trend" + str(rolling_freq)] == one_zero
                # df_port_overview.at[0, "market_trend" + str(rolling_freq) +"_"+ str(one_zero)+"_winrate"] = len(df_port_c[condition_trade & condition_win & condition_one_zero]) / len(df_port_c[condition_trade & condition_one_zero])
            except Exception as e:
                pass

    # overview indicator combination
    for column, a_weight in s_weight1.items():
        df_port_overview[column + "_ascending"] = a_weight[0]
        df_port_overview[column + "_indicator_weight"] = a_weight[1]
        df_port_overview[column + "_asset_weight"] = a_weight[2]

    # add overview for compare asset
    for i in range(len(p_compare), len(p_compare)):  # TODO add it back when nessesary
        competitor_ts_code = p_compare[i][1]
        df_port_overview.at[i + 1, "strategy"] = competitor_ts_code
        df_port_overview.at[i + 1, "pct_chg_mean"] = df_port_c["pct_chg_" + competitor_ts_code].mean()
        df_port_overview.at[i + 1, "pct_chg_std"] = df_port_c["pct_chg_" + competitor_ts_code].std()
        df_port_overview.at[i + 1, "winrate"] = len(df_port_c[(df_port_c["pct_chg_" + competitor_ts_code] >= 0) & (df_port_c["pct_chg_" + competitor_ts_code].notna())]) / len(df_port_c["pct_chg_" + competitor_ts_code].notna())
        df_port_overview.at[i + 1, "period"] = len(df_port_c) - df_port_c["pct_chg_" + competitor_ts_code].isna().sum()
        df_port_overview.at[i + 1, "pct_days_involved"] = 1
        df_port_overview.at[i + 1, "comp_chg"] = df_port_c.at[df_port_c["comp_chg_" + competitor_ts_code].last_valid_index(), "comp_chg_" + competitor_ts_code]
        df_port_overview.at[i + 1, "beta"] = Util.calculate_beta(df_port_c["pct_chg_" + competitor_ts_code], df_port_c["pct_chg" + beta_against])
        df_port_c["tomorrow_pct_chg_" + competitor_ts_code] = df_port_c["pct_chg_" + competitor_ts_code].shift(-1)  # add a future pct_chg 1 for easier target

        # calculate percent change and winrate
        condition_trade = df_port_c["tomorrow_pct_chg_" + competitor_ts_code].notna()
        condition_win = df_port_c["tomorrow_pct_chg_" + competitor_ts_code] >= 0
        for one_zero in [1, 0]:
            for y in current_trend:
                try:
                    df_port_overview.at[i + 1, "market_trend" + str(y) + "_" + str(one_zero) + "_pct_chg_mean"] = df_port_c.loc[df_port_c["market_trend" + str(y)] == one_zero, "tomorrow_pct_chg_" + competitor_ts_code].mean()
                    # condition_one_zero = df_port_c["market_trend" + str(y)] == one_zero
                    # df_port_overview.at[i + 1, "market_trend" + str(y) +"_"+str(one_zero)+"_winrate"] = len(df_port_c[condition_trade & condition_win & condition_one_zero]) / len(df_port_c[condition_trade & condition_one_zero])
                except Exception as e:
                    pass

    # split chart into pct_chg and comp_chg for easier reading
    a_trend_label = ["market_trend" + str(x) for x in current_trend if x != 1]
    df_port_c = df_port_c[["rank_final", "port_pearson", "port_size", "buy", "hold", "sell", "port_cash", "port_close", "all_close", "all_pct_chg", "all_comp_chg"] + ["pct_chg_" + x for x in [x[1] for x in p_compare]] + ["comp_chg_" + x for x in [x[1] for x in p_compare]] + a_trend_label]

    # write portfolio
    portfolio_path = "Market/CN/Backtest_Multiple/Result/Portfolio_" + str(setting["id"])
    Util.to_csv_feather(df=df_port_overview, a_path=Util.a_path(portfolio_path + "/overview"), index=False, skip_feather=True)
    Util.to_csv_feather(df=df_trade_h, a_path=Util.a_path(portfolio_path + "/trade_h"), index=False, skip_feather=True)
    Util.to_csv_feather(df=df_port_c, a_path=Util.a_path(portfolio_path + "/chart"), reset_index=False, skip_feather=True)
    df_setting = pd.DataFrame(setting, index=[0])
    Util.to_csv_feather(df=df_setting, a_path=Util.a_path(portfolio_path + "/setting"), index=False, skip_feather=True)

    print("setting is", setting["s_weight1"])
    print("=" * 50)
    [print(string) for string in a_time]
    print("=" * 50)
    return [df_trade_h, df_port_overview, df_setting]


# TODO try with numba
def try_select(select_from_df, select_size, select_by):
    try:
        return select_from_df.nsmallest(int(select_size), [select_by])
    except Exception as e:  # if sample size bigger than df, ignore
        print("ERROR. less than portfolio p_max_size", e)

def print_and_time(setting_count, phase, dict_trade_h_hold, dict_trade_h_buy, dict_trade_h_sell, p_maxsize, a_time, prev_time):
    # print(f"{setting_count} : {phase} hold {len(dict_trade_h_hold)} bought {len(dict_trade_h_buy)} sold {len(dict_trade_h_sell)} space {p_maxsize - (len(dict_trade_h_hold)+len(dict_trade_h_buy))}")
    print(f"{setting_count} : hold " + '{0: <5}'.format(len(dict_trade_h_hold)) + 'buy {0: <5}'.format(len(dict_trade_h_buy)) + 'sell {0: <5}'.format(len(dict_trade_h_sell)) + 'space {0: <5}'.format(p_maxsize - (len(dict_trade_h_hold) + len(dict_trade_h_buy))))
    now = mytime.time()
    a_time.append('{0: <25}'.format(phase) + f": {now - prev_time}")
    return now

if __name__ == '__main__':
    pass
