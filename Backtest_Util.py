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

pro = ts.pro_api('c473f86ae2f5703f58eecf9864fa9ec91d67edbc01e3294f6a4f9c32')
ts.set_token("c473f86ae2f5703f58eecf9864fa9ec91d67edbc01e3294f6a4f9c32")


def report_portfolio(setting_original, a_port_h, a_trade_h, df_stock_market_all, backtest_start_time, setting_count):
    current_trend = Util.c_rolling_freqs()
    beta_against = "_000001.SH"

    a_time = []
    now = mytime.time()

    # deep copy setting
    setting = copy.deepcopy(setting_original)
    p_compare = setting["p_compare"]
    s_weight_matrix = setting["s_weight_matrix"]

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
    df_trade_h = pd.DataFrame(data=a_trade_h, columns=Util.c_trade_h_label())
    df_port_h = pd.concat(objs=a_port_h, ignore_index=True, sort=False)

    # use trade date on all stock market to see which day was not traded
    df_merge_helper = df_stock_market_all.reset_index(inplace=False, drop=False)
    df_merge_helper = df_merge_helper.loc[df_merge_helper["trade_date"].between(int(setting["start_date"]), int(setting["end_date"])), "trade_date"]
    next_trade_date = DB.get_next_trade_date(freq="D")
    # df_merge_helper=pd.Series(data=df_merge_helper.index.array+[int(next_trade_date)],name="trade_date")
    df_merge_helper = df_merge_helper.append(pd.Series(data=int(next_trade_date), index=[0]), ignore_index=False)  # before merge_ add future trade date as proxy
    df_merge_helper = df_merge_helper.rename("trade_date")

    # merge
    df_trade_h = pd.merge(df_merge_helper, df_trade_h, how='left', on=["trade_date"], suffixes=["", ""], sort=False)
    df_port_h = pd.merge(df_merge_helper, df_port_h, how='left', on=["trade_date"], suffixes=["", ""], sort=False)

    now = print_and_time(setting_count=setting_count, phase="Step1", df_today=pd.DataFrame(), df_tomorrow=pd.DataFrame(), df_today_portfolios=pd.DataFrame(), p_maxsize=30, a_time=a_time, prev_time=now)

    if np.nan in df_trade_h["ts_code"] or float("nan") in df_trade_h:
        print("file has nan ts_codes")
        raise ValueError

    # df_port_c
    # df_port_h.loc[:, df_port_h.columns != "ts_code"] = df_port_h.loc[:, df_port_h.columns != "ts_code"].infer_objects()

    df_port_c_helper_real = df_trade_h[(df_trade_h["trade_type"] == "sell") & (df_trade_h["buy_imp"] == 0)]
    df_port_c_helper_real["pct_chg"] = (df_port_c_helper_real["comp_chg"] - 1) * 100

    df_port_c_helper_imp = df_trade_h[(df_trade_h["trade_type"] == "sell")]
    df_port_c = df_port_c_helper_real.drop("buyout_price", 1).groupby("trade_date").agg("mean")

    df_port_c["portfolio_size_real"] = df_port_c_helper_real[["trade_date"]].groupby("trade_date").size()
    df_port_c["portfolio_size_imp"] = df_port_c_helper_imp[["trade_date"]].groupby("trade_date").size()
    df_port_c["comp_chg"] = Util.column_add_comp_chg(df_port_c["pct_chg"])

    now = print_and_time(setting_count=setting_count, phase="Step2", df_today=pd.DataFrame(), df_tomorrow=pd.DataFrame(), df_today_portfolios=pd.DataFrame(), p_maxsize=30, a_time=a_time, prev_time=now)

    # add competitor
    for competitor in p_compare:
        df_port_c = DB.add_asset_comparison(df=df_port_c, freq=setting["freq"], asset=competitor[0], ts_code=competitor[1], a_compare_label=["pct_chg"])
        df_port_c["comp_chg_" + competitor[1]] = Util.column_add_comp_chg(df_port_c["pct_chg_" + competitor[1]])

    # df_port_c add trend2,10,20,60,240
    a_current_trend_label = []
    for i in current_trend:  # do not add trend1 since it does not exist
        a_current_trend_label.append(f"market_trend{i}")
    df_port_c = pd.merge(left=df_port_c, right=df_stock_market_all[a_current_trend_label], on="trade_date", sort=False)

    now = print_and_time(setting_count=setting_count, phase="Step3", df_today=pd.DataFrame(), df_tomorrow=pd.DataFrame(), df_today_portfolios=pd.DataFrame(), p_maxsize=30, a_time=a_time, prev_time=now)

    # tab_overview
    df_port_overview = pd.DataFrame(float("nan"), index=range(len(p_compare) + 1), columns=[])
    df_port_overview = df_port_overview.astype(object)

    # create ID
    end_time_date = datetime.now()
    day = end_time_date.strftime('%Y/%m/%d')
    time = end_time_date.strftime('%H:%M:%S')
    duration = (end_time_date - backtest_start_time).seconds
    duration, Duration_rest = divmod(duration, 60)

    df_port_c["tomorrow_pct_chg"] = df_port_c["pct_chg"].shift(-1)  # add a future pct_chg 1 for easier target

    now = print_and_time(setting_count=setting_count, phase="Step4", df_today=pd.DataFrame(), df_tomorrow=pd.DataFrame(), df_today_portfolios=pd.DataFrame(), p_maxsize=30, a_time=a_time, prev_time=now)

    # general overview setting
    df_port_overview["SDate"] = day
    df_port_overview["STime"] = time
    df_port_overview["SDuration"] = str(duration) + ":" + str(Duration_rest)
    df_port_overview["strategy"] = setting["id"] = (str(setting_original["id"]) + "__" + str(setting_count))
    df_port_overview["start_date"] = setting["start_date"]
    df_port_overview["end_date"] = setting["end_date"]

    # portfolio strategy specific overview
    df_port_overview.at[0, "period"] = len(df_port_c)
    df_port_overview.at[0, "pct_days_involved"] = 1 - (len(df_port_c[df_port_c["portfolio_size_real"] == 0]) / len(df_port_c))
    df_port_overview.at[0, "beta"] = Util.calculate_beta(df_port_c["pct_chg"], df_port_c["pct_chg" + beta_against])
    df_port_overview.at[0, "winrate"] = len(df_port_c[(df_port_c["tomorrow_pct_chg"] >= 0) & (df_port_c["tomorrow_pct_chg"].notna())]) / len(df_port_c["tomorrow_pct_chg"].notna())

    df_port_overview.at[0, "pct_chg_mean"] = df_port_c["pct_chg"].mean()
    df_port_overview.at[0, "pct_chg_std"] = df_port_c["pct_chg"].std()

    try:
        df_port_overview.at[0, "comp_chg"] = df_port_c.at[df_port_c["comp_chg"].last_valid_index(), "comp_chg"]
    except:
        df_port_overview.at[0, "comp_chg"] = float("nan")

    df_port_overview.at[0, "buy_impossible"] = len(df_trade_h.loc[(df_trade_h["trade_type"] == "buy") & (df_trade_h["comp_chg"] == 1)]) / len(df_trade_h.loc[(df_trade_h["trade_type"] == "buy")])

    for trade_type in ["buy", "sell", "hold"]:
        try:
            df_port_overview.at[0, f"{trade_type}_count"] = df_trade_h["trade_type"].value_counts()[trade_type]
        except:
            df_port_overview.at[0, f"{trade_type}_count"] = float("nan")

    now = print_and_time(setting_count=setting_count, phase="Step5", df_today=pd.DataFrame(), df_tomorrow=pd.DataFrame(), df_today_portfolios=pd.DataFrame(), p_maxsize=30, a_time=a_time, prev_time=now)

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
    for column, a_weight in s_weight_matrix.items():
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
    df_port_c_pct_chg = df_port_c[["portfolio_size_real", "portfolio_size_imp", "rank_final", "pct_chg"] + ["pct_chg_" + x for x in [x[1] for x in p_compare]] + a_trend_label]
    df_port_c_comp_chg = df_port_c[["portfolio_size_real", "portfolio_size_imp", "rank_final", "comp_chg"] + ["comp_chg_" + x for x in [x[1] for x in p_compare]] + a_trend_label]

    now = print_and_time(setting_count=setting_count, phase="Step6", df_today=pd.DataFrame(), df_tomorrow=pd.DataFrame(), df_today_portfolios=pd.DataFrame(), p_maxsize=30, a_time=a_time, prev_time=now)

    # write portfolio
    portfolio_path = "Market/CN/Backtest_Multiple/Result/Portfolio_" + str(setting["id"]) + ".xlsx"
    portfolio_writer = pd.ExcelWriter(portfolio_path, engine='openpyxl')
    df_port_overview.to_excel(portfolio_writer, sheet_name="Overview", index=False)
    df_trade_h.to_excel(portfolio_writer, sheet_name="Trade_History", index=False, encoding='utf-8_sig')
    df_port_h.to_excel(portfolio_writer, sheet_name="Portfolio_History", index=False, encoding='utf-8_sig')
    df_port_c_pct_chg.to_excel(portfolio_writer, sheet_name="Pct_Chg", index=True, encoding='utf-8_sig')
    df_port_c_comp_chg.to_excel(portfolio_writer, sheet_name="Comp_Chg", index=True, encoding='utf-8_sig')
    df_setting = pd.DataFrame(setting, index=[0])
    df_setting.to_excel(portfolio_writer, sheet_name="Setting", index=False, encoding='utf-8_sig')

    now = print_and_time(setting_count=setting_count, phase="Step7", df_today=pd.DataFrame(), df_tomorrow=pd.DataFrame(), df_today_portfolios=pd.DataFrame(), p_maxsize=30, a_time=a_time, prev_time=now)

    print("=" * 50)
    [print(string) for string in a_time]
    print("=" * 50)

    Util.pd_writer_save(portfolio_writer, portfolio_path)

    return [df_trade_h, df_port_overview, df_setting]


# returns a portfolio for tomorrow
def hold_port_h_for_tomorrow(df_today_portfolios, df_date_tomorrow, trade_date_tomorrow):
    # port_h
    if df_today_portfolios.empty:
        return df_today_portfolios

    # use todays portfolio as a base for modify
    df_result = df_today_portfolios.copy()

    # set both df index=ts_code for later serialized operation
    # df_date_tomorrow.index = df_date_tomorrow["ts_code"]
    # df_result.index = df_result["ts_code"]

    # operation done for both trading and non-trading stocks tomorrow
    # TODO in REALIY if a stock does not trade, it counts as bad opportunity cost. So in REAL simulation, the pct_chg here should be changed to 0
    # TODO A strategy that minimizes picking stock with 停牌 is also part of the strategy and hence needs to be regarded。
    df_result["pct_chg"] = np.nan  # set all stocks tomorrows percent change to . If it trades tomorrow, it will be overritten. If not, it stays correct because stock not trading
    df_result["trade_date"] = trade_date_tomorrow
    df_result["hold_days"] = df_result["hold_days"] + 1  # update hold day for both trading and non trading stocks

    try:  # at least one of todays portfolio is trading tomorrow
        # filter df_tomorrow by todays portfolio
        df_date_tomorrow_trade = df_date_tomorrow.loc[df_today_portfolios.index.tolist()]

        # perform two operations ONLY for stocks that trade tomorrow
        # df_result.loc[df_date_tomorrow_trade.index,"pct_chg"]=df_date_tomorrow_trade["pct_chg"]
        # df_result.loc[df_date_tomorrow_trade.index,"comp_chg"]=df_date_tomorrow_trade["close"]/ df_result.loc[df_date_tomorrow_trade.index,"buyout_price"]

        df_date_tomorrow_trade["comp_chg"] = df_date_tomorrow_trade["close"] / df_result["buyout_price"]
        df_result.loc[df_date_tomorrow_trade.index, ["pct_chg", "comp_chg"]] = df_date_tomorrow_trade[["pct_chg", "comp_chg"]]


    except:  # NONE of todays portfolio trade tomorrow. df_date_tomorrow.loc will raise error
        # But I dont need to do anything because pct_chg was already set to 0.0.comp_chg will remain same
        pass

    # print out

    for ts_code, hold_count in zip(df_result.index, range(len(df_result.index))):
        print(f"{hold_count} : hold {ts_code}")

    return df_result


#
# def update_simulate_history(df_portfolio_overview, df_setting):
#     df_result = df_portfolio_overview.head(1)
#
#     # load existing excel
#     path = "Market/CN/Backtest_Multiple/Backtest_Summary.xlsx"
#     portfolio_writer = pd.ExcelWriter(path, engine='xlsxwriter')
#
#     # load sheets of existing excel
#     try:  # if there is an existing history
#         xls = pd.ExcelFile(path)
#         df_overview_h = pd.read_excel(xls, sheet_name="Overview")
#         df_setting_h = pd.read_excel(xls, sheet_name="Setting")
#     except Exception as e:  # if there is no existing history
#         print("ERROR",e)
#         df_overview_h = pd.DataFrame()
#         df_setting_h = pd.DataFrame()
#
#     # to_excel
#     df_overview_h = df_result.append(df_overview_h, sort=False, ignore_index=True)
#     df_setting_h = df_setting.append(df_setting_h, sort=False, ignore_index=True)
#     df_overview_h.to_excel(portfolio_writer, sheet_name="Overview", index=False, encoding='utf-8_sig')
#     df_setting_h.to_excel(portfolio_writer, sheet_name="Setting", index=False, encoding='utf-8_sig')
#     for i in range(0, 10):
#         try:
#             portfolio_writer.save()
#             return
#         except Exception as e:
#             Util.close_file(path)
#             Util.sound("close_excel.mp3")
#             print(e)
#             time.sleep(10)


# possible select func= pd.Dataframe.nsmallest, pd.Dataframe.nlargest, pd.Dataframe.sample

def try_select(select_from_df, select_size, select_by):
    try:
        return select_from_df.nsmallest(int(select_size), [select_by])
    # return numba_select(select_from_df,select_size,select_by)
    # return select_from_df.sort_values(select_by, ascending=True).head(int(select_size))
    except Exception as e:  # if sample size bigger than df, ignore
        print("ERROR. less than portfolio p_max_size", e)


def print_and_time(setting_count, phase, df_today, df_tomorrow, df_today_portfolios, p_maxsize, a_time, prev_time):
    print(f"{setting_count} : {phase} df_today {len(df_today)} df_tomorrow {len(df_tomorrow)} port {len(df_today_portfolios)} space {p_maxsize - len(df_today_portfolios)}")
    now = mytime.time()
    a_time.append(f"{phase}: {now - prev_time}")
    return now


def print_log(setting_count, *args):
    print(setting_count, ": what ", [args], [*args])


if __name__ == '__main__':
    pass
