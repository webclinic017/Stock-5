import tushare as ts
import pandas as pd
import numpy as np
import time as mytime
import math
import talib
import time
import DB
from itertools import combinations
from itertools import permutations
import Util
from progress.bar import Bar
from datetime import datetime
import traceback
import Backtest_Util
import copy
import cProfile
import threading

import matplotlib
from numba import njit
from numba import jit

from sympy.utilities.iterables import multiset_permutations

pd.options.mode.chained_assignment = None  # default='warn'

pro = ts.pro_api('c473f86ae2f5703f58eecf9864fa9ec91d67edbc01e3294f6a4f9c32')
ts.set_token("c473f86ae2f5703f58eecf9864fa9ec91d67edbc01e3294f6a4f9c32")
break_loop = False


def get_input():
    global break_loop
    keystrk = input('Press a key \n')
    break_loop = True


def backtest_once(settings=[{}]):
    # READ FIRST:
    # Assume that all Analysis of stockmarket today happens at evening after stock market closed 15:00
    # Assume that all trades happen at the START of the tomorrow 9:30
    # Assume all sell at start of day keeps yesterdays close price
    # Assume all buy at start of day will be accounted for todays pcg_chg

    break_detector = threading.Thread(target=get_input, daemon=True)
    break_detector.start()

    # 0 PREPARATION
    # 0.1 PREPARATION- META
    backtest_start_time = datetime.now()

    # 0.2 PREPARATION- SETTING = static values that NEVER CHANGE during the loop
    # non changeable for the same period = Strategies that share the same meta info can be put together into one run
    start_date = settings[0]["start_date"]
    end_date = settings[0]["end_date"]
    freq = settings[0]["freq"]
    market = settings[0]["market"]
    assets = settings[0]["assets"]

    # 0.3 PREPARATION- INITIALIZE Iterables derived from Setting
    df_trade_dates = DB.get_trade_date(start_date=start_date, end_date=end_date, freq=freq)

    df_stock_market_all = DB.get_stock_market_all(market)
    df_group_instance_all = DB.get_group_instance_all(assets=["E"])
    last_simulated_date = df_stock_market_all.index[-1]
    df_today_accelerator = pd.DataFrame()

    # 0.4 PREPARATION- INITIALIZE Changeables for the loop
    a_dict_trade_h = [{df_trade_dates.at[0, "trade_date"]: {"sell": [], "hold": [], "buy": []}} for _ in settings]  # trade history for each setting
    dict_capital = {setting_count: {"cash": settings[0]["p_capital"]} for setting_count in range(0, len(settings))}  # only used in iteration, saved in trade_h

    # Main Loop
    for today, tomorrow in zip(df_trade_dates["trade_date"], df_trade_dates["trade_date"].shift(-1).fillna(-1).astype(int)):

        if break_loop:
            print("BREAK loop")
            Util.sound("break.mp3")
            time.sleep(3)
            # adjust date if loop was break
            for setting in settings:
                setting["end_date"] = today
            break

        # initialize df_date
        if tomorrow == -1:
            print("last day reached")
            tomorrow = int(DB.get_next_trade_date(freq=freq))
            df_today = df_today_accelerator
            df_tomorrow = df_today_accelerator.copy()
            df_tomorrow[["open", "high", "low", "close", "pct_chg"]] = np.nan
        else:
            df_tomorrow = DB.get_date(trade_date=tomorrow, assets=assets, freq="D", market=market)
            df_today = DB.get_date(trade_date=today, assets=assets, freq="D", market=market) if df_today_accelerator.empty else df_today_accelerator
            df_today_accelerator = df_tomorrow
            dict_weight_accelerator = {}

        # FOR EACH DAY LOOP OVER SETTING N
        for setting, setting_count, dict_trade_h in zip(settings, range(0, len(settings)), a_dict_trade_h):
            print("\n" * 3)
            a_time = []
            now = mytime.time()
            setting["id"] = datetime.now().strftime('%Y%m%d%H%M%S')

            p_maxsize = setting["p_maxsize"]
            p_feedbackday = setting["p_feedbackday"]
            market_trend = df_stock_market_all.at[int(today), setting["trend"]] if setting["trend"] else 1

            dict_trade_h[tomorrow] = {"sell": [], "hold": [], "buy": []}
            print('{0: <26}'.format("TODAY EVENING ANALYZE") + f"{today} stocks {len(df_today)}")
            print('{0: <26}'.format("TOMORROW MORNING TRADE") + f"{tomorrow} stocks {len(df_tomorrow)}")
            now = Backtest_Util.print_and_time(setting_count=setting_count, phase=f"INIT", dict_trade_h_hold=dict_trade_h[tomorrow]["hold"], dict_trade_h_buy=dict_trade_h[tomorrow]["buy"], dict_trade_h_sell=dict_trade_h[tomorrow]["sell"], p_maxsize=p_maxsize, a_time=a_time, prev_time=now)

            # 1.1 FILTER
            df_today_mod = df_today[df_today["period"] > 240]
            for query_string in setting["f_query"]:  # very slow and expensive for small operation because parsing the string takes long
                df_today_mod.query(query_string, inplace=True)
            now = Backtest_Util.print_and_time(setting_count=setting_count, phase=f"FILTER", dict_trade_h_hold=dict_trade_h[tomorrow]["hold"], dict_trade_h_buy=dict_trade_h[tomorrow]["buy"], dict_trade_h_sell=dict_trade_h[tomorrow]["sell"], p_maxsize=p_maxsize, a_time=a_time, prev_time=now)

            # 2 ECONOMY

            # 3 FINANCIAL MARKET

            # 6 PORTFOLIO = ASSET BALANCING, EFFICIENT FRONTIER, LEAST BETA #TODO

            # 6.1 PORTFOLIO FEEDBACK = Analysis for performance feedback mechanism
            if p_feedbackday > 0:  # not yet to feedback
                p_feedbackday = p_feedbackday - 1
            else:  # feedback day
                # TODO add feedback
                p_feedbackday = 60
                print("give any feedback")

            # 6.2 PORTFOLIO SELL SELECT
            setting_keep = setting["p_keep"]
            hold_count = 1
            sold_count = 1
            for trade_type, a_trade_content in dict_trade_h[today].items():  # NOTE here today means today morning trade
                if trade_type != "sell":  # == in ["hold","buy"]. stocks that kept for 1 night
                    for dict_trade in a_trade_content:

                        # sell decision
                        ts_code = dict_trade["ts_code"]
                        sell = False

                        if dict_trade["hold_days"] >= setting["p_min_holdday"]:  # sellable = consider sell
                            if dict_trade["hold_days"] >= setting["p_max_holdday"]:  # must sell
                                sell = True
                                reason = "max_hold"
                            elif (setting_keep == "winner" and dict_trade["comp_chg"] < 1) or (setting_keep == "loser" and dict_trade["comp_chg"] > 1):
                                sell = True
                                reason = f"not {setting_keep}"
                            else:
                                reason = f"is {setting_keep}"
                        else:
                            reason = f"min_hold"

                        try:
                            tomorrow_open = df_tomorrow.at[ts_code, "open"]
                            tomorrow_close = df_tomorrow.at[ts_code, "close"]
                        except:  # tomorrow 停牌 and no information
                            tomorrow_open = dict_trade["today_close"]
                            tomorrow_close = dict_trade["today_close"]

                            sell = False
                            reason = reason + " not trading"
                            print("probably 停牌", ts_code)

                        if sell:  # Execute sell
                            shares = dict_trade["shares"]
                            realized_value = tomorrow_open * shares
                            dict_capital[setting_count]["cash"] = dict_capital[setting_count]["cash"] + realized_value

                            dict_trade_h[tomorrow]["sell"].append(
                                {"reason": reason, "rank_final": dict_trade["rank_final"], "buy_imp": dict_trade["buy_imp"], "ts_code": dict_trade["ts_code"], "name": dict_trade["name"], "hold_days": dict_trade["hold_days"], "buyout_price": dict_trade["buyout_price"],
                                 "today_open": tomorrow_open, "today_close": np.nan, "sold_price": tomorrow_open, "pct_chg": tomorrow_open / dict_trade["today_close"],
                                 "comp_chg": tomorrow_open / dict_trade["buyout_price"], "shares": shares, "value_open": realized_value, "value_close": np.nan, "port_cash": dict_capital[setting_count]["cash"]})

                        else:  # Execute hold
                            dict_trade_h[tomorrow]["hold"].append(
                                {"reason": reason, "rank_final": dict_trade["rank_final"], "buy_imp": dict_trade["buy_imp"], "ts_code": dict_trade["ts_code"], "name": dict_trade["name"], "hold_days": dict_trade["hold_days"] + 1, "buyout_price": dict_trade["buyout_price"],
                                 "today_open": tomorrow_open, "today_close": tomorrow_close, "sold_price": np.nan, "pct_chg": tomorrow_close / dict_trade["today_close"],
                                 "comp_chg": tomorrow_close / dict_trade["buyout_price"], "shares": dict_trade["shares"], "value_open": tomorrow_open * dict_trade["shares"], "value_close": tomorrow_close * dict_trade["shares"], "port_cash": dict_capital[setting_count]["cash"]})

                        # print out
                        if sell:
                            print(f"{setting_count} : " + '{0: <19}'.format("") + '{0: <9}'.format(f"sell {sold_count}"), (f"{ts_code}"))
                            sold_count = sold_count + 1
                        else:
                            print(f"{setting_count} : " + '{0: <0}'.format("") + '{0: <9}'.format(f"hold {hold_count}"), (f"{ts_code}"))
                            hold_count = hold_count + 1

            now = Backtest_Util.print_and_time(setting_count=setting_count, phase=f"SELL AND HOLD", dict_trade_h_hold=dict_trade_h[tomorrow]["hold"], dict_trade_h_buy=dict_trade_h[tomorrow]["buy"], dict_trade_h_sell=dict_trade_h[tomorrow]["sell"], p_maxsize=p_maxsize, a_time=a_time, prev_time=now)

            # PORTFOLIO BUY BEGIN
            buyable_size = p_maxsize - len(dict_trade_h[tomorrow]["hold"])
            if buyable_size > 0 and int(market_trend) >= 0.7:

                # 6.4 PORTFOLIO BUY SCORE/RANK
                dict_group_instance_weight = Util.c_group_score_weight()
                for column, a_weight in setting["s_weight_matrix"].items():
                    # column use group rank
                    if a_weight[2] != 1:

                        # trick to store and be used for next couple settings running on the same day
                        if column in dict_weight_accelerator:  # TODO maybe add this to dt_date in general once all important indicators are found
                            print("accelerated")
                            df_today_mod = dict_weight_accelerator[column]
                        else:
                            # 1. iterate to see replace value
                            print("NOT accelerated")
                            for group, instance_array in Util.c_groups_dict(assets=["E"], a_ignore=["asset", "industry3"]).items():
                                dict_replace = {}
                                df_today_mod["rank_" + column + "_" + group] = df_today_mod[group]  # to be replaced later by int value
                                for instance in instance_array:
                                    try:
                                        dict_replace[instance] = df_group_instance_all[group + "_" + str(instance)].at[int(today), column]
                                    except Exception as e:
                                        print("(Could be normal if none of these group are trading on that day) ERROR on", today, group, instance)
                                        print(e)
                                        traceback.print_exc()
                                        dict_replace[instance] = 0
                                df_today_mod["rank_" + column + "_" + group].replace(to_replace=dict_replace, inplace=True)
                            dict_weight_accelerator[column] = df_today_mod.copy()

                        # 2. calculate group score
                        df_today_mod["rank_" + column + "_group"] = 0.0
                        for group in Util.c_groups_dict(assets=["E"], a_ignore=["asset", "industry3"]):
                            try:
                                df_today_mod["rank_" + column + "_group"] = df_today_mod["rank_" + column + "_group"] + df_today_mod["rank_" + column + "_" + group] * dict_group_instance_weight[group]
                            except Exception as e:
                                print(e)

                    else:  # column does not use group rank
                        df_today_mod["rank_" + column + "_group"] = 0.0

                    # 3. Create Indicator Rank= indicator_asset+indicator_group
                    df_today_mod[column + "_rank"] = (df_today_mod[column] * a_weight[2] + df_today_mod["rank_" + column + "_group"] * (1 - a_weight[2])).rank(ascending=a_weight[0])

                # 4. Create Rank Final = indicator1+indicator2+indicator3
                df_today_mod["rank_final"] = sum([df_today_mod[column + "_rank"] * a_weight[1]
                                                  for column, a_weight in setting["s_weight_matrix"].items()])  # if final rank is na, nsmallest will not select anyway
                now = Backtest_Util.print_and_time(setting_count=setting_count, phase=f"BUY FINAL RANK", dict_trade_h_hold=dict_trade_h[tomorrow]["hold"], dict_trade_h_buy=dict_trade_h[tomorrow]["buy"], dict_trade_h_sell=dict_trade_h[tomorrow]["sell"], p_maxsize=p_maxsize, a_time=a_time,
                                                   prev_time=now)

                # 6.8 PORTFOLIO BUY FILTER: SELECT PERCENTILE 1
                df_select = Backtest_Util.try_select(select_from_df=df_today_mod, select_size=buyable_size * 3, select_by=setting["f_percentile_column"])

                # 6.6 PORTFOLIO BUY ADD_POSITION: FALSE
                # df_select = df_select[~df_select["ts_code"].isin(df_today_portfolio["ts_code"])]
                df_select = df_select[~df_select.index.isin([trade_info["ts_code"] for trade_info in dict_trade_h[tomorrow]["hold"]])]

                # 6.7 PORTFOLIO BUY SELECT TOMORROW: select Stocks that really TRADES
                df_select_tomorrow = df_tomorrow[df_tomorrow.index.isin(df_select.index)]

                # carry final rank, otherwise the second select will not be able to select
                df_select_tomorrow[["rank_final"]] = df_today_mod[["rank_final"]]

                # 6.8 PORTFOLIO BUY FILTER: SELECT PERCENTILE 2
                df_select_tomorrow = Backtest_Util.try_select(select_from_df=df_select_tomorrow, select_size=buyable_size, select_by=setting["f_percentile_column"])
                now = Backtest_Util.print_and_time(setting_count=setting_count, phase=f"BUY SELECT", dict_trade_h_hold=dict_trade_h[tomorrow]["hold"], dict_trade_h_buy=dict_trade_h[tomorrow]["buy"], dict_trade_h_sell=dict_trade_h[tomorrow]["sell"], p_maxsize=p_maxsize, a_time=a_time, prev_time=now)

                # 6.11 BUY EXECUTE:
                current_capital = dict_capital[setting_count]["cash"]
                single_purchase = round(current_capital / buyable_size, 1)
                for (ts_code, row), hold_count in zip(df_select_tomorrow.iterrows(), range(1, len(df_select_tomorrow) + 1)):
                    # 6.9 BUY WEIGHT: # TODO weight vs score, buy good vs buy none
                    # if trend >0.7, add weight
                    # the market_trend weight = portfolio/cash ratio

                    buy_open = row["open"]
                    buy_close = row["close"]
                    buy_pct_chg_comp_chg = buy_close / buy_open
                    buy_imp = int((row["open"] == row["close"]))
                    share = single_purchase // buy_open
                    value_open = share * buy_open
                    value_close = share * buy_close
                    dict_capital[setting_count]["cash"] = dict_capital[setting_count]["cash"] - value_open

                    dict_trade_h[tomorrow]["buy"].append(
                        {"reason": np.nan, "rank_final": row["rank_final"], "buy_imp": buy_imp, "ts_code": ts_code, "name": row["name"], "hold_days": 1, "buyout_price": buy_open, "today_open": buy_open, "today_close": buy_close, "sold_price": float("nan"), "pct_chg": buy_pct_chg_comp_chg,
                         "comp_chg": buy_pct_chg_comp_chg, "shares": share, "value_open": value_open,
                         "value_close": value_close, "port_cash": dict_capital[setting_count]["cash"]})

                    print(setting_count, ": ", '{0: <9}'.format("") + f"buy {hold_count} {ts_code}")

                now = Backtest_Util.print_and_time(setting_count=setting_count, phase=f"BUY EXECUTE", dict_trade_h_hold=dict_trade_h[tomorrow]["hold"], dict_trade_h_buy=dict_trade_h[tomorrow]["buy"], dict_trade_h_sell=dict_trade_h[tomorrow]["sell"], p_maxsize=p_maxsize, a_time=a_time, prev_time=now)


            else:  # to not buy today
                if market_trend <= 0.7:
                    print("not buying today because no trend")
                else:
                    print("not buying today because no space")

            # 7 PRINT TIME
            if setting["print_log"]:
                print("=" * 50)
                [print(string) for string in a_time]
                print("=" * 50)

    # 10 STEP CREATE REPORT AND BACKTEST OVERVIEW
    a_summary_overview = []
    a_summary_setting = []
    Util.sound("saving.mp3")
    for setting, setting_count, dict_trade_h in zip(settings, range(0, len(settings)), a_dict_trade_h):
        try:
            now = mytime.time()
            df_trade_h, df_portfolio_overview, df_setting = Backtest_Util.report_portfolio(setting_original=setting, dict_trade_h=dict_trade_h, df_stock_market_all=df_stock_market_all, backtest_start_time=backtest_start_time, setting_count=setting_count)
            print("REPORT PORTFOLIO TIME:", mytime.time() - now)

            a_summary_overview.append(df_portfolio_overview.head(1))
            a_summary_setting.append(df_setting)

            # sendmail
            if setting["send_mail"]:
                df_last_simulated_trade = df_trade_h.loc[df_trade_h["trade_date"] == last_simulated_date, ["trade_type", "name"]]
                str_last_simulated_trade = df_last_simulated_trade.to_string(header=False, index=True)
                str_last_simulated_trade = str(last_simulated_date) + ": \n" + setting["name"] + "\n" + str_last_simulated_trade
                Util.send_mail(str_last_simulated_trade)

        except Exception as e:
            Util.sound("error.mp3")
            print("summarizing ERROR:", e)
            traceback.print_exc()

    # read saved backtest summary
    path_summary = "Market/CN/Backtest_Multiple/Backtest_Summary.xlsx"
    summary_writer = pd.ExcelWriter(path_summary, engine='xlsxwriter')
    try:  # if there is an existing history
        xls = pd.ExcelFile(path_summary)
        df_overview_h = pd.read_excel(xls, sheet_name="Overview")
        df_setting_h = pd.read_excel(xls, sheet_name="Setting")
    except:  # if there is no existing history
        df_overview_h, df_setting_h = pd.DataFrame(), pd.DataFrame()

    df_summary_overview_new = pd.concat(a_summary_overview[::-1] + [df_overview_h], sort=False, ignore_index=True)
    df_summary_setting_new = pd.concat(a_summary_setting[::-1] + [df_setting_h], sort=False, ignore_index=True)
    df_summary_overview_new.to_excel(summary_writer, sheet_name="Overview", index=False, encoding='utf-8_sig')
    df_summary_setting_new.to_excel(summary_writer, sheet_name="Setting", index=False, encoding='utf-8_sig')

    Util.pd_writer_save(summary_writer, path_summary)
    return


def change_dict_score_weight(dict_score, a_weight=[]):
    new_dict = {}
    for (key, array), weight in zip(dict_score.items(), a_weight):
        new_dict[key] = [array[0], weight]

    return new_dict


def backtest_multiple(loop_indicator=1):
    # Initialize setting array
    a_settings = []
    setting_base = {
        # general = Non changeable through one run
        "start_date": "20150201",
        "end_date": Util.today(),
        "freq": "D",
        "market": "CN",
        "assets": ["E"],  # E,I,FD

        # meta
        "id": "",  # datetime.now().strftime('%Y%m%d%H%M%S'), but done in backtest_once_loop
        "name": "StrategyX",  # readable strategy name used when describing a strategy
        "change": "",  # placeholder for future fill. Lazy way to check what has been changed in the last setting
        "send_mail": False,
        "print_log": True,

        # buy focus = Select.
        "trend": False,  # possible values: False(all days),trend2,trend3,trend240. Basically trend shown on all_stock_market.csv
        "f_percentile_column": "rank_final",  # {} empty means focus on all percentile. always from small to big. 0%-20% is small.    80%-100% is big. (0 , 18),(18, 50),(50, 82),( 82, 100)
        "f_query": [],  # ,'period > 240' is ALWAYS THERE FOR SPEED REASON, "trend > 0.2", filter everything from group str to price int #TODO create custom ffilter
        "s_weight_matrix": {  # ascending True= small, False is big
            # "pct_chg": [False, 0.2, 1],  # very important
            # "total_mv": [True, 0.1, 1],  # not so useful for this strategy, not more than 10% weight
            # "turnover_rate": [True, 0.1, 0.5],
            # "ivola": [True, 0.4, 1],  # seems important
            "trend": [False, 100, 1],  # very important for this strategy
            "pct_chg": [True, 5, 1],  # very important for this strategy
            "pgain2": [True, 3, 1],  # very important for this strategy
            "pgain5": [True, 2, 1],  # very important for this strategy
            # "pgain10": [True, 3, 1],  # very important for this strategy
            # "pgain20": [True, 2, 1],  # very important for this strategy
            # "pgain60": [True, 1, 1],  # very important for this strategy
            # "pgain240": [True, 1, 1],  # very important for this strategy
            # "candle_net_pos": [True, 0.1, 1],
        },

        # bool: ascending True= small, False is big
        # int: indicator_weight: how each indicator is weight against other indicator. e.g. {"pct_chg": [False, 0.8, 0.2, 0.0]}  =》 df["pct_chg"]*0.8 + df["trend"]*0.2
        # int: asset_weight: how each asset indicator is weighted against its group indicator. e.g. {"pct_chg": [False, 0.8, 0.2, 0.0]}  =》 df["pct_chg"]*0.2+df["pct_chg_group"]*0.8. empty means no group weight
        # int: random_weight: random number spread to be added to asset # TODO add small random weight to each

        # portfolio
        "p_capital": 50000,  # start capital
        # "p_trading_fee": 0.0002,  # 1==100%
        "p_maxsize": 12,  # not too low, otherwise volatility too big
        "p_min_holdday": 1,  # Start consider sell. 0 means trade on next day, aka T+1， = Hold stock for 1 night， 1 means hold for 2 nights. Preferably 0,1,2 for day trading
        "p_max_holdday": 1,  # MUST sell no matter what.
        "p_feedbackday": 60,
        "p_weight": False,
        "p_keep": False,  # options False, "winner","loser"
        "p_add_position": False,
        "p_compare": [["I", "000001.SH"]],  # ["I", "CJ000001.SH"],  ["I", "399001.SZ"], ["I", "399006.SZ"]   compare portfolio against other performance
    }

    # temp
    a_columns = [["total_mv", True, 0.05, 1], ["p5", False, 0.15, 1], ["pct_chg", False, 0.25, 1], ["trend", False, 0.25, 1], ["pjump_up", False, 0.05, 1], ["ivola", True, 0.05, 1], ["candle_net_pos", False, 0.15, 1]]  #
    a_columns = [["pct_chg", False, 1, 1], ["trend", False, 1, 1], ["trend2", False, 1, 1], ["trend10", False, 1, 1], ["candle_net_pos", False, 1, 1], ["candle_net_pos5", False, 1, 1], ["pjump_up", False, 1, 1], ["pjump_up10", False, 1, 1], ["ivola", True, 1, 1], ["ivola5", True, 1, 1],
                 ["pgain2", True, 1, 1], ["pgain5", True, 1, 1], ["pgain60", True, 1, 1], ["pgain240", True, 1, 1], ["turnover_rate", True, 1, 1], ["turnover_rate_pct2", True, 1, 1], ["pb", True, 1, 1], ["dv_ttm", True, 1, 1], ["ps_ttm", True, 1, 1], ["pe_ttm", True, 1, 1], ["total_mv", 1, 1]]  #

    # settings creation
    for max_hold in [1, 2, 3, 5, 8]:
        for p_keep in [False, "winner", "loser"]:
            setting_copy = copy.deepcopy(setting_base)
            s_weight_matrix = {  # ascending True= small, False is big
                # "pct_chg": [False, 0.2, 1],  # very important
                # "total_mv": [True, 0.1, 1],  # not so useful for this strategy, not more than 10% weight
                # "turnover_rate": [True, 0.1, 0.5],
                # "ivola": [True, 0.4, 1],  # seems important
                "trend": [False, 100, 1],  # very important for this strategy
                "pct_chg": [True, 5, 1],  # very important for this strategy
                "pgain2": [True, 3, 1],  # very important for this strategy
                "pgain5": [True, 2, 1],  # very important for this strategy
            }
            setting_copy["s_weight_matrix"] = s_weight_matrix
            setting_copy["p_keep"] = p_keep
            setting_copy["p_max_holdday"] = max_hold
            a_settings.append(setting_copy)
            print(setting_copy["s_weight_matrix"])

    print("Total Settings:", len(a_settings))
    backtest_once(settings=a_settings)


if __name__ == '__main__':
    try:

        pr = cProfile.Profile()
        pr.enable()

        backtest_multiple(5)

        pr.disable()
        pr.print_stats(sort='file')

        pass
    except Exception as e:
        traceback.print_exc()
        Util.sound("error.mp3")
