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
import msvcrt
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
    trade_dates = DB.get_trade_date(start_date=start_date, end_date=end_date, freq=freq)
    df_stock_market_all = DB.get_stock_market_all(market)
    df_group_instance_all = DB.get_group_instance_all(assets=["E"])
    last_simulated_date = df_stock_market_all.index[-1]
    df_today_accelerator = pd.DataFrame()

    # 0.4 PREPARATION- INITIALIZE Changeables for the loop
    a_portfolios = [{"a_port_h": [], "a_trade_h": []} for _ in settings]

    # Main Loop
    for today, tomorrow in zip(trade_dates["trade_date"], trade_dates["trade_date"].shift(-1).fillna(-1).astype(int)):
        if break_loop:
            print("BREAK loop")
            Util.sound("break.mp3")
            time.sleep(3)
            break

        # initialize df_date
        if tomorrow == -1:
            # if tomorrow==-1:
            print("last day reached")
            tomorrow = int(DB.get_next_trade_date(freq=freq))
            df_today = df_today_accelerator
            df_tomorrow = df_today_accelerator.copy()
            df_tomorrow[["open", "high", "low", "close", "pct_chg"]] = np.nan

        else:
            # df_atomorrow = DB.get_date(trade_date=atomorrow, assets=assets, freq="D", market=market)
            df_tomorrow = DB.get_date(trade_date=tomorrow, assets=assets, freq="D", market=market)
            df_today = DB.get_date(trade_date=today, assets=assets, freq="D", market=market) if df_today_accelerator.empty else df_today_accelerator

            # df_tomorrow_accelerator = df_atomorrow
            df_today_accelerator = df_tomorrow
            dict_weight_accelerator = {}

        # FOR EACH DAY LOOP OVER SETTING N
        for setting, setting_count, portfolio in zip(settings, range(0, len(settings)), a_portfolios):
            print()
            print()
            print(f"{setting_count} : START: TODAY EVENING=ANALYZE=> {today}. TOMORROW MORNING=TRADE/ACTION=> {tomorrow}")

            a_time = []
            now = mytime.time()
            setting["id"] = datetime.now().strftime('%Y%m%d%H%M%S')

            p_maxsize = setting["p_maxsize"]
            p_feedbackday = setting["p_feedbackday"]
            market_trend = df_stock_market_all.at[int(today), setting["trend"]] if setting["trend"] else 1
            df_today_portfolio = portfolio["a_port_h"][-1] if len(portfolio["a_port_h"]) > 0 else pd.DataFrame(columns=Util.c_port_h_label())  # today portfolio shall be modified based on yesterdays portfolio

            # 1.1 FILTER  IPO
            df_today_mod = df_today[df_today["period"] > setting["f_ipo"]]  # df_today mod can be modified

            # 1.1 FILTER  GROUP
            if setting["f_groups"] == Util.c_groups_dict(assets):  # focus all groups = default df_date_today
                pass
            else:  # some groups are filtered out
                group_boolean = True
                for column, instance_array in setting["f_groups"].items():
                    print(setting_count, today, "FOCUS on group", column, instance_array)
                    group_boolean = group_boolean & df_today_mod[column].isin(instance_array)

                # put boolean mask on df_today_mod
                df_today_mod = df_today_mod[group_boolean]
            now = Backtest_Util.print_and_time(setting_count=setting_count, phase="META AND BASIC FILTER", df_today=df_today_mod, df_tomorrow=df_tomorrow, df_today_portfolios=df_today_portfolio, p_maxsize=p_maxsize, a_time=a_time, prev_time=now)

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

            # 6.2 PORTFOLIO SELL EXECUTE trade_h
            if market_trend >= 0.7:  # if tomorrow trend is good, consider min_holdday
                df_helper = df_today_portfolio[df_today_portfolio["hold_days"] >= setting["p_min_holdday"]]
                a_sellable_assets = df_helper.index.to_numpy()  # to_numpy is IMPORTANT, otherwise bug
            else:  # if tomorrow trend is bad, sell no matter min_holdday
                a_sellable_assets = df_today_portfolio.index.to_numpy()  # to_numpy is IMPORTANT, otherwise bug

            sellcondition_proxy = True
            a_unsold_assets = []
            for (index, row), sold_counter in zip(df_today_portfolio.iterrows(), range(len(df_today_portfolio))):  # for each stock in portoflio
                if index in a_sellable_assets:  # sellable
                    if sellcondition_proxy:
                        try:  # stock trades tomorrow and sold
                            sell_price = df_tomorrow.at[index, "open"]  # IMPORTANT, use the day TOMORROW  price to sell
                            portfolio["a_trade_h"].append([tomorrow, "sell", row["name"], index, row["hold_days"] + 1, row["buyout_price"], sell_price, sell_price / row["buyout_price"], row["rank_final"], row["buy_imp"]])
                            print(setting_count, ": ", sold_counter, "sold", index)
                        except Exception as e:  # stock not trading tomorrow
                            a_unsold_assets.append(index)
                            print(setting_count, ": ", len(a_unsold_assets), "unsold", index)
                            pass
                    else:  # condition proxy pass
                        pass
                else:  # not sellable
                    a_unsold_assets.append(index)

            df_today_portfolio = df_today_portfolio[df_today_portfolio.index.isin(a_unsold_assets)]
            now = Backtest_Util.print_and_time(setting_count=setting_count, phase="SELL EXECUTE ", df_today=df_today_mod, df_tomorrow=df_tomorrow, df_today_portfolios=df_today_portfolio, p_maxsize=p_maxsize, a_time=a_time, prev_time=now)

            # 6.4 PORTFOLIO HOLD = unsellable and unsold stocks
            df_today_portfolio = Backtest_Util.hold_port_h_for_tomorrow(df_today_portfolio, df_tomorrow, tomorrow)
            [portfolio["a_trade_h"].append([tomorrow, "hold", row["name"], index, row["hold_days"] + 1, row["buyout_price"], float("nan"), row["comp_chg"], row["rank_final"], row["buy_imp"]])
             for index, row in df_today_portfolio.iterrows()]
            now = Backtest_Util.print_and_time(setting_count=setting_count, phase="HOLD EXECUTE", df_today=df_today_mod, df_tomorrow=df_tomorrow, df_today_portfolios=df_today_portfolio, p_maxsize=p_maxsize, a_time=a_time, prev_time=now)

            # PORTFOLIO BUY BEGIN
            if len(df_today_portfolio) != p_maxsize and int(market_trend) >= 0.7:

                # 6.3 PORTFOLIO BUY FILTER:TECHNICAL
                if setting["f_ma"]:
                    print(setting_count, today, "BEFORE FILTER MA", len(df_today_mod))
                    ma_condition = True
                    for key, value in setting["f_ma"].items():
                        if value:
                            ma_condition = ma_condition & (df_today_mod["close"] > df_today_mod[key])
                        else:
                            ma_condition = ma_condition & (df_today_mod["close"] < df_today_mod[key])
                    df_today_mod = df_today_mod[ma_condition]
                    print(setting_count, today, "AFTER FILTER MA", len(df_today_mod))

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
                now = Backtest_Util.print_and_time(setting_count=setting_count, phase="BUY FINAL RANK", df_today=df_today_mod, df_tomorrow=df_tomorrow, df_today_portfolios=df_today_portfolio, p_maxsize=p_maxsize, a_time=a_time, prev_time=now)

                # 6.2 PORTFOLIO BUY FILTER: GET PERCENTILE
                # try:
                #     percentfile_column_name = setting["f_percentile_column"]
                #     helper_column = np.array(df_today_mod[percentfile_column_name])
                #     percentile_between = np.nanpercentile(a=helper_column, q=[setting["f_percentile_min"], setting["f_percentile_max"]], overwrite_input=True)
                #     # print(setting_count, ": ", "BUY PERCENTILE", len(df_today_mod), ":", setting["f_percentile_column"], setting["f_percentile_min"], setting["f_percentile_max"], "translates to", percentile_between)
                #     df_today_mod = df_today_mod[df_today_mod[percentfile_column_name].between(percentile_between[0], percentile_between[1])]
                # except Exception as e:
                #     print("ERROR select percentile")

                # 6.8 PORTFOLIO BUY FILTER: SELECT PERCENTILE 1
                df_select = Backtest_Util.try_select(select_from_df=df_today_mod, select_size=(p_maxsize - len(df_today_portfolio)) * 3, select_by=setting["f_percentile_column"])
                # 6.6 PORTFOLIO BUY ADD_POSITION: FALSE

                # df_select = df_select[~df_select["ts_code"].isin(df_today_portfolio["ts_code"])]
                df_select = df_select[~df_select.index.isin(df_today_portfolio.index)]

                # 6.7 Stocks that open and close price are not the same
                # condition_open_close = df_select["high"]!=df_select["low"]

                # 6.7 PORTFOLIO BUY SELECT TOMORROW: select Stocks that really TRADES
                df_select_tomorrow = df_tomorrow[df_tomorrow.index.isin(df_select.index)]

                # carry final rank, otherwise the second select will not be able to select
                df_select_tomorrow[["rank_final"]] = df_today_mod[["rank_final"]]

                # 6.8 PORTFOLIO BUY FILTER: SELECT PERCENTILE 2
                df_select_tomorrow = Backtest_Util.try_select(select_from_df=df_select_tomorrow, select_size=p_maxsize - len(df_today_portfolio), select_by=setting["f_percentile_column"])
                now = Backtest_Util.print_and_time(setting_count=setting_count, phase="BUY SELECT", df_today=df_today_mod, df_tomorrow=df_tomorrow, df_today_portfolios=df_today_portfolio, p_maxsize=p_maxsize, a_time=a_time, prev_time=now)

                # 6.9 BUY WEIGHT: # TODO weight vs score, buy good vs buy none
                # if trend >0.7, add weight
                # the market_trend weight = portfolio/cash ratio

                # carry calculated score to df_select_tomorrow, so that socre are visible when buy

                # 6.10 BUY EXECUTE port_h: They are selected, traded stocks for tomorrow
                if not df_select_tomorrow.empty:  # buy nothing
                    # df_select_tomorrow.rename(mapper={"close":"buyout_price"},inplace=True)
                    df_select_tomorrow["trade_date"] = tomorrow
                    df_select_tomorrow["hold_days"] = 0
                    df_select_tomorrow["comp_chg"] = df_select_tomorrow["close"] / df_select_tomorrow["open"]  # Important BUY AT OPEN, not CLOSE
                    df_select_tomorrow["pct_chg"] = ((df_select_tomorrow["close"] / df_select_tomorrow["open"]) - 1) * 100
                    df_select_tomorrow["buy_imp"] = ((df_select_tomorrow["open"] == df_select_tomorrow["close"])).astype(int)
                    df_select_tomorrow["buyout_price"] = df_select_tomorrow["open"]
                    # df_select_tomorrow = df_select_tomorrow[ [ Util.c_trade_h_label() not in ["sold_price","trade_type"] ]]
                    # df_select_tomorrow = df_select_tomorrow[Util.c_port_h_label() + ["rank_final"]]
                    # df_select_tomorrow.drop(labels=[x for x in df_select_tomorrow.columns if x not in Util.c_port_h_label() + ["rank_final"]],axis=1, inplace=True)
                    now = Backtest_Util.print_and_time(setting_count=setting_count, phase="BUY EXECUTE 0", df_today=df_today_mod, df_tomorrow=df_tomorrow, df_today_portfolios=df_today_portfolio, p_maxsize=p_maxsize, a_time=a_time, prev_time=now)

                    if df_today_portfolio.empty:  # speed up append process if today portfolio was empty
                        df_today_portfolio = df_select_tomorrow[Util.c_port_h_label() + ["rank_final"]]
                    else:
                        # df_today_portfolio = df_today_portfolio.append(df_select_tomorrow[Util.c_port_h_label() + ["rank_final"]], sort=False, ignore_index=True)  # expensive
                        df_today_portfolio = pd.concat(objs=[df_today_portfolio, df_select_tomorrow], join="inner", sort=False, ignore_index=False)
                        # df_today_portfolio.set_index(keys="ts_code",drop=True,inplace=True)

                    [print(setting_count, ": ", count, "bought", ts_code) for ts_code, count in zip(df_select_tomorrow.index, range(0, len(df_select_tomorrow)))]
                now = Backtest_Util.print_and_time(setting_count=setting_count, phase="BUY EXECUTE 1", df_today=df_today_mod, df_tomorrow=df_tomorrow, df_today_portfolios=df_today_portfolio, p_maxsize=p_maxsize, a_time=a_time, prev_time=now)

                # trade_h: They are selected, traded stocks for tomorrow
                [portfolio["a_trade_h"].append([tomorrow, "buy", row["name"], index, 0, row["buyout_price"], float("nan"), row["comp_chg"], row["rank_final"], row["buy_imp"]])
                 for index, row in df_select_tomorrow.iterrows()]
                now = Backtest_Util.print_and_time(setting_count=setting_count, phase="BUY EXECUTE 2", df_today=df_today_mod, df_tomorrow=df_tomorrow, df_today_portfolios=df_today_portfolio, p_maxsize=p_maxsize, a_time=a_time, prev_time=now)


            else:  # to not buy today
                if market_trend <= 0.7:
                    print("not buying today because no trend")
                else:
                    print("not buying today because no space")

            # 6.9 PORTFOLIO UPDATE: HISTORY
            portfolio["a_port_h"].append(df_today_portfolio)

            # 7 PRINT TIME
            if setting["print_log"]:
                print("=" * 50)
                [print(string) for string in a_time]
                print("=" * 50)

    # 10 STEP CREATE REPORT AND BACKTEST OVERVIEW
    a_summary_overview = []
    a_summary_setting = []
    for setting, setting_count, portfolio in zip(settings, range(0, len(settings)), a_portfolios):
        try:
            now = mytime.time()
            df_trade_h, df_portfolio_overview, df_setting = Backtest_Util.report_portfolio(setting, portfolio["a_port_h"], portfolio["a_trade_h"], df_stock_market_all, backtest_start_time, setting_count=setting_count)
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
    setting = {
        # general = Non changeable through one run
        "start_date": "20200101",
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
        "f_ipo": 240,  # only consider stocks above that period. New IPO stocks less than a year not considered
        "f_groups": {},  # {} empty means focus on all groups, {"industry_L1": ["医疗设备","blablabla"]} means from over group, focus on that
        "f_percentile_column": "rank_final",  # {} empty means focus on all percentile. always from small to big. 0%-20% is small.    80%-100% is big. (0 , 18),(18, 50),(50, 82),( 82, 100)
        # "f_percentile_min": 0.0,
        # "f_percentile_max": 10, #can improve code speed
        "f_ma": {},  # False is better. {}: no filter means all stocks. True: close > ma, False: close < ma. e.g. "close_m5":False. In general, False is better

        # buy filter = Remove

        # buy sort

        "s_weight_matrix": {},
        # bool: ascending True= small, False is big
        # int: indicator_weight: how each indicator is weight against other indicator. e.g. {"pct_chg": [False, 0.8, 0.2, 0.0]}  =》 df["pct_chg"]*0.8 + df["trend"]*0.2
        # int: asset_weight: how each asset indicator is weighted against its group indicator. e.g. {"pct_chg": [False, 0.8, 0.2, 0.0]}  =》 df["pct_chg"]*0.2+df["pct_chg_group"]*0.8. empty means no group weight
        # int: random_weight: random number spread to be added to asset # TODO add small random weight to each

        # portfolio

        # "p_trading_fee": 0.0002,  # 1==100%
        # "p_money": 10000,
        "p_maxsize": 12,  # not too low, otherwise volatility too big
        "p_min_holdday": 0,  # 0 means trade on next day, aka T+1， = Hold stock for 1 night， 1 means hold for 2 nights. Preferably 0,1,2 for day trading
        # "p_max_holdday": 100000,
        "p_feedbackday": 60,
        "p_weight": False,
        # "p_rebalance": False,
        "p_add_position": False,
        # "p_stop_lose": 0.5,  # sell stock if comp_gain < 0.5 times of initial value
        # "p_stop_win": 100,  # sell stock if comp_gain > 100 times of initial value
        "p_compare": [["I", "CJ000001.SH"], ["I", "000001.SH"], ["I", "399001.SZ"], ["I", "399006.SZ"]],  # compare portfolio against other performance
    }

    # ascending True= small, False is big
    s_weight_matrix = {
        # "candle_net_pos": [False, 0.1, 0.5],
        "pct_chg": [False, 0.2, 1],  # very important
        "total_mv": [True, 0.1, 1],  # not so useful for this strategy, not more than 10% weight
        # "turnover_rate": [True, 0.1, 0.5],
        "ivola": [True, 0.4, 1],  # seems important
        "trend": [False, 0.3, 1],  # very important for this strategy
    }

    # TODO add rsi and trend to all assets
    setting["s_weight_matrix"] = s_weight_matrix

    # setting that depend on other seetings. DO NOT CHANGE
    if setting["f_groups"] == {}:
        setting["f_groups"] = Util.c_groups_dict(setting["assets"])

    # Initialize setting array
    a_settings = []

    a_columns = [["total_mv", True, 0.05, 1], ["p5", False, 0.15, 1], ["pct_chg", False, 0.25, 1], ["trend", False, 0.25, 1], ["pjump_up", False, 0.05, 1], ["ivola", True, 0.05, 1], ["candle_net_pos", False, 0.15, 1]]  #
    a_columns = [["pct_chg", False, 1, 1], ["trend", False, 1, 1], ["candle_net_pos", False, 1, 1], ["pjump_up", False, 1, 1], ["ivola", True, 1, 1], ["pgain2", True, 1, 1], ["pgain5", True, 1, 1], ["turnover_rate", True, 1, 1], ["pb", True, 1, 1], ["total_mv", 1, 1]]  #

    for label1, label2 in combinations(a_columns, 2):

        for label1_true in [True, False]:
            for label2_true in [True, False]:
                setting_copy = copy.deepcopy(setting)

                print(label1, label1_true)
                s_weight_matrix = {
                    label1[0]: [label1_true, 1, 1],
                    label2[0]: [label2_true, 1, 1]
                }

                setting_copy["s_weight_matrix"] = s_weight_matrix
                a_settings.append(setting_copy)
                print(s_weight_matrix)

    print("Total Settings:", len(a_settings))
    backtest_once(settings=a_settings)


if __name__ == '__main__':
    try:

        pr = cProfile.Profile()
        pr.enable()

        backtest_multiple(5)

        pr.disable()
        # pr.print_stats(sort='file')

        pass
    except Exception as e:
        traceback.print_exc()
        Util.sound("error.mp3")
