import tushare as ts
import pandas as pd
import numpy as np
import time as mytime
import time
import DB
from itertools import combinations
from itertools import permutations
import LB
from datetime import datetime
import traceback
from scipy.stats import gmean
import copy
import cProfile
import operator
import threading
import ICreate
import matplotlib
from numba import njit
from numba import jit
from sympy.utilities.iterables import multiset_permutations
pd.options.mode.chained_assignment = None  # default='warn'
pro = ts.pro_api('c473f86ae2f5703f58eecf9864fa9ec91d67edbc01e3294f6a4f9c32')
ts.set_token("c473f86ae2f5703f58eecf9864fa9ec91d67edbc01e3294f6a4f9c32")


def btest_portfolio(setting_original, dict_trade_h, df_stock_market_all, backtest_start_time, setting_count):
    #init
    current_trend = LB.c_bfreq()
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
            setting[key] = LB.groups_dict_to_string_iterable(setting[key])
        elif type(value) in [list, np.ndarray]:
            if (key == "p_compare"):
                setting["p_compare"] = ', '.join([x[1] for x in setting["p_compare"]])
            else:
                setting[key] = ', '.join([str(x) for x in value])

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

    # merge with all dates (in case some days did not trade)
    df_trade_h = pd.merge(df_merge_helper, df_trade_h, how='left', on=["trade_date"], suffixes=["", ""], sort=False)

    #check if all index are valid
    if np.nan in df_trade_h["ts_code"] or float("nan") in df_trade_h:
        print("file has nan ts_codes")
        raise ValueError

    #create chart
    df_port_c = df_trade_h.groupby("trade_date").agg("mean")  # TODO remove inefficient apply and groupby and loc
    df_port_c=df_port_c.iloc[:-1]#exclude last row because last row predicts future
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

    # chart add competitor
    for competitor in p_compare:
        df_port_c = DB.add_asset_comparison(df=df_port_c, freq=setting["freq"], asset=competitor[0], ts_code=competitor[1], a_compare_label=["pct_chg"])
        df_port_c["comp_chg_" + competitor[1]] = ICreate.column_add_comp_chg(df_port_c["pct_chg_" + competitor[1]])

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
    df_port_overview.at[0, "asset_sell_winrate"] = len(df_trade_h.loc[(df_trade_h["trade_type"] == "sell") & (df_trade_h["comp_chg"] > 1)]) / len(df_trade_h.loc[(df_trade_h["trade_type"] == "sell")])
    df_port_overview.at[0, "all_daily_winrate"] = len(df_port_c.loc[df_port_c["all_pct_chg"] >= 1]) / len(df_port_c)

    df_port_overview.at[0, "asset_sell_gmean"] = gmean(df_trade_h.loc[(df_trade_h["trade_type"] == "sell"), "comp_chg"].dropna()) #everytime you sell a stock
    df_port_overview.at[0, "all_gmean"] = gmean(df_port_c["all_pct_chg"].dropna()) #everyday

    df_port_overview.at[0, "asset_sell_pct_chg"] = df_trade_h.loc[(df_trade_h["trade_type"] == "sell"), "comp_chg"].mean() #everytime you sell a stock
    df_port_overview.at[0, "all_pct_chg"] = df_port_c["all_pct_chg"].mean() #everyday

    df_port_overview.at[0, "asset_sell_pct_chg_std"] = df_trade_h.loc[(df_trade_h["trade_type"] == "sell"), "comp_chg"].std()
    df_port_overview.at[0, "all_pct_chg_std"] = df_port_c["all_pct_chg"].std()

    try:
        df_port_overview.at[0, "all_comp_chg"] = df_port_c.at[df_port_c["all_comp_chg"].last_valid_index(), "all_comp_chg"]
    except:
        df_port_overview.at[0, "all_comp_chg"] = float("nan")

    df_port_overview.at[0, "port_beta"] = LB.calculate_beta(df_port_c["all_pct_chg"], df_port_c["pct_chg" + beta_against])
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
    for i in range(len(p_compare), len(p_compare)):
        competitor_ts_code = p_compare[i][1]
        df_port_overview.at[i + 1, "strategy"] = competitor_ts_code
        df_port_overview.at[i + 1, "pct_chg_mean"] = df_port_c["pct_chg_" + competitor_ts_code].mean()
        df_port_overview.at[i + 1, "pct_chg_std"] = df_port_c["pct_chg_" + competitor_ts_code].std()
        df_port_overview.at[i + 1, "winrate"] = len(df_port_c[(df_port_c["pct_chg_" + competitor_ts_code] >= 0) & (df_port_c["pct_chg_" + competitor_ts_code].notna())]) / len(df_port_c["pct_chg_" + competitor_ts_code].notna())
        df_port_overview.at[i + 1, "period"] = len(df_port_c) - df_port_c["pct_chg_" + competitor_ts_code].isna().sum()
        df_port_overview.at[i + 1, "pct_days_involved"] = 1
        df_port_overview.at[i + 1, "comp_chg"] = df_port_c.at[df_port_c["comp_chg_" + competitor_ts_code].last_valid_index(), "comp_chg_" + competitor_ts_code]
        df_port_overview.at[i + 1, "beta"] = LB.calculate_beta(df_port_c["pct_chg_" + competitor_ts_code], df_port_c["pct_chg" + beta_against])
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
    a_trend_label = [f"close.market.trend{x}" for x in current_trend if x != 1]
    a_trend_label = []
    df_port_c = df_port_c[["rank_final", "port_pearson", "port_size", "buy", "hold", "sell", "port_cash", "port_close", "all_close", "all_pct_chg", "all_comp_chg"] + ["pct_chg_" + x for x in [x[1] for x in p_compare]] + ["comp_chg_" + x for x in [x[1] for x in p_compare]] + a_trend_label]

    # write portfolio
    portfolio_path = "Market/CN/Backtest_Multiple/Result/Portfolio_" + str(setting["id"])
    LB.to_csv_feather(df=df_port_overview, a_path=LB.a_path(portfolio_path + "/overview"), skip_feather=True)
    LB.to_csv_feather(df=df_trade_h, a_path=LB.a_path(portfolio_path + "/trade_h"), skip_feather=True)
    LB.to_csv_feather(df=df_port_c, a_path=LB.a_path(portfolio_path + "/chart"), index_relevant=True, skip_feather=True)
    df_setting = pd.DataFrame(setting, index=[0])
    LB.to_csv_feather(df=df_setting, a_path=LB.a_path(portfolio_path + "/setting"), index_relevant=False, skip_feather=True)

    print("setting is", setting["s_weight1"])
    print("=" * 50)
    [print(string) for string in a_time]
    print("=" * 50)
    return [df_trade_h, df_port_overview, df_setting]


def btest_once(settings=[{}]):
    # inside functions
    class btest_break:  # static method inside class is required to break the loop because two threads
        btest_break = False

    def get_input():
        input('Press a key \n')  # no need to store input, any key will trigger break
        btest_break.btest_break = True

    @LB.try_ignore
    def try_select(select_from_df, select_size, select_by):
        return select_from_df.nsmallest(int(select_size), [select_by])

    def print_and_time(setting_count, phase, dict_trade_h_hold, dict_trade_h_buy, dict_trade_h_sell, p_maxsize, a_time, prev_time):
        print(f"{setting_count} : hold " + '{0: <5}'.format(len(dict_trade_h_hold)) + 'buy {0: <5}'.format(len(dict_trade_h_buy)) + 'sell {0: <5}'.format(len(dict_trade_h_sell)) + 'space {0: <5}'.format(p_maxsize - (len(dict_trade_h_hold) + len(dict_trade_h_buy))))
        now = mytime.time()
        a_time.append('{0: <25}'.format(phase) + f": {now - prev_time}")
        return now


    # READ FIRST:
    # Assume that all Analysis of stockmarket today happens at evening after stock market closed 15:00
    # Assume that all trades happen at the START of the tomorrow 9:30
    # Assume all sell at start of day keeps yesterdays close price
    # Assume all buy at start of day will be accounted for todays pcg_chg

    # 0 PREPARATION
    # 0.1 PREPARATION- META
    backtest_start_time = datetime.now()
    break_detector = threading.Thread(target=get_input, daemon=True)
    break_detector.start()

    # 0.2 PREPARATION- SETTING = static values that NEVER CHANGE during the loop
    start_date = settings[0]["start_date"]
    end_date = settings[0]["end_date"]
    freq = settings[0]["freq"]
    market = settings[0]["market"]
    assets = settings[0]["assets"]

    # 0.3 PREPARATION- INITIALIZE Iterables derived from Setting
    df_trade_dates = DB.get_trade_date(start_date=start_date, end_date=end_date, freq=freq)
    df_trade_dates["tomorrow"] = df_trade_dates.index.values
    df_trade_dates["tomorrow"] = df_trade_dates["tomorrow"].shift(-1).fillna(-1).astype(int)
    df_stock_market_all = DB.get_stock_market_all(market)
    df_group_instance_all = DB.preload_groups(assets=["E"])
    print("what", df_stock_market_all.index)
    last_simulated_date = df_stock_market_all.index[-1]
    df_today_accelerator = pd.DataFrame()

    # 0.4 PREPARATION- INITIALIZE Changeables for the loop
    a_dict_trade_h = [{df_trade_dates.index[0]: {"sell": [], "hold": [], "buy": []}} for _ in settings]  # trade history for each setting
    dict_capital = {setting_count: {"cash": settings[0]["p_capital"]} for setting_count in range(0, len(settings))}  # only used in iteration, saved in trade_h

    # Main Loop
    for today, tomorrow in zip(df_trade_dates.index, df_trade_dates["tomorrow"]):

        if btest_break.btest_break:
            print("BREAK loop")
            LB.sound("break.mp3")
            time.sleep(5)
            # adjust date if loop was break
            for setting in settings:
                setting["end_date"] = today
            break

        # initialize df_date
        if tomorrow == -1:
            print("last day reached")
            tomorrow = int(DB.get_next_trade_date(freq=freq))
            df_today = df_today_accelerator
            df_tomorrow = df_today_accelerator.copy()  # bug here?
            df_tomorrow[["high", "low", "close", "pct_chg"]] = np.nan
        else:
            df_tomorrow = DB.get_date(trade_date=tomorrow, assets=assets, freq="D", market=market)
            df_today = DB.get_date(trade_date=today, assets=assets, freq="D", market=market) if df_today_accelerator.empty else df_today_accelerator
            df_today_accelerator = df_tomorrow
            dict_weight_accelerator = {}

        # FOR EACH DAY LOOP OVER SETTING N
        for setting_count, (setting, dict_trade_h) in enumerate(zip(settings, a_dict_trade_h)):
            print("\n" * 3)
            a_time = []
            now = mytime.time()
            setting["id"] = datetime.now().strftime('%Y%m%d%H%M%S')

            a_p_maxsize = setting["p_maxsize"]
            p_maxsize = a_p_maxsize[0] if a_p_maxsize[1] else int((a_p_maxsize[0]/100) * len(df_today)) #if true take fixed value. False takes percentage of all trading stocks today

            p_feedbackday = setting["p_feedbackday"]
            market_trend = df_stock_market_all.at[int(today), setting["trend"]] if setting["trend"] else 1

            dict_trade_h[tomorrow] = {"sell": [], "hold": [], "buy": []}
            print('{0: <26}'.format("TODAY EVENING ANALYZE") + f"{today} stocks {len(df_today)}")
            print('{0: <26}'.format("TOMORROW MORNING TRADE") + f"{tomorrow} stocks {len(df_tomorrow)}")
            now = print_and_time(setting_count=setting_count, phase=f"INIT", dict_trade_h_hold=dict_trade_h[tomorrow]["hold"], dict_trade_h_buy=dict_trade_h[tomorrow]["buy"], dict_trade_h_sell=dict_trade_h[tomorrow]["sell"], p_maxsize=p_maxsize, a_time=a_time, prev_time=now)

            # 1.1 FILTER
            a_filter = True
            for column, a_op in setting["f_query_asset"].items():  # very slow and expensive for small operation because parsing the string takes long
                func = a_op[0]
                if a_op[2]:
                    argument2 = a_op[1]
                    print(f"filter {column} {func.__name__} {argument2}", )
                else:
                    argument2= df_today[a_op[1]] # compare against fixed value(True) or against another series(False)
                    print(f"filter {column} {func.__name__} {argument2.name}", )

                a_filter = a_filter & func(df_today[column], argument2)

            df_today_mod = df_today[a_filter]
            print("today after filter", len(df_today_mod))
            now = print_and_time(setting_count=setting_count, phase=f"FILTER", dict_trade_h_hold=dict_trade_h[tomorrow]["hold"], dict_trade_h_buy=dict_trade_h[tomorrow]["buy"], dict_trade_h_sell=dict_trade_h[tomorrow]["sell"], p_maxsize=p_maxsize, a_time=a_time, prev_time=now)

            # 2 ECONOMY
            # 3 FINANCIAL MARKET
            # 6 PORTFOLIO = ASSET BALANCING, EFFICIENT FRONTIER, LEAST BETA #TODO
            # 6.1 PORTFOLIO FEEDBACK = Analysis for performance feedback mechanism
            if p_feedbackday > 0:  # not yet to feedback
                p_feedbackday = p_feedbackday - 1
            else:  # feedback day
                # TODO add feedback
                p_feedbackday = 60

            # 6.2 PORTFOLIO SELL SELECT
            p_winner_abv = setting["p_winner_abv"]
            p_loser_und = setting["p_loser_und"]
            hold_count = 1
            sell_count = 1
            for trade_type, a_trade_content in dict_trade_h[today].items():  # NOTE here today means today morning trade
                if trade_type != "sell":  # == in ["hold","buy"]. last day stocks that was kept for over night
                    for dict_trade in a_trade_content:

                        # sell meta
                        ts_code = dict_trade["ts_code"]
                        hold_day_overnight = dict_trade["T+"] + 1  # simulates the night when deciding to sell tomorrow
                        sell = False

                        # sell decision
                        if hold_day_overnight >= setting["p_min_T+"]:  # sellable = consider sell
                            if hold_day_overnight >= setting["p_max_T+"]:  # must sell
                                sell = True
                                reason = f"> T+{hold_day_overnight}"
                            elif p_winner_abv:
                                if dict_trade["comp_chg"] > p_winner_abv:
                                    sell = True
                                    reason = f"sell winner comp_chg above {p_winner_abv}"
                            elif p_loser_und:
                                if dict_trade["comp_chg"] < p_loser_und:
                                    sell = True
                                    reason = f"sell loser comp_chg under {p_loser_und}"
                            else:
                                reason = f"sellable - but no"
                        else:
                            reason = f"< T+{hold_day_overnight}"

                        # try to get tomorrow price. if not ,then not trading
                        try:
                            tomorrow_open = df_tomorrow.at[ts_code, "open"]
                            tomorrow_close = df_tomorrow.at[ts_code, "close"]
                        except:  # tomorrow 停牌 and no information
                            tomorrow_open = dict_trade["today_close"]
                            tomorrow_close = dict_trade["today_close"]
                            sell = False
                            reason = reason + "- not trading"
                            print("probably 停牌", ts_code)

                        if sell:  # Execute sell
                            shares = dict_trade["shares"]
                            realized_value = tomorrow_open * shares
                            fee = setting["p_fee"] * realized_value
                            dict_capital[setting_count]["cash"] = dict_capital[setting_count]["cash"] + realized_value - fee

                            dict_trade_h[tomorrow]["sell"].append(
                                {"reason": reason, "rank_final": dict_trade["rank_final"], "buy_imp": dict_trade["buy_imp"], "ts_code": dict_trade["ts_code"], "name": dict_trade["name"], "T+": hold_day_overnight, "buyout_price": dict_trade["buyout_price"],
                                 "today_open": tomorrow_open, "today_close": np.nan, "sold_price": tomorrow_open, "pct_chg": tomorrow_open / dict_trade["today_close"],
                                 "comp_chg": tomorrow_open / dict_trade["buyout_price"], "shares": shares, "value_open": realized_value, "value_close": np.nan, "port_cash": dict_capital[setting_count]["cash"]})

                        else:  # Execute hold
                            dict_trade_h[tomorrow]["hold"].append(
                                {"reason": reason, "rank_final": dict_trade["rank_final"], "buy_imp": dict_trade["buy_imp"], "ts_code": dict_trade["ts_code"], "name": dict_trade["name"], "T+": hold_day_overnight, "buyout_price": dict_trade["buyout_price"],
                                 "today_open": tomorrow_open, "today_close": tomorrow_close, "sold_price": np.nan, "pct_chg": tomorrow_close / dict_trade["today_close"],
                                 "comp_chg": tomorrow_close / dict_trade["buyout_price"], "shares": dict_trade["shares"], "value_open": tomorrow_open * dict_trade["shares"], "value_close": tomorrow_close * dict_trade["shares"], "port_cash": dict_capital[setting_count]["cash"]})

                        # print out
                        if sell:
                            print(f"{setting_count} : " + '{0: <19}'.format("") + '{0: <9}'.format(f"sell {sell_count}"), (f"{ts_code}"))
                            sell_count = sell_count + 1
                        else:
                            print(f"{setting_count} : " + '{0: <0}'.format("") + '{0: <9}'.format(f"hold {hold_count}"), (f"{ts_code}"))
                            hold_count = hold_count + 1

            now = print_and_time(setting_count=setting_count, phase=f"SELL AND HOLD", dict_trade_h_hold=dict_trade_h[tomorrow]["hold"], dict_trade_h_buy=dict_trade_h[tomorrow]["buy"], dict_trade_h_sell=dict_trade_h[tomorrow]["sell"], p_maxsize=p_maxsize, a_time=a_time, prev_time=now)

            # PORTFOLIO BUY SELECT BEGIN
            buyable_size = p_maxsize - len(dict_trade_h[tomorrow]["hold"])
            if buyable_size > 0 and int(market_trend) >= 0.7 and len(df_today_mod)>0:

                # 6.4 PORTFOLIO BUY SCORE/RANK
                #if selet final rank by a defined criteria
                if setting["s_weight1"]:
                    dict_group_instance_weight = LB.c_group_score_weight()
                    for column, a_weight in setting["s_weight1"].items():
                        print("select column", column)
                        # column use group rank
                        if a_weight[2] != 1:

                            # trick to store and be used for next couple settings running on the same day
                            if column in dict_weight_accelerator:  # TODO maybe add this to dt_date in general once all important indicators are found
                                print("accelerated")
                                df_today_mod = dict_weight_accelerator[column]
                            else:
                                # 1. iterate to see replace value
                                print("NOT accelerated")
                                for group, instance_array in LB.c_groups_dict(assets=["E"], a_ignore=["asset", "industry3"]).items():
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
                            for group in LB.c_groups_dict(assets=["E"], a_ignore=["asset", "industry3"]):
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
                                                      for column, a_weight in setting["s_weight1"].items()])  # if final rank is na, nsmallest will not select anyway

                # sweight does not exist. using random values
                else:
                    print("select using random criteria")
                    df_today_mod["rank_final"] = np.random.randint(low=0,high=len(df_today_mod), size=len(df_today_mod))

                now = print_and_time(setting_count=setting_count, phase=f"BUY FINAL RANK", dict_trade_h_hold=dict_trade_h[tomorrow]["hold"], dict_trade_h_buy=dict_trade_h[tomorrow]["buy"], dict_trade_h_sell=dict_trade_h[tomorrow]["sell"], p_maxsize=p_maxsize, a_time=a_time,
                                     prev_time=now)




                # 6.8 PORTFOLIO BUY FILTER: SELECT PERCENTILE 1
                for i in range(1, len(df_today_mod) + 1):
                    df_select = try_select(select_from_df=df_today_mod, select_size=buyable_size * 10 * i, select_by=setting["f_percentile_column"])

                    # 6.6 PORTFOLIO BUY ADD_POSITION: FALSE
                    df_select = df_select[~df_select.index.isin([trade_info["ts_code"] for trade_info in dict_trade_h[tomorrow]["hold"]])]

                    # 6.7 PORTFOLIO BUY SELECT TOMORROW: select Stocks that really TRADES
                    df_select_tomorrow = df_tomorrow[df_tomorrow.index.isin(df_select.index)]


                    #if count stocks that trade tomorrow
                    if len(df_select_tomorrow) >= buyable_size:
                        break
                    else:
                        print(f"selection failed, reselect {i}")

                #if have not found enough stocks that trade tomorrow
                else:
                    #this probably means none of the stocks meats the criteria due to filter and rank.
                    #for none of the selected stocks trade tomorrow
                    #or something wrong with the ranking
                    #or you buy less than the size you want to buy
                    #LB.sound("error.mp3")
                    #df_select_tomorrow=pd.DataFrame()

                    #two options.
                    # case 1 normal test: if max port size is 10 but only 5 stocks met that criteria: carry and buy 5

                    # case 2 single stock test: if max port size is 1 but only 0 stocks met that criteria: not carry and buy 0
                    pass

                # carry final rank, otherwise the second select will not be able to select
                df_select_tomorrow["rank_final"] = df_today_mod.loc[df_select_tomorrow.index, "rank_final"]

                # 6.8 PORTFOLIO BUY FILTER: SELECT PERCENTILE 2
                df_select_tomorrow = try_select(select_from_df=df_select_tomorrow, select_size=buyable_size, select_by=setting["f_percentile_column"])

                now = print_and_time(setting_count=setting_count, phase=f"BUY SELECT", dict_trade_h_hold=dict_trade_h[tomorrow]["hold"], dict_trade_h_buy=dict_trade_h[tomorrow]["buy"], dict_trade_h_sell=dict_trade_h[tomorrow]["sell"], p_maxsize=p_maxsize, a_time=a_time, prev_time=now)



                # 6.11 BUY EXECUTE:
                p_fee = setting["p_fee"]
                current_capital = dict_capital[setting_count]["cash"]
                if setting["p_proportion"] == "prop":
                    df_select_tomorrow["reserved_capital"] = (df_select_tomorrow["rank_final"].sum() / df_select_tomorrow["rank_final"])
                    df_select_tomorrow["reserved_capital"] = current_capital * (df_select_tomorrow["reserved_capital"] / df_select_tomorrow["reserved_capital"].sum())
                elif setting["p_proportion"] == "fibo":
                    df_select_tomorrow["reserved_capital"] = current_capital * pd.Series(data=LB.fibonacci_weight(len(df_select_tomorrow))[::-1], index=df_select_tomorrow.index.to_numpy())
                else:
                    df_select_tomorrow["reserved_capital"] = current_capital / buyable_size

                for hold_count, (ts_code, row) in enumerate(df_select_tomorrow.iterrows(), start=1):
                    buy_open = row["open"]
                    buy_close = row["close"]
                    buy_pct_chg_comp_chg = buy_close / buy_open
                    buy_imp = int((row["open"] == row["close"]))
                    shares = row["reserved_capital"] // buy_open
                    value_open = shares * buy_open
                    value_close = shares * buy_close
                    fee = p_fee * value_open
                    dict_capital[setting_count]["cash"] = dict_capital[setting_count]["cash"] - value_open - fee

                    dict_trade_h[tomorrow]["buy"].append(
                        {"reason": np.nan, "rank_final": row["rank_final"], "buy_imp": buy_imp, "T+": 0, "ts_code": ts_code, "name": row["name"], "buyout_price": buy_open, "today_open": buy_open, "today_close": buy_close, "sold_price": float("nan"), "pct_chg": buy_pct_chg_comp_chg,
                         "comp_chg": buy_pct_chg_comp_chg, "shares": shares, "value_open": value_open,
                         "value_close": value_close, "port_cash": dict_capital[setting_count]["cash"]})

                    print(setting_count, ": ", '{0: <9}'.format("") + f"buy {hold_count} {ts_code}")

                now = print_and_time(setting_count=setting_count, phase=f"BUY EXECUTE", dict_trade_h_hold=dict_trade_h[tomorrow]["hold"], dict_trade_h_buy=dict_trade_h[tomorrow]["buy"], dict_trade_h_sell=dict_trade_h[tomorrow]["sell"], p_maxsize=p_maxsize, a_time=a_time, prev_time=now)

            else:  # to not buy today
                if len(df_today_mod)==0:
                    print("no stock meets criteria after basic filtering (like IPO)")

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

    a_summary_merge = []
    LB.sound("saving.mp3")
    for setting_count, (setting, dict_trade_h) in enumerate(zip(settings, a_dict_trade_h)):
        try:
            now = mytime.time()
            df_trade_h, df_portfolio_overview, df_setting = btest_portfolio(setting_original=setting, dict_trade_h=dict_trade_h, df_stock_market_all=df_stock_market_all, backtest_start_time=backtest_start_time, setting_count=setting_count)
            print("REPORT PORTFOLIO TIME:", mytime.time() - now)
            a_summary_merge.append(pd.merge(left=df_portfolio_overview.head(1), right=df_setting, left_on="strategy", right_on="id", sort=False))

            # sendmail
            if setting["send_mail"]:
                df_last_simulated_trade = df_trade_h.loc[df_trade_h["trade_date"] == last_simulated_date, ["trade_type", "name"]]
                str_last_simulated_trade = df_last_simulated_trade.to_string(header=False, index=True)
                str_last_simulated_trade = str(last_simulated_date) + ": \n" + setting["name"] + "\n" + str_last_simulated_trade
                LB.send_mail(str_last_simulated_trade)

        except Exception as e:
            LB.sound("error.mp3")
            print("summarizing ERROR:", e)
            traceback.print_exc()

    path = LB.a_path("Market/CN/Backtest_Multiple/Backtest_Summary")
    df_backtest_summ = pd.concat(a_summary_merge[::-1], sort=False, ignore_index=True)
    df_backtest_summ = df_backtest_summ.append(DB.get_file(path[0]), sort=False)
    LB.to_csv_feather(df_backtest_summ, a_path=path, skip_feather=True, index_relevant=False)


def btest_multiple(loop_indicator=1):
    # Initialize setting array
    a_settings = []
    setting_base = {
        # general = Non changeable through one run
        "start_date": "20050101",
        "end_date": "20200101",
        "freq": "D",
        "market": "CN",
        "assets": ["I"],  # E,I,FD,G

        # meta
        "id": "",  # datetime.now().strftime('%Y%m%d%H%M%S'), but done in backtest_once_loop
        "name": "StrategyX",  # readable strategy name used when describing a strategy
        "change": "",  # placeholder for future fill. Lazy way to check what has been changed in the last setting
        "send_mail": False,
        "print_log": True,

        # buy focus = Select.
        "trend": False,  # possible values: False(all days),trend2,trend3,trend240. Basically trend shown on all_stock_market.csv
        "f_percentile_column": "rank_final",  # always from small to big. 0%-20% is small.    80%-100% is big. (0 , 18),(18, 50),(50, 82),( 82, 100)
        "f_query_asset": {"period": [operator.ge, 240, True]},  # ,'period > 240' is ALWAYS THERE FOR SPEED REASON, "trend > 0.2", filter everything from group str to price int
        "f_query_date": {},  # filter days vs filter assets

        "s_weight1": {  # ascending True= small, False is big
            # "pct_chg": [False, 0.2, 1],  # very important
            # "total_mv": [True, 0.1, 1],  # not so useful for this strategy, not more than 10% weight
            # "turnover_rate": [True, 0.1, 0.5],
            # "ivola": [True, 0.4, 1],  # seems important
            "trend5": [False, 100, 1],  # very important for this strategy
            "pct_chg": [True, 5, 1],  # very important for this strategy
            "pgain2": [True, 3, 1],  # very important for this strategy
            "pgain5": [True, 2, 1],  # very important for this strategy
            # "pgain10": [True, 3, 1],  # very important for this strategy
            # "pgain20": [True, 2, 1],  # very important for this strategy
            # "pgain60": [True, 1, 1],  # very important for this strategy
            # "pgain240": [True, 1, 1],  # very important for this strategy
            # "candle_net_pos": [True, 0.1, 1],
        },
        "s_weight2": {  # ascending True= small, False is big
            "trend": [False, 100, 1],  # very important for this strategy
        },
        # bool: ascending True= small, False is big
        # int: indicator_weight: how each indicator is weight against other indicator. e.g. {"pct_chg": [False, 0.8, 0.2, 0.0]}  =》 df["pct_chg"]*0.8 + df["trend"]*0.2
        # int: asset_weight: how each asset indicator is weighted against its group indicator. e.g. {"pct_chg": [False, 0.8, 0.2, 0.0]}  =》 df["pct_chg"]*0.2+df["pct_chg_group"]*0.8. empty means no group weight
        # int: random_weight: random number spread to be added to asset # TODO add small random weight to each

        # portfolio
        "p_capital": 1000000,  # start capital
        "p_fee": 0.0000,  # 1==100%
        "p_maxsize": [30,True],  # True for fixed number, False for Percent. e.g. 30 stocks vs 30% of todays trading stock
        "p_min_T+": 1,  # Start consider sell. 1 means trade on next day, aka T+1， = Hold stock for 1 night， 2 means hold for 2 nights. Preferably 0,1,2 for day trading
        "p_max_T+": 1,  # MUST sell no matter what.
        "p_feedbackday": 20,
        "p_proportion": False,  # False = evenly weighted, "prop" = Score propotional weighted #fiboTODO Add fibonacci proportion
        "p_winner_abv": False,  # options False, >1. e.g. 1.2 means if one stock gained 20%, sell
        "p_loser_und": False,  # options False, <1. e.g. 0.8 means if one stocks gained -20%, sell
        "p_add_position": False,
        "p_compare": [["I", "000001.SH"]],  # ["I", "CJ000001.SH"],  ["I", "399001.SZ"], ["I", "399006.SZ"]   compare portfolio against other performance
    }

    # temp
    a_columns = [["total_mv", True, 0.05, 1], ["p5", False, 0.15, 1], ["pct_chg", False, 0.25, 1], ["trend", False, 0.25, 1], ["pjump_up", False, 0.05, 1], ["ivola", True, 0.05, 1], ["candle_net_pos", False, 0.15, 1]]  #
    a_columns = [["pct_chg", False, 1, 1], ["trend", False, 1, 1], ["trend2", False, 1, 1], ["trend10", False, 1, 1], ["candle_net_pos", False, 1, 1], ["candle_net_pos5", False, 1, 1], ["pjump_up", False, 1, 1], ["pjump_up10", False, 1, 1], ["ivola", True, 1, 1], ["ivola5", True, 1, 1],
                 ["pgain2", True, 1, 1], ["pgain5", True, 1, 1], ["pgain60", True, 1, 1], ["pgain240", True, 1, 1], ["turnover_rate", True, 1, 1], ["turnover_rate_pct2", True, 1, 1], ["pb", True, 1, 1], ["dv_ttm", True, 1, 1], ["ps_ttm", True, 1, 1], ["pe_ttm", True, 1, 1], ["total_mv", 1, 1]]  #


    name_sample=DB.get_ts_code()
    name_sample=list(name_sample["name"].sample(10))

    # settings creation

    # "ema2_close.(1, 10, 20)"
    for ema in ["ema1_close.(1, 5, 10)","ema1_close.(1, 10, 20)","ema1_close.(1, 240, 300)"]: #"zlmacd_close.(1, 5, 10)", "zlmacd_close.(1, 10, 20)", "zlmacd_close.(1, 240, 300)",
    #for zlmacd in ["zlmacd_close.(1, 240, 300)","zlmacd_close.(1, 5, 10)"]: #"zlmacd_close.(1, 5, 10)", "zlmacd_close.(1, 10, 20)", "zlmacd_close.(1, 240, 300)",
        for signal in [operator.gt,operator.lt]:
            #for exchange in ["创业板","中小板","主板"]:
                setting_copy = copy.deepcopy(setting_base)

                #filter
                setting_copy["f_query_asset"] ={"period": [operator.ge, 1000,True],
                                                "close" : [signal, ema, False],
                                                #"exchange": [operator.eq, exchange,True],
                                                }
                                                #"name": [operator.eq, "平安银行"],}

                #s weight
                setting_copy["s_weight1"] ={}
                a_settings.append(setting_copy)
                print(setting_copy["s_weight1"])


    print("Total Settings:", len(a_settings))
    btest_once(settings=a_settings)


if __name__ == '__main__':
    try:
        pr = cProfile.Profile()
        pr.enable()

        #
        # df_asset=DB.get_asset(ts_code="600499.SH")
        # df_asset = df_asset[(df_asset["period"] > 2000)]
        #
        # df_asset["tomorrow1"]=1+ df_asset["open.fgain1"].shift(-1)
        # gmeanis = gmean(df_asset["tomorrow1"].dropna())
        # print(gmeanis)
        #
        #
        # pct_chg=LB.my_gmean(df_asset["pct_chg"])
        # print(pct_chg)


        btest_multiple(5)

        pr.disable()
        # pr.print_stats(sort='file')

        pass
    except Exception as e:
        traceback.print_exc()
        LB.sound("error.mp3")
