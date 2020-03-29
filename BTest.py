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


def btest_portfolio(setting_original, d_trade_h, df_stock_market_all, backtest_start_time, setting_count):
    # init
    a_freqs = LB.c_bfreq()
    beta_against = "_000001.SH"
    a_time = []
    now = mytime.time()

    # deep copy setting
    setting = copy.deepcopy(setting_original)
    p_compare = setting["p_compare"]
    s_weight1 = setting["s_weight1"]

    # convertes dict in dict to string
    for key, value in setting.items():
        if type(value) == dict:
            setting[key] = LB.groups_d_to_string_iterable(setting[key])
        elif type(value) in [list, np.ndarray]:
            if (key == "p_compare"):
                setting["p_compare"] = ', '.join([x[1] for x in setting["p_compare"]])
            else:
                setting[key] = ', '.join([str(x) for x in value])

    # create trade_h and port_h from array
    # everything under this line has integer as index
    # everything above this line has ts_code and trade_date as index
    trade_h_helper = []
    for day, info in d_trade_h.items():
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

    # check if all index are valid
    if np.nan in df_trade_h["ts_code"] or float("nan") in df_trade_h:
        print("file has nan ts_codes")
        raise ValueError

    # create chart
    df_port_c = df_trade_h.groupby("trade_date").agg("mean")  # TODO remove inefficient apply and groupby and loc
    df_port_c = df_port_c.iloc[:-1]  # exclude last row because last row predicts future
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
        df_port_c[f"comp_chg_{competitor[1]}"] = ICreate.column_add_comp_chg(df_port_c[f"pct_chg_{competitor[1]}"])

    # tab_overview
    df_port_overview = pd.DataFrame(float("nan"), index=range(len(p_compare) + 1), columns=[]).astype(object)
    end_time_date = datetime.now()
    day = end_time_date.strftime('%Y/%m/%d')
    time = end_time_date.strftime('%H:%M:%S')
    duration = (end_time_date - backtest_start_time).seconds
    duration, Duration_rest = divmod(duration, 60)
    df_port_overview["SDate"] = day
    df_port_overview["STime"] = time
    df_port_overview["SDuration"] = f"{duration}:{Duration_rest}"
    df_port_overview["strategy"] = setting["id"] = f"{setting_original['id']}__{setting_count}"
    df_port_overview["start_date"] = setting["start_date"]
    df_port_overview["end_date"] = setting["end_date"]

    # portfolio strategy specific overview
    df_port_c["tomorrow_pct_chg"] = df_port_c["all_pct_chg"].shift(-1)  # add a future pct_chg 1 for easier target
    period = len(df_trade_h.groupby("trade_date"))
    df_port_overview.at[0, "period"] = period
    df_port_overview.at[0, "pct_days_involved"] = 1 - (len(df_port_c[df_port_c["port_size"] == 0]) / len(df_port_c))

    df_port_overview.at[0, "sell_mean_T+"] = df_trade_h.loc[(df_trade_h["trade_type"] == "sell"), "T+"].mean()
    df_port_overview.at[0, "asset_sell_winrate"] = len(df_trade_h.loc[(df_trade_h["trade_type"] == "sell") & (df_trade_h["comp_chg"] > 1)]) / len(df_trade_h.loc[(df_trade_h["trade_type"] == "sell")])
    df_port_overview.at[0, "all_daily_winrate"] = len(df_port_c.loc[df_port_c["all_pct_chg"] > 1]) / len(df_port_c)

    df_port_overview.at[0, "asset_sell_gmean"] = gmean(df_trade_h.loc[(df_trade_h["trade_type"] == "sell"), "comp_chg"].dropna())  # everytime you sell a stock
    df_port_overview.at[0, "all_gmean"] = gmean(df_port_c["all_pct_chg"].dropna())  # everyday

    df_port_overview.at[0, "asset_sell_pct_chg"] = df_trade_h.loc[(df_trade_h["trade_type"] == "sell"), "comp_chg"].mean()  # everytime you sell a stock
    df_port_overview.at[0, "all_pct_chg"] = df_port_c["all_pct_chg"].mean()  # everyday

    df_port_overview.at[0, "asset_sell_pct_chg_std"] = df_trade_h.loc[(df_trade_h["trade_type"] == "sell"), "comp_chg"].std()
    df_port_overview.at[0, "all_pct_chg_std"] = df_port_c["all_pct_chg"].std()

    try:
        df_port_overview.at[0, "all_comp_chg"] = df_port_c.at[df_port_c["all_comp_chg"].last_valid_index(), "all_comp_chg"]
    except:
        df_port_overview.at[0, "all_comp_chg"] = float("nan")

    df_port_overview.at[0, "port_beta"] = LB.calculate_beta(df_port_c["all_pct_chg"], df_port_c[f"pct_chg{beta_against}"])
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
        for rolling_freq in a_freqs:
            try:  # calculate the pct_chg of all stocks 1 day after the trend shows buy signal
                df_port_overview.at[0, f"market_trend{rolling_freq}_{one_zero}_pct_chg_mean"] = df_port_c.loc[df_port_c[f"market_trend{rolling_freq}"] == one_zero, "tomorrow_pct_chg"].mean()
                # condition_one_zero=df_port_c["market_trend" + str(rolling_freq)] == one_zero
                # df_port_overview.at[0, "market_trend" + str(rolling_freq) +"_"+ str(one_zero)+"_winrate"] = len(df_port_c[condition_trade & condition_win & condition_one_zero]) / len(df_port_c[condition_trade & condition_one_zero])
            except Exception as e:
                pass

    # overview indicator combination
    for column, a_weight in s_weight1.items():
        df_port_overview[f"{column}_ascending"] = a_weight[0]
        df_port_overview[f"{column}_indicator_weight"] = a_weight[1]
        df_port_overview[f"{column}_asset_weight"] = a_weight[2]

    # add overview for compare asset
    for i in range(len(p_compare), len(p_compare)):
        competitor_ts_code = p_compare[i][1]
        df_port_overview.at[i + 1, "strategy"] = competitor_ts_code
        df_port_overview.at[i + 1, "pct_chg_mean"] = df_port_c[f"pct_chg_{competitor_ts_code}"].mean()
        df_port_overview.at[i + 1, "pct_chg_std"] = df_port_c[f"pct_chg_{competitor_ts_code}"].std()
        df_port_overview.at[i + 1, "winrate"] = len(df_port_c[(df_port_c[f"pct_chg_{competitor_ts_code}"] >= 0) & (df_port_c[f"pct_chg_{competitor_ts_code}"].notna())]) / len(df_port_c[f"pct_chg_{competitor_ts_code}"].notna())
        df_port_overview.at[i + 1, "period"] = len(df_port_c) - df_port_c[f"pct_chg_{competitor_ts_code}"].isna().sum()
        df_port_overview.at[i + 1, "pct_days_involved"] = 1
        df_port_overview.at[i + 1, "comp_chg"] = df_port_c.at[df_port_c[f"comp_chg_{competitor_ts_code}"].last_valid_index(), f"comp_chg_{competitor_ts_code}"]
        df_port_overview.at[i + 1, "beta"] = LB.calculate_beta(df_port_c[f"pct_chg_{competitor_ts_code}"], df_port_c[f"pct_chg{beta_against}"])
        df_port_c[f"tomorrow_pct_chg_{competitor_ts_code}"] = df_port_c[f"pct_chg_{competitor_ts_code}"].shift(-1)  # add a future pct_chg 1 for easier target

        # calculate percent change and winrate
        condition_trade = df_port_c[f"tomorrow_pct_chg_{competitor_ts_code}"].notna()
        condition_win = df_port_c[f"tomorrow_pct_chg_{competitor_ts_code}"] >= 0
        for one_zero in [1, 0]:
            for y in a_freqs:
                try:
                    df_port_overview.at[i + 1, f"market_trend{y}_{one_zero}_pct_chg_mean"] = df_port_c.loc[df_port_c[f"market_trend{y}"] == one_zero, f"tomorrow_pct_chg_{competitor_ts_code}"].mean()
                    # condition_one_zero = df_port_c["market_trend" + str(y)] == one_zero
                    # df_port_overview.at[i + 1, "market_trend" + str(y) +"_"+str(one_zero)+"_winrate"] = len(df_port_c[condition_trade & condition_win & condition_one_zero]) / len(df_port_c[condition_trade & condition_one_zero])
                except Exception as e:
                    pass

    # split chart into pct_chg and comp_chg for easier reading
    a_trend_label = [f"close.market.trend{x}" for x in a_freqs if x != 1]
    a_trend_label = []  # TODO trend is excluded, add it back or remove it
    df_port_c = df_port_c[["rank_final", "port_pearson", "port_size", "buy", "hold", "sell", "port_cash", "port_close", "all_close", "all_pct_chg", "all_comp_chg"] + [f"pct_chg_{x}" for x in [x[1] for x in p_compare]] + [f"comp_chg_{x}" for x in [x[1] for x in p_compare]]]

    # write portfolio
    portfolio_path = f"Market/CN/Btest/Result/Portfolio_{setting['id']}"
    LB.to_csv_feather(df=df_port_overview, a_path=LB.a_path(f"{portfolio_path}/overview"), skip_feather=True)
    LB.to_csv_feather(df=df_trade_h, a_path=LB.a_path(f"{portfolio_path}/trade_h"), skip_feather=True)
    LB.to_csv_feather(df=df_port_c, a_path=LB.a_path(f"{portfolio_path}/chart"), index_relevant=True, skip_feather=True)
    df_setting = pd.DataFrame(setting, index=[0])
    LB.to_csv_feather(df=df_setting, a_path=LB.a_path(f"{portfolio_path}/setting"), index_relevant=False, skip_feather=True)

    print("setting is", setting["s_weight1"])
    print("=" * 50)
    [print(string) for string in a_time]
    print("=" * 50)
    return [df_trade_h, df_port_overview, df_setting]


def btest_once(settings=[{}]):
    # inside functions
    @LB.deco_try_ignore
    def try_select(select_from_df, select_size, select_by):
        return select_from_df.nsmallest(int(select_size), [select_by])

    def print_and_time(setting_count, phase, d_trade_h_hold, d_trade_h_buy, d_trade_h_sell, p_maxsize, a_time, prev_time):
        print(f"{setting_count} : hold " + '{0: <5}'.format(len(d_trade_h_hold)) + 'buy {0: <5}'.format(len(d_trade_h_buy)) + 'sell {0: <5}'.format(len(d_trade_h_sell)) + 'space {0: <5}'.format(p_maxsize - (len(d_trade_h_hold) + len(d_trade_h_buy))))
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
    LB.interrupt_start()

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
    df_group_instance_all = DB.preload(asset="G", d_queries_ts_code={"G": ["on_asset == 'E'", "group != 'industry3'"]})
    last_simulated_date = df_stock_market_all.index[-1]
    df_today_accelerator = pd.DataFrame()

    # 0.4 PREPARATION- INITIALIZE Changeables for the loop
    a_d_trade_h = [{df_trade_dates.index[0]: {"sell": [], "hold": [], "buy": []}} for _ in settings]  # trade history for each setting
    d_capital = {setting_count: {"cash": settings[0]["p_capital"]} for setting_count in range(0, len(settings))}  # only used in iteration, saved in trade_h

    # Main Loop
    for today, tomorrow in zip(df_trade_dates.index, df_trade_dates["tomorrow"]):

        if LB.interrupt_confirmed():
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
            df_tomorrow = DB.get_date(trade_date=tomorrow, a_assets=assets, freq="D", market=market)
            df_today = DB.get_date(trade_date=today, a_assets=assets, freq="D", market=market) if df_today_accelerator.empty else df_today_accelerator
            df_today_accelerator = df_tomorrow
            d_weight_accelerator = {}

        # FOR EACH DAY LOOP OVER SETTING N
        for setting_count, (setting, d_trade_h) in enumerate(zip(settings, a_d_trade_h)):
            # 0.0 INIT
            print("\n" * 3)
            a_time = []
            now = mytime.time()
            setting["id"] = datetime.now().strftime('%Y%m%d%H%M%S')
            p_maxsize = setting["p_maxsize"]

            d_trade_h[tomorrow] = {"sell": [], "hold": [], "buy": []}
            print(f"Assets {setting['assets']}, Market {setting['market']}")
            print('{0: <26}'.format("TODAY EVENING ANALYZE") + f"{today} stocks {len(df_today)}")
            print('{0: <26}'.format("TOMORROW MORNING TRADE") + f"{tomorrow} stocks {len(df_tomorrow)}")
            now = print_and_time(setting_count=setting_count, phase=f"INIT", d_trade_h_hold=d_trade_h[tomorrow]["hold"], d_trade_h_buy=d_trade_h[tomorrow]["buy"], d_trade_h_sell=d_trade_h[tomorrow]["sell"], p_maxsize=p_maxsize, a_time=a_time, prev_time=now)

            # 1.1 FILTER
            final_filter = True
            for eval_string in setting["f_query_asset"]:  # very slow and expensive for small operation because parsing the string takes long
                final_filter &= eval(eval_string)
                print(f"eval string: {eval_string}")
            df_today_mod = df_today[final_filter]
            print("today after filter", len(df_today_mod))
            now = print_and_time(setting_count=setting_count, phase=f"FILTER", d_trade_h_hold=d_trade_h[tomorrow]["hold"], d_trade_h_buy=d_trade_h[tomorrow]["buy"], d_trade_h_sell=d_trade_h[tomorrow]["sell"], p_maxsize=p_maxsize, a_time=a_time, prev_time=now)

            # 2 ECONOMY
            # 3 FINANCIAL MARKET
            # 6 PORTFOLIO = ASSET BALANCING, EFFICIENT FRONTIER, LEAST BETA #TODO

            # 6.2 PORTFOLIO SELL SELECT
            p_winner_abv = setting["p_winner_abv"]
            p_loser_und = setting["p_loser_und"]
            hold_count = 1
            sell_count = 1
            for trade_type, a_trade_content in d_trade_h[today].items():  # NOTE here today means today morning trade
                if trade_type != "sell":  # == in ["hold","buy"]. last day stocks that was kept for over night
                    for d_trade in a_trade_content:

                        # sell meta
                        ts_code = d_trade["ts_code"]
                        hold_day_overnight = d_trade["T+"] + 1  # simulates the night when deciding to sell tomorrow
                        sell = False

                        # sell decision
                        if hold_day_overnight >= setting["p_min_T+"]:  # sellable = consider sell
                            if hold_day_overnight >= setting["p_max_T+"]:  # must sell
                                sell = True
                                reason = f"> T+{hold_day_overnight}"
                            elif p_winner_abv:
                                if d_trade["comp_chg"] > p_winner_abv:
                                    sell = True
                                    reason = f"sell winner comp_chg above {p_winner_abv}"
                            elif p_loser_und:
                                if d_trade["comp_chg"] < p_loser_und:
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
                            tomorrow_open = d_trade["today_close"]
                            tomorrow_close = d_trade["today_close"]
                            sell = False
                            reason = f"{reason} - not trading"
                            print("probably 停牌", ts_code)

                        if sell:  # Execute sell
                            shares = d_trade["shares"]
                            realized_value = tomorrow_open * shares
                            fee = setting["p_fee"] * realized_value
                            d_capital[setting_count]["cash"] += realized_value - fee

                            d_trade_h[tomorrow]["sell"].append(
                                {"reason": reason, "rank_final": d_trade["rank_final"], "buy_imp": d_trade["buy_imp"], "ts_code": d_trade["ts_code"], "name": d_trade["name"], "T+": hold_day_overnight, "buyout_price": d_trade["buyout_price"],
                                 "today_open": tomorrow_open, "today_close": np.nan, "sold_price": tomorrow_open, "pct_chg": tomorrow_open / d_trade["today_close"],
                                 "comp_chg": tomorrow_open / d_trade["buyout_price"], "shares": shares, "value_open": realized_value, "value_close": np.nan, "port_cash": d_capital[setting_count]["cash"]})

                        else:  # Execute hold
                            d_trade_h[tomorrow]["hold"].append(
                                {"reason": reason, "rank_final": d_trade["rank_final"], "buy_imp": d_trade["buy_imp"], "ts_code": d_trade["ts_code"], "name": d_trade["name"], "T+": hold_day_overnight, "buyout_price": d_trade["buyout_price"],
                                 "today_open": tomorrow_open, "today_close": tomorrow_close, "sold_price": np.nan, "pct_chg": tomorrow_close / d_trade["today_close"],
                                 "comp_chg": tomorrow_close / d_trade["buyout_price"], "shares": d_trade["shares"], "value_open": tomorrow_open * d_trade["shares"], "value_close": tomorrow_close * d_trade["shares"], "port_cash": d_capital[setting_count]["cash"]})

                        # print out
                        if sell:
                            print(f"{setting_count} : " + '{0: <19}'.format("") + '{0: <9}'.format(f"sell {sell_count}"), (f"{ts_code}"))
                            sell_count = sell_count + 1
                        else:
                            print(f"{setting_count} : " + '{0: <0}'.format("") + '{0: <9}'.format(f"hold {hold_count}"), (f"{ts_code}"))
                            hold_count = hold_count + 1

            now = print_and_time(setting_count=setting_count, phase=f"SELL AND HOLD", d_trade_h_hold=d_trade_h[tomorrow]["hold"], d_trade_h_buy=d_trade_h[tomorrow]["buy"], d_trade_h_sell=d_trade_h[tomorrow]["sell"], p_maxsize=p_maxsize, a_time=a_time, prev_time=now)

            # PORTFOLIO BUY SELECT BEGIN #Rank Final as the final rank name
            buyable_size = p_maxsize - len(d_trade_h[tomorrow]["hold"])
            if buyable_size > 0 and len(df_today_mod) > 0:

                # 6.4 PORTFOLIO BUY SCORE/RANK
                # if selet final rank by a defined criteria
                if setting["s_weight1"]:

                    d_group_instance_weight = LB.c_group_score_weight()

                    for column, a_weight in setting["s_weight1"].items():
                        print("select column", column)

                        # column use group rank
                        if a_weight[2] != 1:

                            # trick to store and be used for next couple settings running on the same day
                            if column in d_weight_accelerator:  # TODO maybe add this to dt_date in general once all important indicators are found
                                print("accelerated")
                                df_today_mod = d_weight_accelerator[column]
                            else:
                                # 1. iterate to see replace value
                                print("NOT accelerated")  # TODO add groups for all assets
                                for group, instance_array in LB.c_d_groups(assets=["E"], a_ignore=["asset", "industry3"]).items():
                                    d_replace = {}
                                    df_today_mod[f"rank_{column}_{group}"] = df_today_mod[group]  # to be replaced later by int value
                                    for instance in instance_array:
                                        try:
                                            d_replace[instance] = df_group_instance_all[f"{group}_{instance}"].at[int(today), column]
                                        except Exception as e:
                                            print("(Could be normal if none of these group are trading on that day) ERROR on", today, group, instance)
                                            print(e)
                                            traceback.print_exc()
                                            d_replace[instance] = 0
                                    df_today_mod[f"rank_{column}_{group}"].replace(to_replace=d_replace, inplace=True)
                                d_weight_accelerator[column] = df_today_mod.copy()

                            # 2. calculate group score
                            df_today_mod[f"rank_{column}_group"] = 0.0
                            for group in LB.c_d_groups(assets=["E"], a_ignore=["asset", "industry3"]):
                                try:
                                    df_today_mod[f"rank_{column}_group"] += df_today_mod[f"rank_{column}_{group}"] * d_group_instance_weight[group]
                                except Exception as e:
                                    print(e)

                        else:  # column does not use group rank
                            df_today_mod[f"rank_{column}_group"] = 0.0

                        # 3. Create Indicator Rank= indicator_asset+indicator_group
                        df_today_mod[f"{column}_rank"] = (df_today_mod[column] * a_weight[2] + df_today_mod[f"rank_{column}_group"] * (1 - a_weight[2])).rank(ascending=a_weight[0])

                    # 4. Create Rank Final = indicator1+indicator2+indicator3
                    df_today_mod["rank_final"] = sum([df_today_mod[f"{column}_rank"] * a_weight[1]
                                                      for column, a_weight in setting["s_weight1"].items()])  # if final rank is na, nsmallest will not select anyway

                # sweight does not exist. using random values
                else:
                    print("select using random criteria")
                    df_today_mod["rank_final"] = np.random.randint(low=0, high=len(df_today_mod), size=len(df_today_mod))

                now = print_and_time(setting_count=setting_count, phase=f"BUY FINAL RANK", d_trade_h_hold=d_trade_h[tomorrow]["hold"], d_trade_h_buy=d_trade_h[tomorrow]["buy"], d_trade_h_sell=d_trade_h[tomorrow]["sell"], p_maxsize=p_maxsize, a_time=a_time,
                                     prev_time=now)

                # 6.8 PORTFOLIO BUY FILTER: SELECT PERCENTILE 1
                for select_size in [buyable_size * 3, len(df_today_mod)]:
                    df_select = try_select(select_from_df=df_today_mod, select_size=select_size, select_by="rank_final")

                    # 6.6 PORTFOLIO BUY ADD_POSITION: FALSE
                    df_select = df_select[~df_select.index.isin([trade_info["ts_code"] for trade_info in d_trade_h[tomorrow]["hold"]])]

                    # 6.7 PORTFOLIO BUY SELECT TOMORROW: select Stocks that really TRADES
                    df_select_tomorrow = df_tomorrow[df_tomorrow.index.isin(df_select.index)]

                    # if count stocks that trade tomorrow
                    if len(df_select_tomorrow) >= buyable_size:
                        break
                    else:
                        print(f"selection failed, reselect {select_size}")

                # if have not found enough stocks that trade tomorrow
                else:
                    # this probably means less of the stocks meats the criteria: buyable stock < p_maxsize
                    # for none of the selected stocks trade tomorrow
                    # current solution: if max port size is 10 but only 5 stocks met that criteria: carry and buy 5
                    pass

                # carry final rank, otherwise the second select will not be able to select
                df_select_tomorrow["rank_final"] = df_today_mod.loc[df_select_tomorrow.index, "rank_final"]

                # 6.8 PORTFOLIO BUY FILTER: SELECT PERCENTILE 2
                df_select_tomorrow = try_select(select_from_df=df_select_tomorrow, select_size=buyable_size, select_by="rank_final")
                now = print_and_time(setting_count=setting_count, phase=f"BUY SELECT", d_trade_h_hold=d_trade_h[tomorrow]["hold"], d_trade_h_buy=d_trade_h[tomorrow]["buy"], d_trade_h_sell=d_trade_h[tomorrow]["sell"], p_maxsize=p_maxsize, a_time=a_time, prev_time=now)

                # 6.11 BUY EXECUTE:
                p_fee = setting["p_fee"]
                current_capital = d_capital[setting_count]["cash"]
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
                    d_capital[setting_count]["cash"] = d_capital[setting_count]["cash"] - value_open - fee

                    d_trade_h[tomorrow]["buy"].append(
                        {"reason": np.nan, "rank_final": row["rank_final"], "buy_imp": buy_imp, "T+": 0, "ts_code": ts_code, "name": row["name"], "buyout_price": buy_open, "today_open": buy_open, "today_close": buy_close, "sold_price": float("nan"), "pct_chg": buy_pct_chg_comp_chg,
                         "comp_chg": buy_pct_chg_comp_chg, "shares": shares, "value_open": value_open,
                         "value_close": value_close, "port_cash": d_capital[setting_count]["cash"]})

                    print(setting_count, ": ", '{0: <9}'.format("") + f"buy {hold_count} {ts_code}")

                now = print_and_time(setting_count=setting_count, phase=f"BUY EXECUTE", d_trade_h_hold=d_trade_h[tomorrow]["hold"], d_trade_h_buy=d_trade_h[tomorrow]["buy"], d_trade_h_sell=d_trade_h[tomorrow]["sell"], p_maxsize=p_maxsize, a_time=a_time, prev_time=now)

            else:  # to not buy today
                if len(df_today_mod) == 0:
                    print("no stock meets criteria after basic filtering (like IPO)")
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
    for setting_count, (setting, d_trade_h) in enumerate(zip(settings, a_d_trade_h)):
        try:
            now = mytime.time()
            df_trade_h, df_portfolio_overview, df_setting = btest_portfolio(setting_original=setting, d_trade_h=d_trade_h, df_stock_market_all=df_stock_market_all, backtest_start_time=backtest_start_time, setting_count=setting_count)
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

    path = LB.a_path(f"Market/CN/Btest/Backtest_Summary")
    df_backtest_summ = pd.concat(a_summary_merge[::-1], sort=False, ignore_index=True)
    df_backtest_summ = df_backtest_summ.append(DB.get_file(path[0]), sort=False)
    LB.to_csv_feather(df_backtest_summ, a_path=path, skip_feather=True, index_relevant=False)


def btest_multiple(loop_indicator=1):
    # Initialize settings
    setting_base = {
        # general = Non changeable through one run
        "start_date": "20000101",
        "end_date": "20200101",
        "freq": "D",
        "market": "CN",
        "assets": ["E"],  # E,I,FD,G,F

        # meta
        "id": "",  # datetime.now().strftime('%Y%m%d%H%M%S'), but done in backtest_once_loop
        "send_mail": False,
        "print_log": True,

        # buy focus = Select.
        "f_query_asset": ["df_today['period']>240"],  # ,'period > 240' is ALWAYS THERE FOR SPEED REASON, "trend > 0.2", filter everything from group str to price int
        "f_query_date": [],  # filter days vs filter assets. restrict some days to only but or only sell

        # selection weight
        # bool: ascending True= small, False is big
        # int: indicator_weight: how each indicator is weight against other indicator. e.g. {"pct_chg": [False, 0.8, 0.2]}  =》 df["pct_chg"]*0.8 + df["trend"]*0.2
        # int: asset_weight: how each asset indicator is weighted against its group indicator. e.g. {"pct_chg": [False, 0.8, 0.2]}  =》 df["pct_chg"]*0.2+df["pct_chg_group"]*0.8. empty means no group weight
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

        # portfolio
        "p_capital": 1000000,  # start capital
        "p_fee": 0.0000,  # 1==100%
        "p_maxsize": 10,  # 1: fixed number,2: percent. 3: percent with top_limit. e.g. 30 stocks vs 30% of todays trading stock
        "p_min_T+": 1,  # Start consider sell. 1 means trade on next day, aka T+1， = Hold stock for 1 night， 2 means hold for 2 nights. Preferably 0,1,2 for day trading
        "p_max_T+": 1,  # MUST sell no matter what.
        "p_proportion": False,  # choices: False(evenly), prop(proportional to rank), fibo(fibonacci)
        "p_winner_abv": False,  # options False, >1. e.g. 1.2 means if one stock gained 20%, sell
        "p_loser_und": False,  # options False, <1. e.g. 0.8 means if one stocks gained -20%, sell
        "p_add_position": False,
        "p_compare": [["I", "000001.SH"]],  # ["I", "CJ000001.SH"],  ["I", "399001.SZ"], ["I", "399006.SZ"]   compare portfolio against other performance
    }

    # settings creation  #
    for asset in ["E","I","FD","F","G"]:
        a_settings = []

        # is_max is_min
        # for thresh in [x/100 for x in range(0,100,5)]: #"zlmacd_close.(1, 5, 10)", "zlmacd_close.(1, 10, 20)", "zlmacd_close.(1, 240, 300)",
        #     setting_copy = copy.deepcopy(setting_base)
        #     setting_copy["assets"] = [asset]
        #     setting_copy["f_query_asset"] =[f"df_today['period']>240",
        #                                       f"(df_today['e_min']/df_today['close']).between({thresh},{thresh+0.05})"]
        #     setting_copy["s_weight1"] ={}
        #     a_settings.append(setting_copy)
        #

        # Fresh stock by final rank
        # for period1 in [250,500,750,1000]:
        #     for period2 in [750,1500,7000]:  # "zlmacd_close.(1, 5, 10)", "zlmacd_close.(1, 10, 20)", "zlmacd_close.(1, 240, 300)",
        #         if period1<period2:
        #             setting_copy = copy.deepcopy(setting_base)
        #             setting_copy["assets"] = [asset]
        #             setting_copy["f_query_asset"] = [
        #                 f"df_today['period'].between({period1},{period2})",
        #                 f"(df_today['e_min']/df_today['close']).between(0.7,1)"
        #             ]
        #             setting_copy["s_weight1"] = {"bull": [True, 1, 1], }
        #             a_settings.append(setting_copy)

        # macd
        # for macd in ["zlmacd_close.(1, 5, 10)","zlmacd_close.(1, 10, 20)","zlmacd_close.(1, 240, 300)","zlmacd_close.(1, 300, 500)"]:
        #     for macd_bool in [10,-10]:
        #         setting_copy = copy.deepcopy(setting_base)
        #         setting_copy["assets"]=[asset]
        #
        #         setting_copy["f_query_asset"] = [f"df_today['period']>240",
        #                                          f"(df_today['{macd}']=={macd_bool})"]
        #
        #         setting_copy["s_weight1"] ={}
        #         a_settings.append(setting_copy)

        #random choose from any asset type
        setting_copy = copy.deepcopy(setting_base)
        setting_copy["assets"] = [asset]
        setting_copy["f_query_asset"] = [f"df_today['period']>240"]
        setting_copy["s_weight1"] = {}
        a_settings.append(setting_copy)

        print("Total Settings:", len(a_settings))
        btest_once(settings=a_settings)


if __name__ == '__main__':
    try:
        pr = cProfile.Profile()
        pr.enable()

        btest_multiple(5)

        pr.disable()
        # pr.print_stats(sort='file')

        pass
    except Exception as e:
        traceback.print_exc()
        LB.sound("error.mp3")
