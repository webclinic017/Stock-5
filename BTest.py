import tushare as ts
import pandas as pd
import numpy as np
import time as mytime
import time
import DB
import LB
import os
from datetime import datetime
import traceback
from scipy.stats import gmean
import copy
import cProfile
import Alpha
from numba import njit
from numba import jit

pd.options.mode.chained_assignment = None  # default='warn'
pro = ts.pro_api('c473f86ae2f5703f58eecf9864fa9ec91d67edbc01e3294f6a4f9c32')
ts.set_token("c473f86ae2f5703f58eecf9864fa9ec91d67edbc01e3294f6a4f9c32")

"""
Atest (Assettest): 
= Test strategy on individual asset and then mean them 
= COMPARE past time to now (relative to past)
= NOT COMPARE what other stocks do (NOT relative to time/market)

Btest (Backtest):
= COMPARE past time to now (relative to past)
= COMPARE assets with other (relative to other)
"""

def btest_portfolio(setting_original, d_trade_h, df_stock_market_all, backtest_start_time, setting_count):
    # init
    a_freqs = LB.c_bfreq()
    beta_against = "_000001.SH"
    a_time = []
    now = mytime.time()
    for key,items in setting_original.items():
        print(f"{setting_count}: {key}: {items}")

    # copy settin signals betfore converting to st
    setting = copy.deepcopy(setting_original)
    p_compare = setting["p_compare"]
    s_weight1 = setting["s_weight1"]
    auto= setting["auto"] if setting["auto"] else []

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

    #TODO handle this maybe better. if this produces error, it means the strategy has never traded once.
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

    # PORT_C
    df_port_c = df_trade_h.groupby("trade_date").agg("mean")
    df_port_c = df_port_c.iloc[:-1]  # exclude last row because last row predicts future

    #helper function to make it clear and faster
    df_sell=df_trade_h[df_trade_h["trade_type"]== "sell"]
    df_sell_grouped=df_sell.groupby("trade_date")

    #standard data
    df_port_c["port_rank_pearson"] = df_sell_grouped.apply(lambda x: x["rank_final"].corr(x["pct_chg"],method="pearson"))
    df_port_c["port_rank_spearman"] = df_sell_grouped.apply(lambda x: x["rank_final"].corr(x["pct_chg"],method="spearman"))
    df_port_c["port_size"] = df_trade_h[df_trade_h["trade_type"].isin(["hold", "buy"])].groupby("trade_date").size()
    df_port_c["port_cash"] = df_trade_h.groupby("trade_date").apply(lambda x: x.at[x.last_valid_index(), "port_cash"])
    df_port_c["buy"] = df_trade_h[df_trade_h["trade_type"].isin(["buy"])].groupby("trade_date").size()
    df_port_c["hold"] = df_trade_h[df_trade_h["trade_type"].isin(["hold"])].groupby("trade_date").size()
    df_port_c["sell"] = df_trade_h[df_trade_h["trade_type"].isin(["sell"])].groupby("trade_date").size()

    #chart based on stocks sold today
    df_port_c["port_sell_pct_chg"] = df_sell_grouped.mean()["comp_chg"]
    df_port_c["port_sell_comp_chg"] = df_port_c["port_sell_pct_chg"].cumprod()

    # chart based on stocks hold overnight
    df_port_c["port_close"] = df_trade_h[df_trade_h["trade_type"].isin(["hold", "buy"])].groupby("trade_date").sum()["value_close"]
    df_port_c["all_close"] = df_port_c["port_close"] + df_port_c["port_cash"]
    df_port_c["all_pct_chg"] = df_port_c["all_close"].pct_change().fillna(0) + 1
    df_port_c["all_comp_chg"] = df_port_c["all_pct_chg"].cumprod()
    df_port_c["tomorrow_pct_chg"] = df_port_c["all_pct_chg"].shift(-1)  # add a future pct_chg 1 for easier target

    # chart add competitor
    for competitor in p_compare:
        df_port_c = DB.add_asset_comparison(df=df_port_c, freq=setting["freq"], asset=competitor[0], ts_code=competitor[1], a_compare_label=["pct_chg"])
        #old df_port_c[f"comp_chg_{competitor[1]}"] = Alpha.column_add_comp_chg(df_port_c[f"pct_chg_{competitor[1]}"])
        df_port_c[f"comp_chg_{competitor[1]}"] = Alpha.comp_chg(df=df_port_c,abase=f"pct_chg_{competitor[1]}",inplace=False)

    #PORT_C final
    df_port_c = df_port_c[["rank_final", "port_rank_pearson", "port_rank_spearman", "port_size", "buy", "hold", "sell", "port_cash", "port_close", "all_close", "all_pct_chg", "all_comp_chg", "port_sell_pct_chg", "port_sell_comp_chg"] + [f"pct_chg_{x}" for x in [x[1] for x in p_compare]] + [f"comp_chg_{x}" for x in [x[1] for x in p_compare]]]

    # OVERVIEW
    df_overview = pd.DataFrame(float("nan"), index=range(len(p_compare) + 1), columns=[]).astype(object)
    end_time_date = datetime.now()
    day = end_time_date.strftime('%Y/%m/%d')
    time = end_time_date.strftime('%H:%M:%S')
    duration = (end_time_date - backtest_start_time).seconds
    duration, Duration_rest = divmod(duration, 60)

    df_overview["SDate"] = day
    df_overview["STime"] = time
    df_overview["SDuration"] = f"{duration}:{Duration_rest}"
    df_overview["strategy"] = setting["id"] = f"{setting_original['id']}__{setting_count}"
    df_overview["start_date"] = setting["start_date"]
    df_overview["end_date"] = setting["end_date"]

    # portfolio strategy specific overview
    #helper
    group_object=df_trade_h.groupby("trade_date")
    s_sell_comp = df_trade_h.loc[(df_trade_h["trade_type"] == "sell"), "comp_chg"].dropna()

    #basic
    df_overview.at[0, "period"] = len(group_object)
    df_overview.at[0, "pct_days_involved"] = 1 - (len(df_port_c[df_port_c["port_size"] == 0]) / len(df_port_c))
    df_overview.at[0, "sell_mean_T+"] = df_trade_h.loc[(df_trade_h["trade_type"] == "sell"), "T+"].mean()
    df_overview.at[0, "stock_age"] = df_trade_h["stock_age"].mean()
    df_overview.at[0, "total_mv"] = (group_object.mean()["total_mv"]).mean()
    df_overview.at[0, "buy_imp"] = len(df_trade_h.loc[(df_trade_h["buy_imp"] == 1) & (df_trade_h["trade_type"] == "buy")]) / len(df_trade_h.loc[(df_trade_h["trade_type"] == "buy")])

    #return
    df_overview.at[0, "asset_sell_winrate"] = len(s_sell_comp[s_sell_comp > 1]) / len(df_trade_h.loc[(df_trade_h["trade_type"] == "sell")])
    df_overview.at[0, "all_daily_winrate"] = len(df_port_c.loc[df_port_c["all_pct_chg"] > 1]) / len(df_port_c)
    df_overview.at[0, "asset_sell_gmean"] = gmean(s_sell_comp)  # everytime you sell a stock
    df_overview.at[0, "all_gmean"] = gmean(df_port_c["all_pct_chg"].dropna())  # everyday
    df_overview.at[0, "asset_sell_pct_chg"] = s_sell_comp.mean()  # everytime you sell a stock
    df_overview.at[0, "all_pct_chg"] = df_port_c["all_pct_chg"].mean()  # everyday
    df_overview.at[0, "asset_sell_sharp"] = s_sell_comp.mean()/s_sell_comp.std()
    df_overview.at[0, "all_sharp"] = df_port_c["all_pct_chg"].mean()/df_port_c["all_pct_chg"].std()  # everyday
    # df_overview.at[0, "asset_sell_pct_chg_std"] = s_sell_comp.std()
    # df_overview.at[0, "all_pct_chg_std"] = df_port_c["all_pct_chg"].std()
    if not df_port_c.empty:
        df_overview.at[0, "all_comp_chg"] = df_port_c.at[df_port_c["all_comp_chg"].last_valid_index(), "all_comp_chg"]
        #df_overview.at[0, "all_comp_chg/sh_index"] = df_port_c.at[(len(df_port_c) - 1), "all_comp_chg"] / df_port_c.at[len(df_port_c) - 1, "comp_chg_000001.SH"]
        df_overview.at[0, "all_comp_chg/sh_index"] = df_port_c["all_comp_chg"].iat[-1] / df_port_c["comp_chg_000001.SH"].iat[- 1]
    else:
        df_overview.at[0, "all_comp_chg"] =np.nan
        df_overview.at[0, "all_comp_chg/sh_index"] =np.nan

    #correlation
    df_overview.at[0, "port_corr_pearson"] = df_port_c["all_pct_chg"].corr(df_port_c[f"pct_chg{beta_against}"], method="pearson")
    df_overview.at[0, "port_corr_spearman"] = df_port_c["all_pct_chg"].corr(df_port_c[f"pct_chg{beta_against}"], method="spearman")
    df_overview.at[0, "port_rank_pearson"] = df_port_c["port_rank_pearson"].mean()
    df_overview.at[0, "port_rank_spearman"] = df_port_c["port_rank_spearman"].mean()

    valuecounts=df_trade_h["trade_type"].value_counts()
    for trade_type in ["buy", "sell", "hold"]:
        df_overview.at[0, f"{trade_type}_count"] = valuecounts[trade_type] if trade_type in valuecounts.index else 0

    for lower_year, upper_year in [(20000101, 20050101), (20050101, 20100101), (20100101, 20150101), (20150101, 20200101)]:
        df_overview.at[0, f"all_pct_chg{upper_year}"] = df_port_c.loc[(df_port_c.index > lower_year) & (df_port_c.index < upper_year), "all_pct_chg"].mean()

    # overview indicator combination
    for column, a_weight in s_weight1.items():
        df_overview[f"{column}_ascending"] = a_weight[0]
        df_overview[f"{column}_indicator_weight"] = a_weight[1]
        df_overview[f"{column}_asset_weight"] = a_weight[2]


    # add overview for compare asset
    for i in range(len(p_compare), len(p_compare)):
        competitor_ts_code = p_compare[i][1]
        # df_overview.at[i + 1, "strategy"] = competitor_ts_code
        # df_overview.at[i + 1, "pct_chg_mean"] = df_port_c[f"pct_chg_{competitor_ts_code}"].mean()
        # df_overview.at[i + 1, "pct_chg_std"] = df_port_c[f"pct_chg_{competitor_ts_code}"].std()
        # df_overview.at[i + 1, "winrate"] = len(df_port_c[(df_port_c[f"pct_chg_{competitor_ts_code}"] >= 0) & (df_port_c[f"pct_chg_{competitor_ts_code}"].notna())]) / len(df_port_c[f"pct_chg_{competitor_ts_code}"].notna())
        # df_overview.at[i + 1, "period"] = len(df_port_c) - df_port_c[f"pct_chg_{competitor_ts_code}"].isna().sum()
        # df_overview.at[i + 1, "pct_days_involved"] = 1
        # df_overview.at[i + 1, "comp_chg"] = df_port_c.at[df_port_c[f"comp_chg_{competitor_ts_code}"].last_valid_index(), f"comp_chg_{competitor_ts_code}"]
        # df_overview.at[i + 1, "beta"] = LB.calculate_beta(df_port_c[f"pct_chg_{competitor_ts_code}"], df_port_c[f"pct_chg{beta_against}"])




    # write portfolio
    if setting["auto"]:
        print("setting auto",auto)
        portfolio_path = f"Market/CN/Btest/auto/comb_{len(auto)}/{'_'.join(auto)}/{setting['assets']}/result/{setting['id']}"
    else:
        print("setting manu")
        portfolio_path = f"Market/CN/Btest/manu/result/{setting['id']}"

    #add link to port overview
    a_links_label=["overview","trade_h","chart","setting"]
    for label in a_links_label:
        df_overview.insert(0, label,value=np.nan, allow_duplicates=False)
        df_overview[label]=f'=HYPERLINK("{LB.c_root()+portfolio_path}/{label}_{setting["id"]}.csv")'

    #save to drive
    LB.to_csv_feather(df=df_overview, a_path=LB.a_path(f"{portfolio_path}/overview_{setting['id']}"), skip_feather=True,index_relevant=False)
    LB.to_csv_feather(df=df_trade_h, a_path=LB.a_path(f"{portfolio_path}/trade_h_{setting['id']}"), skip_feather=True,index_relevant=False)
    LB.to_csv_feather(df=df_port_c, a_path=LB.a_path(f"{portfolio_path}/chart_{setting['id']}"),  skip_feather=True,index_relevant=True)

    df_setting = pd.DataFrame(setting,index=[0])

    LB.to_csv_feather(df=df_setting, a_path=LB.a_path(f"{portfolio_path}/setting_{setting['id']}"),  skip_feather=True,index_relevant=False)

    print("=" * 50)
    [print(string) for string in a_time]
    print("=" * 50)
    return [df_trade_h, df_overview, df_setting]


def btest(settings=[{}]):
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
    print("df_stock_market_all",df_stock_market_all)
    #d_G_df = DB.preload(asset="G", d_queries_ts_code=LB.c_G_queries())
    d_G_df = {}
    df_today_accelerator = pd.DataFrame()

    # 0.4 PREPARATION- INITIALIZE Changeables for the loop
    a_d_trade_h = [{int(df_trade_dates.index[0]): {"sell": [], "hold": [], "buy": []}} for _ in settings]  # trade history for each setting
    d_capital = {setting_count: {"cash": settings[0]["p_capital"]} for setting_count in range(0, len(settings))}  # only used in iteration, saved in trade_h

    # Main Loop
    for today, tomorrow in zip(df_trade_dates.index, df_trade_dates["tomorrow"]):

        today=int(today)
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
            df_tomorrow = DB.get_date(trade_date=tomorrow, a_asset=assets, freq="D", market=market)

            #to be removed
            df_tomorrow["name"]=df_tomorrow.index
            df_today = DB.get_date(trade_date=today, a_asset=assets, freq="D", market=market) if df_today_accelerator.empty else df_today_accelerator
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
            print(f"Auto {setting['auto']}, Assets {setting['assets']}, Market {setting['market']}")
            print('{0: <26}'.format("TODAY EVENING ANALYZE") + f"{today} stocks {len(df_today)}")
            print('{0: <26}'.format("TOMORROW MORNING TRADE") + f"{tomorrow} stocks {len(df_tomorrow)}")
            now = print_and_time(setting_count=setting_count, phase=f"INIT", d_trade_h_hold=d_trade_h[tomorrow]["hold"], d_trade_h_buy=d_trade_h[tomorrow]["buy"], d_trade_h_sell=d_trade_h[tomorrow]["sell"], p_maxsize=p_maxsize, a_time=a_time, prev_time=now)

            # 1.1 FILTER
            final_filter = True

            for eval_string in setting["f_query_asset"]:  # very slow and expensive for small operation because parsing the string takes long
                print(f"eval string: {eval_string}")
                final_filter &= eval(eval_string)

            df_today_mod = df_today[final_filter]
            print("today after filter", len(df_today_mod))
            now = print_and_time(setting_count=setting_count, phase=f"FILTER", d_trade_h_hold=d_trade_h[tomorrow]["hold"], d_trade_h_buy=d_trade_h[tomorrow]["buy"], d_trade_h_sell=d_trade_h[tomorrow]["sell"], p_maxsize=p_maxsize, a_time=a_time, prev_time=now)

            # 2 ECONOMY
            # 3 FINANCIAL MARKET
            # 6 PORTFOLIO = ASSET BALANCING, EFFICIENT FRONTIER, LEAST BETA
            # #TODO. Basically create portfolio with low beta when market is bad and high beta when market is good

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
                                {"reason": reason, "rank_final": d_trade["rank_final"], "buy_imp": d_trade["buy_imp"], "ts_code": d_trade["ts_code"], "name": d_trade["name"], "stock_age":d_trade["stock_age"], "total_mv":d_trade["total_mv"], "T+": hold_day_overnight, "buyout_price": d_trade["buyout_price"],
                                 "today_open": tomorrow_open, "today_close": np.nan, "sold_price": tomorrow_open, "pct_chg": tomorrow_open / d_trade["today_close"],
                                 "comp_chg": tomorrow_open / d_trade["buyout_price"], "shares": shares, "value_open": realized_value, "value_close": np.nan, "port_cash": d_capital[setting_count]["cash"]})

                        else:  # Execute hold
                            d_trade_h[tomorrow]["hold"].append(
                                {"reason": reason, "rank_final": d_trade["rank_final"], "buy_imp": d_trade["buy_imp"], "ts_code": d_trade["ts_code"], "name": d_trade["name"], "stock_age":d_trade["stock_age"], "total_mv":d_trade["total_mv"],"T+": hold_day_overnight, "buyout_price": d_trade["buyout_price"],
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
                                            d_replace[instance] = d_G_df[f"{group}_{instance}"].at[int(today), column]
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
                if len(df_select_tomorrow) !=0:
                    p_fee = setting["p_fee"]
                    cash_available = d_capital[setting_count]["cash"]
                    if setting["p_proportion"] == "prop":
                        df_select_tomorrow["reserved_capital"] = (df_select_tomorrow["rank_final"].sum() / df_select_tomorrow["rank_final"])
                        df_select_tomorrow["reserved_capital"] = cash_available * (df_select_tomorrow["reserved_capital"] / df_select_tomorrow["reserved_capital"].sum())
                    elif setting["p_proportion"] == "fibo":
                        df_select_tomorrow["reserved_capital"] = cash_available * pd.Series(data=LB.fibonacci_weight(len(df_select_tomorrow))[::-1], index=df_select_tomorrow.index.to_numpy())
                    else:
                        #former bug here. not divide reserved capital by p_maxsize, rather divide it by available stocks to buy.
                        df_select_tomorrow["reserved_capital"] = cash_available / len(df_select_tomorrow)


                    for hold_count, (ts_code, row) in enumerate(df_select_tomorrow.iterrows(), start=1):

                        #must have columns int
                        buy_open = row["open"]
                        buy_close = row["close"]
                        buy_pct_chg_comp_chg = buy_close / buy_open
                        buy_imp = int((row["open"] == row["close"]))

                        #portfolio calculation
                        shares = row["reserved_capital"] // buy_open
                        value_open = shares * buy_open
                        value_close = shares * buy_close
                        fee = p_fee * value_open
                        d_capital[setting_count]["cash"] -= value_open - fee

                        #nice-to-have columns
                        total_mv = row["total_mv"] if "total_mv" in row.index else 888
                        stock_age = row["period"] if "period" in row.index else np.nan

                        d_trade_h[tomorrow]["buy"].append(
                            {"reason": np.nan, "rank_final": row["rank_final"], "buy_imp": buy_imp, "T+": 0, "ts_code": ts_code, "name": row["name"], "stock_age":stock_age,"total_mv":total_mv, "buyout_price": buy_open, "today_open": buy_open, "today_close": buy_close, "sold_price": float("nan"), "pct_chg": buy_pct_chg_comp_chg,
                             "comp_chg": buy_pct_chg_comp_chg, "shares": shares, "value_open": value_open,
                             "value_close": value_close, "port_cash": d_capital[setting_count]["cash"]})
                        print(setting_count, ": ", '{0: <9}'.format("") + f"buy {hold_count} {ts_code}")

                    now = print_and_time(setting_count=setting_count, phase=f"BUY EXECUTE", d_trade_h_hold=d_trade_h[tomorrow]["hold"], d_trade_h_buy=d_trade_h[tomorrow]["buy"], d_trade_h_sell=d_trade_h[tomorrow]["sell"], p_maxsize=p_maxsize, a_time=a_time, prev_time=now)

            else:  # to not buy today
                if len(df_today_mod) == 0:
                    print("no stock meets criteria after basic filtering (like IPO)")
                else:
                    print("not buying today because no space")

            cash_available=d_capital[setting_count]["cash"]
            print(f"cash at end of day {cash_available}")

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

            # sendmail
            if setting["send_mail"]:
                df_last_simulated_trade = df_trade_h.loc[df_trade_h["trade_date"] == str_last_simulated_trade, ["trade_type", "name"]]
                str_last_simulated_trade = df_last_simulated_trade.to_string(header=False, index=True)
                str_last_simulated_trade = str(str_last_simulated_trade) + ": \n" + setting["name"] + "\n" + str_last_simulated_trade
                LB.send_mail(str_last_simulated_trade)

        except Exception as e:
            LB.sound("error.mp3")
            print("summarizing ERROR:", e)
            traceback.print_exc()




def btest_setting_master():
    return {
        # general = Non changeable through one run
        "start_date": "20180101",
        "end_date": "20180601",
        "freq": "D",
        "market": "CN",
        "assets": ["E"],  # E,I,FD,G,F

        # meta
        "id": "",  # datetime.now().strftime('%Y%m%d%H%M%S'), but done in backtest_once_loop
        "send_mail": False,
        "print_log": True,
        "auto": {},

        # buy focus = Select.
        "f_query_asset": ["df_today['period']>240"],  # ,'period > 240' is ALWAYS THERE FOR SPEED REASON, "trend > 0.2", filter everything from group str to price int
        "f_query_date": [],  # filter days vs filter assets. restrict some days to only but or only sell

        # selection weight  doc- bool: ascending True= small, False is big. 1st int: indicator_weight: how each indicator is weight against other indicator. e.g. {"pct_chg": [False, 0.8, 0.2]}  =》 df["pct_chg"]*0.8 + df["trend"]*0.2. 2nd int: asset_weight: how each asset indicator is weighted against its group indicator. e.g. {"pct_chg": [False, 0.8, 0.2]}  =》 df["pct_chg"]*0.2+df["pct_chg_group"]*0.8. empty means no group weight
        "s_weight1": {
            "pct_chg": [True, 5, 1],  # very important for this strategy
            "pgain2": [True, 3, 1],  # very important for this strategy
            "pgain5": [True, 2, 1],
        },

        # portfolio
        "p_capital": 1000000,  # start capital
        "p_fee": 0.0000,  # 1==100%
        "p_maxsize": 1000,  # 1: fixed number,2: percent. 3: percent with top_limit. e.g. 30 stocks vs 30% of todays trading stock
        "p_min_T+": 1,  # Start consider sell. 1 means trade on next day, aka T+1， = Hold stock for 1 night， 2 means hold for 2 nights. Preferably 0,1,2 for day trading
        "p_max_T+": 1,  # MUST sell no matter what.
        "p_proportion": False,  # choices: False(evenly), prop(proportional to rank), fibo(fibonacci)
        "p_winner_abv": False,  # options False, >1. e.g. 1.2 means if one stock gained 20%, sell
        "p_loser_und": False,  # options False, <1. e.g. 0.8 means if one stocks gained -20%, sell
        "p_add_position": False,
        "p_compare": [["I", "000001.SH"]],  # ["I", "CJ000001.SH"],  ["I", "399001.SZ"], ["I", "399006.SZ"]   compare portfolio against other performance
    }

def btest_manu(setting_master = btest_setting_master()):
    setting_master["auto"] = {} #important {} and not ()
    # settings creation  #,"G"
    for asset in ["E"]:

        setting_asset = copy.deepcopy(setting_master)
        setting_asset["f_query_asset"] += [f"df_today['group'].isin({['concept', 'industry1', 'industry2']})"] if asset == "G" else []
        setting_asset["assets"] = [asset]
        a_setting_instance = []


        # treats history of custom stocks
        custom_string="002019.SZ,002029.SZ,600409.SH,600525.SH,002157.SZ,600682.SH,002022.SZ,600036.SH,002223.SZ,600741.SH,600563.SH,002158.SZ,600990.SH,002129.SZ,600085.SH,600482.SH,600549.SH,002151.SZ,600420.SH,002201.SZ,600271.SH,000895.SZ,600507.SH,600114.SH,600201.SH,002156.SZ,002010.SZ,600660.SH,600486.SH,002273.SZ,000963.SZ,600490.SH,002222.SZ,002167.SZ,002009.SZ,002113.SZ,600985.SH,600993.SH,600585.SH,002002.SZ,002146.SZ,002108.SZ,600446.SH,300024.SZ,600352.SH,000858.SZ,002245.SZ,600426.SH,600438.SH,600522.SH,002030.SZ,600887.SH,002020.SZ,002034.SZ,002028.SZ,002126.SZ,002042.SZ,600452.SH,600987.SH,002198.SZ,002043.SZ,300015.SZ,002168.SZ,002120.SZ,002221.SZ,002081.SZ,600521.SH,600066.SH,002003.SZ,002219.SZ,002139.SZ,002262.SZ,600518.SH,002180.SZ,000651.SZ,600535.SH,002311.SZ,002127.SZ,002141.SZ,002176.SZ,600967.SH,600406.SH,002174.SZ,002085.SZ,600309.SH,002044.SZ,600572.SH,002138.SZ,002049.SZ,600570.SH,002013.SZ,002230.SZ,002007.SZ,002050.SZ,000538.SZ,002001.SZ,002038.SZ,002179.SZ,002241.SZ,002185.SZ,002210.SZ,600487.SH,002271.SZ,600276.SH,600436.SH,002178.SZ,002252.SZ,002032.SZ,002008.SZ,002035.SZ,600340.SH,600519.SH,002236.SZ"
        a_names=custom_string.split(",")
        setting_asset["f_query_asset"] += [f"df_today.index.isin({a_names})"]


        for column1 in ["close"]:
            setting_instance = copy.deepcopy(setting_asset)
            setting_instance["s_weight1"] = {}
            a_setting_instance.append(setting_instance)

        # 0 00000000000000000is_min
        # for thresh in [x/100 for x in range(0,100,5)]: #"zlmacd_close.(1, 5, 10)", "zlmacd_close.(1, 10, 20)", "zlmacd_close.(1, 240, 300)",
        #     setting_instance = copy.deepcopy(setting_asset)
        #     setting_instance["f_query_asset"] +=[f"(df_today['e_min']/df_today['close']).between({thresh},{thresh+0.05})"]
        #     setting_instance["s_weight1"] ={}

        #     a_setting_instance.append(setting_instance)
        #

        # Fresh stock by final rank
        # for period1 in [250,500,750,1000]:
        #     for period2 in [750,1500,7000]:  # "zlmacd_close.(1, 5, 10)", "zlmacd_close.(1, 10, 20)", "zlmacd_close.(1, 240, 300)",
        #         if period1<period2:
        #             setting_instance = copy.deepcopy(setting_asset)
        #             setting_instance["f_query_asset"] += [
        #                 f"df_today['period'].between({period1},{period2})",
        #                 f"(df_today['e_min']/df_today['close']).between(0.7,1)"
        #             ]
        #             setting_instance["s_weight1"] = {"bull": [True, 1, 1], }
        #
        #             a_setting_instance.append(setting_instance)

        # macd
        # for macd in ["zlmacd_close.(1, 5, 10)","zlmacd_close.(1, 10, 20)","zlmacd_close.(1, 240, 300)","zlmacd_close.(1, 300, 500)"]:
        #     for macd_bool in [10,-10]:
        #         setting_instance = copy.deepcopy(setting_asset)        #
        #         setting_instance["f_query_asset"] += [f"(df_today['{macd}']=={macd_bool})"]
        #         setting_instance["s_weight1"] ={}

        #         a_setting_instance.append(setting_instance)


        #general: quantile, ,"turnover_rate","total_mv","pe_ttm","ps_ttm","total_share","pb","vol"
        #for column in ["period","close.pgain5","close.pgain10","close.pgain20","close.pgain60","close.pgain120","close.pgain240","close","ivola"]:

        #for column1 in ["close"]:
        #    for low_quant1,high_quant1 in LB.custom_pairwise_overlap(LB.drange(0,101,4)):
        #                setting_instance = copy.deepcopy(setting_asset)
        #                setting_instance["f_query_asset"] += [f"df_today['{column1}'].between(  **LB.btest_quantile(df_today['{column1}'].quantile([{low_quant1},{high_quant1}])))"]
        #                setting_instance["s_weight1"] = {}

        #                a_setting_instance.append(setting_instance)



        #general: binary
        # for column in ["exchange"]:
        #     for column_val in ["创业板","中小板","主板"]:
        #         setting_instance = copy.deepcopy(setting_master)
        #         setting_instance["f_query_asset"] += [f"df_today['{column}']== '{column_val}'"]
        #         setting_instance["s_weight1"] = {}

        #         a_setting_instance.append(setting_instance)

        LB.print_iterables([(x["assets"], x["f_query_asset"]) for x in a_setting_instance])
        print("Total Settings:", len(a_setting_instance))
        time.sleep(10)

        if a_setting_instance: btest(settings=a_setting_instance)
    btest_overview_master(mode="manu")


def btest_auto(pair=1, setting_master = btest_setting_master()):
    """
    this btest is to single test each indicator on each asset, and put them into organized category.

    1. create combination pairs of columns. e.g. (period,),(period,close),(period,close,total_mv)
    2. create master quantile combinations. e.g. [(0.0,0,2),(0.2,0.4),(0.4,0.6),(0.6,0.8),(0.8,1.0)]
    3. create cartesian product of quantile combinations. e.g. [((0.0,0,2),(0.0,0.2)), ((0.0,0,2),(0.2,0.4)), ((0.0,0,2),(0.4,0.6)), ((0.0,0,2),(0.4,0.8)),]
    4. create dict iterables e.g. [{col1: (0.0,0,2), col2: (0.0,0.2)}, {col1: (0.0,0,2), col2: (0.2,0.4)}]
    5. iterate over the results
    The result is same as many many for loops like this:
    for x in array:
        for y in array:
            for z in array:
                for...
    This has been simplified using cartesian product and is generic adjustable with variable pair
    """

    #done E
    #todo "FD","I","F"
    for asset in ["G"]:

        #copy master setting
        setting_asset = copy.deepcopy(setting_master)
        setting_asset["f_query_asset"] += [f"df_today['group'].isin({['concept', 'industry1', 'industry2']})"] if asset == "G" else []
        setting_asset["assets"] = [asset]
        setting_asset["p_min_T+"]= 1 # to faster calculate

        setting_asset["p_max_T+"]= 1 # to faster calculate
        setting_asset["p_maxsize"]= 10 # to faster calculate
        a_setting_instance = []
        a_example_column=DB.get_example_column(asset=asset,freq="D",numeric_only=True)

        #remove unessesary columns:
        a_columns=[]
        for column in a_example_column:
            for exclude_column in ["fgain"]:
                if exclude_column not in column:
                    a_columns.append(column)

        #step 1 TODO do with pair 1,2,3
        for col_comb in LB.custom_pairwise_combination(a_array=a_columns,n=pair):

            #skip if file already exists
            portfolio_path = f"Market/CN/Btest/auto/comb_{len(col_comb)}/{'_'.join(col_comb)}/{asset}/result"
            folders = 0
            for _, dirnames, filenames in os.walk(portfolio_path):
                folders += len(dirnames)
            if folders > 0:
                print( f"{col_comb} combination exists for {asset}")
                continue

            #step 2 and 3
            q_master=LB.custom_pairwise_overlap(LB.drange(0, 101, 20))
            q_cartesian = LB.custom_pairwise_cartesian(q_master, n=pair)

            #step 4
            d_iterables=[]
            for q_comb in q_cartesian:
                result={}
                for counter,q in enumerate(q_comb):
                    result[col_comb[counter]]=q_comb[counter]
                d_iterables.append(result)

            #step 5
            for d_one_setting in d_iterables:
                setting_instance = copy.deepcopy(setting_asset)
                setting_instance["auto"] = tuple(col_comb)
                for col, q_tuple in d_one_setting.items():
                    setting_instance["f_query_asset"] += [f"df_today['{col}'].between(  **LB.btest_quantile(df_today['{col}'].quantile([{q_tuple[0]},{q_tuple[1]}])))"]
                setting_instance["s_weight1"] = {}
                a_setting_instance.append(setting_instance)

        LB.print_iterables([((f"pair {pair}"),x["assets"],x["f_query_asset"]) for x in a_setting_instance])
        print("Total Settings:", len(a_setting_instance))
        time.sleep(10)

        if a_setting_instance:
            btest(settings=a_setting_instance)
            print("finished saving all. PC lag now")
            btest_overview_master(mode="auto",pair=pair)



def btest_overview_master(mode="manu",pair=1):
    """
    1. Updates ALL overview for a mode after a complete run
    2. Creates new file every time by loading directly from result
    """

    path = f"Market/CN/Btest/manu/" if mode == "manu" else f"Market/CN/Btest/auto/comb_{pair}"
    a_df_overview=[]
    for root, dirnames, filenames in os.walk(path):
        overview_path = setting_path = ""
        for file in filenames:
            if "overview" in file:
                overview_path=os.path.join(root,file)
            if "setting" in file:
                setting_path=os.path.join(root,file)

        if overview_path and setting_path:
            print(f"summarizing..{root}")
            df_overview=pd.read_csv(overview_path)
            df_setting=pd.read_csv(setting_path)
            print(f"{len(df_overview),len(df_setting)}")
            df_overview_setting =pd.merge(left=df_overview.head(1), right=df_setting, left_on="strategy", right_on="id", sort=False)
            a_df_overview.append(df_overview_setting)

    df_btest_ov = pd.concat(a_df_overview, sort=False, ignore_index=True)
    path+= f"manu_summary" if mode == "manu" else f"/comb_{pair}_summary"
    LB.to_csv_feather(df_btest_ov, a_path=LB.a_path(path), skip_feather=True, index_relevant=False)


def btest_validation(column="total_mv",a_assets=["E","FD","I","G"]):
    """
    by logic:
    1. Btest is just looping over all date_df.
    2. Looping manuly over date_df should be the same for easy operation

    Checks if buy and trade instances are the same (they are)
    Checks if buy and sell price are the same (they are not)
    """

    for asset in a_assets:
        q_setting = LB.drange(0,101,20)
        d_preload = DB.preload(asset=asset,on_asset=False, step=1)
        d_trade_h = {}
        df_result = pd.DataFrame()

        for today,tomorrow in LB.custom_pairwise_overlap([x for x in d_preload.keys()]):
            df_today=d_preload[today]

            df_tomorrow=d_preload[tomorrow]

            for counter,(key, df_q) in enumerate(LB.custom_quantile(df=df_today, column=column, key_val=False,p_setting=q_setting).items()):

                selected_index=df_q.index

                df_future = df_tomorrow.loc[selected_index]
                df_future.dropna(how="all",inplace=True)

                print(asset,tomorrow,key)
                #add instance of today q
                # if counter==0:
                #     print("add index for ",key)
                #     d_trade_h[tomorrow]=pd.Series(list(selected_index))

                #calculate instance mean
                df_result.at[tomorrow, f"open.fgain1_{key}"] = 1+ df_future["open.fgain1"].mean()


        for x,y in LB.custom_pairwise_overlap(LB.drange(0,101,20)):
            key=f"{x},{y}"
            df_result[ f"open.fgain1_{key}_cumprod"]=df_result[f"open.fgain1_{key}"].cumprod()

        a_path=LB.a_path(f"Market/CN/Btest/valid/{asset}/{column}_q")
        LB.to_csv_feather(df=df_result,a_path=a_path,skip_feather=True)
        #
        # df_trade_h=pd.DataFrame(d_trade_h)
        # LB.to_csv_feather(df=df_trade_h,a_path=LB.a_path(f"Market/CN/Btest/Validation/{asset}/{column}_trade_h"),skip_feather=True)





if __name__ == '__main__':
        pr = cProfile.Profile()
        pr.enable()

        #btest_validation(column="bull")
        btest_manu()
        #btest_overview_master(mode="manu",pair=1)
        for n in (1,):
        #     #btest_overview_master(mode="auto",pair=n)
            #btest_auto(pair=n)
            pass


        pr.disable()
        # pr.print_stats(sort='file')

