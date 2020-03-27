import pandas as pd
import tushare as ts
import numpy as np
import API_Tushare
import LB
import os.path
import inspect
from itertools import combinations
import operator
import math
from numba import njit
import traceback
import cProfile
from scipy.stats import gmean
from tqdm import tqdm
import ICreate
import Sandbox

from LB import *

pd.options.mode.chained_assignment = None  # default='warn'
pro = ts.pro_api('c473f86ae2f5703f58eecf9864fa9ec91d67edbc01e3294f6a4f9c32')
ts.set_token("c473f86ae2f5703f58eecf9864fa9ec91d67edbc01e3294f6a4f9c32")


def update_general_trade_cal(start_date="19900101", end_date="20250101"):
    df = API_Tushare.my_trade_cal(start_date=start_date, end_date=end_date).set_index(keys="cal_date", inplace=False, drop=True)
    LB.to_csv_feather(df, LB.a_path("Market/CN/General/trade_cal_D"))


# measures which day of week/month performs generally better
def update_date_seasonal_stats(group_instance="asset_E"):
    """this needs to be performed first before using the seasonal matrix"""
    path = f"Market/CN/Backtest_Single/seasonal/all_date_seasonal_{group_instance}.xlsx"
    pdwriter = pd.ExcelWriter(path, engine='xlsxwriter')

    # perform seasonal stats for all stock market or for some groups only
    df_group = get_stock_market_all().reset_index() if group_instance == "" else get_asset(ts_code=group_instance, asset="G").reset_index()

    # get all different groups
    a_groups = [[LB.get_trade_date_datetime_dayofweek, "dayofweek"],
                [LB.get_trade_date_datetime_d, "dayofmonth"],
                [LB.get_trade_date_datetime_weekofyear, "weekofyear"],
                [LB.get_trade_date_datetime_dayofyear, "dayofyear"],
                [LB.get_trade_date_datetime_m, "monthofyear"],
                [LB.get_trade_date_datetime_s, "seasonofyear"],]

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


def update_general_trade_date(freq="D", market="CN"):
    def update_general_trade_date_stockcount(df, market="CN"):
        df.index = df["trade_date"].astype(int)
        for asset in c_assets():
            df_ts_codes = get_ts_code(a_asset=[asset])
            df_ts_codes = df_ts_codes.rename(columns={"list_date": "trade_date"})
            df_grouped = df_ts_codes[["name", "trade_date"]].groupby(by="trade_date").count()
            # vecorized approach faster than loop over individual date
            df[f"{asset}_count"] = df_grouped["name"].astype(int).cumsum()
            df[f"{asset}_count"] = df[f"{asset}_count"].fillna(method="ffill")
            df[f"{asset}_count"] = df[f"{asset}_count"].fillna(0)
        return df

    def update_general_trade_date_seasonal_score(df_trade_date, freq="D", market="CN"):
        """this requires the date_seasonal matrix to be updated first """
        # get all indicator for each day
        path_indicator = "Market/CN/Backtest_Single/seasonal/all_date_seasonal.xlsx"
        xls = pd.ExcelFile(path_indicator)

        # get all different groups
        a_groups = [[LB.get_trade_date_datetime_dayofweek, "dayofweek"],  # 1-5
                    [LB.get_trade_date_datetime_m, "monthofyear"],  # 1-12
                    [LB.get_trade_date_datetime_d, "dayofmonth"],  # 1-31
                    [LB.get_trade_date_datetime_weekofyear, "weekofyear"],  # 1-51
                    [LB.get_trade_date_datetime_dayofyear, "dayofyear"],  # 1-365
                    [LB.get_trade_date_datetime_s, "seasonofyear"],  # 1-365
                    ]

        # transform all trade_date into different format
        for group in a_groups:
            df_trade_date[group[1]] = df_trade_date["trade_date"].apply(lambda x: group[0](x)).astype(str)
            df_score_board = pd.read_excel(xls, sheet_name=group[1], converters={group[1]: lambda x: str(x)})
            df_score_board = df_score_board.fillna(0.0)  # if no information about certain period, assume the change is 0
            df_score_board[group[1]] = df_score_board[group[1]].astype(str)

            df_score_board = df_score_board.rename(columns={"pct_chg": f"{group[1]}_score"})
            df_trade_date[group[1]] = df_trade_date[group[1]].astype(str)
            df_trade_date = pd.merge(df_trade_date, df_score_board, how='left', on=group[1], suffixes=["", ""], sort=False)
            df_trade_date[f"{group[1]}_score"] = df_trade_date[f"{group[1]}_score"].fillna(0.0)  # if information not enough, assume the pct_chg is 0.0

        df_trade_date["seasonal_score"] = df_trade_date["dayofweek_score"] * 0.15 + \
                                          df_trade_date["dayofmonth_score"] * 0.25 + \
                                          df_trade_date["weekofyear_score"] * 0.20 + \
                                          df_trade_date["dayofyear_score"] * 0.05 + \
                                          df_trade_date["monthofyear_score"] * 0.35

        df_trade_date["seasonal_score"] = df_trade_date["seasonal_score"].rolling(2).mean()
        return df_trade_date

    if freq in ["D", "W"]:
        a_path = LB.a_path(f"Market/{market}/General/trade_date_{freq}")
        df = API_Tushare.my_pro_bar(ts_code="000001.SH", start_date="00000000", end_date=today(), freq=freq, asset="I")
        df = LB.df_reverse_reindex(df)

        df = df[["trade_date"]]
        df = update_general_trade_date_stockcount(df)  # adds E,I,FD count
        # df = update_general_trade_date_seasonal_score(df, freq, market)  # TODO adds seasonal score for each day
        LB.to_csv_feather(df, a_path, index_relevant=False)


def update_general_ts_code(asset="E", market="CN", big_update=True):
    print("start update general ts_code ", asset)
    if (asset == "E"):
        df = API_Tushare.my_stockbasic(is_hs="", list_status="L", exchange="").set_index("ts_code")

        # add exchange info for each stock
        df["exchange"] = ["创业板" if x[0:3] in ["300"] else "中小板" if x[0:3] in ["002"] else "主板" if x[0:2] in ["00", "60"] else float("nan") for x in df.index]

        # add SW industry info for each stock
        for level in c_industry_level():
            df_industry_instance = pd.DataFrame()
            df_industry_member = get_industry_member(level)
            for industry_index in df_industry_member.index:
                df_industry_instance = df_industry_instance.append(get_industry_index(level=level, index=industry_index), sort=False)
            df_industry_instance.index.name = "ts_code"
            df[f"industry{level}"] = df_industry_instance["index_code"].replace(to_replace=df_industry_member[f"industry{level}"].to_dict())

        # LB.columns_remove(df, ["index_code" + x for x in c_industry_level()])

        # add State Government for each stock
        df["state_company"] = False
        for ts_code in df.index:
            print("update state_company", ts_code)
            df_government = get_assets_top_holder(ts_code=ts_code, columns=["holder_name", "hold_ratio"])
            if df_government.empty:  # if empty, assume it is False
                continue
            df_government_grouped = df_government.groupby(by="holder_name").mean()
            df_government_grouped = df_government_grouped["hold_ratio"].nlargest(n=1)  # look at the top 4 share holders

            counter = 0
            for top_holder_name in df_government_grouped.index:
                if ("公司" in top_holder_name) or (len(top_holder_name) > 3):
                    counter += 1
            df.at[ts_code, "state_company"] = True if counter >= 1 else False

    elif (asset == "I"):
        df_SSE = API_Tushare.my_index_basic(market='SSE')
        df_SZSE = API_Tushare.my_index_basic(market='SZSE')
        df = df_SSE.append(df_SZSE, sort=False).set_index("ts_code")
    elif (asset == "FD"):
        df_E = API_Tushare.my_fund_basic(market='E')
        df_O = API_Tushare.my_fund_basic(market='O')
        df = df_E.append(df_O, sort=False).set_index("ts_code")
    elif (asset == "G"):
        df = pd.DataFrame()
        for on_asset in c_assets():
            for group, a_instance in c_d_groups(assets=[on_asset]).items():
                for instance in a_instance:
                    df.at[f"{group}_{instance}", "name"] = f"{group}_{instance}"
                    df.at[f"{group}_{instance}", "on_asset"] = on_asset
                    df.at[f"{group}_{instance}", "group"] = str(group)
                    df.at[f"{group}_{instance}", "instance"] = str(instance)
        df.index.name = "ts_code"
    elif (asset== "F"):
        df=API_Tushare.my_fx_daily(start_date=get_last_trade_date(freq="D")).set_index("ts_code")
        df=df[~df.index.duplicated(keep="last")]
        df["list_date"]=np.nan
        df=df[["list_date"]]
    elif (asset == "B"):
        #可转债，相当于股票。可以选择换成股票，也可以选择换成利率。网上信息很少，几乎没人玩
        df = API_Tushare.my_cb_basic().set_index("ts_code")
        df["list_date"] = np.nan

        #中债，国债
        #only yield curve, no daily data



    df["asset"] = asset
    df["list_date"] = df["list_date"].fillna(method='ffill') if asset not in ["G","F"] else np.nan
    a_path = LB.a_path(f"Market/{market}/General/ts_code_{asset}")
    LB.to_csv_feather(df, a_path)


@LB.deco_only_big_update
def update_general_industry(level, market="CN", big_update=True):
    # industry member list
    df_member = API_Tushare.my_index_classify(f"L{level}")
    df = df_member[["index_code", "industry_name"]].rename(columns={"industry_name": f"industry{level}"}).set_index("index_code")
    LB.to_csv_feather(df, a_path=LB.a_path(f"Market/{market}/General/industry_{level}"))

    # industry instance
    for index in df_member["index_code"]:
        df = API_Tushare.my_index_member(index).set_index(keys="con_code", drop=True)
        LB.to_csv_feather(df, LB.a_path(f"Market/{market}/General/industry/{level}/{index}"))


def update_assets_EIFD_D(asset="E", freq="D", market="CN", step=1, big_update=True):

    def merge_saved_df_helper(df,df_saved):
        df.set_index("trade_date",inplace=True)
        return df_saved.append(df, sort=False)
        # df = LB.df_drop_duplicated_reindex(df, "trade_date")

    def set_index_helper(df):
        df.set_index(keys="trade_date", drop=True, inplace=True)
        df.index = df.index.astype(int)
        return df

    a_path_empty = LB.a_path("Market/CN/General/ts_code_ignore")

    # init
    df_ts_codes = get_ts_code(a_asset=[asset])
    real_latest_trade_date = get_last_trade_date(freq)

    # iteratve over ts_code
    for ts_code in df_ts_codes.index[::step]:
        start_date, middle_date, end_date, complete_new_update, df_saved = ("00000000", "20050101", today(), True, pd.DataFrame())  # True means update from 00000000 to today, False means update latest_trade_date to today
        a_path = LB.a_path(f"Market/{market}/Asset/{asset}/{freq}/{ts_code}")

        # file exists--> check latest_trade_date, else update completely new
        if os.path.isfile(a_path[1]):  # or os.path.isfile(a_path[1])
            try:
                df_saved = get_asset(ts_code=ts_code, asset=asset, freq=freq, market=market)  # get latest file trade_date
                asset_latest_trade_date = str(df_saved.index[-1])
            except:
                asset_latest_trade_date = start_date
                print(asset, ts_code, freq, end_date, "EMPTY - START UPDATE", asset_latest_trade_date, " to today")

            # file exist and on latest date--> finish, else update
            if (str(asset_latest_trade_date) == str(real_latest_trade_date)):

                print(asset, ts_code, freq, end_date, "Up-to-date", real_latest_trade_date)
                continue
            else:  # file exists and not on latest date
                # file exists and not on latest date, AND stock trades--> update
                complete_new_update = False

        # file not exist or not on latest_trade_date --> update
        if (asset == "E" and freq == "D"):
            # 1.1 get df
            if complete_new_update:
                df1 = update_assets_EI_D_reindex_reverse(ts_code, freq, asset, start_date, middle_date, adj="hfq")
                df2 = update_assets_EI_D_reindex_reverse(ts_code, freq, asset, middle_date, end_date, adj="hfq")
                df = df1.append(df2, ignore_index=True, sort=False)
                df = set_index_helper(df)
            else:
                df = update_assets_EI_D_reindex_reverse(ts_code, freq, asset, asset_latest_trade_date, end_date, adj="hfq")



            # 1.2 get adj factor because tushare is too dump to calculate it on its own
            df_adj = API_Tushare.my_query(api_name='adj_factor', ts_code=ts_code, start_date=start_date, end_date=end_date)
            if df_adj.empty:
                print(asset, ts_code, freq, start_date, end_date, "has no adj_factor yet. skipp")
            else:
                latest_adj = df_adj.at[0, "adj_factor"]
                df[["open", "high", "low", "close"]] = df[["open", "high", "low", "close"]] / latest_adj

            # 2.1 get daily basic
            if complete_new_update:
                df_fun_1 = API_Tushare.my_query(api_name="daily_basic", ts_code=ts_code, start_date=start_date, end_date=middle_date)
                df_fun_2 = API_Tushare.my_query(api_name="daily_basic", ts_code=ts_code, start_date=middle_date, end_date=end_date)
                df_fun = df_fun_1.append(df_fun_2, ignore_index=True, sort=False)
            else:
                df_fun = API_Tushare.my_query(api_name="daily_basic", ts_code=ts_code, start_date=asset_latest_trade_date, end_date=end_date)

            try:  # new stock can cause error here
                df_fun = df_fun[["trade_date", "turnover_rate", "pe_ttm", "pb", "ps_ttm", "dv_ttm", "total_share", "total_mv"]]
                df_fun["total_share"] = df_fun["total_share"] * 10000
                df_fun["total_mv"] = df_fun["total_mv"] * 10000
                df_fun["trade_date"] = df_fun["trade_date"].astype(int)
                df = pd.merge(df, df_fun, how='left', on=["trade_date"], suffixes=[False, False], sort=False).set_index("trade_date")
            except Exception as e:
                print("error fun", e)

            # 2.2 add FUN financial report aka fina
            # 流动资产合计,非流动资产合计,资产合计，   流动负债合计,非流动负债合计,负债合计
            df_balancesheet = get_assets_E_D_Fun("balancesheet", ts_code=ts_code, columns=["total_cur_assets", "total_assets", "total_cur_liab", "total_liab"])
            # 营业活动产生的现金流量净额	，投资活动产生的现金流量净额,筹资活动产生的现金流量净额
            df_cashflow = get_assets_E_D_Fun("cashflow", ts_code=ts_code, columns=["n_cashflow_act", "n_cashflow_inv_act", "n_cash_flows_fnc_act"])
            # 扣除非经常性损益后的净利润,净利润同比增长(netprofit_yoy instead of q_profit_yoy on the doc)，营业收入同比增长,销售毛利率，销售净利率,资产负债率,存货周转天数(should be invturn_days,but casted to turn_days in the api for some reasons)
            df_indicator = get_assets_E_D_Fun("fina_indicator", ts_code=ts_code, columns=["profit_dedt", "netprofit_yoy", "or_yoy", "grossprofit_margin", "netprofit_margin", "debt_to_assets", "turn_days"])
            # 股权质押比例
            df_pledge_stat = get_assets_pledge_stat(ts_code=ts_code, columns=["pledge_ratio"])

            # add fina to df

            for df_fun, fun_name in zip([df_balancesheet, df_cashflow, df_indicator, df_pledge_stat], ["bala", "cash", "indi", "pledge"]):
                df_fun.index.name = "trade_date"
                df_fun.index = df_fun.index.astype(int)
                df[list(df_fun.columns)] = df_fun[list(df_fun.columns)]
                # df = pd.merge(df, df_fun, how="left", on="trade_date", sort=False).set_index("trade_date")  # just added might be wrong

            # append old df and drop duplicates
            if not df_saved.empty:
                df = df_saved.append(df, sort=False)
            df = df[~df.index.duplicated(keep="last")]  # important

            # interpolate/fill between empty fina and pledge_stat values
            all_report_label = list(df_balancesheet.columns.values) + list(df_cashflow.columns.values) + list(df_indicator.columns.values) + ["pledge_ratio"]
            for label in all_report_label:
                try:
                    df[label] = df[label].fillna(method='ffill')
                except:
                    pass

        elif (asset == "I" and freq == "D"):
            if complete_new_update:
                df = update_assets_EI_D_reindex_reverse(ts_code, freq, asset, start_date, end_date, adj="qfq")
                df = set_index_helper(df)
            else:
                df = update_assets_EI_D_reindex_reverse(ts_code, freq, asset, asset_latest_trade_date, end_date, adj="qfq")
                df=merge_saved_df_helper(df=df,df_saved=df_saved)

        elif (asset == "FD" and freq == "D"):
            if complete_new_update:
                df = update_assets_FD_D_reindex_reverse(ts_code, freq, asset, start_date, end_date)
                df = set_index_helper(df)
            else:
                df = update_assets_FD_D_reindex_reverse(ts_code, freq, asset, asset_latest_trade_date, end_date)
                df= merge_saved_df_helper(df=df,df_saved=df_saved)
        elif (asset == "F" and freq == "D"):
            if complete_new_update:
                df = update_assets_F_D_reindex_reverse(ts_code, start_date, end_date)
                df = set_index_helper(df)
            else:
                df = update_assets_F_D_reindex_reverse(ts_code, asset_latest_trade_date, end_date)
                df= merge_saved_df_helper(df=df,df_saved=df_saved)

        #duplicated index ward
        df=df.loc[~df.index.duplicated(keep="first")]

        """the problem above is df(new) has trade date as column. df_saved has trade_date as index right?"""
        # 3. add my derivative indices and save it
        if not df.empty:
            update_assets_EIFD_D_technical(df=df, asset=asset)
            update_assets_EIFD_D_expanding(df=df, ts_code=ts_code)
        LB.to_csv_feather(df=df, a_path=a_path)
        print(asset, ts_code, freq, end_date, "UPDATED!", real_latest_trade_date)


# For all Pri indices and derivates. ordered after FILO
def update_assets_EIFD_D_technical(df, asset="E", bfreq=c_bfreq()):
    for rolling_freq in bfreq[::-1]:
        ICreate.pgain(df=df, ibase="close", freq=rolling_freq)  # past gain includes today = today +yesterday comp_gain
    for rolling_freq in bfreq[::-1]:
        ICreate.fgain(df=df, ibase="close", freq=rolling_freq)  # future gain does not include today = tomorrow+atomorrow comp_gain

    for rolling_freq in [1, 2, 5][::-1]:
        ICreate.pgain(df=df, ibase="open", freq=rolling_freq)  # past gain includes today = today +yesterday comp_gain
    for rolling_freq in [1, 2, 5][::-1]:
        ICreate.fgain(df=df, ibase="open", freq=rolling_freq)  # future gain does not include today = tomorrow+atomorrow comp_gain

    ICreate.ivola(df=df)  # 0.890578031539917 for 300 loop
    ICreate.period(df=df)  # 0.2 for 300 loop
    ICreate.pjup(df=df)  # 1.0798187255859375 for 300 loop
    ICreate.pjdown(df=df)  # 1.05 independend for 300 loop
    ICreate.co_pct_chg(df=df)

    # ICreate.cdl(df,ibase="cdl")  # VERY SLOW. NO WAY AROUND. 120 sec for 300 loop
    # if asset == "E":  # else sh_index will try to get corr wit himself during update
    #     ICreate.deri_sta(df=df, ibase="close", freq=BFreq.f5, ideri=ICreate.IDeri.corr, re=ICreate.RE.r)
    #     ICreate.deri_sta(df=df, ibase="close", freq=BFreq.f10, ideri=ICreate.IDeri.corr, re=ICreate.RE.r)

    # trend support and resistance
    # add trend for individual stocks
    # ICreate.trend(df=df, ibase="close")
    # df = support_resistance_horizontal(df_asset=df)

    # macd for strategy
    for sfreq, bfreq in [(5, 10), (10, 20), (240, 300), (300, 500)]:
        # for sfreq, bfreq in LB.custom_pairwise_combination([5, 10, 20, 40, 60, 120, 180, 240, 300, 500], 2):
        if sfreq < bfreq:
            Sandbox.my_macd(df=df, ibase="close", sfreq=sfreq, bfreq=bfreq, type=1, score=10)


def update_assets_EIFD_D_expanding(df, ts_code):
    """
    for each asset, calculate expanding information. Standalone from actual information to prevent repeated calculation
        EXPANDING
        1. GMEAN
        2. BETA
        3. ENTROPY
        4. VOLATILITY
        5. days above ma5
        """

    a_freqs = [5, 20, 60, 240]

    # 1. Geomean.
    # one time calculation
    df["e_gmean"] = 1 + (df["pct_chg"] / 100)
    df["e_gmean"] = df["e_gmean"].expanding(240).apply(gmean, raw=False)
    print(f"{ts_code} gmean finished")

    # 2. times above ma, bigger better
    for freq in a_freqs:
        # one time calculation
        df[f"hp{freq}"] = Sandbox.highpass(df["close"], freq)
        df[f"lp{freq}"] = df["close"] - df[f"hp{freq}"]
        df[f"ma{freq}"] = df["close"].rolling(freq).mean()
        df[f"abv_ma{freq}"] = (df["close"] > df[f"ma{freq}"]).astype(int)
        df[f"abv_lp{freq}"] = (df["close"] > df[f"ma{freq}"]).astype(int)

        # expanding
        df[f"e_abv_ma{freq}"] = df[f"abv_ma{freq}"].expanding(freq).mean()
    print(f"{ts_code} abv_ma finished")

    # 3. trend swap. how long a trend average lasts too slow
    # for freq in a_freqs:
    #     #expanding
    #     for until_index,df_expand in custom_expand(df,freq).items():
    #         df.at[until_index,f"e_abv_ma_days{freq}"]=LB.trend_swap(df_expand, f"abv_ma{freq}", 1)
    # print(f"{ts_code} abv_ma_days finished")

    # 4. volatility of the high pass, the smaller the better
    for freq in a_freqs:
        df[f"e_hp_mean{freq}"] = df[f"hp{freq}"].expanding(freq).mean()
        df[f"e_hp_std{freq}"] = df[f"hp{freq}"].expanding(freq).std()
    print(f"{ts_code} highpass_mean finished")

    # volatility pct_ chg, less than better
    df["rapid_down"] = (df["pct_chg"] < -5).astype(int)
    df["e_rapid_down"] = df["rapid_down"].expanding(240).mean()
    print(f"{ts_code} rapid_down finished")

    # beta, lower the better too slow
    # df_sh_index = get_asset(ts_code="000001.SH", asset="I")
    # df_sh_index["sh_close"] = df_sh_index["close"]
    # df_cy_index = get_asset(ts_code="399006.SZ", asset="I")
    # df_cy_index["cy_close"] = df_cy_index["close"]
    # for until_index, df_expand in custom_expand(df, 240).items():
    #     beta_sh=LB.calculate_beta(df_expand["close"], df_sh_index["sh_close"])
    #     beta_cy=LB.calculate_beta(df_expand["close"], df_cy_index["cy_close"])
    #     df.at[until_index, "e_beta_sh"] = beta_sh
    #     df.at[until_index, "e_beta_cy"] = beta_cy
    #     df.at[until_index, "e_beta"] = abs(beta_sh) * abs(beta_cy)
    # print(f"{ts_code} beta finished")

    # is_max. How long the current price is around the all time high. higher better
    df["e_max"] = df["close"].expanding(240).max()
    df["e_max_pct"] = (df["close"] / df["e_max"]).between(0.9, 1.1).astype(int)
    df["e_max_pct"] = df["e_max_pct"].expanding(240).mean()

    # is_min
    df["e_min"] = df["close"].expanding(240).min()
    df["e_min_pct"] = (df["close"] / df["e_min"]).between(0.9, 1.1).astype(int)
    df["e_min_pct"] = df["e_min_pct"].expanding(240).mean()
    print(f"{ts_code} max finished")


# def update_ts_code_expanding(step=1):
#     d_all_df=preload(load="asset", step=step, query="")
#     for counter, (ts_code, df) in enumerate(d_all_df.items()):
#         a_path = LB.a_path(f"Market\CN\Asset\E\D_Expanding/{ts_code}")
#
#         #skip if file exists:
#         if os.path.isfile(a_path[1]):
#             print(counter,ts_code, "expanding up-to-date")
#             continue
#         else:
#             df_expanding=update_assets_EIFD_D_expanding(df, ts_code)
#             LB.to_csv_feather(df_expanding,a_path=a_path)
#             print(counter, ts_code, "expanding updated")




def break_tushare_limit_helper(func,kwargs,limit=1000):
    """for some reason tushare only allows fund，forex to be given at max 1000 entries per request"""
    df =func(**kwargs)
    len_df_this = len(df)
    df_last = df
    while len_df_this == limit:  # TODO if this is fixed or add another way to loop
        kwargs["end_date"]=df_last.at[len(df_last) - 1, "trade_date"]
        df_this=func(**kwargs)
        if (df_this.equals(df_last)):
            break
        df = df.append(df_this, sort=False, ignore_index=True).drop_duplicates(subset="trade_date")
        len_df_this = len(df_this)
        df_last = df_this
    return df

# For E,I,FD
def update_assets_EI_D_reindex_reverse(ts_code, freq, asset, start_date, end_date, adj="qfq", market="CN"):
    df = API_Tushare.my_pro_bar(ts_code=ts_code, freq=freq, asset=asset, start_date=str(start_date), end_date=str(end_date), adj=adj)
    df = LB.df_reverse_reindex(df)
    LB.columns_remove(df, ["pre_close", "amount", "change"])
    return df

# For F Tested and works
def update_assets_F_D_reindex_reverse(ts_code, start_date,end_date):
    df=break_tushare_limit_helper(func=API_Tushare.my_fx_daily,kwargs={"ts_code":ts_code,"start_date":f"{start_date}", "end_date":f"{end_date}"},limit=1000)
    df = LB.df_reverse_reindex(df)
    for column in ["open","high","low","close"]:
        df[column]=(df[f"bid_{column}"]+df[f"ask_{column}"])/2
    df["pct_chg"]=df["close"].pct_change()*100
    return df[["trade_date","ts_code","open", "high", "low", "close","pct_chg","tick_qty"]]

# For FD D
def update_assets_FD_D_reindex_reverse(ts_code, freq, asset, start_date, end_date, adj=None, market="CN"):
    df = break_tushare_limit_helper(func=API_Tushare.my_pro_bar, kwargs={"ts_code": ts_code, "start_date": f"{start_date}", "end_date": f"{end_date}","adj":adj, "freq":freq, "asset":asset},limit=1000)

    #TODO to be deletet if breaker works
    #df = API_Tushare.my_pro_bar(ts_code=ts_code, freq=freq, asset=asset, start_date=str(start_date), end_date=str(end_date), adj=adj)
    # request_df_len = len(df)
    # last_df2 = df
    #
    # """for some reason tushare only allows fund to be given at max 1000 entries per request"""
    # while request_df_len == 1000:  # TODO if this is fixed or add another way to loop
    #     middle_date = last_df2.at[len(last_df2) - 1, "trade_date"]
    #     df2 = API_Tushare.my_pro_bar(ts_code=ts_code, freq=freq, asset=asset, start_date=str(start_date), end_date=str(middle_date), adj=adj)
    #     if (df2.equals(last_df2)):
    #         break
    #     df = df.append(df2, sort=False, ignore_index=True)
    #     df = df.drop_duplicates(subset="trade_date")
    #     request_df_len = len(df2)
    #     last_df2 = df2

    """
    不复权 = 无
    前复权 = 当日收盘价 × 当日复权因子 / 最新复权因子 (This is the standard Xueqiu also uses)
    后复权 = 当日收盘价 x 当日复权因子
    Fund 前复权was tested and compared to xueqiu and it was CORRECT
    """
    if not df.empty:
        df.index = df["trade_date"]
        df_adj_factor = API_Tushare.my_query(api_name='fund_adj', ts_code=ts_code, start_date=start_date, end_date=end_date)
        if df_adj_factor.empty:
            print(asset, ts_code, freq, "has no adj_factor, skip")
        else:
            df_adj_factor.index = df_adj_factor.index.astype(int)
            latest_adj = df_adj_factor.at[0, "adj_factor"]
            df_adj_factor.index = df_adj_factor["trade_date"]
            df["adj_factor"] = df_adj_factor["adj_factor"]
            df["adj_factor"] = df["adj_factor"].interpolate()  # interpolate between alues because some dates are missing from tushare

            #df[["open", "high", "low", "close"]] = df[["open", "high", "low", "close"]] * df.adj_factor / latest_adj
            for column in ["open", "high", "low", "close"]:
                #for debug include 不复权and后复权
                #df[f"{column}_不复权"]=df[column]
                #df[f"{column}_后复权"]=df[column] * df["adj_factor"]
                df[column] = df[column] * df["adj_factor"] / latest_adj
        df = LB.df_reverse_reindex(df)
        LB.columns_remove(df, ["pre_close", "amount", "change", "adj_factor"])
    return df


# recommended update frequency W
def update_assets_E_W_pledge_stat(start_date="00000000", market="CN", step=1, big_update=True):
    for counter, ts_code in enumerate(get_ts_code(["E"]).index[::step]):
        a_path = LB.a_path(f"Market/{market}/Asset/E/W_pledge_stat/{ts_code}")
        if os.path.isfile(a_path[1]) and (not big_update):
            print(counter, ts_code, "pledge_stat Up-to-date")
            continue
        else:  # always update completely new
            df = API_Tushare.my_pledge_stat(ts_code=ts_code)
            df = LB.df_reverse_reindex(df).set_index("end_date")
            LB.to_csv_feather(df, a_path)
            print(counter, ts_code, "pledge_stat UPDATED")


def update_assets_E_D_top_holder(big_update=True, market="CN", step=1):
    df_ts_codes = API_Tushare.my_stockbasic(is_hs="", list_status="L", exchange="").set_index("ts_code")
    for counter, ts_code in enumerate(df_ts_codes.index[::step]):
        a_path = LB.a_path(f"Market/{market}/Asset/E/D_top_holder/{ts_code}")
        if os.path.isfile(a_path[1]) and (not big_update):  # skipp if small update or file exist
            print(counter, ts_code, "top_holder Up-to-date")
            continue
        else:  # always update completely new
            df = API_Tushare.my_query(api_name='top10_holders', ts_code=ts_code, start_date='20190101', end_date=today())  # NO INDEX SET HERE
            df = LB.df_reverse_reindex(df)  # .set_index()
            LB.to_csv_feather(df, a_path, index_relevant=False)
            print(counter, ts_code, "top_holder UPDATED")


def update_assets_E_D_Fun(start_date="0000000", end_date=LB.today(), market="CN", step=1, big_update=True):
    for counter, ts_code in enumerate(get_ts_code(["E"]).index[::step]):
        for fina_name, fina_function in c_assets_fina_function_dict().items():
            a_path = LB.a_path(f"Market/{market}/Asset/E/D_Fun/{fina_name}/{ts_code}")
            if os.path.isfile(a_path[1]) and (not big_update):  # skipp if small update or file exist
                print(counter, ts_code, f"{ts_code} {fina_name} Up-to-date")
                continue
            else:  # always update completely new
                df = fina_function(ts_code=ts_code, start_date=start_date, end_date=end_date)
                df = LB.df_reverse_reindex(df).set_index("end_date")
                LB.to_csv_feather(df, a_path)
                print(counter, ts_code, f"{ts_code} {fina_name} UPDATED")


def update_assets_G_D(assets=["E"], big_update=True):
    """there are 2 approach to do this
    1. Asset to industry group and then create date
    2. Asset to date and then filter only stocks that belongs to one industry

    This approach is using procedure 2. But both approach should produce same result
    Requiement: df_date
    """

    # initialize all group as dict
    d_group_instance_update = {}  # dict of array
    d_group_instance_saved = {}  # dict of array
    for group, a_instance in c_d_groups(assets=assets).items():
        for instance in a_instance:
            d_group_instance_update[f"{group}_{instance}"] = []
            d_group_instance_saved[f"{group}_{instance}"] = get_asset(f"{group}_{instance}", asset="G")

    # get last saved trade_date on df_saved
    last_trade_date = get_last_trade_date("D")
    example_df = d_group_instance_saved["asset_E"]
    try:
        last_saved_date = example_df.index[-1]
    except:
        last_saved_date = "19990101"

    # if small update and saved_date==last trade_date
    if last_trade_date == last_saved_date and (not big_update):
        return print("ALL GROUP Up-to-date")

    # initialize trade_date
    df_trade_date = get_trade_date(end_date=today(), freq="D")
    df_trade_date = df_trade_date[df_trade_date.index > int(last_saved_date)]
    print("START UPDATE GROUP since", last_saved_date)

    # loop over date and get mean
    for trade_date in df_trade_date.index:  # for each day
        print(trade_date, "Updating GROUP")
        df_date = get_date(trade_date=trade_date, a_assets=assets, freq="D")
        for group, a_instance in c_d_groups(assets=assets).items():  # for each group
            df_date_grouped = df_date.groupby(by=group, ).mean()  # calculate mean
            for instance, row in df_date_grouped.iterrows():  # append to dict
                d_group_instance_update[f"{group}_{instance}"].append(row)

    # save all to df
    for (key, a_update_instance), (key_saved, df_saved) in zip(d_group_instance_update.items(), d_group_instance_saved.items()):
        df_update = pd.DataFrame(a_update_instance)
        if not df_saved.empty:
            df_update = pd.concat(objs=[df_saved, df_update], sort=False, ignore_index=True)
        if not df_update.empty:
            df_update.set_index(keys="trade_date", drop=True, inplace=True)  # reset index after group
        df_update.insert(0, "ts_code", key)
        LB.to_csv_feather(df_update, LB.a_path(f"Market/CN/Asset/G/D/{key}"))
        print(key, "UPDATED")



def update_date(asset="E", freq="D", market="CN", big_update=True, step=1, naive=False):
    """step -1 might be wrong if trade dates and asset are updated seperately. then they will not align
        step 1 always works
        naive: approach always works, but is extremly slow
    """
    if step not in [1,-1] :
        return print("STEP only 1 or -1 !!!!")

    # get the latest column of the asset file
    if asset=="E":
        code = "000001.SZ"
    elif asset == "I":
        code = "000001.SH"
    elif asset == "FD":
        code = "150008.SZ"
    elif asset == "F":
        code = "AUDCAD.FXCM"
    elif asset == "G":
        code = "area_安徽"
    else:
        code = "000001.SZ"
    example_column = list(get_asset(code, asset, freq).columns)

    df_static_data = get_ts_code(a_asset=[asset])
    df_trade_dates = get_trade_date("00000000", LB.today(), freq,market=market)

    d_list_date = {ts_code: row["list_date"] for ts_code, row in get_ts_code(a_asset=[asset]).iterrows()}
    d_queries_ts_code=c_G_queries() if asset=="G" else {}
    d_preload = preload(asset=asset,step=1,period_abv=240,d_queries_ts_code=d_queries_ts_code)
    d_lookup_table = {ts_code: (0 if step==1 else len(df) - 1) for ts_code, df in d_preload.items()}

    for trade_date in df_trade_dates.index[::step]:  # IMPORTANT! do not modify step, otherwise lookup will not work
        a_path = LB.a_path(f"Market/{market}/Date/{asset}/{freq}/{trade_date}")
        a_date_result = []

        # date file exists AND not big_update. If big_update, then always overwrite
        if os.path.isfile(a_path[0]) and (not big_update):

            if naive:#fallback strategies is the naive approach
                continue
            else:
                # update lookup table before continue. So that skipped days still match
                for ts_code, df_asset in d_preload.items():
                    row_number = d_lookup_table[ts_code]
                    if step == 1 and row_number > len(df_asset) - 1:
                        continue
                    if step == -1 and row_number < 0:
                        continue
                    if int(df_asset.index[row_number]) == int(trade_date):
                        d_lookup_table[ts_code] += step
                print(asset, freq, trade_date, "date file Up-to-date")

        # date file does not exist or big_update
        else:
            for counter, (ts_code, df_asset) in enumerate(d_preload.items()):

                if naive:
                    try:
                        a_date_result.append(df_asset.loc[trade_date].to_numpy().flatten())
                    except:
                        #it is totally normal to have not that day for each df_asset
                        pass
                else:
                    if len(df_asset)==0:
                        print(f"{trade_date} skip ts_code {ts_code} because len is 0")
                        continue

                    #if asset list date is in future: not IPO yet
                    list_date=d_list_date[ts_code]
                    if type(list_date) in [str, int]:
                        if int(list_date) > int(trade_date):
                            print(f"{trade_date} skip ts_code {ts_code} because IPO in future")
                            continue

                    row_number = d_lookup_table[ts_code]  # lookup table can not be changed while iterating over it.

                    #this df_asset is already at last row. Is finished
                    if step==1 and row_number>len(df_asset)-1:
                        continue
                    if step==-1 and row_number<0:
                        continue

                    if int(df_asset.index[row_number]) == int(trade_date):
                        a_date_result.append(df_asset.loc[trade_date].to_numpy().flatten())
                        d_lookup_table[ts_code] += step
                        #print(f"IN {trade_date} counter {counter}, step {step}, ts_code {ts_code}, len {len(df_asset)}  row number {row_number}")
                    else:
                        pass
                        #print(f"OUT {trade_date} counter {counter}, step {step}, ts_code {ts_code}, len {len(df_asset)}  row number {row_number} associated date {int(df_asset.index[row_number])}")

            #create df_date from a_date_result
            df_date = pd.DataFrame(data=a_date_result, columns=example_column)

            #remove duplicate columns that also exist in static data. Then merge
            no_duplicate_cols = df_date.columns.difference(df_static_data.columns)
            df_date = pd.merge(df_date[no_duplicate_cols], df_static_data, how='left', on=["ts_code"], suffixes=["", ""], sort=False).set_index("ts_code")#add static data
            df_date.insert(loc=0, column='trade_date', value=int(trade_date))

            #create final rank
            df_date["bull"] = df_date["e_gmean"].rank(ascending=False) * 0.70 \
                              + df_date["e_max_pct"].rank(ascending=False) * 0.08 \
                              + df_date["period"].rank(ascending=False) * 0.03 \
                              + df_date["e_hp_mean240"].rank(ascending=False) * 0.01 \
                              + df_date["e_hp_mean60"].rank(ascending=False) * 0.01 \
                              + df_date["e_hp_mean20"].rank(ascending=False) * 0.01 \
                              + df_date["e_hp_mean5"].rank(ascending=False) * 0.01 \
                              + df_date["e_hp_std240"].rank(ascending=True) * 0.01 \
                              + df_date["e_hp_std60"].rank(ascending=True) * 0.01 \
                              + df_date["e_hp_std20"].rank(ascending=True) * 0.01 \
                              + df_date["e_hp_std5"].rank(ascending=True) * 0.01 \
                              + df_date["e_abv_ma240"].rank(ascending=False) * 0.01 \
                              + df_date["e_abv_ma60"].rank(ascending=False) * 0.01 \
                              + df_date["e_abv_ma20"].rank(ascending=False) * 0.01 \
                              + df_date["e_abv_ma5"].rank(ascending=False) * 0.01 \
                              + df_date["e_rapid_down"].rank(ascending=True) * 0.03
            LB.to_csv_feather(df_date, a_path)
            print(asset, freq, trade_date, "date updated")


def update_date_E_Oth(asset="E", freq="D", market="CN", big_update=True, step=1):
    trade_dates = get_trade_date(start_date="00000000", end_date=today(), freq=freq)
    for trade_date in trade_dates["trade_date"][::step]:
        d_oth_names = c_date_oth()
        d_oth_paths = {name: LB.a_path(f"Market/{market}/Date/{asset}/{name}/{trade_date}") for name, function in d_oth_names.items()}
        for name, function in d_oth_names.items():
            if os.path.isfile(d_oth_paths[name][0]):
                print(trade_date, asset, freq, name, "Up-to-date")
            else:
                df_oth = function(trade_date)
                LB.to_csv_feather(df_oth, d_oth_paths[name])
                print(trade_date, asset, freq, name, "UPDATED")


def update_date_Oth_analysis_market_D(freq, market="CN"):  # TODO add it to all date
    path = f"Market/{market}/Date/market/{freq}.csv"
    trade_dates = get_trade_date(start_date="00000000", end_date=today(), freq=freq)
    df_result = trade_dates.copy()

    oth = c_date_oth()  # changed from array to dict
    a_block_trade_count = []
    a_holdertrade_in_count = []
    a_holdertrade_de_count = []
    a_repurchase_count = []
    a_repurchase_amount = []
    a_share_float_count = []
    for trade_date in trade_dates.trade_date:
        print("calculating market for", trade_date)
        df_block_trade = get_date_E_oth(trade_date, oth[0])
        if (df_block_trade.empty):
            a_block_trade_count.append(0)
            a_holdertrade_in_count.append(0)
            a_holdertrade_de_count.append(0)
            a_repurchase_count.append(0)
            a_repurchase_amount.append(0)
            a_share_float_count.append(0)
            continue

        a_block_trade_count.append(df_block_trade['ts_code'].nunique())

        df_holdertrade = get_date_E_oth(trade_date, oth[1])
        df_holdertrade = df_holdertrade.drop_duplicates(subset="ts_code")
        df_count = df_holdertrade.in_de.value_counts()
        try:
            a_holdertrade_in_count.append(df_count["IN"])
        except Exception:
            a_holdertrade_in_count.append(0)

        try:
            a_holdertrade_de_count.append(df_count["DE"])
        except Exception:
            a_holdertrade_de_count.append(0)

        df_repurchase = get_date_E_oth(trade_date, oth[2])
        a_repurchase_count.append(len(df_repurchase))
        a_repurchase_amount.append(df_repurchase.amount.sum())

        df_share_float = get_date_E_oth(trade_date, oth[3])
        try:
            a_share_float_count.append(df_share_float['ts_code'].nunique())
        except Exception:
            a_share_float_count.append(0)

    df_result["q_block_trade_count"] = (pd.Series(a_block_trade_count) / df_result.E) * 100
    df_result["q_holdertrade_in_count"] = (pd.Series(a_holdertrade_in_count) / df_result.E) * 100
    df_result["q_holdertrade_de_count"] = (pd.Series(a_holdertrade_de_count) / df_result.E) * 100
    df_result["q_repurchase_count"] = (pd.Series(a_repurchase_count) / df_result.E) * 100
    df_result["q_share_float_count"] = (pd.Series(a_share_float_count) / df_result.E) * 100
    df_result.to_csv(path, index=False, encoding='utf-8_sig')


def update_date_stock_market_all(start_date="00000000", end_date=today(), assets=["E"], freq="D", market="CN", comparison_index=["000001.SH", "399001.SZ", "399006.SZ"], big_update=False):
    # check if big update and if the a previous saved file exists
    last_saved_date = "19990104"
    last_trade_date = get_last_trade_date("D")
    if not big_update:
        try:
            df_saved = get_stock_market_all()
            last_saved_date = df_saved.index[len(df_saved) - 1]
            df_saved = df_saved.reset_index()
            print(last_saved_date)

            if last_saved_date == last_trade_date:
                print("Date_Base Up-to_date")
                return
        except:
            df_saved = pd.DataFrame()
    else:
        df_saved = pd.DataFrame()

    print("last saved day is", last_saved_date)

    # meta preparation
    a_path = LB.a_path(f"Market/{market}/Btest/Setup/Stock_market/all_stock_market")
    a_result = []
    df_sh_index = get_asset(ts_code="000001.SH", asset="I", freq=freq, market="CN")
    df_sh_index = df_sh_index.loc[int(last_saved_date):int(last_trade_date)]
    df_sh_index = df_sh_index.loc[df_sh_index.index > int(last_saved_date)]  # exclude last saved date from update

    # loop through all trade dates and add them together
    for trade_date, sh_pct_chg in zip(df_sh_index.index, df_sh_index["pct_chg"]):
        print(trade_date, "being added to all_stock_market_base")
        df_date = get_date(str(trade_date), a_assets=assets, freq=freq, market="CN")
        df_date = df_date[df_date["period"] >= 240]  # IMPORTANT disregard ipo stocks

        trading_stocks = len(df_date)
        df_date_mean = df_date.mean()
        df_date_mean["trading"] = trading_stocks
        df_date_mean["trade_date"] = trade_date

        df_date_mean["up_limit"] = len(df_date[df_date["pct_chg"] >= 8.0])
        df_date_mean["down_limit"] = len(df_date[df_date["pct_chg"] <= -8.0])
        df_date_mean["net_limit_ratio"] = (df_date_mean["up_limit"] - df_date_mean["down_limit"]) / trading_stocks

        df_date_mean["winner"] = len(df_date[df_date["pct_chg"] > 0]) / trading_stocks
        df_date_mean["loser"] = len(df_date[df_date["pct_chg"] < 0]) / trading_stocks
        df_date_mean["beat_sh_index"] = len(df_date[df_date["pct_chg"] >= sh_pct_chg]) / trading_stocks

        a_result.append(df_date_mean)

    # array to dataframe
    df_result = pd.DataFrame(a_result)

    # if small update, append new data to old data
    if (not df_saved.empty) and (not big_update):
        df_result = df_saved.append(df_result, sort=False)

    # add comp chg and index
    df_result["comp_chg"] = ICreate.column_add_comp_chg(df_result["pct_chg"])
    for ts_code in comparison_index:
        df_result = add_asset_comparison(df=df_result, freq=freq, asset="I", ts_code=ts_code, a_compare_label=["open", "high", "low", "close", "pct_chg"])
        df_result[f"comp_chg_{ts_code}"] = ICreate.column_add_comp_chg(df_result[f"pct_chg_{ts_code}"])

    LB.to_csv_feather(df_result, a_path, index_relevant=False)
    print("Date_Base UPDATED")


@LB.deco_except_empty_df
def get_file(path):
    return pd.read_csv(path)


def get(a_path=[], set_index=""):  # read feather first
    for counter, func in [(1, pd.read_feather), (0, pd.read_csv)]:
        try:
            df = func(a_path[counter])
            if set_index:
                if set_index in ["trade_date", "cal_date", "end_date"]:
                    df[set_index] = df[set_index].astype(int)
                df.set_index(keys=set_index, drop=True, inplace=True)
            return df
        except Exception as e:
            pass  # print(f"read error {func.__name__}", e)
    else:
        print("DB READ File Not Exist!", f"{a_path[0]}.feather")
        return pd.DataFrame()


def get_ts_code(a_asset=["E"], market="CN", d_queries={}):
    """d_query contains only entries that are TRUE. e.g. {"E": ["industry1 == '医疗设备'", "period > 240 "]}"""
    a_result = []
    for asset in a_asset:
        df = get(LB.a_path(f"Market/{market}/General/ts_code_{asset}"), set_index="ts_code")
        if (asset == "FD"):
            df = df[df["delist_date"].isna()]
            # df = df[df["type"]=="契约型开放式"] #契约型开放式 and 契约型封闭式 都可以买 在线交易，封闭式不能随时赎回，但是可以在二级市场上专卖。 开放式更加资本化，发展的好可以扩大盘面，发展的不好可以随时赎回。所以开放式的盘面大小很重要。越大越稳重
            df = df[df["market"] == "E"]  # for now, only consider Equity market traded funds

        if d_queries:
            a_queries = d_queries[asset]
            for query in a_queries:
                df = df.query(query)
        a_result.append(df)
    return pd.concat(a_result, sort=False)


def get_asset(ts_code="000002.SZ", asset="E", freq="D", market="CN"):
    return get(LB.a_path(f"Market/{market}/Asset/{asset}/{freq}/{ts_code}"), set_index="trade_date")


# should be fully replaced by get asset by now. Waiting for bug confirmation
# def get_group_instance(ts_code="asset_E", market="CN", freq="D"):
#     return get(LB.a_path(f"Market/{market}/Asset/G/{freq}/{ts_code}"), set_index="trade_date")


def get_assets_E_D_Fun(query, ts_code, columns=["end_date"], market="CN"):
    df = get(LB.a_path(f"Market/{market}/Asset/E/D_Fun/{query}/{ts_code}"), set_index="end_date")
    if df.empty:
        print("Error get_assets_E_D_Fun ", query, "not exist for", ts_code)
        return LB.empty_df(query)[columns]
    else:
        df = df[~df.index.duplicated(keep="last")]
        return df[columns]


def get_assets_pledge_stat(ts_code, columns, market="CN"):
    df = get(LB.a_path(f"Market/{market}/Asset/E/W_pledge_stat/{ts_code}"), set_index="end_date")
    if df.empty:
        print("Error get_assets_pledge_stat not exist for", ts_code)
        df = LB.empty_df("pledge_stat")
    df = df[~df.index.duplicated(keep="last")]
    return df[columns]


def get_assets_top_holder(ts_code, columns, market="CN"):
    df = get(LB.a_path(f"Market/{market}/Asset/E/D_top_holder/{ts_code}"), set_index="end_date")
    if df.empty:
        print("Error get_assets_top_holder not exist for", ts_code)
        df = LB.empty_df("top_holder")
    return df[columns]


def get_trade_date(start_date="000000", end_date=today(), freq="D", market="CN"):
    df = get(LB.a_path(f"Market/{market}/General/trade_date_{freq}"), set_index="trade_date")
    return df[(df.index >= int(start_date)) & (df.index <= int(end_date))]


def get_trade_cal_D(start_date="19900101", end_date="88888888", a_is_open=[1]):
    df = get(LB.a_path("Market/CN/General/trade_cal_D"), set_index="cal_date")
    df.index.name = "trade_date"
    return df[(df["is_open"].isin(a_is_open)) & (df.index >= int(start_date)) & (df.index <= int(end_date))]


def get_last_trade_date(freq="D", market="CN", type=str):
    df_trade_date = get_trade_date(start_date="00000000", end_date=LB.today(), freq=freq, market=market)
    return type(df_trade_date.index[-1])


def get_next_trade_date(freq="D", market="CN"):  # TODO might be wrong
    df = get_trade_cal_D(a_is_open=[1])  # todo next trade date should be set to 17:00 after tushare has released its new data
    last_trade_date = get_last_trade_date(freq, market)
    df = df[df.index > int(last_trade_date)].reset_index()
    return df.at[0, "trade_date"]


def get_stock_market_all(market="CN"):
    return get(LB.a_path(f"Market/{market}/Btest/Setup/Stock_Market/all_stock_market"), set_index="trade_date")


def get_industry_member(level, market="CN"):
    return get(LB.a_path(f"Market/{market}/General/industry_{level}"), set_index="index_code")


def get_industry_index(index, level, market="CN"):
    return get(LB.a_path(f"Market/{market}/General/industry/{level}/{index}"), set_index="con_code")

def get_date(trade_date, a_assets=["E"], freq="D", market="CN"): # might need get_date_all for efficiency
    a_df=[]
    for asset in a_assets:
        a_df.append(get(LB.a_path(f"Market/{market}/Date/{asset}/{freq}/{trade_date}"), set_index="ts_code"))
    return pd.concat(a_df,sort=False) if len(a_df)>1 else a_df[0]

def get_date_E_oth(trade_date, oth_name, market="CN"):
    return get(LB.a_path(f"Market/{market}/Date/E/{oth_name}/{trade_date}"), set_index="")  # nothing to set

# path =["column_name", True]
def to_excel_with_static_data(df_ts_code, path, sort: list = [], asset=["I", "E", "FD"], group_result=True):
    df_ts_code = add_static_data(df_ts_code, assets=asset)
    d_df = {"Overview": df_ts_code}

    # tab group
    if group_result:
        for group_column in LB.c_d_groups(asset).keys():
            try:
                df_groupbyhelper = df_ts_code.groupby(group_column)
                df_group = df_groupbyhelper.mean()
                df_group["count"] = df_groupbyhelper.size()
                print("not bug until here")
                if sort:
                    df_group.sort_values(by=sort[0], ascending=sort[1], inplace=True)
                d_df[group_column] = df_group
            except Exception as e:
                print("error in group results", e)
    LB.to_excel(path=path, d_df=d_df)


# needs to be optimized for speed and efficiency
def add_static_data(df, assets=["E", "I", "FD"], market="CN"):
    df_result = pd.DataFrame()
    for asset in assets:
        df_asset = get_ts_code(a_asset=[asset])
        df_result = df_result.append(df_asset, sort=False, ignore_index=False)
    df.index.name = "ts_code"  # important, otherwise merge will fail
    return pd.merge(df, df_result, how='left', on=["ts_code"], suffixes=[False, False], sort=False)


def add_asset_comparison(df, freq, asset, ts_code, a_compare_label=["open", "high", "low", "close", "pct_chg"]):
    """ require: trade_date
        function: adds another asset close price"""
    d_rename = {column: f"{column}_{ts_code}" for column in a_compare_label}
    df_compare = get_asset(ts_code, asset, freq)[a_compare_label]
    df_compare.rename(columns=d_rename, inplace=True)
    LB.columns_remove(df, [f"{label}_{ts_code}" for label in a_compare_label])
    return pd.merge(df, df_compare, how='left', on=["trade_date"], suffixes=["", ""], sort=False)


def add_asset_final_analysis_rank(df, assets, freq, analysis="bullishness", market="CN"):
    path = f"Market/CN/Backtest_Single/{analysis}/EIFD_D_final.xlsx"
    df_analysis = pd.read_excel(path, sheet_name="Overview")
    final_score_label = ["ts_code"] + [s for s in df_analysis.columns if f"final_{analysis}_rank" in s]
    df_analysis = df_analysis[final_score_label]
    return pd.merge(df, df_analysis, how='left', on=["ts_code"], suffixes=[False, False], sort=False)


# TODO preload also for E, FD, I
def preload(asset="E", step=1, query_df="", period_abv=240, d_queries_ts_code={},reset_index=False):
    """
    query_on_df: filters df_asset/df_date by some criteria. If the result is empty dataframe, it will NOT be included in d_result
    """
    d_result = {}
    df_listing = get_ts_code(a_asset=[asset], d_queries=d_queries_ts_code)[::step] if asset in LB.c_assets_big() else get_trade_date(start_date="20000101")[::step]
    func = get_asset if asset in LB.c_assets_big() else get_date
    kwargs = {"asset": asset} if asset in LB.c_assets_big() else {}

    bar = tqdm(range(len(df_listing)))
    for index, i in zip(df_listing.index, bar):
        bar.set_description(f"{i}: {asset}: {index}")
        try:
            df = func(index, **kwargs)
            if asset in ["E","I","FD"]: #not work for G, F
                df = df[(df["period"] > period_abv)]
            if query_df:
                df = df.query(expr=query_df)
            if df.empty:
                continue
            else:  # only take df that satisfy ALL conditions and is non empty
                d_result[index] = df.reset_index() if reset_index else df
        except Exception as e:
            print("preload exception", e)
    bar.close()
    print(f"really loaded {len(d_result)}")
    return d_result


"""shuld be fully replaced by preload by now"""


# def preload_groups(assets=["E"]):
#     d_result = {}
#     d_group_label_pair = c_groups_dict(assets=assets, a_ignore=["asset", "industry3"])
#
#     bar = tqdm(range(len(d_group_label_pair)))
#     bar.set_description(f"loading groups...")
#     for (group, instance_array),i in zip(d_group_label_pair.items(), bar):
#         for instance in instance_array:
#             d_result[group + "_" + str(instance)] = get_asset(ts_code=group + "_" + str(instance), asset="G")
#     bar.close()
#     return d_result


def update_all_in_one(big_update=False):
    # there are 3 types of update

    # TODO update completely is to delete all folder. Else update smart
    # 0. ALWAYS UPDATE
    # 1. ONLY ON BIG UPDATE: OVERRIDES EVERY THING EVERY TIME
    # 2. ON BOTH BIG AND SMALL UPDATE: OVERRIDES EVERYTHING EVERY TIME
    # 3. SMART: BIG OR SMALL UPDATE DOES NOT MATTER, ALWAYS CHECK IF FILE NEEDS TO BE UPDATED

    big_steps = [1, 2, 3, 5, 7, 9, - 1, -2, -3, -5, -7, -9]
    middle_steps = [1, 2, 3, 5, -1, -2, -3, -5]
    small_steps = [1, 2, -1, -2]

    # 1.0. GENERAL - CAL_DATE
    # update_general_trade_cal()  # always update
    #
    # # 1.1. GENERAL - INDUSTRY
    # for level in c_industry_level():
    #    update_general_industry(level, big_update=True)  # ONLY ON BIG UPDATE

    # 1.2. GENERAL - TOP HOLDER
    # multi_process(func=update_assets_E_top_holder, a_kwargs={"big_update": False}, a_steps=small_steps)  # SMART
    #
    # # 1.3. GENERAL - TS_CODE
    # for asset in c_assets() + ["G","F"]:
    #     update_general_ts_code(asset)  # ALWAYS UPDATE
    #
    # # 1.5. GENERAL - TRADE_DATE
    # for freq in ["D", "W"]:  # Currently only update D and W, because W is needed for pledge stats
    #     update_general_trade_date(freq)  # ALWAYS UPDATE

    # 2.1. ASSET - FUNDAMENTALS
    # multi_process(func=update_assets_E_D_Fun, a_kwargs={"start_date": "00000000", "end_date": today(), "big_update": False}, a_steps=big_steps)  # SMART
    # multi_process(func=update_assets_E_W_pledge_stat, a_kwargs={"start_date": "00000000", "big_update": False}, a_steps=small_steps)  # SMART

    # 2.2. ASSET - DF
    # multi_process(func=update_assets_EIFD_D, a_kwargs={"asset": "I", "freq": "D", "market": "CN", "big_update": False}, a_steps=[1])  # SMART
    #multi_process(func=update_assets_EIFD_D, a_kwargs={"asset": "FD", "freq": "D", "market": "CN", "big_update": False}, a_steps=[1])  # SMART
    # multi_process(func=update_assets_EIFD_D, a_kwargs={"asset": "E", "freq": "D", "market": "CN", "big_update": False}, a_steps=middle_steps)  # SMART
    multi_process(func=update_assets_EIFD_D, a_kwargs={"asset": "F", "freq": "D", "market": "CN", "big_update": False}, a_steps=[1])  # SMART

    # 3.1. DATE - OTH
    # multi_process(func=update_date_E_Oth, a_kwargs={"asset": "E", "freq": "D", "big_update": big_update}, a_steps=[1, -1])  # big: smart decide - small: smart decide

    # 3.2. DATE - DF
    date_step = [-1, 1] if big_update else [-1, 1]
    # multi_process(func=update_date, a_kwargs={"asset": "I", "freq": "D", "market": "CN", "big_update": False}, a_steps=date_step)  # SMART
    # multi_process(func=update_date, a_kwargs={"asset": "FD", "freq": "D", "market": "CN", "big_update": False}, a_steps=date_step)  # SMART
    # multi_process(func=update_date, a_kwargs={"asset": "E", "freq": "D", "market": "CN", "big_update": False}, a_steps=date_step)  # SMART

    # # 3.3. DATE - BASE
    # update_stock_market_all(start_date="19990101", end_date=today(), big_update=big_update, assets=["E"])  # SMART
    #
    # # 3.4. DATE - TREND
    # df = get_stock_market_all()  # ALWAS
    # ICreate.trend(df=df, ibase="close", market_suffix="market.")  # big: override - small: override
    # LB.to_csv_feather(df, a_path=LB.a_path("Market/CN/Btest/Setup/Stock_Market/all_stock_market"))
    #
    # # 4.1. CUSTOM - INDEX
    # update_assets_G_D(big_update=big_update)


# speed order=remove apply and loc, use vectorize where possible
# 1. vectorized
# 2. List comprehension
# 3. apply
# 4. loc
if __name__ == '__main__':
    # TODO update all in one so that it can run from zero to hero in one run
    # TODO make update general ts_code state_company faster and not update very damn time
    pr = cProfile.Profile()
    pr.enable()
    try:
        df_asset = get_asset()
        big_update = False
        #multi_process(func=update_assets_EIFD_D, a_kwargs={"asset": "F", "freq": "D", "market": "CN", "big_update": False}, a_steps=[1])  # SMART

        for asset in ["G"]:
            multi_process(func=update_date, a_kwargs={"asset": asset, "freq": "D", "market": "CN", "big_update": False,"naive":False}, a_steps=[1])  # SMART

        #update_general_ts_code(asset="B")
        #update_all_in_one(big_update=big_update)


    except Exception as e:
        traceback.print_exc()
        print(e)
        LB.sound("error.mp3")
    pr.disable()
    # pr.print_stats(sort='file')

# slice
# a[-1]    # last item in the array
# a[-2:]   # last two items in the array
# a[:-2]   # everything except the last two items
# a[::-1]    # all items in the array, reversed
# a[1::-1]   # the first two items, reversed
# a[:-3:-1]  # the last two items, reversed
# a[-3::-1]  # everything except the last two items, reversed


# '…'

"""excel sheets

detect duplicates  =COUNTIF(A:A, A2)>1

"""
