import pandas as pd
import tushare as ts
import numpy as np
import _API_Tushare
import LB
import os.path
import inspect
from itertools import combinations
import operator
import math
from numba import njit
import traceback
import cProfile
import threading
from scipy.stats import gmean
from tqdm import tqdm
import Alpha
import Atest
import _API_JQ
import ffn

from LB import *

pd.options.mode.chained_assignment = None  # default='warn'
pro = ts.pro_api('c473f86ae2f5703f58eecf9864fa9ec91d67edbc01e3294f6a4f9c32')
ts.set_token("c473f86ae2f5703f58eecf9864fa9ec91d67edbc01e3294f6a4f9c32")


def update_trade_cal(start_date="19900101", end_date="30250101",market="CN"):
    if market=="CN":
        exchange="SSE"
    elif market=="HK":
        exchange="XHKG"

    df = _API_Tushare.my_trade_cal(start_date=start_date, end_date=end_date, exchange=exchange).set_index(keys="cal_date", inplace=False, drop=True)
    LB.to_csv_feather(df, LB.a_path(f"Market/{market}/General/trade_cal_D"))


# measures which day of week/month performs generally better
def update_date_seasonal_stats(group_instance="asset_E"):
    """this needs to be performed first before using the seasonal matrix"""
    path = f"Market/CN/Atest/seasonal/all_date_seasonal_{group_instance}.xlsx"
    pdwriter = pd.ExcelWriter(path, engine='xlsxwriter')

    # perform seasonal stats for all stock market or for some groups only
    df_group = get_stock_market_all().reset_index() if group_instance == "" else get_asset(ts_code=group_instance, asset="G").reset_index()

    # get all different groups
    a_groups = [[LB.get_trade_date_datetime_dayofweek, "dayofweek"],
                [LB.get_trade_date_datetime_d, "dayofmonth"],
                [LB.get_trade_date_datetime_weekofyear, "weekofyear"],
                [LB.get_trade_date_datetime_dayofyear, "dayofyear"],
                [LB.get_trade_date_datetime_m, "monthofyear"],
                [LB.get_trade_date_datetime_s, "seasonofyear"], ]

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


def update_trade_date(freq="D", market="CN"):
    def update_trade_date_stockcount(df, market="CN"):
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

    def update_trade_date_seasonal_score(df_trade_date, freq="D", market="CN"):
        """this requires the date_seasonal matrix to be updated first """
        # get all indicator for each day
        path_indicator = "Market/CN/Atest/seasonal/all_date_seasonal.xlsx"
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


    #here function starts
    if freq in ["D", "W"]:
        a_path = LB.a_path(f"Market/{market}/General/trade_date_{freq}")
        df = get_trade_cal_D(market=market)

        if market=="CN":
            df = update_trade_date_stockcount(df)  # adds E,I,FD count
        # df = update_trade_date_seasonal_score(df, freq, market)  # TODO adds seasonal score for each day
        LB.to_csv_feather(df, a_path, index_relevant=True)


def update_ts_code(asset="E", market="CN", big_update=True):
    print("start update general ts_code ",market, asset)

    #CN MARKET
    if market=="CN":
        if (asset == "E"):
            df = _API_Tushare.my_stockbasic(is_hs="", list_status="L", exchange="").set_index("ts_code")

            # add exchange info for each stock
            #df["exchange"] = ["创业板" if x[0:3] in ["300"] else "中小板" if x[0:3] in ["002"] else "主板" if x[0:2] in ["00", "60"] else float("nan") for x in df.index]

            # add SW industry info for each stock
            for level in c_industry_level():
                df_industry_member = get_ts_code(a_asset=[f"industry{level}"])
                df[f"industry{level}"] = df_industry_member[f"industry{level}"]

            # add concept
            df_concept = get_ts_code(a_asset=["concept"])
            df_grouped_concept = df_concept.groupby("ts_code").agg(lambda column: ", ".join(column))
            df_grouped_concept.to_csv("grouped.csv", encoding='utf-8_sig')
            df["concept"] = df_grouped_concept["concept"]
            df["concept_code"] = df_grouped_concept["code"]

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
            df_SSE = _API_Tushare.my_index_basic(market='SSE')
            df_SZSE = _API_Tushare.my_index_basic(market='SZSE')
            df = df_SSE.append(df_SZSE, sort=False).set_index("ts_code")
        elif (asset == "FD"):
            df_E = _API_Tushare.my_fund_basic(market='E')
            df_O = _API_Tushare.my_fund_basic(market='O')
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
        elif (asset == "F"):
            df = _API_Tushare.my_fx_daily(start_date=get_last_trade_date(freq="D"))
            df["name"] = df["ts_code"]
            print(df)
            df = df.set_index("ts_code")
            df = df.loc[~df.index.duplicated(keep="last")]
            df = df[["name"]]
        elif (asset == "B"):
            # 可转债，相当于股票。可以选择换成股票，也可以选择换成利率。网上信息很少，几乎没人玩
            df = _API_Tushare.my_cb_basic().set_index("ts_code")

            # 中债，国债
            # only yield curve, no daily data
        elif (asset == "industry"):
            for level in c_industry_level():
                # industry member list
                df_member = _API_Tushare.my_index_classify(f"L{level}")
                df_member = df_member.rename(columns={"industry_name": f"industry{level}"}).set_index("index_code")

                # industry instance
                a_df_instances = []
                for index in df_member.index:
                    df_instance = _API_Tushare.my_index_member(index)
                    df_instance.rename(columns={"con_code": "ts_code"}, inplace=True)
                    a_df_instances.append(df_instance)
                df = pd.concat(a_df_instances, sort=False)
                df = pd.merge(df, df_member, how="left", on=["index_code"], suffixes=[False, False], sort=False)
                df = df.set_index("ts_code", drop=True)
                LB.to_csv_feather(df, a_path=LB.a_path(f"Market/{market}/General/ts_code_industry{level}"))
            return
        elif asset == "concept":
            df_member = _API_Tushare.my_concept()
            a_concepts = []
            for code in df_member["code"]:
                df_instance = _API_Tushare.my_concept_detail(id=code)
                a_concepts.append(df_instance)
            df = pd.concat(a_concepts, sort=False)

            # remove column name it is in both df
            df_member_col = [x for x in df_member if x not in df.columns]
            df["code"] = df["id"]
            df = pd.merge(df, df_member[df_member_col], how="left", on=["code"], suffixes=[False, False], sort=False)

            #change name with / to:
            df.rename(columns={"concept_name": "concept"}, inplace=True)
            df["concept"]=df["concept"].str.replace(pat="/",repl="_")
            df = df.set_index("ts_code", drop=True)
            LB.to_csv_feather(df, a_path=LB.a_path(f"Market/{market}/General/ts_code_concept"))
            return

    #HK MARKET
    elif market=="HK":
        if (asset == "E"):
            df = _API_Tushare.my_hk_basic(list_status="L").set_index("ts_code")
            df=df.loc[:, df.columns !="curr_type"]

    #TODO add US market
    else:
        pass


    df["asset"] = asset
    if "list_date" not in df.columns:
        df["list_date"] = np.nan
    df["list_date"] = df["list_date"].fillna(method='ffill') if asset not in ["G", "F"] else np.nan
    a_path = LB.a_path(f"Market/{market}/General/ts_code_{asset}")
    LB.to_csv_feather(df, a_path)


# @LB.deco_only_big_update
# def update_general_industry(level, market="CN", big_update=True):
#     # industry member list
#     df_member = API_Tushare.my_index_classify(f"L{level}")
#     df_member = df_member.rename(columns={"industry_name": f"industry{level}"}).set_index("index_code")
#
#     # industry instance
#     a_df_instances=[]
#     for index in df_member.index:
#         df_instance = API_Tushare.my_index_member(index)
#         df_instance.rename(columns={"con_code":"ts_code"},inplace=True)
#         a_df_instances.append(df_instance)
#     df_instances=pd.concat(a_df_instances,sort=False)
#     df_instances=pd.merge(df_instances,df_member,how="left",  on=["index_code"], suffixes=[False, False],sort=False)
#     df_instances=df_instances.set_index("ts_code",drop=True)
#     LB.to_csv_feather(df_instances, a_path=LB.a_path(f"Market/{market}/General/ts_code_industry{level}"))
#

def update_assets_EIFD_D(asset="E", freq="D", market="CN", step=1, big_update=True):
    def merge_saved_df_helper(df, df_saved):
        df.set_index("trade_date", inplace=True)
        return df_saved.append(df, sort=False)
        # df = LB.df_drop_duplicated_reindex(df, "trade_date")

    def set_index_helper(df):
        df.set_index(keys="trade_date", drop=True, inplace=True)
        df = df.loc[~df.index.duplicated(keep="last")]  # important
        df=df[df.index.notna()]
        df.index = df.index.astype(int)
        return df

    #a_path_empty = LB.a_path(f"Market/{market}/General/ts_code_ignore")

    # init
    df_ts_codes = get_ts_code(a_asset=[asset],market=market)

    #TODO needs to adjusted for other markets like hk and us
    real_latest_trade_date = get_last_trade_date(freq,market=market)

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

            #only for CN
            if market=="CN":
                # 1.2 get adj factor because tushare is too dump to calculate it on its own
                df_adj = _API_Tushare.my_query(api_name='adj_factor', ts_code=ts_code, start_date=start_date, end_date=end_date)
                if df_adj.empty:
                    print(asset, ts_code, freq, start_date, end_date, "has no adj_factor yet. skipp")
                else:
                    latest_adj = df_adj.at[0, "adj_factor"]
                    df[["open", "high", "low", "close"]] = df[["open", "high", "low", "close"]] / latest_adj

                # 2.1 get daily basic
                if complete_new_update:
                    df_fun_1 = _API_Tushare.my_query(api_name="daily_basic", ts_code=ts_code, start_date=start_date, end_date=middle_date)
                    df_fun_2 = _API_Tushare.my_query(api_name="daily_basic", ts_code=ts_code, start_date=middle_date, end_date=end_date)
                    df_fun = df_fun_1.append(df_fun_2, ignore_index=True, sort=False)
                else:
                    df_fun = _API_Tushare.my_query(api_name="daily_basic", ts_code=ts_code, start_date=asset_latest_trade_date, end_date=end_date)

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
                df = df.loc[~df.index.duplicated(keep="last")]  # important

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
                df = merge_saved_df_helper(df=df, df_saved=df_saved)

        elif (asset == "FD" and freq == "D"):
            if complete_new_update:
                df = update_assets_FD_D_reindex_reverse(ts_code, freq, asset, start_date, end_date)
                df = set_index_helper(df)
            else:
                df = update_assets_FD_D_reindex_reverse(ts_code, freq, asset, asset_latest_trade_date, end_date)
                df = merge_saved_df_helper(df=df, df_saved=df_saved)
        elif (asset == "F" and freq == "D"):
            if complete_new_update:
                df = update_assets_F_D_reindex_reverse(ts_code, start_date, end_date)
                df = set_index_helper(df)
            else:
                df = update_assets_F_D_reindex_reverse(ts_code, asset_latest_trade_date, end_date)
                df = merge_saved_df_helper(df=df, df_saved=df_saved)

        # duplicated index ward
        df = df.loc[~df.index.duplicated(keep="first")]

        """the problem above is df(new) has trade date as column. df_saved has trade_date as index right?"""
        # 3. add my derivative indices and save it
        if not df.empty:
            update_assets_EIFD_D_rolling(df=df, asset=asset)
            update_assets_EIFD_D_expanding(df=df, asset=asset)
            update_assets_EIFD_D_point(df=df, asset=asset)
        LB.to_csv_feather(df=df, a_path=a_path)
        print(asset, ts_code, freq, end_date, "UPDATED!", real_latest_trade_date)


# For all Pri indices and derivates. ordered after FILO
def update_assets_EIFD_D_rolling(df, asset, bfreq=c_bfreq()):
    # close pgain fgain
    for rolling_freq in [x for x in LB.BFreq][::-1]:
        Alpha.pgain(df=df, abase="close", freq=rolling_freq)  # past gain includes today = today +yesterday comp_gain
    for rolling_freq in [x for x in LB.BFreq][::-1]:
        Alpha.fgain(df=df, abase="close", freq=rolling_freq)  # future gain does not include today = tomorrow+atomorrow comp_gain

    # open pgain fgain
    for rolling_freq in [LB.BFreq.f1, LB.BFreq.f2, LB.BFreq.f5][::-1]:
        Alpha.pgain(df=df, abase="open", freq=rolling_freq)  # past gain includes today = today +yesterday comp_gain
    for rolling_freq in [LB.BFreq.f1, LB.BFreq.f2, LB.BFreq.f5][::-1]:
        Alpha.fgain(df=df, abase="open", freq=rolling_freq)  # future gain does not include today = tomorrow+atomorrow comp_gain

    # alpha and Beta, lower the better. too slow
    if asset in ["I", "E", "FD"]:
        for ts_code, df_index in Global.d_index.items():
            for freq in [LB.BFreq.f20, LB.BFreq.f60, LB.BFreq.f240][::-1]:
                beta_corr = Alpha.corr(df=df, abase="close", freq=freq, re=Alpha.RE.r, corr_with=ts_code, corr_series=df_index["close"])
                df[f"alpha{freq}_{ts_code}"] = df["pct_chg"] - df[beta_corr] * df_index["pct_chg"]
                df[f"e_alpha{freq}_{ts_code}"]=df[f"alpha{freq}_{ts_code}"].expanding(240).mean()

    # sharp ratio
    for freq in [LB.BFreq.f20, LB.BFreq.f60, LB.BFreq.f240][::-1]:
        sharp_col=Alpha.sharp(df=df,abase="pct_chg",freq=freq)
        df[f"{sharp_col}.m{freq.value}"]=df[sharp_col].rolling(freq.value).mean()

    # gmean
    for freq in [LB.BFreq.f20, LB.BFreq.f60, LB.BFreq.f240][::-1]:
        df[f"gmean{freq}"] = 1 + (df["pct_chg"] / 100)
        df[f"gmean{freq}"] = df[f"gmean{freq}"].rolling(240).apply(my_sharp, raw=False)

    # Testing - 高送转
    #1. total share/ first day share
    #2. difference between today share and tomorrow share
    if "total_share"in df.columns:
        df["norm_total_share"]=df["total_share"]/ df.at[df["total_share"].first_valid_index,"total_share"]
        df["pct_total_share"]=df["pct_total_share"].pct_change()


    #TODO
    #buy if pct_chg.sharp20 is under e_gmean



def update_assets_EIFD_D_point(df, asset, bfreq=c_bfreq()):

    # other alpha indicators
    Alpha.ivola(df=df)  # 0.890578031539917 for 300 loop
    Alpha.period(df=df)  # 0.2 for 300 loop
    Alpha.pjup(df=df)  # 1.0798187255859375 for 300 loop
    Alpha.pjdown(df=df)  # 1.05 independend for 300 loop
    Alpha.co_pct_chg(df=df)

    # Alpha.cdl(df,abase="cdl")  # VERY SLOW. NO WAY AROUND. 120 sec for 300 loop
    # trend support and resistance
    # add trend for individual stocks
    # Alpha.trend(df=df, abase="close")
    # df = support_resistance_horizontal(df_asset=df)

    # macd for strategy
    for sfreq, bfreq in [(5, 10), (10, 20), (240, 300), (300, 500)]:
        # for sfreq, bfreq in LB.custom_pairwise_combination([5, 10, 20, 40, 60, 120, 180, 240, 300, 500], 2):
        if sfreq < bfreq:
            Atest.my_macd(df=df, abase="close", sfreq=sfreq, bfreq=bfreq, type=1, score=10)


def update_assets_EIFD_D_expanding(df, asset):
    """
    for each asset, calculate expanding information. Standalone from actual information to prevent repeated calculation
        EXPANDING
        1. GMEAN
        3. ENTROPY (todo)
        4. VOLATILITY
        5. days above ma5
        """

    a_freqs = [5, 20, 60, 240, 500,750]

    # 1. Geomean.
    # one time calculation
    df["e_gmean"] = 1 + (df["pct_chg"] / 100)
    df["e_gmean"] = df["e_gmean"].expanding(240).apply(gmean, raw=False)

    #sharp ratio
    df["e_sharp"] = df["pct_chg"].expanding(240).apply(LB.my_sharp, raw=False)
    #e_sharp and e_gmean are pretty similar

    # 2. times above ma, bigger better
    for freq in a_freqs:
        # one time calculation
        df[f"hp{freq}"] = Atest.highpass(df["close"], freq)
        df[f"lp{freq}"] = df["close"] - df[f"hp{freq}"]
        df[f"ma{freq}"] = df["close"].rolling(freq).mean()
        df[f"abv_ma{freq}"] = (df["close"] > df[f"ma{freq}"]).astype(int)
        df[f"abv_lp{freq}"] = (df["close"] > df[f"ma{freq}"]).astype(int)
        df[f"rsi{freq}"]=talib.RSI(df["close"],freq)

        # expanding
        df[f"e_abv_ma{freq}"] = df[f"abv_ma{freq}"].expanding(freq).mean()

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

    # volatility pct_ chg, less than better
    df["rapid_down"] = (df["pct_chg"] < -5).astype(int)
    df["e_rapid_down"] = df["rapid_down"].expanding(240).mean()

    # is_max. How long the current price is around the all time high. higher better
    df["e_max"] = df["close"].expanding(240).max()
    df["e_max_pct"] = (df["close"] / df["e_max"]).between(0.9, 1.1).astype(int)
    df["e_max_pct"] = df["e_max_pct"].expanding(240).mean()

    # is_min
    df["e_min"] = df["close"].expanding(240).min()
    df["e_min_pct"] = (df["close"] / df["e_min"]).between(0.9, 1.1).astype(int)
    df["e_min_pct"] = df["e_min_pct"].expanding(240).mean()


def break_tushare_limit_helper(func, kwargs, limit=1000):
    """for some reason tushare only allows fund，forex to be given at max 1000 entries per request"""
    df = func(**kwargs)
    len_df_this = len(df)
    df_last = df
    while len_df_this == limit:  # TODO if this is fixed or add another way to loop
        kwargs["end_date"] = df_last.at[len(df_last) - 1, "trade_date"]
        df_this = func(**kwargs)
        if (df_this.equals(df_last)):
            break
        df = df.append(df_this, sort=False, ignore_index=True).drop_duplicates(subset="trade_date")
        len_df_this = len(df_this)
        df_last = df_this
    return df


# For E,I,FD
def update_assets_EI_D_reindex_reverse(ts_code, freq, asset, start_date, end_date, adj="qfq", market="CN"):
    if ".SZ" in ts_code or ".SH" in ts_code:
        df = _API_Tushare.my_pro_bar(ts_code=ts_code, freq=freq, asset=asset, start_date=str(start_date), end_date=str(end_date), adj=adj)
    elif ".HK" in ts_code:
        df = _API_Tushare.my_hk_daily(ts_code=ts_code, start_date=str(start_date), end_date=str(end_date))


    df = LB.df_reverse_reindex(df)
    LB.columns_remove(df, ["pre_close", "amount", "change"])
    return df


# For F Tested and works
def update_assets_F_D_reindex_reverse(ts_code, start_date, end_date):
    df = break_tushare_limit_helper(func=_API_Tushare.my_fx_daily, kwargs={"ts_code": ts_code, "start_date": f"{start_date}", "end_date": f"{end_date}"}, limit=1000)
    df = LB.df_reverse_reindex(df)
    for column in ["open", "high", "low", "close"]:
        df[column] = (df[f"bid_{column}"] + df[f"ask_{column}"]) / 2
    df["pct_chg"] = df["close"].pct_change() * 100
    return df[["trade_date", "ts_code", "open", "high", "low", "close", "pct_chg", "tick_qty"]]


# For FD D
def update_assets_FD_D_reindex_reverse(ts_code, freq, asset, start_date, end_date, adj=None, market="CN"):
    if ".OF" in str(ts_code):
        # 场外基金

        df = break_tushare_limit_helper(func=_API_Tushare.my_fund_nav, kwargs={"ts_code": ts_code}, limit=1000)
        LB.df_reverse_reindex(df)

        df_helper=pd.DataFrame()
        for ohlc in ["open","high","low","close"]:
            df_helper[ohlc]=df["unit_nav"].astype(int)
        df_helper["trade_date"]=df["ann_date"]
        df_helper["pct_chg"]=df_helper["open"].pct_change()
        df=df_helper
        df = LB.df_reverse_reindex(df)

        # df.to_csv(f"{ts_code}.csv")
        # print(df)
        # df.index=df["trade_date"]
        # LB.columns_remove(df,["trade_date"])
        # df = df.loc[~df.index.duplicated(keep="last")]
        # df=df.loc[df.index.notna()]

        return df

    else:
        #场内基金
        df = break_tushare_limit_helper(func=_API_Tushare.my_fund_daily, kwargs={"ts_code": ts_code, "start_date": f"{start_date}", "end_date": f"{end_date}"}, limit=1000)
        if df.empty:
            print("not good")
        # TODO to be deletet if breaker works
        # df = API_Tushare.my_pro_bar(ts_code=ts_code, freq=freq, asset=asset, start_date=str(start_date), end_date=str(end_date), adj=adj)
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
            df_adj_factor = _API_Tushare.my_query(api_name='fund_adj', ts_code=ts_code, start_date=start_date, end_date=end_date)
            if df_adj_factor.empty:
                print(asset, ts_code, freq, "has no adj_factor, skip")
            else:
                df_adj_factor.index = df_adj_factor.index.astype(int)
                latest_adj = df_adj_factor.at[0, "adj_factor"]
                df_adj_factor.index = df_adj_factor["trade_date"]
                df["adj_factor"] = df_adj_factor["adj_factor"]
                df["adj_factor"] = df["adj_factor"].interpolate()  # interpolate between alues because some dates are missing from tushare

                # df[["open", "high", "low", "close"]] = df[["open", "high", "low", "close"]] * df.adj_factor / latest_adj
                for column in ["open", "high", "low", "close"]:
                    # for debug include 不复权and后复权
                    # df[f"{column}_不复权"]=df[column]
                    # df[f"{column}_后复权"]=df[column] * df["adj_factor"]
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
            df = _API_Tushare.my_pledge_stat(ts_code=ts_code)
            df = LB.df_reverse_reindex(df).set_index("end_date")
            LB.to_csv_feather(df, a_path)
            print(counter, ts_code, "pledge_stat UPDATED")


def update_assets_E_D_top_holder(big_update=True, market="CN", step=1):
    df_ts_codes = _API_Tushare.my_stockbasic(is_hs="", list_status="L", exchange="").set_index("ts_code")
    for counter, ts_code in enumerate(df_ts_codes.index[::step]):
        a_path = LB.a_path(f"Market/{market}/Asset/E/D_top_holder/{ts_code}")
        if os.path.isfile(a_path[1]) and (not big_update):  # skipp if small update or file exist
            print(counter, ts_code, "top_holder Up-to-date")
            continue
        else:  # always update completely new
            df = _API_Tushare.my_query(api_name='top10_holders', ts_code=ts_code, start_date='20190101', end_date=today())  # NO INDEX SET HERE
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

def update_assets_G_D(assets=["E"], big_update=True, step=1):
    """
    add all assets together and then save them as date
    """
    df_ts_code=get_ts_code(a_asset=["E"])
    d_preload=preload(asset="E")
    example_column=get_example_column(asset="E",freq="D", numeric_only=True)

    for group, a_instance in c_d_groups(assets=["E"]).items():
        for instance in a_instance:

            #get member of that instance
            if group=="concept":
                #TODO check in and out concept date
                df_members=df_ts_code[df_ts_code["concept"].str.contains(instance)==True]
            else:
                df_members=df_ts_code[df_ts_code[group]==instance]

            #loop over all members
            df_instance=pd.DataFrame()
            for counter, ts_code in enumerate(df_members.index):
                try:
                    df_asset=d_preload[ts_code]
                except Exception as e:
                    pass # in df_ts_code but not in preload. maybe not met condition of preload like period > 240

                if df_asset.empty:
                    continue

                print(f"creating group: {group,instance}: {counter} - {ts_code} - len {len(df_asset)}")
                df_asset=LB.get_numeric_df(df_asset)
                df_asset["divide_helper"]=1#counts how many assets are there at one day
                df_instance=df_instance.add(df_asset,fill_value=0)

            #save df_instance as asset
            if not df_instance.empty:
                df_instance=df_instance.div(df_instance["divide_helper"],axis=0)
                df_instance=df_instance[example_column] #align columns
                df_instance.insert(0, "ts_code", f"{group}_{instance}")
            LB.to_csv_feather(df_instance, LB.a_path(f"Market/CN/Asset/G/D/{group}_{instance}"))
            print(f"{group}_{instance} UPDATED")


def update_assets_G_D2(assets=["E"], big_update=True,step=1):
    """
    DEPRECATED!

    there are 2 approach to do this
    1. Asset to industry group and then create date

    This approach is using procedure 2. But both approach should produce same result
    Requirement: df_date
    """

    LB.interrupt_start()

    # initialize all group as dict
    d_group_instance_update = {}  # dict of array
    d_group_instance_saved = {}  # dict of array
    for group, a_instance in c_d_groups(assets=assets).items():
        print("what",group, a_instance)
        for instance in a_instance:
            d_group_instance_update[f"{group}_{instance}"] = []
            d_group_instance_saved[f"{group}_{instance}"] = get_asset(f"{group}_{instance}", asset="G")

    # # get last saved trade_date on df_saved
    # last_trade_date = get_last_trade_date("D")
    # example_df = d_group_instance_saved["asset_E"]
    # try:
    #     last_saved_date = example_df.index[-1]
    # except:
    #     last_saved_date = "19990101"
    #
    #
    # # if small update and saved_date==last trade_date
    # if last_trade_date == last_saved_date and (not big_update):
    #     return print("ALL GROUP Up-to-date")
    #
    # #     # initialize trade_date
    df_trade_date = get_trade_date(end_date=today(), freq="D")
    # df_trade_date = df_trade_date[df_trade_date.index > int(last_saved_date)]
    # print("START UPDATE GROUP since", last_saved_date)
    #
    #     # create a dict for concept in advance
    #     d_concept = {}
    #     df_grouped_concept = get_ts_code(a_asset=["concept"]).groupby("concept")
    #     for concept, row in df_grouped_concept:
    #         d_concept[concept] = row.index
    #
    #     # loop over date and get mean
    for trade_date in df_trade_date.index:  # for each day

        if LB.interrupt_confirmed():
            break

        print(trade_date, "Updating GROUP")
        df_date = get_date(trade_date=trade_date, a_assets=assets, freq="D")
        for group, a_instance in c_d_groups(assets=assets).items():  # for each group

            # one stock can have many concept groups. 1:N Relation
            if group == "concept":
                continue
                for instance in a_instance:
                    try:
                        df_date_grouped = df_date.loc[d_concept[instance]]
                    except Exception as e:
                        print(f"{trade_date} {group} {instance}: {e}")
                        continue

                    if df_date_grouped.empty:
                        continue
                    else:
                        row = df_date_grouped.mean()
                        d_group_instance_update[f"{group}_{instance}"].append(row)

            # one stock can only be in one group. 1:1 Relation
            else:
                df_date_grouped = df_date.groupby(by=group).mean()
                for instance, row in df_date_grouped.iterrows():
                    if f"{group}_{instance}" != "asset_G":
                        d_group_instance_update[f"{group}_{instance}"].append(row)


    # save all to df
    for (key, a_update_instance), (key_saved, df_saved) in zip(d_group_instance_update.items(), d_group_instance_saved.items()):
        df_update = pd.DataFrame(a_update_instance)
        if df_update.empty:
            print(key, "EMPTY. CONTINUE")
            continue
        else:
            df_update.set_index(keys="trade_date", drop=True, inplace=True)  # reset index after group

        if not df_saved.empty:
            df_update = pd.concat(objs=[df_saved, df_update], sort=False, ignore_index=True)

        if "ts_code" not in df_update:
            df_update.insert(0, "ts_code", key)
        LB.to_csv_feather(df_update, LB.a_path(f"Market/CN/Asset/G/D/{key}"))
        print(key, "UPDATED")


def update_asset_intraday(asset="I",freq="15m"):
    df_ts_code=get_ts_code(a_asset=[asset])
    df_ts_code=["000001.SH","399001.SZ","399006.SZ"]
    for ts_code in df_ts_code:
        print(f"update intraday {asset,freq,ts_code}")
        a_path = LB.a_path(f"Market/CN/Asset/{asset}/{freq}/{ts_code}")
        if not os.path.isfile(a_path[0]):
            jq_code=LB.ts_code_switcher(ts_code)
            df=_API_JQ.my_get_bars(jq_code=jq_code,freq=freq)
            df["ts_code"]=ts_code
            df=df.set_index("date",drop=True)
            LB.to_csv_feather(df=df,a_path=a_path)


def update_date(asset="E", freq="D", market="CN", big_update=True, step=1):
    """step -1 might be wrong if trade dates and asset are updated seperately. then they will not align
        step 1 always works
        naive: approach always works, but is extremly slow
    """
    if step not in [1, -1]:
        return print("STEP only 1 or -1 !!!!")

    #latest example column
    naive = True if asset == "F" else False
    example_column = get_example_column(asset=asset,freq=freq)

    #init df
    df_static_data = get_ts_code(a_asset=[asset])
    df_trade_dates = get_trade_date("00000000", LB.today(), freq, market=market)

    #init dict
    d_list_date = {ts_code: row["list_date"] for ts_code, row in get_ts_code(a_asset=[asset]).iterrows()}
    d_queries_ts_code = c_G_queries() if asset == "G" else {}
    d_preload = preload(asset=asset, step=1, period_abv=240, d_queries_ts_code=d_queries_ts_code)
    d_lookup_table = {ts_code: (0 if step == 1 else len(df) - 1) for ts_code, df in d_preload.items()}

    for trade_date in df_trade_dates.index[::step]:  # IMPORTANT! do not modify step, otherwise lookup will not work
        a_path = LB.a_path(f"Market/{market}/Date/{asset}/{freq}/{trade_date}")
        a_date_result = []

        # date file exists AND not big_update. If big_update, then always overwrite
        if os.path.isfile(a_path[0]) and (not big_update):

            if naive:  # fallback strategies is the naive approach
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
                        # it is totally normal to have not that day for each df_asset
                        pass
                else:
                    if len(df_asset) == 0:
                        print(f"{trade_date} skip ts_code {ts_code} because len is 0")
                        continue

                    # if asset list date is in future: not IPO yet
                    list_date = d_list_date[ts_code]
                    if type(list_date) in [str, int]:
                        if int(list_date) > int(trade_date):
                            print(f"{trade_date} skip ts_code {ts_code} because IPO in future")
                            continue

                    row_number = d_lookup_table[ts_code]  # lookup table can not be changed while iterating over it.

                    # this df_asset is already at last row. Is finished
                    if step == 1 and row_number > len(df_asset) - 1:
                        continue
                    if step == -1 and row_number < 0:
                        continue

                    if int(df_asset.index[row_number]) == int(trade_date):
                        a_date_result.append(df_asset.loc[trade_date].to_numpy().flatten())
                        d_lookup_table[ts_code] += step
                        # print(f"IN {trade_date} counter {counter}, step {step}, ts_code {ts_code}, len {len(df_asset)}  row number {row_number}")
                    else:
                        pass
                        # print(f"{int(df_asset.index[row_number])}",trade_date)
                        # print(f"OUT {trade_date} counter {counter}, step {step}, ts_code {ts_code}, len {len(df_asset)}  row number {row_number} associated date {int(df_asset.index[row_number])}")

            # create df_date from a_date_result
            df_date = pd.DataFrame(data=a_date_result, columns=example_column)

            # remove duplicate columns that also exist in static data. Then merge
            no_duplicate_cols = df_date.columns.difference(df_static_data.columns)
            df_date = pd.merge(df_date[no_duplicate_cols], df_static_data, how='left', on=["ts_code"], suffixes=["", ""], sort=False).set_index("ts_code")  # add static data
            df_date.insert(loc=0, column='trade_date', value=int(trade_date))

            # create final rank. TODO. make it same as in bullishness
            df_date["dat_bull"] = df_date["e_gmean"].rank(ascending=False) * 0.70 \
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

            df_date["dat_bull2"]= df_date["e_gmean"].rank(ascending=False) * 0.5 \
                              + df_date["e_sharp"].rank(ascending=False) * 0.5

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


def update_assets_stock_market_all(start_date="00000000", end_date=today(), assets=["E"], freq="D", market="CN", comparison_index=["000001.SH", "399001.SZ", "399006.SZ"], big_update=False):
    """ In theory this should be exactly same as Asset_E
    """#TODO make this same as asset_E

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
    a_path = LB.a_path(f"Market/{market}/Asset/G/D/all_stock_market")
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
    df_result["comp_chg"] = Alpha.column_add_comp_chg(df_result["pct_chg"])
    for ts_code in comparison_index:
        df_result = add_asset_comparison(df=df_result, freq=freq, asset="I", ts_code=ts_code, a_compare_label=["open", "high", "low", "close", "pct_chg"])
        df_result[f"comp_chg_{ts_code}"] = Alpha.column_add_comp_chg(df_result[f"pct_chg_{ts_code}"])

    LB.to_csv_feather(df_result, a_path, index_relevant=False)
    print("Date_Base UPDATED")


def update_margin_total():
    df_trade_date=get_trade_date()
    a_result=[]
    for trade_date in df_trade_date.index:
        print(f"update_margin_total {trade_date}")
        a_path = LB.a_path(f"Market/CN/Asset/E_Market/margin_total/{trade_date}")
        if not os.path.isfile(a_path[0]):
            df_margintotal=_API_JQ.my_margin(date=trade_date)
            LB.to_csv_feather(df_margintotal,a_path=a_path)
        else:
            df_margintotal =get(a_path=a_path)
        a_result.append(df_margintotal)

    df_result=pd.concat(a_result, sort=False)
    df_sh=get_asset("000001.SH",asset="I")
    df_sh=LB.ohlc(df_sh)
    for market in [f"XSHE", f"XSHG"]:
        df_market=df_result[df_result["exchange_code"]==market]
        df_market["trade_date"]=df_market["date"].apply(LB.trade_date_switcher)
        df_market["trade_date"]=df_market["trade_date"].astype(int)
        df_market=pd.merge(df_sh,df_market,how="left",on="trade_date",sort=False)
        a_path=LB.a_path(f"Market/CN/Asset/E_Market/margin_total/{market}")
        LB.to_csv_feather(df=df_market,a_path=a_path)


@LB.deco_except_empty_df
def get_file(path):
    return pd.read_excel(path)


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
        if df.empty:
            continue

        if (asset == "FD"):
            df = df[df["delist_date"].isna()]
            # df = df[df["type"]=="契约型开放式"] #契约型开放式 and 契约型封闭式 都可以买 在线交易，封闭式不能随时赎回，但是可以在二级市场上专卖。 开放式更加资本化，发展的好可以扩大盘面，发展的不好可以随时赎回。所以开放式的盘面大小很重要。越大越稳重
            df = df[df["market"] == "E"]#TODO manually change it to E  # for now, only consider Equity market traded funds

        if d_queries:
            a_queries = d_queries[asset]
            for query in a_queries:
                #when query index use name or "index"? A: both are working
                df = df.query(query)
        a_result.append(df)
    if a_result:
        return pd.concat(a_result, sort=False)
    else:
        return pd.DataFrame()


def get_asset(ts_code="000002.SZ", asset="E", freq="D", market="CN"):
    return get(LB.a_path(f"Market/{market}/Asset/{asset}/{freq}/{ts_code}"), set_index="trade_date")


# should be fully replaced by get asset by now. Waiting for bug confirmation
# def get_group_instance(ts_code="asset_E", market="CN", freq="D"):
#     return get(LB.a_path(f"Market/{market}/Asset/G/{freq}/{ts_code}"), set_index="trade_date")

def update_date_news_tushare():
    df_trade_date=get_trade_date()
    for today,tomorrow in LB.custom_pairwise_overlap(list(df_trade_date.index)):
        a_path = LB.a_path(f"Market/CN/Date/News/tushare/{today}")
        print(today)
        if not os.path.isfile(a_path[0]):
            df=_API_Tushare.my_major_news(start_date=f"{LB.trade_date_switcher(today)} 00:00:00",end_date=f"{LB.trade_date_switcher(tomorrow)} 00:00:00")
            if df.empty:
                continue
            LB.to_csv_feather(df=df,a_path=a_path,skip_feather=True)

def update_date_news_tushare_cctv():
    df_trade_date=get_trade_date()
    for today in df_trade_date.index[::-1]:
        a_path = LB.a_path(f"Market/CN/Date/News/tushare_cctv/{today}")
        print(today)
        if not os.path.isfile(a_path[0]):
            df=_API_Tushare.my_cctv_news(date=today)
            if df.empty:
                continue
            LB.to_csv_feather(df=df,a_path=a_path,skip_feather=True)


def update_date_news_jq():
    df_trade_date=get_trade_date()
    for today in df_trade_date.index[::-1]:
        a_path = LB.a_path(f"Market/CN/Date/News/jq/{today}")
        print(today)
        if not os.path.isfile(a_path[0]):
            df=_API_JQ.my_cctv_news(day=LB.trade_date_switcher(today))
            if df.empty:
                continue
            LB.to_csv_feather(df=df,a_path=a_path,skip_feather=True)


def get_date_news(trade_date, api="tushare"):
    return get(LB.a_path(f"Market/CN/Date/News/{api}/{trade_date}"))

def update_date_news_count(api="tushare"):
    df_sh=get_asset("000001.SH",asset="I")
    df_sh=df_sh[["open","high","low","close"]]
    for trade_date in df_sh.index:
        df_news=get_date_news(trade_date, api)
        str_df=df_news.to_string()
        for keyword in ["股市","股份","股份公司","增长","好","涨幅"]:
            df_sh.at[trade_date,keyword]=str_df.count(keyword)/len(str_df)
    a_path=LB.a_path(f"Market/CN/Date/News/{api}/count/count")
    LB.to_csv_feather(df=df_sh,a_path=a_path)





def get_assets_E_D_Fun(query, ts_code, columns=["end_date"], market="CN"):
    df = get(LB.a_path(f"Market/{market}/Asset/E/D_Fun/{query}/{ts_code}"), set_index="end_date")
    if df.empty:
        print("Error get_assets_E_D_Fun ", query, "not exist for", ts_code)
        return LB.empty_df(query)[columns]
    else:
        df = df.loc[~df.index.duplicated(keep="last")]
        return df[columns]


def get_assets_pledge_stat(ts_code, columns, market="CN"):
    df = get(LB.a_path(f"Market/{market}/Asset/E/W_pledge_stat/{ts_code}"), set_index="end_date")
    if df.empty:
        print("Error get_assets_pledge_stat not exist for", ts_code)
        df = LB.empty_df("pledge_stat")
    df = df.loc[~df.index.duplicated(keep="last")]
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


def get_trade_cal_D(start_date="00000000", end_date="88888888", a_is_open=[1], market="CN"):
    df = get(LB.a_path(f"Market/{market}/General/trade_cal_D"), set_index="cal_date")
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
    return get(LB.a_path(f"Market/{market}/Asset/G/D/all_stock_market"), set_index="trade_date")


def get_date(trade_date, a_assets=["E"], freq="D", market="CN"):  # might need get_date_all for efficiency
    a_df = []
    for asset in a_assets:
        a_df.append(get(LB.a_path(f"Market/{market}/Date/{asset}/{freq}/{trade_date}"), set_index="ts_code"))
    return pd.concat(a_df, sort=False) if len(a_df) > 1 else a_df[0]


def get_date_E_oth(trade_date, oth_name, market="CN"):
    return get(LB.a_path(f"Market/{market}/Date/E/{oth_name}/{trade_date}"), set_index="")  # nothing to set

def get_example_column(asset="E",freq="D", numeric_only=False, notna=True):
    # get the latest column of the asset file
    if asset == "E":
        ts_code = "000001.SZ"
    elif asset == "I":
        ts_code = "000001.SH"
    elif asset == "FD":
        ts_code = "150008.SZ"
    elif asset == "F":
        ts_code = "AUDCAD.FXCM"
    elif asset == "G":
        ts_code = "area_安徽"
    else:
        ts_code = "000001.SZ"

    df=get_asset(ts_code, asset, freq)
    if notna:
        df=df.dropna(how="all",axis=1)

    #nummeric only or not
    return list(get_numeric_df(df).columns) if numeric_only else list(df.columns)



# path =["column_name", True]
def to_excel_with_static_data(df_ts_code, path, sort: list = [], a_assets=["I", "E", "FD", "F", "G"], group_result=True, market="CN"):
    df_ts_code = add_static_data(df_ts_code, assets=a_assets,market=market)
    d_df = {"Overview": df_ts_code}

    # tab group
    if group_result:
        for group, a_instance in LB.c_d_groups(a_assets,market=market).items():
            if group=="concept":
                df_group=pd.DataFrame()
                for instance in a_instance:
                    df_instance=df_ts_code[df_ts_code["concept"].str.contains(instance)==True]
                    s=df_instance.mean()
                    s["count"]=len(df_instance)
                    s.name=instance
                    df_group=df_group.append(s,sort=False)
                df_group.index.name="concept"
                df_group=df_group[["count"]+list(LB.get_numeric_df(df_ts_code).columns)]
            else:
                df_groupbyhelper = df_ts_code.groupby(group)
                df_group = df_groupbyhelper.mean()
                if not df_group.empty:
                    df_group["count"] = df_groupbyhelper.size()

            if sort and sort[0] in df_group.columns:
                df_group=df_group.sort_values(by=sort[0], ascending=sort[1])
            d_df[group] = df_group
    LB.to_excel(path=path, d_df=d_df, index=True)


# needs to be optimized for speed and efficiency
def add_static_data(df, assets=["E", "I", "FD"], market="CN"):
    df_result = pd.DataFrame()
    for asset in assets:
        df_asset = get_ts_code(a_asset=[asset],market=market)
        if df_asset.empty:
            continue
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
    path = f"Market/CN/Atest/{analysis}/EIFD_D_final.xlsx"
    df_analysis = pd.read_excel(path, sheet_name="Overview")
    final_score_label = ["ts_code"] + [s for s in df_analysis.columns if f"final_{analysis}_rank" in s]
    df_analysis = df_analysis[final_score_label]
    return pd.merge(df, df_analysis, how='left', on=["ts_code"], suffixes=[False, False], sort=False)


# TODO preload also for E, FD, I
def preload(asset="E", on_asset=True, step=1, query_df="", period_abv=240, d_queries_ts_code={}, reset_index=False):
    """
    query_on_df: filters df_asset/df_date by some criteria. If the result is empty dataframe, it will NOT be included in d_result
    """
    d_result = {}
    df_index = get_ts_code(a_asset=[asset], d_queries=d_queries_ts_code)[::step] if on_asset else get_trade_date(start_date="20000101")[::step]
    func = get_asset if on_asset else get_date
    kwargs = {"asset": asset} if on_asset else {"a_assets":[asset]}

    bar = tqdm(range(len(df_index)))
    for index, i in zip(df_index.index, bar):
        bar.set_description(f"{i}: {asset}: {index}")
        try:
            df = func(index, **kwargs)
            if asset in ["E", "I", "FD"]:  # not work for G, F
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


def update_date_beta_table(a_asset=["E"], a_freqs=[240]):

    """
    This calculates pairwise beta between all stocks for all days
    for all trade_dates
        for all assets
            for all past beta frequencies
    """

    allowed_assets=["E","I","FD","G"]
    for asset in a_asset:
        if asset not in allowed_assets:
            continue

        #get the index
        d_preload=preload(asset=asset,step=1)
        for freq in a_freqs:
            for asset_counter1, (ts_code1,df_asset1) in enumerate(d_preload.items()):
                a_path = LB.a_path(f"{LB.c_root_beta()}Market/CN/Asset/Beta/{asset}/{freq}/{ts_code1}")
                if os.path.isfile(a_path[0]):
                    continue

                df_result = df_asset1[["close"]]
                for asset_counter2, (ts_code2,df_asset2) in enumerate(d_preload.items()):
                    print(f"freq: {freq}. beta: {ts_code1} - {ts_code2}")
                    df_result[ts_code2]=df_result["close"].rolling(freq,min_periods=2).corr(df_asset2["close"])
                LB.to_csv_feather(df=df_result,a_path=a_path,skip_feather=True)

def update_date_beta_table_I(a_freqs=[240]):

    """
    This calculates pairwise beta between Index and all stocks for all days
    for all trade_dates
        for all assets
            for all past beta frequencies
    """

    for asset in ["E"]:

        #get the index
        d_preload_index = preload(asset="I",step=1,d_queries_ts_code=LB.c_I_queries())
        d_preload = preload(asset=asset, step=1)

        for freq in a_freqs:
            for index_counter, (ts_code_index,df_index) in enumerate(d_preload_index.items()):
                a_path = LB.a_path(f"{LB.c_root_beta()}Market/CN/Asset/Beta/I/{freq}/{ts_code_index}")
                if os.path.isfile(a_path[0]):
                    continue

                df_result = df_index[["close"]]
                for asset_counter, (ts_code_asset,df_asset) in enumerate(d_preload.items()):
                    print(f"freq: {freq}. beta: {ts_code_index} - {ts_code_asset}")
                    df_result[ts_code_asset]=df_result["close"].rolling(freq,min_periods=2).corr(df_asset["close"])
                LB.to_csv_feather(df=df_result,a_path=a_path,skip_feather=True)

def intraday_analysis():
    var = 15
    asset="I"
    for ts_code in ["000001.SH","399006.SZ","399001.SZ"]:
        df = pd.read_csv(f"D:\Stock\Market\CN\Asset\{asset}\{var}m/{ts_code}.csv")

        df["pct_chg"] = df["close"].pct_change()

        df["day"] = df["date"].str.slice(0, 10)
        df["intraday"] = df["date"].str.slice(11, 22)
        df["h"] = df["intraday"].str.slice(0, 2)
        df["m"] = df["intraday"].str.slice(3, 5)
        df["s"] = df["intraday"].str.slice(6, 8)

        df_result = pd.DataFrame()
        a_intraday = list(df["intraday"].unique())
        #1.part stats about mean and volatility
        for intraday in a_intraday:
            df_filter = df[df["intraday"] == intraday]
            mean = df_filter["pct_chg"].mean()
            pct_chg_pos=len(df_filter[df_filter["pct_chg"]>0])/len(df_filter)
            pct_chg_neg=len(df_filter[df_filter["pct_chg"]<0])/len(df_filter)
            std = df_filter["pct_chg"].std()
            sharp = mean / std
            df_result.at[intraday, "mean"] = mean
            df_result.at[intraday, "pos"] = pct_chg_pos
            df_result.at[intraday, "neg"] = pct_chg_neg
            df_result.at[intraday, "std"] = std
            df_result.at[intraday, "sharp"] = sharp
        df_result.to_csv(f"intraday{ts_code}.csv")


        #2.part:prediction. first 15 min predict today
        a_results=[]
        for intraday in a_intraday:
            df_day=get_asset(ts_code=ts_code,asset=asset)
            df_filter = df[df["intraday"] == intraday]
            df_filter["trade_date"]=df_filter["day"].apply(LB.trade_date_switcher)
            df_filter["trade_date"]=df_filter["trade_date"].astype(int)
            df_final=pd.merge(LB.ohlc(df=df_day),df_filter,on="trade_date",suffixes=["_d","_15m"],sort=False)

            df_final["pct_chg_d"] = df_final["pct_chg_d"].shift(-1)
            df_final.to_csv(f"intraday_prediction_{ts_code}.csv")

            len_df=len(df_final)

            TT= len(df_final[(df_final["pct_chg_15m"]>0) & (df_final["pct_chg_d"]>0)])/len_df
            TF= len(df_final[(df_final["pct_chg_15m"]>0) & (df_final["pct_chg_d"]<0)])/len_df
            FT= len(df_final[(df_final["pct_chg_15m"]<0) & (df_final["pct_chg_d"]>0)])/len_df
            FF= len(df_final[(df_final["pct_chg_15m"]<0) & (df_final["pct_chg_d"]<0)])/len_df

            #rolling version
            rolling=5
            df_final[f"pct_chg_15m_r{rolling}"]=df_final[f"pct_chg_15m"].rolling(rolling).mean()
            pearson=df_final[f"pct_chg_15m_r{rolling}"].corr(df_final["pct_chg_d"])
            spearman=df_final[f"pct_chg_15m_r{rolling}"].corr(df_final["pct_chg_d"],method="spearman")

            s=pd.Series({"intraday":intraday,"TT":TT,"TF":TF,"FT":FT,"FF":FF,"pearson":pearson,"sparman":spearman})
            a_results.append(s)
        df_predict_result=pd.DataFrame(a_results)
        df_predict_result.to_csv(f"intraday_prediction_result_{ts_code}.csv")

def update_all_in_one_hk(big_update=False):
    #update trade calendar
    update_trade_cal(market="HK")

    #update trade_date
    update_trade_date(market="HK")

    #get all hk ts_code
    update_ts_code(asset="E",market="HK")

    #update all stock assets
    a_steps=[1,2,-1,-2]
    multi_process(func=update_assets_EIFD_D, a_kwargs={"asset": "E", "freq": "D", "market": "HK", "big_update": False}, a_steps=a_steps)  # SMART



def update_all_in_one_cn(big_update=False):
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

    # # 1.5. GENERAL - TRADE_DATE
    # for freq in ["D", "W"]:  # Currently only update D and W, because W is needed for pledge stats
    #     update_trade_date(freq)  # ALWAYS UPDATE

    # 1.2. GENERAL - TOP HOLDER
    # multi_process(func=update_assets_E_top_holder, a_kwargs={"big_update": False}, a_steps=small_steps)  # SMART
    #
    # # 1.3. GENERAL - TS_CODE
    # for asset in ["industry", "concept"] + c_assets() + ["G","F"]:
    #     update_general_ts_code(asset)  # ALWAYS UPDATE
    #

    # 2.1. ASSET - FUNDAMENTALS
    # multi_process(func=update_assets_E_D_Fun, a_kwargs={"start_date": "00000000", "end_date": today(), "big_update": False}, a_steps=big_steps)  # SMART
    # multi_process(func=update_assets_E_W_pledge_stat, a_kwargs={"start_date": "00000000", "big_update": False}, a_steps=small_steps)  # SMART

    # 2.2. ASSET - DF
    # multi_process(func=update_assets_EIFD_D, a_kwargs={"asset": "I", "freq": "D", "market": "CN", "big_update": False}, a_steps=[1])  # SMART
    multi_process(func=update_assets_EIFD_D, a_kwargs={"asset": "FD", "freq": "D", "market": "CN", "big_update": False}, a_steps=[1])  # SMART
    # multi_process(func=update_assets_EIFD_D, a_kwargs={"asset": "E", "freq": "D", "market": "CN", "big_update": False}, a_steps=middle_steps)  # SMART
    # multi_process(func=update_assets_EIFD_D, a_kwargs={"asset": "F", "freq": "D", "market": "CN", "big_update": False}, a_steps=[1])  # SMART


    # 3.1. DATE - OTH
    # multi_process(func=update_date_E_Oth, a_kwargs={"asset": "E", "freq": "D", "big_update": big_update}, a_steps=[1, -1])  # big: smart decide - small: smart decide

    # 3.2. DATE - DF
    date_step = [-1, 1] if big_update else [-1, 1]
    # multi_process(func=update_date, a_kwargs={"asset": "I", "freq": "D", "market": "CN", "big_update": False}, a_steps=date_step)  # SMART
    # multi_process(func=update_date, a_kwargs={"asset": "FD", "freq": "D", "market": "CN", "big_update": False}, a_steps=date_step)  # SMART
    # multi_process(func=update_date, a_kwargs={"asset": "E", "freq": "D", "market": "CN", "big_update": False}, a_steps=date_step)  # SMART

    # # 4.1. CUSTOM - INDEX
    # update concept is very very slow. = Night shift
    # update_assets_G_D(big_update=big_update)

    # # 3.3. DATE - BASE
    #update_assets_stock_market_all(start_date="19990101", end_date=today(), big_update=big_update, assets=["E"])  # SMART



class Global:
    df_asset= get_asset()
    d_index = preload(asset="I", d_queries_ts_code=LB.c_I_queries())


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
        big_update = False

        update_all_in_one_hk()

        #update_date_news_tushare_cctv()
        # for api in ["tushare","jq","tushare_cctv"]:
        #     update_date_news_count(api)
        #update_all_in_one(big_update)
        #update_asset_intraday(asset="I",freq="15m")

        intraday_analysis()
        #update_date_beta_table(a_asset=["G"])

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
